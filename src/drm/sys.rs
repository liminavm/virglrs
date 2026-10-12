// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! The kernel interface for native contexts, and the only unsafe code they have.
//!
//! Every call into a DRM driver goes through here, as a typed function over a [`Device`]: the rest
//! of `drm` never sees a raw descriptor, a request number or a pointer the kernel will follow.
//! libdrm is not linked; these are the `ioctl`s libdrm and libdrm_amdgpu make, with the structs
//! from the pinned tree's `drm-uapi` headers (`uapi.rs`).

use super::uapi::{
    AMDGPU_CTX_OP_ALLOC_CTX, AMDGPU_CTX_OP_FREE_CTX, AMDGPU_GEM_METADATA_OP_GET_METADATA,
    AMDGPU_GEM_METADATA_OP_SET_METADATA, AMDGPU_GEM_OP_GET_GEM_CREATE_INFO, AmdgpuBoListEntry,
    AmdgpuBoListIn, AmdgpuCsChunk, AmdgpuCsChunkDep, AmdgpuCsChunkFence, AmdgpuCsChunkIb,
    AmdgpuCsIn, AmdgpuCtxIn, AmdgpuGemCreate, AmdgpuGemMetadata, AmdgpuGemOp, AmdgpuGemVa,
    AmdgpuInfo, AmdgpuVmIn, AmdgpuWaitCsIn, chunk,
};
use super::uapi::{
    AMDGPU_CTX_OP_GET_STABLE_PSTATE, AMDGPU_CTX_OP_SET_STABLE_PSTATE, AMDGPU_VM_OP_RESERVE_VMID,
    AMDGPU_VM_OP_UNRESERVE_VMID,
};
use bytemuck::Pod;
use std::ffi::{c_char, c_int};
use std::io;
use std::os::fd::{AsRawFd, BorrowedFd, FromRawFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

/// `_IOC` as Linux encodes it on every architecture this builds for: direction in the top two
/// bits, then the argument's size, then the driver's letter -- `'d'` for DRM -- and the number.
const fn ioc(dir: u32, nr: u32, size: usize) -> libc::Ioctl {
    ((dir << 30) | ((size as u32) << 16) | ((b'd' as u32) << 8) | nr) as libc::Ioctl
}
const WRITE: u32 = 1;
const READ: u32 = 2;
/// `DRM_COMMAND_BASE`: where the driver-specific requests start.
const COMMAND_BASE: u32 = 0x40;

/// `struct drm_version`. Its pointers are where the kernel copies the driver's strings, so it is
/// built here and nowhere else.
#[repr(C)]
struct DrmVersion {
    version_major: c_int,
    version_minor: c_int,
    version_patchlevel: c_int,
    name_len: usize,
    name: *mut c_char,
    date_len: usize,
    date: *mut c_char,
    desc_len: usize,
    desc: *mut c_char,
}

/// `struct drm_set_client_name`.
#[repr(C)]
struct DrmSetClientName {
    name_len: u64,
    name: u64,
}

const DRM_IOCTL_VERSION: libc::Ioctl = ioc(READ | WRITE, 0x00, size_of::<DrmVersion>());
const DRM_IOCTL_SET_CLIENT_NAME: libc::Ioctl =
    ioc(READ | WRITE, 0xd1, size_of::<DrmSetClientName>());
/// `struct drm_gem_close`.
#[repr(C)]
struct DrmGemClose {
    handle: u32,
    pad: u32,
}

/// `struct drm_prime_handle`.
#[repr(C)]
struct DrmPrimeHandle {
    handle: u32,
    flags: u32,
    fd: i32,
}

/// `struct drm_syncobj_create` and `struct drm_syncobj_destroy`, which share a shape.
#[repr(C)]
struct DrmSyncobjHandleOnly {
    handle: u32,
    flags_or_pad: u32,
}

/// `struct drm_syncobj_handle`.
#[repr(C)]
struct DrmSyncobjHandle {
    handle: u32,
    flags: u32,
    fd: i32,
    pad: u32,
}

/// `DRM_SYNCOBJ_HANDLE_TO_FD_FLAGS_EXPORT_SYNC_FILE`.
const SYNCOBJ_EXPORT_SYNC_FILE: u32 = 1;

const DRM_IOCTL_SYNCOBJ_CREATE: libc::Ioctl =
    ioc(READ | WRITE, 0xbf, size_of::<DrmSyncobjHandleOnly>());
const DRM_IOCTL_SYNCOBJ_DESTROY: libc::Ioctl =
    ioc(READ | WRITE, 0xc0, size_of::<DrmSyncobjHandleOnly>());
const DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD: libc::Ioctl =
    ioc(READ | WRITE, 0xc1, size_of::<DrmSyncobjHandle>());
const DRM_IOCTL_AMDGPU_CTX: libc::Ioctl =
    ioc(READ | WRITE, COMMAND_BASE + 0x02, size_of::<AmdgpuCtxIn>());
const DRM_IOCTL_AMDGPU_CS: libc::Ioctl =
    ioc(READ | WRITE, COMMAND_BASE + 0x04, size_of::<AmdgpuCsIn>());
const DRM_IOCTL_GEM_CLOSE: libc::Ioctl = ioc(WRITE, 0x09, size_of::<DrmGemClose>());
const DRM_IOCTL_PRIME_HANDLE_TO_FD: libc::Ioctl =
    ioc(READ | WRITE, 0x2d, size_of::<DrmPrimeHandle>());
const DRM_IOCTL_PRIME_FD_TO_HANDLE: libc::Ioctl =
    ioc(READ | WRITE, 0x2e, size_of::<DrmPrimeHandle>());
const DRM_IOCTL_AMDGPU_GEM_CREATE: libc::Ioctl =
    ioc(READ | WRITE, COMMAND_BASE, size_of::<AmdgpuGemCreate>());
/// `union drm_amdgpu_gem_mmap`: a handle in, the mapping's offset in the device file out.
const DRM_IOCTL_AMDGPU_GEM_MMAP: libc::Ioctl = ioc(READ | WRITE, COMMAND_BASE + 0x01, 8);
const DRM_IOCTL_AMDGPU_INFO: libc::Ioctl = ioc(WRITE, COMMAND_BASE + 0x05, size_of::<AmdgpuInfo>());
const DRM_IOCTL_AMDGPU_GEM_METADATA: libc::Ioctl =
    ioc(READ | WRITE, COMMAND_BASE + 0x06, size_of::<AmdgpuGemMetadata>());
const DRM_IOCTL_AMDGPU_GEM_OP: libc::Ioctl =
    ioc(READ | WRITE, COMMAND_BASE + 0x10, size_of::<AmdgpuGemOp>());
const DRM_IOCTL_AMDGPU_VM: libc::Ioctl =
    ioc(READ | WRITE, COMMAND_BASE + 0x13, size_of::<AmdgpuVmIn>());
const DRM_IOCTL_AMDGPU_WAIT_CS: libc::Ioctl =
    ioc(READ | WRITE, COMMAND_BASE + 0x09, size_of::<AmdgpuWaitCsIn>());
const DRM_IOCTL_AMDGPU_GEM_VA: libc::Ioctl =
    ioc(WRITE, COMMAND_BASE + 0x08, size_of::<AmdgpuGemVa>());

/// The most a single `DRM_AMDGPU_INFO` answers into: the C's own cap on a guest's query.
pub const MAX_INFO: usize = 1 << 20;

/// A DRM driver's name and version, as `DRM_IOCTL_VERSION` gives them.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Version {
    pub name: String,
    pub major: u32,
    pub minor: u32,
    pub patchlevel: u32,
}

/// An open DRM device file. Closing it is dropping it.
///
/// One per native context: a GPU driver gives each open file its own GPU address space, and a
/// guest context must not share one with another.
pub struct Device {
    fd: OwnedFd,
    path: PathBuf,
    /// The `Gem` that owns each handle in this file. A handle is closed once however many times
    /// the kernel handed it out, and it hands an existing one out again for a buffer the file
    /// already holds; this is how a second import shares the owner instead of becoming another.
    /// Held across every call that makes or closes a handle, so a number cannot be closed and
    /// reissued between the lookup and the use.
    live: Mutex<std::collections::BTreeMap<u32, Weak<Gem>>>,
}

impl Device {
    pub fn open(path: &Path) -> io::Result<Device> {
        // `std` opens close-on-exec, which is what a descriptor no child should inherit wants.
        let file = std::fs::OpenOptions::new().read(true).write(true).open(path)?;
        Ok(Device { fd: file.into(), path: path.to_owned(), live: Mutex::default() })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The driver's name and version.
    pub fn version(&self) -> io::Result<Version> {
        let mut v = DrmVersion {
            version_major: 0,
            version_minor: 0,
            version_patchlevel: 0,
            name_len: 0,
            name: std::ptr::null_mut(),
            date_len: 0,
            date: std::ptr::null_mut(),
            desc_len: 0,
            desc: std::ptr::null_mut(),
        };
        // SAFETY: `v` is a `drm_version` with every length zero, so the kernel writes the lengths
        // and versions into it and copies no string anywhere.
        unsafe { self.ioctl(DRM_IOCTL_VERSION, &mut v) }?;
        let mut name = vec![0u8; v.name_len];
        v.name = name.as_mut_ptr().cast();
        v.date_len = 0;
        v.desc_len = 0;
        // SAFETY: `v.name` points at `name`, which is `v.name_len` bytes long and outlives the
        // call; the other two lengths are zero, so nothing is copied through their pointers.
        unsafe { self.ioctl(DRM_IOCTL_VERSION, &mut v) }?;
        name.truncate(v.name_len.min(name.len()));
        Ok(Version {
            name: String::from_utf8_lossy(&name).into_owned(),
            major: v.version_major as u32,
            minor: v.version_minor as u32,
            patchlevel: v.version_patchlevel as u32,
        })
    }

    /// Name the file for the kernel's per-client accounting (`fdinfo`, debugfs), as the C does
    /// with the guest's context name.
    pub fn set_client_name(&self, name: &str) -> io::Result<()> {
        let mut arg = DrmSetClientName { name_len: name.len() as u64, name: name.as_ptr() as u64 };
        // SAFETY: `name` is a live string of `name_len` bytes the kernel only reads, and it
        // outlives the call.
        unsafe { self.ioctl(DRM_IOCTL_SET_CLIENT_NAME, &mut arg) }
    }

    /// `DRM_AMDGPU_INFO`: ask the driver `query`, with the query's `params`, for an answer of up
    /// to `out.len()` bytes written into `out`.
    pub fn amdgpu_info(&self, query: u32, params: [u32; 4], out: &mut [u8]) -> io::Result<()> {
        assert!(out.len() <= MAX_INFO, "an answer is bounded before it is asked for");
        let mut arg = AmdgpuInfo {
            return_pointer: out.as_mut_ptr() as u64,
            return_size: out.len() as u32,
            query,
            params,
        };
        // SAFETY: the kernel writes at most `return_size` bytes through `return_pointer`, which is
        // `out`: live, exclusively borrowed and exactly that long for the whole call. Whatever
        // `query` and `params` ask, the answer is bounded by that size, which is why a guest's
        // query can be passed through.
        unsafe { self.ioctl(DRM_IOCTL_AMDGPU_INFO, &mut arg) }
    }

    /// `DRM_AMDGPU_GEM_CREATE`: a new buffer, as the guest described it. The kernel checks every
    /// field; nothing here can make it allocate outside this file's own accounting.
    pub fn amdgpu_gem_create(self: &Arc<Self>, args: AmdgpuGemCreate) -> io::Result<Arc<Gem>> {
        let mut live = self.live.lock().expect("the handle set is never held across a panic");
        let mut arg = args;
        // SAFETY: `arg` is the union's `in` half, as large as the union, and holds no pointer.
        unsafe { self.ioctl(DRM_IOCTL_AMDGPU_GEM_CREATE, &mut arg) }?;
        // The kernel answers in `out`, whose `handle` overlays the first word.
        let handle = arg.bo_size as u32;
        // A handle still in the table is still open, so the kernel cannot have issued it again.
        let gem = Arc::new(Gem { device: Arc::clone(self), handle });
        assert!(live.insert(handle, Arc::downgrade(&gem)).is_none(), "a new buffer, a new handle");
        Ok(gem)
    }

    /// `DRM_AMDGPU_VM`: reserve this file a VMID of its own, or give it back.
    pub fn amdgpu_reserve_vmid(&self, reserve: bool) -> io::Result<()> {
        let op = if reserve { AMDGPU_VM_OP_RESERVE_VMID } else { AMDGPU_VM_OP_UNRESERVE_VMID };
        let mut arg = AmdgpuVmIn { op, flags: 0 };
        // SAFETY: the union's `in` half, as large as the union, with no pointer in it.
        unsafe { self.ioctl(DRM_IOCTL_AMDGPU_VM, &mut arg) }
    }

    /// `DRM_AMDGPU_GEM_VA`: map, unmap, clear or replace a range of this file's GPU address
    /// space. `gem` is `None` for an operation on no buffer -- a sparse range, or a clear. The
    /// kernel checks the range against this file's own address space.
    pub fn amdgpu_gem_va(&self, gem: Option<&Gem>, mut va: AmdgpuGemVa) -> io::Result<()> {
        if let Some(g) = gem {
            assert!(std::ptr::eq(&*g.device, self), "a buffer is mapped in its own file");
        }
        va.handle = gem.map_or(0, |g| g.handle);
        // SAFETY: `va` is a `drm_amdgpu_gem_va` and holds no pointer.
        unsafe { self.ioctl(DRM_IOCTL_AMDGPU_GEM_VA, &mut va) }
    }

    /// `DRM_AMDGPU_CTX` `ALLOC_CTX`: a GPU context, the scheduling entity submissions run in.
    pub fn amdgpu_ctx_create(self: &Arc<Self>, priority: i32) -> io::Result<AmdgpuCtx> {
        let mut arg = AmdgpuCtxIn { op: AMDGPU_CTX_OP_ALLOC_CTX, priority, ..Default::default() };
        // SAFETY: the union's `in` half, as large as the union, with no pointer in it.
        unsafe { self.ioctl(DRM_IOCTL_AMDGPU_CTX, &mut arg) }?;
        // The answer's `alloc.ctx_id` overlays the first word.
        Ok(AmdgpuCtx { device: Arc::clone(self), id: arg.op })
    }

    /// A dma-buf made a buffer in this file.
    ///
    /// The kernel answers a buffer this file already has a handle for with that same handle, and
    /// a GEM handle is closed once however many times it was handed out. So a buffer the file
    /// already holds comes back as the `Gem` that owns it, shared; whoever drops it last closes it.
    ///
    /// An owner whose last reference is gone but whose drop has not yet taken the table still
    /// holds an open handle. The new `Gem` takes the handle over, and that drop leaves it alone.
    pub fn import_dmabuf(self: &Arc<Self>, fd: BorrowedFd<'_>) -> io::Result<Arc<Gem>> {
        let mut live = self.live.lock().expect("the handle set is never held across a panic");
        let mut arg = DrmPrimeHandle { handle: 0, flags: 0, fd: fd.as_raw_fd() };
        // SAFETY: a `drm_prime_handle` with no pointer, naming a descriptor borrowed for the call.
        unsafe { self.ioctl(DRM_IOCTL_PRIME_FD_TO_HANDLE, &mut arg) }?;
        if let Some(owner) = live.get(&arg.handle).and_then(Weak::upgrade) {
            return Ok(owner);
        }
        let gem = Arc::new(Gem { device: Arc::clone(self), handle: arg.handle });
        live.insert(arg.handle, Arc::downgrade(&gem));
        Ok(gem)
    }

    /// A syncobj, for a submission to signal.
    pub fn syncobj_create(self: &Arc<Self>) -> io::Result<Syncobj> {
        let mut arg = DrmSyncobjHandleOnly { handle: 0, flags_or_pad: 0 };
        // SAFETY: a `drm_syncobj_create` with no pointer in it.
        unsafe { self.ioctl(DRM_IOCTL_SYNCOBJ_CREATE, &mut arg) }?;
        Ok(Syncobj { device: Arc::clone(self), handle: arg.handle })
    }

    /// `DRM_AMDGPU_CS`: submit `chunks` in `ctx`, and answer the sequence number the kernel gave
    /// the submission.
    ///
    /// The kernel follows a pointer to an array of pointers to chunks, some of which point
    /// further; all of that is built here from the typed chunks and lives until the call returns.
    /// Every buffer, syncobj and context must belong to this file -- a handle is a number in one
    /// file's table, and in another it names something else.
    pub fn amdgpu_cs(&self, ctx: &AmdgpuCtx, chunks: &[Chunk<'_>]) -> io::Result<u64> {
        assert!(std::ptr::eq(&*ctx.device, self), "a context submits in its own file");
        let mine = |d: &Arc<Device>| assert!(std::ptr::eq(&**d, self), "a chunk names this file");
        // What the kernel reads for each chunk, as eight-byte words so every struct in it is
        // aligned. The buffer lists are built first and only read after, so their heap storage
        // does not move once a list's header has pointed at it.
        let entries: Vec<Vec<AmdgpuBoListEntry>> = chunks
            .iter()
            .filter_map(|c| match c {
                Chunk::BoList(list) => Some(
                    list.iter()
                        .map(|&(gem, bo_priority)| {
                            mine(&gem.device);
                            AmdgpuBoListEntry { bo_handle: gem.handle, bo_priority }
                        })
                        .collect(),
                ),
                _ => None,
            })
            .collect();
        let mut lists = entries.iter();
        let data: Vec<(u32, Vec<u64>)> = chunks
            .iter()
            .map(|c| match *c {
                Chunk::Ib(ib) => (chunk::IB, words(std::slice::from_ref(&ib))),
                Chunk::Dependencies(deps) => (chunk::DEPENDENCIES, words(deps)),
                Chunk::Fence { gem, offset } => {
                    mine(&gem.device);
                    let f = AmdgpuCsChunkFence { handle: gem.handle, offset };
                    (chunk::FENCE, words(std::slice::from_ref(&f)))
                }
                Chunk::BoList(_) => {
                    let list = lists.next().expect("one entry list per BO_HANDLES chunk");
                    let head = AmdgpuBoListIn {
                        operation: !0,
                        list_handle: !0,
                        bo_number: list.len() as u32,
                        bo_info_size: size_of::<AmdgpuBoListEntry>() as u32,
                        bo_info_ptr: list.as_ptr() as u64,
                    };
                    (chunk::BO_HANDLES, words(std::slice::from_ref(&head)))
                }
                Chunk::SyncobjOut(sync) => {
                    mine(&sync.device);
                    (chunk::SYNCOBJ_OUT, words(std::slice::from_ref(&sync.handle)))
                }
            })
            .collect();
        let headers: Vec<AmdgpuCsChunk> = data
            .iter()
            .zip(chunks)
            .map(|((id, words), c)| AmdgpuCsChunk {
                chunk_id: *id,
                length_dw: c.length_dw(),
                chunk_data: words.as_ptr() as u64,
            })
            .collect();
        let pointers: Vec<u64> = headers.iter().map(|h| std::ptr::from_ref(h) as u64).collect();
        let mut arg = AmdgpuCsIn {
            ctx_id: ctx.id,
            bo_list_handle: 0,
            num_chunks: u32::try_from(pointers.len()).expect("a bounded chunk count"),
            flags: 0,
            chunks: pointers.as_ptr() as u64,
        };
        // SAFETY: `arg` points at `pointers`, which point at `headers`, which point into `data`
        // and, for a buffer list, on into `entries`. All four are alive and unmoved until this
        // returns, every length was computed from the storage it describes, and the kernel only
        // reads through them.
        unsafe { self.ioctl(DRM_IOCTL_AMDGPU_CS, &mut arg) }?;
        // The answer's `handle` overlays the first eight bytes.
        Ok(u64::from_ne_bytes(bytemuck::bytes_of(&arg)[..8].try_into().expect("eight bytes")))
    }

    /// One `ioctl`, retried when interrupted as libdrm's `drmIoctl` retries it.
    ///
    /// # Safety
    ///
    /// `request` must be a request whose argument is a `T`, and every pointer inside `arg` must be
    /// valid for whatever that request does with it.
    unsafe fn ioctl<T>(&self, request: libc::Ioctl, arg: &mut T) -> io::Result<()> {
        loop {
            // SAFETY: the descriptor is open for as long as `self`, and the caller vouches for the
            // request and its argument.
            let r = unsafe { libc::ioctl(self.fd.as_raw_fd(), request, std::ptr::from_mut(arg)) };
            if r == 0 {
                return Ok(());
            }
            let e = io::Error::last_os_error();
            match e.raw_os_error() {
                Some(libc::EINTR) | Some(libc::EAGAIN) => continue,
                _ => return Err(e),
            }
        }
    }
}

/// A GEM handle in one device file, and the buffer it keeps alive. Closing it is dropping it.
///
/// It holds its device: a handle is a number in one file's table, so it must not outlive the
/// file, and the file must not close while a buffer still needs it.
pub struct Gem {
    device: Arc<Device>,
    handle: u32,
}

impl Gem {
    /// Whether this is a handle in `device`'s file.
    pub fn is_in(&self, device: &Arc<Device>) -> bool {
        Arc::ptr_eq(&self.device, device)
    }

    /// The buffer mapped into this process, `len` bytes from its start, through the fake offset
    /// `DRM_AMDGPU_GEM_MMAP` gives for it in its device file.
    pub fn amdgpu_mmap(&self, len: usize) -> io::Result<Mapping> {
        let mut arg: u64 = self.handle as u64;
        // SAFETY: the argument is the 8-byte union: the handle in its first word on the way in,
        // the offset on the way out. No pointer.
        unsafe { self.device.ioctl(DRM_IOCTL_AMDGPU_GEM_MMAP, &mut arg) }?;
        let offset =
            libc::off_t::try_from(arg).map_err(|_| io::Error::from_raw_os_error(libc::EINVAL))?;
        // SAFETY: a fresh shared mapping at an address the kernel chooses, over the offset the
        // kernel just gave for this buffer in this file. It aliases nothing in this process, and
        // the kernel keeps the buffer alive for as long as it is mapped.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                self.device.fd.as_raw_fd(),
                offset,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        let ptr = NonNull::new(ptr.cast()).expect("mmap answers MAP_FAILED, never null");
        Ok(Mapping { ptr, len })
    }

    /// `DRM_AMDGPU_GEM_METADATA` `SET`: the buffer's tiling and the driver's opaque description
    /// of it, for whoever imports it next. `data` is at most 256 bytes.
    pub fn amdgpu_set_metadata(&self, flags: u64, tiling_info: u64, data: &[u8]) -> io::Result<()> {
        let mut arg = AmdgpuGemMetadata {
            handle: self.handle,
            op: AMDGPU_GEM_METADATA_OP_SET_METADATA,
            flags,
            tiling_info,
            data_size_bytes: data.len() as u32,
            data: [0; 64],
            _pad: 0,
        };
        bytemuck::cast_slice_mut::<u32, u8>(&mut arg.data)[..data.len()].copy_from_slice(data);
        // SAFETY: a `drm_amdgpu_gem_metadata` with no pointer in it.
        unsafe { self.device.ioctl(DRM_IOCTL_AMDGPU_GEM_METADATA, &mut arg) }
    }

    /// The buffer as the kernel describes it: how it was created, and its metadata.
    pub fn amdgpu_query(&self) -> io::Result<(AmdgpuGemCreate, AmdgpuGemMetadata)> {
        let mut meta = AmdgpuGemMetadata {
            handle: self.handle,
            op: AMDGPU_GEM_METADATA_OP_GET_METADATA,
            ..bytemuck::Zeroable::zeroed()
        };
        // SAFETY: a `drm_amdgpu_gem_metadata` with no pointer in it.
        unsafe { self.device.ioctl(DRM_IOCTL_AMDGPU_GEM_METADATA, &mut meta) }?;
        let mut info = AmdgpuGemCreate::default();
        let mut op = AmdgpuGemOp {
            handle: self.handle,
            op: AMDGPU_GEM_OP_GET_GEM_CREATE_INFO,
            value: std::ptr::from_mut(&mut info) as u64,
        };
        // SAFETY: `value` points at `info`, a `drm_amdgpu_gem_create_in` that lives until this
        // returns, which is what the kernel writes for this op.
        unsafe { self.device.ioctl(DRM_IOCTL_AMDGPU_GEM_OP, &mut op) }?;
        Ok((info, meta))
    }

    /// A dma-buf of the buffer, which another device or process can import.
    pub fn export(&self) -> io::Result<OwnedFd> {
        let mut arg = DrmPrimeHandle {
            handle: self.handle,
            flags: (libc::O_CLOEXEC | libc::O_RDWR) as u32,
            fd: -1,
        };
        // SAFETY: a `drm_prime_handle` with no pointer in it; the kernel writes the new fd.
        unsafe { self.device.ioctl(DRM_IOCTL_PRIME_HANDLE_TO_FD, &mut arg) }?;
        // SAFETY: the kernel just made this descriptor for us, and nothing else holds it.
        Ok(unsafe { OwnedFd::from_raw_fd(arg.fd) })
    }
}

impl Drop for Gem {
    fn drop(&mut self) {
        let mut live =
            self.device.live.lock().expect("the handle set is never held across a panic");
        let entry = live.get(&self.handle).expect("an open handle is in the table");
        if !std::ptr::eq(entry.as_ptr(), self) {
            // An import took the handle over while this was on its way out: it is that `Gem`'s
            // to close now.
            return;
        }
        live.remove(&self.handle);
        let mut arg = DrmGemClose { handle: self.handle, pad: 0 };
        // SAFETY: a `drm_gem_close` with no pointer, naming a handle this value owns and nothing
        // else closes.
        if let Err(e) = unsafe { self.device.ioctl(DRM_IOCTL_GEM_CLOSE, &mut arg) } {
            // The handle came from this file and is closed once, here; the kernel refusing is a
            // host invariant broken.
            panic!("GEM handle {} would not close: {e}", self.handle);
        }
    }
}

/// One chunk of a command submission, as the kernel will be told it. Each names what it refers to
/// by the owning type, never by a raw handle.
pub enum Chunk<'a> {
    /// An indirect buffer, by GPU address.
    Ib(AmdgpuCsChunkIb),
    /// Earlier submissions to wait for.
    Dependencies(&'a [AmdgpuCsChunkDep]),
    /// Where the GPU writes the user fence: `offset` bytes into `gem`.
    Fence { gem: &'a Gem, offset: u32 },
    /// The buffers the submission uses, each with its migration priority.
    BoList(&'a [(&'a Gem, u32)]),
    /// A syncobj the submission signals when it completes.
    SyncobjOut(&'a Syncobj),
}

impl Chunk<'_> {
    fn length_dw(&self) -> u32 {
        let bytes = match self {
            Chunk::Ib(_) => size_of::<AmdgpuCsChunkIb>(),
            Chunk::Dependencies(deps) => size_of_val(*deps),
            Chunk::Fence { .. } => size_of::<AmdgpuCsChunkFence>(),
            Chunk::BoList(_) => size_of::<AmdgpuBoListIn>(),
            Chunk::SyncobjOut(_) => size_of::<u32>(),
        };
        u32::try_from(bytes / 4).expect("a chunk the protocol bounded")
    }
}

/// `v`'s bytes in eight-byte words, zero-padded.
fn words<T: Pod>(v: &[T]) -> Vec<u64> {
    let bytes: &[u8] = bytemuck::cast_slice(v);
    let mut out = vec![0u64; bytes.len().div_ceil(8)];
    bytemuck::cast_slice_mut::<u64, u8>(&mut out)[..bytes.len()].copy_from_slice(bytes);
    out
}

/// A GPU context in one device file. Freed when dropped.
pub struct AmdgpuCtx {
    device: Arc<Device>,
    id: u32,
}

impl AmdgpuCtx {
    /// The id the kernel gave it, which is what the guest names it by.
    pub fn id(&self) -> u32 {
        self.id
    }

    /// The context's stable power state: set to `flags` when `set`, and answered either way.
    /// Only these two operations of `DRM_AMDGPU_CTX` are reachable here; the others make and
    /// free contexts, which only this type does.
    pub fn stable_pstate(&self, set: bool, flags: u32) -> io::Result<u32> {
        let op =
            if set { AMDGPU_CTX_OP_SET_STABLE_PSTATE } else { AMDGPU_CTX_OP_GET_STABLE_PSTATE };
        let mut arg = AmdgpuCtxIn { op, flags, ctx_id: self.id, priority: 0 };
        // SAFETY: the union's `in` half, as large as the union, with no pointer in it.
        unsafe { self.device.ioctl(DRM_IOCTL_AMDGPU_CTX, &mut arg) }?;
        // The answer's `pstate.flags` overlays the first word.
        Ok(arg.op)
    }

    /// `DRM_AMDGPU_WAIT_CS`: wait up to `timeout` for submission `seq` on a ring of this
    /// context. `true` while it is still running.
    pub fn wait_cs(
        &self,
        ip_type: u32,
        ip_instance: u32,
        ring: u32,
        seq: u64,
        timeout: Duration,
    ) -> io::Result<bool> {
        let mut now = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        // SAFETY: `now` is a live timespec the call writes.
        let r = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut now) };
        assert_eq!(r, 0, "CLOCK_MONOTONIC is always readable");
        let now = now.tv_sec as u64 * 1_000_000_000 + now.tv_nsec as u64;
        let mut arg = AmdgpuWaitCsIn {
            handle: seq,
            timeout: now.saturating_add(timeout.as_nanos() as u64),
            ip_type,
            ip_instance,
            ring,
            ctx_id: self.id,
        };
        // SAFETY: the union's `in` half, as large as the union, with no pointer in it.
        unsafe { self.device.ioctl(DRM_IOCTL_AMDGPU_WAIT_CS, &mut arg) }?;
        // The answer's `status` overlays `handle`.
        Ok(arg.handle != 0)
    }
}

impl Drop for AmdgpuCtx {
    fn drop(&mut self) {
        let mut arg =
            AmdgpuCtxIn { op: AMDGPU_CTX_OP_FREE_CTX, ctx_id: self.id, ..Default::default() };
        // SAFETY: the union's `in` half with no pointer, naming a context this value owns.
        if let Err(e) = unsafe { self.device.ioctl(DRM_IOCTL_AMDGPU_CTX, &mut arg) } {
            panic!("GPU context {} would not free: {e}", self.id);
        }
    }
}

/// A DRM syncobj in one device file. Destroyed when dropped.
pub struct Syncobj {
    device: Arc<Device>,
    handle: u32,
}

impl Syncobj {
    /// The fence the syncobj holds now, as a sync_file.
    pub fn export_sync_file(&self) -> io::Result<OwnedFd> {
        let mut arg = DrmSyncobjHandle {
            handle: self.handle,
            flags: SYNCOBJ_EXPORT_SYNC_FILE,
            fd: -1,
            pad: 0,
        };
        // SAFETY: a `drm_syncobj_handle` with no pointer; the kernel writes the new fd.
        unsafe { self.device.ioctl(DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD, &mut arg) }?;
        // SAFETY: the kernel just made this descriptor for us, and nothing else holds it.
        Ok(unsafe { OwnedFd::from_raw_fd(arg.fd) })
    }
}

impl Drop for Syncobj {
    fn drop(&mut self) {
        let mut arg = DrmSyncobjHandleOnly { handle: self.handle, flags_or_pad: 0 };
        // SAFETY: a `drm_syncobj_destroy` with no pointer, naming a syncobj this value owns.
        if let Err(e) = unsafe { self.device.ioctl(DRM_IOCTL_SYNCOBJ_DESTROY, &mut arg) } {
            panic!("syncobj {} would not destroy: {e}", self.handle);
        }
    }
}

/// Wait until at least one of `fds` is readable, or in error, and say which are.
pub fn wait_readable(fds: &[BorrowedFd<'_>]) -> io::Result<Vec<bool>> {
    let mut polled: Vec<libc::pollfd> = fds
        .iter()
        .map(|fd| libc::pollfd { fd: fd.as_raw_fd(), events: libc::POLLIN, revents: 0 })
        .collect();
    loop {
        // SAFETY: `polled` is a live array of `polled.len()` pollfds, each naming a descriptor
        // borrowed for the length of this call.
        let r = unsafe { libc::poll(polled.as_mut_ptr(), polled.len() as libc::nfds_t, -1) };
        if r >= 0 {
            break;
        }
        let e = io::Error::last_os_error();
        if e.raw_os_error() != Some(libc::EINTR) {
            return Err(e);
        }
    }
    Ok(polled.iter().map(|p| p.revents != 0).collect())
}

/// A buffer mapped into this process. Unmapped when dropped.
pub struct Mapping {
    ptr: NonNull<u8>,
    len: usize,
}

// SAFETY: the mapping is plain shared memory, valid from any thread for its whole life; this type
// hands out its address and length and never a reference into it.
unsafe impl Send for Mapping {}
// SAFETY: as `Send`.
unsafe impl Sync for Mapping {}

impl Mapping {
    pub fn addr(&self) -> usize {
        self.ptr.as_ptr() as usize
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: exactly the range `mmap` returned, unmapped once.
        let r = unsafe { libc::munmap(self.ptr.as_ptr().cast(), self.len) };
        assert_eq!(r, 0, "munmap of our own mapping: {}", io::Error::last_os_error());
    }
}

/// The render nodes on this host, in minor order -- the order libdrm's `drmOpenWithType` tries
/// them in.
pub fn render_nodes() -> Vec<PathBuf> {
    let Ok(dir) = std::fs::read_dir("/dev/dri") else { return Vec::new() };
    let mut nodes: Vec<(u32, PathBuf)> = dir
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let minor = e.file_name().to_str()?.strip_prefix("renderD")?.parse().ok()?;
            Some((minor, e.path()))
        })
        .collect();
    nodes.sort();
    nodes.into_iter().map(|(_, p)| p).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_numbers_are_the_headers() {
        // From the C headers' macros, as a C compiler on Linux evaluates them.
        assert_eq!(DRM_IOCTL_VERSION as u32, 0xc040_6400);
        assert_eq!(DRM_IOCTL_SET_CLIENT_NAME as u32, 0xc010_64d1);
        assert_eq!(DRM_IOCTL_AMDGPU_INFO as u32, 0x4020_6445);
        assert_eq!(DRM_IOCTL_GEM_CLOSE as u32, 0x4008_6409);
        assert_eq!(DRM_IOCTL_PRIME_HANDLE_TO_FD as u32, 0xc00c_642d);
        assert_eq!(DRM_IOCTL_PRIME_FD_TO_HANDLE as u32, 0xc00c_642e);
        assert_eq!(DRM_IOCTL_AMDGPU_GEM_CREATE as u32, 0xc020_6440);
        assert_eq!(DRM_IOCTL_AMDGPU_GEM_MMAP as u32, 0xc008_6441);
        assert_eq!(DRM_IOCTL_AMDGPU_GEM_VA as u32, 0x4028_6448);
        assert_eq!(DRM_IOCTL_AMDGPU_CTX as u32, 0xc010_6442);
        assert_eq!(DRM_IOCTL_AMDGPU_VM as u32, 0xc008_6453);
        assert_eq!(DRM_IOCTL_AMDGPU_WAIT_CS as u32, 0xc020_6449);
        assert_eq!(DRM_IOCTL_AMDGPU_GEM_METADATA as u32, 0xc120_6446);
        assert_eq!(DRM_IOCTL_AMDGPU_GEM_OP as u32, 0xc010_6450);
        assert_eq!(DRM_IOCTL_AMDGPU_CS as u32, 0xc018_6444);
        assert_eq!(DRM_IOCTL_SYNCOBJ_CREATE as u32, 0xc008_64bf);
        assert_eq!(DRM_IOCTL_SYNCOBJ_DESTROY as u32, 0xc008_64c0);
        assert_eq!(DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD as u32, 0xc010_64c1);
    }
}
