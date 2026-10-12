// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! The kernel interface for native contexts, and the only unsafe code they have.
//!
//! Every call into a DRM driver goes through here, as a typed function over a [`Device`]: the rest
//! of `drm` never sees a raw descriptor, a request number or a pointer the kernel will follow.
//! libdrm is not linked; these are the `ioctl`s libdrm and libdrm_amdgpu make, with the structs
//! from the pinned tree's `drm-uapi` headers (`uapi.rs`).

use super::uapi::{AmdgpuGemCreate, AmdgpuGemVa, AmdgpuInfo};
use std::ffi::{c_char, c_int};
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::Arc;

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

const DRM_IOCTL_GEM_CLOSE: libc::Ioctl = ioc(WRITE, 0x09, size_of::<DrmGemClose>());
const DRM_IOCTL_PRIME_HANDLE_TO_FD: libc::Ioctl =
    ioc(READ | WRITE, 0x2d, size_of::<DrmPrimeHandle>());
const DRM_IOCTL_AMDGPU_GEM_CREATE: libc::Ioctl =
    ioc(READ | WRITE, COMMAND_BASE, size_of::<AmdgpuGemCreate>());
/// `union drm_amdgpu_gem_mmap`: a handle in, the mapping's offset in the device file out.
const DRM_IOCTL_AMDGPU_GEM_MMAP: libc::Ioctl = ioc(READ | WRITE, COMMAND_BASE + 0x01, 8);
const DRM_IOCTL_AMDGPU_INFO: libc::Ioctl = ioc(WRITE, COMMAND_BASE + 0x05, size_of::<AmdgpuInfo>());
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
}

impl Device {
    pub fn open(path: &Path) -> io::Result<Device> {
        // `std` opens close-on-exec, which is what a descriptor no child should inherit wants.
        let file = std::fs::OpenOptions::new().read(true).write(true).open(path)?;
        Ok(Device { fd: file.into(), path: path.to_owned() })
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
    pub fn amdgpu_gem_create(self: &Arc<Self>, args: AmdgpuGemCreate) -> io::Result<Gem> {
        let mut arg = args;
        // SAFETY: `arg` is the union's `in` half, as large as the union, and holds no pointer.
        unsafe { self.ioctl(DRM_IOCTL_AMDGPU_GEM_CREATE, &mut arg) }?;
        // The kernel answers in `out`, whose `handle` overlays the first word.
        Ok(Gem { device: Arc::clone(self), handle: arg.bo_size as u32 })
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
        assert_eq!(DRM_IOCTL_AMDGPU_GEM_CREATE as u32, 0xc020_6440);
        assert_eq!(DRM_IOCTL_AMDGPU_GEM_MMAP as u32, 0xc008_6441);
        assert_eq!(DRM_IOCTL_AMDGPU_GEM_VA as u32, 0x4028_6448);
    }
}
