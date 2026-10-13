// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! DRM native contexts: the guest's GPU driver talks to the host kernel's, through this.
//!
//! The design is `docs/native-context.md`. A context's commands are the kernel driver's own
//! operations, framed by the guest; this side checks each request at the trust boundary, runs it
//! against a render-node file of the context's own, and answers through a shared response region.
//!
//! The wire, the capset and the framing are plain data and build everywhere, so their tests run on
//! every host. What reaches a kernel -- `sys.rs` and the backends' devices -- is Linux-only, and
//! on any other host [`Native::probe`] finds nothing.

pub mod amdgpu;
pub mod capset;
pub mod context;
pub mod proto;
pub mod region;
#[cfg(all(target_os = "linux", feature = "drm"))]
pub mod timeline;
/// No backend, so no rings: a type with no values, so code that holds one compiles and none runs.
#[cfg(not(all(target_os = "linux", feature = "drm")))]
pub mod timeline {
    pub enum Timelines {}

    impl Timelines {
        pub fn push(
            &self,
            _ring: crate::ids::RingIdx,
            _sync: Option<std::os::fd::OwnedFd>,
            _fence: crate::ids::FenceId,
        ) {
            match *self {}
        }
    }
}
#[cfg(all(target_os = "linux", feature = "drm"))]
#[allow(unsafe_code)]
pub mod sys;
pub mod uapi;

use crate::guest_mem::GuestMap;
use crate::ids::{BlobId, ContextId, FenceId, ResourceHandle, RingIdx};
use std::sync::Arc;

pub use capset::Capset;

/// Why a native context refused something the guest asked of it.
///
/// Each is the guest's doing. None poisons the context: the C answers a failed request by dropping
/// it and the rest of its submission, and the guest's driver is written against exactly that.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Refused {
    /// A request's length is shorter than its header, longer than what is left of the
    /// submission, or not a multiple of the backend's alignment.
    RequestLength(u32),
    /// A response offset is not a multiple of the backend's alignment.
    ResponseOffset(u32),
    /// Bytes left over after the last whole request.
    Trailing(usize),
    /// A command this backend does not have.
    UnknownCommand(u32),
    /// A command this backend has, but this build does not serve yet.
    Unserved(u32),
    /// An answer was asked for before the guest created the response region.
    NoRegion,
    /// An answer would not fit in the response region at the offset the request named.
    ResponseOutOfBounds { off: u32, len: usize },
    /// The response region's blob was not created mappable, or was created with more.
    RegionFlags(u32),
    /// A second response region; a context has one.
    SecondRegion,
    /// A response region too small for its header, or too large to address.
    RegionSize(u64),
    /// A query asked for more than the C ever answers in one response.
    ResponseTooLarge(u32),
    /// The host could not open a device for the context; the error is the kernel's.
    NoDevice(i32),
    /// A blob id the guest cannot have made: zero, or wider than the protocol's 32 bits.
    BlobId(u64),
    /// A buffer under a blob id that already has one waiting to be bound.
    BlobIdInUse(u64),
    /// No buffer waits under that blob id.
    NoSuchBlob(u64),
    /// A buffer is already bound to that resource in this context.
    ResourceBound(u32),
    /// No buffer is bound to that resource in this context.
    NoBuffer(u32),
    /// A blob larger than the buffer it would publish.
    BlobTooLarge { asked: u64, have: u64 },
    /// A ring index the context has no ring for.
    Ring(u32),
    /// No GPU context has that id in this native context.
    NoGpuContext(u32),
    /// A submission with more chunks than the protocol allows.
    TooManyChunks(u32),
    /// A submission chunk that is out of bounds, misaligned, the wrong size, repeated, or of a
    /// kind no guest sends.
    Chunk(u32),
    /// A field the protocol reserves was not zero, or a flag it does not define was set.
    Reserved(u64),
    /// The kernel refused a command the protocol answers by dropping the submission; the error is
    /// the kernel's.
    Kernel(i32),
}

impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Refused::RequestLength(n) => write!(f, "a request {n} bytes long does not frame"),
            Refused::ResponseOffset(n) => write!(f, "response offset {n:#x} is misaligned"),
            Refused::Trailing(n) => write!(f, "{n} bytes after the last request"),
            Refused::UnknownCommand(c) => write!(f, "no command {c}"),
            Refused::Unserved(c) => write!(f, "command {c} is not served yet"),
            Refused::NoRegion => f.write_str("an answer before the response region exists"),
            Refused::ResponseOutOfBounds { off, len } => {
                write!(f, "a {len}-byte answer at {off:#x} is outside the response region")
            }
            Refused::RegionFlags(flags) => {
                write!(f, "the response region must be mappable and only that, not {flags:#x}")
            }
            Refused::SecondRegion => f.write_str("a second response region"),
            Refused::RegionSize(n) => write!(f, "a response region of {n} bytes"),
            Refused::ResponseTooLarge(n) => write!(f, "a {n}-byte answer is too large"),
            Refused::NoDevice(e) => {
                write!(f, "no device: {}", std::io::Error::from_raw_os_error(*e))
            }
            Refused::BlobId(id) => write!(f, "blob id {id:#x} is not one a guest makes"),
            Refused::BlobIdInUse(id) => write!(f, "blob id {id} already has a buffer waiting"),
            Refused::NoSuchBlob(id) => write!(f, "no buffer waits under blob id {id}"),
            Refused::ResourceBound(r) => write!(f, "resource {r} already has a buffer"),
            Refused::NoBuffer(r) => write!(f, "resource {r} has no buffer in this context"),
            Refused::BlobTooLarge { asked, have } => {
                write!(f, "a {asked}-byte blob of a {have}-byte buffer")
            }
            Refused::Ring(r) => write!(f, "no ring {r}"),
            Refused::NoGpuContext(id) => write!(f, "no GPU context {id}"),
            Refused::TooManyChunks(n) => write!(f, "{n} chunks in one submission"),
            Refused::Chunk(id) => write!(f, "a malformed or unknown chunk {id}"),
            Refused::Reserved(v) => write!(f, "a reserved field or flag is set: {v:#x}"),
            Refused::Kernel(e) => {
                write!(f, "the kernel refused: {}", std::io::Error::from_raw_os_error(*e))
            }
        }
    }
}

/// A buffer a native context made, as a resource holds it: the buffer, its mapping in this process
/// when the VMM asked for one, and the buffer imported into each other context that attached it.
///
/// The owner of all of them: a context only names a buffer, and the name stops resolving when the
/// resource goes. A resource can outlive its context, and its buffer then keeps the context's
/// device file open, which is what a handle into that file needs.
pub struct Buffer {
    #[cfg(all(target_os = "linux", feature = "drm"))]
    object: Arc<Object>,
    /// The same buffer in other contexts' device files.
    #[cfg(all(target_os = "linux", feature = "drm"))]
    imports: Imports,
    /// Made the first time the VMM asks, as the C maps at `resource_map`: most buffers are never
    /// mapped, and one the CPU cannot reach refuses it. A refusal is kept as the answer.
    #[cfg(all(target_os = "linux", feature = "drm"))]
    map: std::sync::OnceLock<Option<sys::Mapping>>,
    /// The dma-buf the guest asked for at the create, when it did.
    #[cfg(all(target_os = "linux", feature = "drm"))]
    dmabuf: Option<std::os::fd::OwnedFd>,
    #[cfg(not(all(target_os = "linux", feature = "drm")))]
    _never: std::convert::Infallible,
}

impl Buffer {
    #[cfg(all(target_os = "linux", feature = "drm"))]
    fn new(object: Arc<Object>, dmabuf: Option<std::os::fd::OwnedFd>) -> Buffer {
        Buffer { object, imports: Imports::default(), map: std::sync::OnceLock::new(), dmabuf }
    }

    /// Where the buffer is mapped in this process, and how far: mapped now if it was not yet.
    /// `None` when the kernel will not map it, which it says once.
    pub fn host_mapping(&self) -> Option<(usize, u64)> {
        #[cfg(all(target_os = "linux", feature = "drm"))]
        return self
            .map
            .get_or_init(|| {
                let len = self.object.extent()?;
                self.object
                    .gem
                    .amdgpu_mmap(len)
                    .inspect_err(|e| {
                        eprintln!("[virglrs] native: a {len}-byte buffer will not map: {e}")
                    })
                    .ok()
            })
            .as_ref()
            .map(|m| (m.addr(), m.len() as u64));
        #[cfg(not(all(target_os = "linux", feature = "drm")))]
        match self._never {}
    }

    /// Whether the host maps the buffer write-combined, so the guest must not read it back
    /// through a cache. Everything but GTT the guest did not ask write-combined is.
    pub fn write_combining(&self) -> bool {
        #[cfg(all(target_os = "linux", feature = "drm"))]
        return self.object.write_combining;
        #[cfg(not(all(target_os = "linux", feature = "drm")))]
        match self._never {}
    }

    /// The dma-buf the guest asked to share the buffer as, lent for as long as the resource.
    pub fn descriptor(&self) -> Option<std::os::fd::BorrowedFd<'_>> {
        #[cfg(all(target_os = "linux", feature = "drm"))]
        return self.dmabuf.as_ref().map(std::os::fd::AsFd::as_fd);
        #[cfg(not(all(target_os = "linux", feature = "drm")))]
        match self._never {}
    }

    /// A dma-buf of the buffer, made now: the C's one-time export of a buffer the guest did not
    /// ask to share at the create. Most of radeonsi's buffers refuse it, in the kernel: a buffer
    /// created always-valid in its context's address space cannot leave it.
    pub fn export(&self) -> std::io::Result<std::os::fd::OwnedFd> {
        #[cfg(all(target_os = "linux", feature = "drm"))]
        return self.object.gem.export();
        #[cfg(not(all(target_os = "linux", feature = "drm")))]
        match self._never {}
    }
}

#[cfg(all(target_os = "linux", feature = "drm"))]
impl Object {
    /// How much memory the buffer has: what the guest asked for, in whole pages, as the kernel
    /// allocates it. The guest's kernel rounds a blob the same way, so this is what a blob of the
    /// buffer may cover and what a mapping of it spans.
    pub fn extent(&self) -> Option<usize> {
        usize::try_from(self.size).ok().and_then(crate::guest_mem::page_round)
    }
}

/// A resource's storage imported into native contexts' device files: at most one copy per file,
/// owned here, so a context's name for one stops resolving when the resource goes.
#[derive(Default)]
pub struct Imports {
    #[cfg(all(target_os = "linux", feature = "drm"))]
    copies: Vec<Arc<Object>>,
}

/// What a context attached to a resource can name: the resource's native buffer, or the dma-buf
/// of storage some other renderer holds, with the copies already imported from it.
pub enum Source<'a> {
    Native(&'a mut Buffer),
    Foreign { imports: &'a mut Imports, dmabuf: std::os::fd::OwnedFd },
}

/// A kernel buffer and what was known of it when it was made. Shared by the context that made it
/// and the resource it was bound to.
#[cfg(all(target_os = "linux", feature = "drm"))]
pub struct Object {
    pub gem: Arc<sys::Gem>,
    /// What the guest asked for, which is what a mapping covers.
    pub size: u64,
    pub write_combining: bool,
    /// What the buffer is counted as, for as long as it lives. An import is the same memory
    /// again, and is not counted twice.
    pub _charge: Option<crate::budget::Charge>,
}

/// The native-context root: the backend this host has, its capset, and the contexts open on it.
pub struct Native {
    host: Host,
    contexts: crate::Map<ContextId, context::Context>,
    /// What the buffers contexts make are counted against.
    #[cfg_attr(
        not(all(target_os = "linux", feature = "drm-amdgpu")),
        expect(dead_code, reason = "no backend makes buffers")
    )]
    budget: Arc<crate::budget::Budget>,
    /// Where each context's waiter retires its fences.
    #[cfg_attr(
        not(all(target_os = "linux", feature = "drm-amdgpu")),
        expect(dead_code, reason = "no backend has rings")
    )]
    fences: crate::fence::Handle,
    /// Whether a context's guest may set its GPU's power state.
    #[cfg_attr(
        not(all(target_os = "linux", feature = "drm-amdgpu")),
        expect(dead_code, reason = "no backend has a power state")
    )]
    pstate: Pstate,
}

/// Who sets a GPU's stable power state. See [`crate::Config::native_pstate`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Pstate {
    /// The host: a guest may read the state, and is refused setting it.
    Host,
    /// The guest, for its own context, as the kernel lets any client.
    Guest,
}

/// The backend found at probe, and how to open a device for a new context.
enum Host {
    #[cfg(all(target_os = "linux", feature = "drm-amdgpu"))]
    Amdgpu(amdgpu::Probed),
}

impl Native {
    /// Find a GPU this build has a backend for, and describe it.
    ///
    /// `None` is a host with no such GPU, or a build or platform with no backend: the renderer
    /// then serves no native context, and the capset is not offered.
    #[cfg_attr(
        not(all(target_os = "linux", feature = "drm-amdgpu")),
        expect(unused_variables, reason = "no backend, so nothing to count")
    )]
    pub fn probe(
        budget: &Arc<crate::budget::Budget>,
        fences: &crate::fence::Handle,
        guest_pstate: bool,
    ) -> Option<Native> {
        #[cfg(all(target_os = "linux", feature = "drm-amdgpu"))]
        if let Some(p) = amdgpu::Probed::find() {
            return Some(Native {
                host: Host::Amdgpu(p),
                contexts: Default::default(),
                budget: Arc::clone(budget),
                fences: fences.clone(),
                pstate: if guest_pstate { Pstate::Guest } else { Pstate::Host },
            });
        }
        None
    }

    /// What the guest is told about the device, in its capset.
    pub fn capset(&self) -> Capset {
        match self.host {
            #[cfg(all(target_os = "linux", feature = "drm-amdgpu"))]
            Host::Amdgpu(ref p) => p.capset,
        }
    }

    /// The device the contexts are opened on, for the init line.
    pub fn describe(&self) -> String {
        match self.host {
            #[cfg(all(target_os = "linux", feature = "drm-amdgpu"))]
            Host::Amdgpu(ref p) => p.describe(),
        }
    }

    /// Open a context: a device file of its own, so its GPU address space is its own, and a
    /// waiter for its rings' fences.
    pub fn context_create(&mut self, id: ContextId, name: String) -> Result<(), Refused> {
        let context = self.open(id, name)?;
        self.contexts.insert(id, context);
        Ok(())
    }

    #[cfg_attr(
        not(all(target_os = "linux", feature = "drm-amdgpu")),
        expect(unused_variables, reason = "no backend, so nothing to name")
    )]
    fn open(&self, id: ContextId, name: String) -> Result<context::Context, Refused> {
        let backend: Box<dyn context::Backend> = match self.host {
            #[cfg(all(target_os = "linux", feature = "drm-amdgpu"))]
            Host::Amdgpu(ref p) => Box::new(p.open(&name, &self.budget, self.pstate)?),
        };
        #[cfg(all(target_os = "linux", feature = "drm"))]
        {
            let timelines = timeline::Timelines::start(id, backend.rings(), self.fences.clone())
                .map_err(|e| Refused::NoDevice(e.raw_os_error().unwrap_or(libc::EIO)))?;
            Ok(context::Context::new(name, backend, Some(timelines)))
        }
        #[cfg(not(all(target_os = "linux", feature = "drm")))]
        #[expect(unreachable_code, reason = "no backend, so nothing reaches here")]
        return Ok(context::Context::new(name, backend, None));
    }

    /// Order `fence` behind `ctx`'s work on `ring`. `Ok(false)` is ring 0, the CPU's, whose
    /// fences have nothing to wait for and are the caller's to retire.
    pub fn submit_fence(
        &mut self,
        ctx: ContextId,
        ring: RingIdx,
        fence: FenceId,
    ) -> Option<Result<bool, Refused>> {
        Some(self.contexts.get_mut(&ctx)?.submit_fence(ring, fence))
    }

    pub fn context_destroy(&mut self, id: ContextId) {
        self.contexts.remove(&id);
    }

    pub fn submit(&mut self, id: ContextId, buf: &[u8]) -> Option<Result<(), Refused>> {
        Some(self.contexts.get_mut(&id)?.submit(buf))
    }

    /// Hand a context its response region; see [`context::Context::attach_region`].
    pub fn attach_region(
        &mut self,
        id: ContextId,
        map: Arc<GuestMap>,
        size: u64,
        flags: u32,
    ) -> Option<Result<(), Refused>> {
        Some(self.contexts.get_mut(&id)?.attach_region(map, size, flags))
    }

    /// Bind the buffer `ctx` made under `blob` to resource `handle`, and hand the resource its
    /// share. `size` and `flags` are the blob's, as the guest asked for it.
    pub fn get_blob(
        &mut self,
        ctx: ContextId,
        handle: ResourceHandle,
        blob: BlobId,
        size: u64,
        flags: u32,
    ) -> Option<Result<Buffer, Refused>> {
        Some(self.contexts.get_mut(&ctx)?.get_blob(handle, blob, size, flags))
    }

    /// `ctx` was attached to `handle`, whose storage is `source`. Storage the context does not
    /// name yet is named, imported onto its device file first when it is not already there.
    pub fn attach(
        &mut self,
        ctx: ContextId,
        handle: ResourceHandle,
        source: Source<'_>,
    ) -> Option<Result<(), Refused>> {
        Some(self.contexts.get_mut(&ctx)?.attach(handle, source))
    }

    /// `ctx` detached `handle`: the guest closed it there. The resource keeps its share.
    pub fn detach(&mut self, ctx: ContextId, handle: ResourceHandle) {
        if let Some(c) = self.contexts.get_mut(&ctx) {
            c.forget(handle);
        }
    }

    pub fn context_name(&self, id: ContextId) -> Option<&str> {
        self.contexts.get(&id).map(|c| c.name.as_str())
    }
}
