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
#[allow(unsafe_code)]
pub mod sys;
pub mod uapi;

use crate::guest_mem::GuestMap;
use crate::ids::ContextId;
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
        }
    }
}

/// The native-context root: the backend this host has, its capset, and the contexts open on it.
pub struct Native {
    host: Host,
    contexts: crate::Map<ContextId, context::Context>,
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
    pub fn probe() -> Option<Native> {
        #[cfg(all(target_os = "linux", feature = "drm-amdgpu"))]
        if let Some(p) = amdgpu::Probed::find() {
            return Some(Native { host: Host::Amdgpu(p), contexts: Default::default() });
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

    /// Open a context: a device file of its own, so its GPU address space is its own.
    pub fn context_create(&mut self, id: ContextId, name: String) -> Result<(), Refused> {
        let backend = self.open(&name)?;
        self.contexts.insert(id, context::Context::new(name, backend));
        Ok(())
    }

    #[cfg_attr(
        not(all(target_os = "linux", feature = "drm-amdgpu")),
        expect(unused_variables, reason = "no backend, so nothing to name")
    )]
    fn open(&self, name: &str) -> Result<Box<dyn context::Backend>, Refused> {
        match self.host {
            #[cfg(all(target_os = "linux", feature = "drm-amdgpu"))]
            Host::Amdgpu(ref p) => Ok(Box::new(p.open(name)?)),
        }
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

    pub fn context_name(&self, id: ContextId) -> Option<&str> {
        self.contexts.get(&id).map(|c| c.name.as_str())
    }
}
