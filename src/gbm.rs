// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! Linear buffers from the render node's own allocator, through Mesa's libgbm.
//!
//! One of the named unsafe modules (CLAUDE.md). It exists for one allocation the GL driver will
//! not make on request: a 2D texture whose storage is guaranteed `LINEAR`. A guest that presents
//! a shared buffer through virtio-gpu KMS needs exactly that -- the kernel driver takes no other
//! modifier -- and a venus context importing the same buffer has to be told its true layout. GL
//! tiles its own textures as it likes, and nothing in GLES asks it not to; GBM takes
//! `GBM_BO_USE_LINEAR` and answers for what it made. The C reaches the same buffer the same way
//! (`vrend_resource_gbm_init`).
//!
//! What leaves this module is a [`crate::dmabuf::Surface`]: a descriptor and the layout GBM
//! reported, which the rest of the tree already knows how to hold, image and lend. The buffer
//! object itself never leaves. It is destroyed as soon as the descriptor is taken, because the
//! descriptor alone keeps the storage alive and a second owner of one allocation is a lifetime
//! question nobody downstream should have to answer.

use crate::dmabuf::{PixelFormat, Surface, buffer_size};
use crate::surface::{Layout, PlaneLayout, PlaneLayouts};
use core::ffi::c_int;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::path::Path;

#[repr(C)]
struct GbmDevice {
    _opaque: [u8; 0],
}

#[repr(C)]
struct GbmBo {
    _opaque: [u8; 0],
}

const GBM_BO_USE_RENDERING: u32 = 1 << 2;
const GBM_BO_USE_LINEAR: u32 = 1 << 4;

#[link(name = "gbm")]
unsafe extern "C" {
    fn gbm_create_device(fd: c_int) -> *mut GbmDevice;
    fn gbm_device_destroy(gbm: *mut GbmDevice);
    fn gbm_device_is_format_supported(gbm: *mut GbmDevice, format: u32, flags: u32) -> c_int;
    fn gbm_bo_create(
        gbm: *mut GbmDevice,
        width: u32,
        height: u32,
        format: u32,
        flags: u32,
    ) -> *mut GbmBo;
    fn gbm_bo_destroy(bo: *mut GbmBo);
    fn gbm_bo_get_fd(bo: *mut GbmBo) -> c_int;
    fn gbm_bo_get_stride(bo: *mut GbmBo) -> u32;
    fn gbm_bo_get_offset(bo: *mut GbmBo, plane: c_int) -> u32;
    fn gbm_bo_get_modifier(bo: *mut GbmBo) -> u64;
    fn gbm_bo_get_plane_count(bo: *mut GbmBo) -> c_int;
}

/// Why a linear buffer was not made. Each is the allocator's answer, not the guest's doing: a
/// guest asking for a shared buffer is asking for something ordinary, and a refusal here means
/// the resource keeps the GL storage it would have had anyway.
#[derive(Debug)]
pub enum NoBuffer {
    /// The device node would not open.
    Open(std::io::Error),
    /// libgbm would not make a device of it.
    Device,
    /// The device cannot render to this format linearly.
    Unsupported { fourcc: u32 },
    /// The allocation itself failed.
    Create,
    /// The buffer has a layout this side cannot hold: more planes than a layout takes, or a
    /// descriptor the kernel reports no size for.
    Layout,
    /// GBM was asked for `LINEAR` and answered with another modifier. A driver contradicting
    /// itself; the buffer is refused rather than described wrongly.
    NotLinear { modifier: u64 },
    /// No descriptor of the buffer.
    Export,
}

impl core::fmt::Display for NoBuffer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            NoBuffer::Open(e) => write!(f, "the render node will not open: {e}"),
            NoBuffer::Device => f.write_str("gbm_create_device failed"),
            NoBuffer::Unsupported { fourcc } => {
                write!(f, "no linear render target in fourcc {fourcc:#010x}")
            }
            NoBuffer::Create => f.write_str("gbm_bo_create failed"),
            NoBuffer::Layout => f.write_str("the buffer's layout cannot be held"),
            NoBuffer::NotLinear { modifier } => {
                write!(f, "asked for LINEAR and given modifier {modifier:#x}")
            }
            NoBuffer::Export => f.write_str("gbm_bo_get_fd failed"),
        }
    }
}

/// A GBM device on one render node, and the node's descriptor it was made from.
pub struct Allocator {
    device: *mut GbmDevice,
    /// Kept open for the device's life: libgbm does not dup it.
    _node: OwnedFd,
}

// SAFETY: a GBM device has no thread affinity; what it must not have is two threads in it at
// once, which `Send` alone does not allow -- a holder that shares one wraps it in a lock. The
// pointer is freed only in `Drop`, which runs once, with no borrow outstanding.
unsafe impl Send for Allocator {}

impl Allocator {
    /// Open a device on `node`, which must be the node the GL driver renders on: a buffer made
    /// on one GPU and imaged on another is a copy at best and a refusal at worst.
    pub fn open(node: &Path) -> Result<Allocator, NoBuffer> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(node)
            .map_err(NoBuffer::Open)?;
        let node = OwnedFd::from(file);
        // SAFETY: a descriptor this value owns and keeps open for as long as the device lives.
        let device = unsafe { gbm_create_device(node.as_raw_fd()) };
        if device.is_null() {
            return Err(NoBuffer::Device);
        }
        Ok(Allocator { device, _node: node })
    }

    /// A linear, renderable buffer of `width` by `height` in `format`, as the descriptor and
    /// layout the rest of the tree holds.
    pub fn linear(
        &self,
        width: u32,
        height: u32,
        format: PixelFormat,
    ) -> Result<Surface, NoBuffer> {
        let fourcc = format.fourcc();
        let flags = GBM_BO_USE_RENDERING | GBM_BO_USE_LINEAR;
        // SAFETY: a device this value made and has not destroyed.
        if unsafe { gbm_device_is_format_supported(self.device, fourcc, flags) } == 0 {
            return Err(NoBuffer::Unsupported { fourcc });
        }
        // SAFETY: as above; the extent is the caller's and GBM refuses one it cannot make.
        let bo = unsafe { gbm_bo_create(self.device, width, height, fourcc, flags) };
        if bo.is_null() {
            return Err(NoBuffer::Create);
        }
        let described = describe(bo, width, height, fourcc);
        // SAFETY: the buffer made above, destroyed once. A descriptor taken of it keeps the
        // storage alive on its own, so nothing that leaves this call depends on the object.
        unsafe { gbm_bo_destroy(bo) };
        described
    }
}

/// The descriptor and layout of a buffer GBM just made.
fn describe(bo: *mut GbmBo, width: u32, height: u32, fourcc: u32) -> Result<Surface, NoBuffer> {
    // SAFETY: a live buffer the caller made and destroys after this returns.
    let modifier = unsafe { gbm_bo_get_modifier(bo) };
    if modifier != crate::surface::DRM_FORMAT_MOD_LINEAR {
        return Err(NoBuffer::NotLinear { modifier });
    }
    // SAFETY: as above.
    let count = unsafe { gbm_bo_get_plane_count(bo) };
    let count = usize::try_from(count).map_err(|_| NoBuffer::Layout)?;
    // SAFETY: as above; the stride is the first plane's, which is the only one a 32-bit RGB
    // format has.
    let pitch = unsafe { gbm_bo_get_stride(bo) };
    let planes: Vec<PlaneLayout> = (0..count)
        .map(|plane| PlaneLayout {
            // SAFETY: as above, for a plane index below the buffer's own count.
            offset: u64::from(unsafe { gbm_bo_get_offset(bo, plane as c_int) }),
            pitch,
        })
        .collect();
    let planes = PlaneLayouts::new(&planes).map_err(|_| NoBuffer::Layout)?;
    // SAFETY: as above. The descriptor returned is a new one the caller owns.
    let raw = unsafe { gbm_bo_get_fd(bo) };
    if raw < 0 {
        return Err(NoBuffer::Export);
    }
    // SAFETY: a descriptor `gbm_bo_get_fd` just made, owned here and nowhere else.
    let fd = unsafe { OwnedFd::from_raw_fd(raw) };
    let alloc_size = buffer_size(std::os::fd::AsFd::as_fd(&fd)).ok_or(NoBuffer::Layout)?;
    Ok(Surface::exported(fd, Layout { width, height, fourcc, modifier, planes, alloc_size }))
}

impl Drop for Allocator {
    fn drop(&mut self) {
        // SAFETY: the device this value made, destroyed once; its node closes after, as the
        // field drops.
        unsafe { gbm_device_destroy(self.device) };
    }
}
