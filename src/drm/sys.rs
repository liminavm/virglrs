// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! The kernel interface for native contexts, and the only unsafe code they have.
//!
//! Every call into a DRM driver goes through here, as a typed function over a [`Device`]: the rest
//! of `drm` never sees a raw descriptor, a request number or a pointer the kernel will follow.
//! libdrm is not linked; these are the `ioctl`s libdrm and libdrm_amdgpu make, with the structs
//! from the pinned tree's `drm-uapi` headers (`uapi.rs`).

use super::uapi::AmdgpuInfo;
use std::ffi::{c_char, c_int};
use std::io;
use std::os::fd::{AsRawFd, OwnedFd};
use std::path::{Path, PathBuf};

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
const DRM_IOCTL_AMDGPU_INFO: libc::Ioctl = ioc(WRITE, COMMAND_BASE + 0x05, size_of::<AmdgpuInfo>());

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
    }
}
