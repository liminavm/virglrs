// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! The amdgpu backend: radeonsi and RADV in the guest, the host's `amdgpu` kernel driver.

#[cfg(all(target_os = "linux", feature = "drm-amdgpu"))]
mod device;
pub mod ids;
pub mod proto;

#[cfg(all(target_os = "linux", feature = "drm-amdgpu"))]
pub use device::{Amdgpu, Probed};
