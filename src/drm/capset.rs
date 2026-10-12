// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! The native-context capset: which kernel driver the guest is talking to, and what that driver
//! told the host about the GPU. The guest's Mesa configures itself from these bytes, so they have
//! to be the C's, byte for byte -- `struct virgl_renderer_capset_drm` in the pinned `drm_hw.h`.

use super::uapi::{AmdgpuBufferSizeAlignments, AmdgpuGpuInfo};
use bytemuck::{Pod, Zeroable};

/// Which kernel driver a native context speaks to. `VIRTGPU_DRM_CONTEXT_*`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ContextType {
    Amdgpu,
}

impl ContextType {
    fn wire(self) -> u32 {
        match self {
            ContextType::Amdgpu => 2,
        }
    }
}

/// The amdgpu arm of the capset's union.
///
/// `has_vm_always_valid` is a one-bit C bitfield in the header, the low bit of its word on every
/// ABI the C builds for; it is that word here.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct Amdgpu {
    pub address32_hi: u32,
    pub has_vm_always_valid: u32,
    pub alignments: AmdgpuBufferSizeAlignments,
    pub gpu_info: AmdgpuGpuInfo,
    pub marketing_name: [u8; 128],
}

/// `struct virgl_renderer_capset_drm`, as built with the amdgpu backend.
///
/// The union is as wide as its widest arm, and amdgpu's is the widest, so with amdgpu built in the
/// struct is exactly the header plus that arm. The C sizes the union by which backends it was
/// built with, which is why the capset is compared against a C leg built with amdgpu.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct Capset {
    pub wire_format_version: u32,
    pub version_major: u32,
    pub version_minor: u32,
    pub version_patchlevel: u32,
    pub context_type: u32,
    pub pad: u32,
    pub amdgpu: Amdgpu,
}

/// The capset's version, which the C never raised past zero.
pub const VERSION: u32 = 0;

const _: () = {
    assert!(size_of::<Amdgpu>() == 568);
    assert!(size_of::<Capset>() == 592);
    assert!(std::mem::offset_of!(Capset, amdgpu) == 24);
    assert!(std::mem::offset_of!(Amdgpu, gpu_info) == 24);
    assert!(std::mem::offset_of!(Amdgpu, marketing_name) == 440);
};

/// The kernel driver's version, as `DRM_IOCTL_VERSION` reports it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct DriverVersion {
    pub major: u32,
    pub minor: u32,
    pub patchlevel: u32,
}

impl Capset {
    /// The capset for an amdgpu device.
    ///
    /// The marketing name is cut at 127 bytes, so the last byte of the field is always the
    /// terminator the guest reads it up to, as the C's `strncpy` leaves it.
    pub fn amdgpu(
        version: DriverVersion,
        address32_hi: u32,
        alignments: AmdgpuBufferSizeAlignments,
        gpu_info: AmdgpuGpuInfo,
        marketing_name: &str,
    ) -> Capset {
        let mut name = [0u8; 128];
        let bytes = marketing_name.as_bytes();
        let len = bytes.len().min(name.len() - 1);
        name[..len].copy_from_slice(&bytes[..len]);
        Capset {
            // The C leaves amdgpu's at zero, where msm, asahi and i915 set 2.
            wire_format_version: 0,
            version_major: version.major,
            version_minor: version.minor,
            version_patchlevel: version.patchlevel,
            context_type: ContextType::Amdgpu.wire(),
            pad: 0,
            amdgpu: Amdgpu {
                address32_hi,
                // Every kernel this backend opens supports `VM_ALWAYS_VALID`, and the C says so
                // unconditionally.
                has_vm_always_valid: 1,
                alignments,
                gpu_info,
                marketing_name: name,
            },
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_marketing_name_keeps_its_terminator() {
        let long = "x".repeat(300);
        let c = Capset::amdgpu(
            DriverVersion { major: 3, minor: 64, patchlevel: 0 },
            0,
            AmdgpuBufferSizeAlignments::default(),
            AmdgpuGpuInfo::zeroed(),
            &long,
        );
        assert_eq!(&c.amdgpu.marketing_name[..127], &[b'x'; 127][..]);
        assert_eq!(c.amdgpu.marketing_name[127], 0);
    }

    #[test]
    fn the_header_carries_the_driver_and_the_context_type() {
        let c = Capset::amdgpu(
            DriverVersion { major: 3, minor: 64, patchlevel: 1 },
            0xffff_8000,
            AmdgpuBufferSizeAlignments { size_local: 0x1000, size_remote: 0x1000 },
            AmdgpuGpuInfo::zeroed(),
            "AMD Radeon 890M Graphics",
        );
        let b = c.as_bytes();
        let word = |at: usize| u32::from_le_bytes(b[at..at + 4].try_into().unwrap());
        assert_eq!([word(0), word(4), word(8), word(12), word(16)], [0, 3, 64, 1, 2]);
        assert_eq!(word(24), 0xffff_8000);
        assert_eq!(word(28), 1);
        assert_eq!(&b[464..488], b"AMD Radeon 890M Graphics");
        assert_eq!(b[488], 0);
    }
}
