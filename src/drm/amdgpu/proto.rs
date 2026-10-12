// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! amdgpu's native-context wire: the pinned tree's `amdgpu_virtio_proto.h`.

use crate::drm::proto::ReqHeader;
use crate::drm::uapi::AmdgpuInfo;
use bytemuck::{Pod, Zeroable};

/// `enum amdgpu_ccmd`.
pub mod cmd {
    pub const QUERY_INFO: u32 = 1;
    pub const GEM_NEW: u32 = 2;
    pub const BO_VA_OP: u32 = 3;
    pub const CS_SUBMIT: u32 = 4;
    pub const SET_METADATA: u32 = 5;
    pub const BO_QUERY_INFO: u32 = 6;
    pub const CREATE_CTX: u32 = 7;
    pub const RESERVE_VMID: u32 = 8;
    pub const SET_PSTATE: u32 = 9;
    pub const CS_QUERY_FENCE_STATUS: u32 = 10;
}

/// Every request length and response offset is a multiple of this: the protocol's structs are
/// all a multiple of eight long and no more than eight-aligned.
pub const ALIGNMENT: usize = 8;

/// `struct amdgpu_ccmd_rsp`, which leads every answer: the vdrm length, then the result.
pub const RSP_HEADER: usize = 8;
/// Where in an answer its result goes.
pub const RSP_RET: usize = 4;

/// `struct amdvgpu_shmem`: the response region's header.
pub mod shmem {
    /// A count of failures the guest learns of no other way, from commands that have no answer.
    pub const ASYNC_ERROR: usize = 8;
    pub const GTT: usize = 16;
    pub const VRAM: usize = 40;
    pub const VIS_VRAM: usize = 64;
    pub const LEN: usize = 88;
}

/// `struct amdgpu_ccmd_query_info_req`: a `DRM_AMDGPU_INFO`, passed through.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct QueryInfoReq {
    pub hdr: ReqHeader,
    pub info: AmdgpuInfo,
}

const _: () = {
    assert!(size_of::<QueryInfoReq>() == 48);
    assert!(shmem::VIS_VRAM + size_of::<crate::drm::uapi::AmdgpuHeapInfo>() == shmem::LEN);
};
