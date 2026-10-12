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

/// `struct amdgpu_ccmd_gem_new_req`: a buffer, to be bound to a resource under `blob_id`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct GemNewReq {
    pub hdr: ReqHeader,
    pub blob_id: u64,
    pub alloc_size: u64,
    pub phys_alignment: u64,
    pub preferred_heap: u32,
    pub pad: u32,
    pub flags: u64,
}

/// `struct amdgpu_ccmd_bo_va_op_req`: a `DRM_AMDGPU_GEM_VA` on a resource's buffer.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct BoVaOpReq {
    pub hdr: ReqHeader,
    pub va: u64,
    pub vm_map_size: u64,
    /// `AMDGPU_VM_PAGE_*`, for the kernel.
    pub flags: u64,
    /// `AMDGPU_CCMD_BO_VA_OP_*`, for the host.
    pub flags2: u64,
    pub offset: u64,
    pub res_id: u32,
    pub op: u32,
}

/// `AMDGPU_CCMD_BO_VA_OP_SPARSE_BO`: the range is mapped to no buffer.
pub const BO_VA_OP_SPARSE_BO: u64 = 1;

/// `struct amdgpu_ccmd_create_ctx_req`: a GPU context, or with [`CREATE_CTX_DESTROY`] the end of
/// one. The first word is the new context's priority, or the id of the one to destroy.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct CreateCtxReq {
    pub hdr: ReqHeader,
    pub priority_or_id: u32,
    pub flags: u32,
}

/// `AMDGPU_CCMD_CREATE_CTX_DESTROY`.
pub const CREATE_CTX_DESTROY: u32 = 1;

/// `struct amdgpu_ccmd_create_ctx_rsp`: the result, then the new context's id.
pub const CREATE_CTX_RSP: usize = 16;
pub const CREATE_CTX_RSP_ID: usize = 8;

/// `struct amdgpu_ccmd_cs_submit_req` up to its payload, which is `num_chunks` [`ChunkDesc`]s and
/// then the chunks' data, each at its descriptor's offset from the end of the descriptors.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct CsSubmitReq {
    pub hdr: ReqHeader,
    pub ctx_id: u32,
    pub num_chunks: u32,
    pub pad: u32,
    pub ring_idx: u32,
}

/// One chunk's place in a submission's payload.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct ChunkDesc {
    pub chunk_id: u16,
    pub length_dw: u16,
    pub offset: u32,
}

/// `struct amdgpu_ccmd_set_metadata_req` up to its metadata, which follows: `size_metadata`
/// bytes of it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct SetMetadataReq {
    pub hdr: ReqHeader,
    pub flags: u64,
    pub tiling_info: u64,
    pub res_id: u32,
    pub size_metadata: u32,
}

/// The most metadata a buffer holds, `sizeof(amdgpu_bo_metadata::umd_metadata)`.
pub const MAX_METADATA: u32 = 256;

/// `struct amdgpu_ccmd_bo_query_info_req`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct BoQueryInfoReq {
    pub hdr: ReqHeader,
    pub res_id: u32,
    pub pad: u32,
}

/// `struct amdgpu_ccmd_bo_query_info_rsp`: libdrm's `amdgpu_bo_info`, padded.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct BoQueryInfoRsp {
    pub len: u32,
    pub ret: i32,
    pub alloc_size: u64,
    pub phys_alignment: u64,
    pub preferred_heap: u32,
    pub pad: u32,
    pub alloc_flags: u64,
    pub flags: u64,
    pub tiling_info: u64,
    pub size_metadata: u32,
    pub umd_metadata: [u32; 64],
    pub pad2: u32,
}

/// `AMDGPU_CCMD_CS_SUBMIT_MAX_NUM_CHUNKS`.
pub const MAX_CHUNKS: u32 = 128;

const _: () = {
    assert!(size_of::<QueryInfoReq>() == 48);
    assert!(size_of::<CreateCtxReq>() == 24);
    assert!(size_of::<CsSubmitReq>() == 32);
    assert!(size_of::<ChunkDesc>() == 8);
    assert!(size_of::<SetMetadataReq>() == 40);
    assert!(size_of::<BoQueryInfoReq>() == 24);
    assert!(size_of::<BoQueryInfoRsp>() == 320);
    assert!(size_of::<GemNewReq>() == 56);
    assert!(size_of::<BoVaOpReq>() == 64);
    assert!(shmem::VIS_VRAM + size_of::<crate::drm::uapi::AmdgpuHeapInfo>() == shmem::LEN);
};
