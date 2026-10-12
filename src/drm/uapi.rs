// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! The kernel's structs, laid out as the pinned tree's `src/drm/drm-uapi/` headers lay them out,
//! and libdrm_amdgpu's own where the native-context wire carries one.
//!
//! Plain data, and on every host: nothing here calls the kernel, which is `sys.rs`'s. Every
//! struct is `Pod` with its padding spelled out as fields, so a layout that drifts from the header
//! is a compile error here or a failed size assertion below, never a wrong `ioctl`.

use bytemuck::{Pod, Zeroable};

/// `struct drm_amdgpu_info`, the `DRM_AMDGPU_INFO` argument. `params` is the header's anonymous
/// union, whose widest member (`query_fw`, `read_mmr_reg`) is four words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default, PartialEq, Eq)]
pub struct AmdgpuInfo {
    pub return_pointer: u64,
    pub return_size: u32,
    pub query: u32,
    pub params: [u32; 4],
}

/// `AMDGPU_INFO_*` queries this side issues itself.
pub mod query {
    pub const ACCEL_WORKING: u32 = 0x00;
    pub const HW_IP_INFO: u32 = 0x02;
    pub const VRAM_USAGE: u32 = 0x10;
    pub const GTT_USAGE: u32 = 0x11;
    pub const VRAM_GTT: u32 = 0x14;
    pub const READ_MMR_REG: u32 = 0x15;
    pub const DEV_INFO: u32 = 0x16;
    pub const VIS_VRAM_USAGE: u32 = 0x17;
}

/// `AMDGPU_HW_IP_NUM`: how many hardware IP types there are to ask about.
pub const AMDGPU_HW_IP_NUM: u32 = 10;
/// `AMDGPU_IDS_FLAGS_FUSION`: an APU, whose GPU shares the CPU's package and name.
pub const AMDGPU_IDS_FLAGS_FUSION: u64 = 0x1;
/// `AMDGPU_FAMILY_CI`, the first family with a second raster config register.
pub const AMDGPU_FAMILY_CI: u32 = 120;
/// `AMDGPU_FAMILY_AI` (Vega), the first family whose tiling the kernel describes rather than the
/// registers.
pub const AMDGPU_FAMILY_AI: u32 = 141;

/// `struct drm_amdgpu_info_device`, the answer to `AMDGPU_INFO_DEV_INFO`.
///
/// The kernel writes as much of its own version of this struct as the request has room for, so a
/// kernel newer or older than the header is answered correctly in every field named here.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct AmdgpuDeviceInfo {
    pub device_id: u32,
    pub chip_rev: u32,
    pub external_rev: u32,
    pub pci_rev: u32,
    pub family: u32,
    pub num_shader_engines: u32,
    pub num_shader_arrays_per_engine: u32,
    pub gpu_counter_freq: u32,
    pub max_engine_clock: u64,
    pub max_memory_clock: u64,
    pub cu_active_number: u32,
    pub cu_ao_mask: u32,
    pub cu_bitmap: [[u32; 4]; 4],
    pub enabled_rb_pipes_mask: u32,
    pub num_rb_pipes: u32,
    pub num_hw_gfx_contexts: u32,
    pub pcie_gen: u32,
    pub ids_flags: u64,
    pub virtual_address_offset: u64,
    pub virtual_address_max: u64,
    pub virtual_address_alignment: u32,
    pub pte_fragment_size: u32,
    pub gart_page_size: u32,
    pub ce_ram_size: u32,
    pub vram_type: u32,
    pub vram_bit_width: u32,
    pub vce_harvest_config: u32,
    pub gc_double_offchip_lds_buf: u32,
    pub prim_buf_gpu_addr: u64,
    pub pos_buf_gpu_addr: u64,
    pub cntl_sb_buf_gpu_addr: u64,
    pub param_buf_gpu_addr: u64,
    pub prim_buf_size: u32,
    pub pos_buf_size: u32,
    pub cntl_sb_buf_size: u32,
    pub param_buf_size: u32,
    pub wave_front_size: u32,
    pub num_shader_visible_vgprs: u32,
    pub num_cu_per_sh: u32,
    pub num_tcc_blocks: u32,
    pub gs_vgt_table_depth: u32,
    pub gs_prim_buffer_depth: u32,
    pub max_gs_waves_per_vgt: u32,
    pub pcie_num_lanes: u32,
    pub cu_ao_bitmap: [[u32; 4]; 4],
    pub high_va_offset: u64,
    pub high_va_max: u64,
    pub pa_sc_tile_steering_override: u32,
    pub _pad0: u32,
    pub tcc_disabled_mask: u64,
    pub min_engine_clock: u64,
    pub min_memory_clock: u64,
    pub tcp_cache_size: u32,
    pub num_sqc_per_wgp: u32,
    pub sqc_data_cache_size: u32,
    pub sqc_inst_cache_size: u32,
    pub gl1c_cache_size: u32,
    pub gl2c_cache_size: u32,
    pub mall_size: u64,
    pub enabled_rb_pipes_mask_hi: u32,
    pub shadow_size: u32,
    pub shadow_alignment: u32,
    pub csa_size: u32,
    pub csa_alignment: u32,
    pub _pad1: u32,
}

/// `struct drm_amdgpu_info_hw_ip`, the answer to `AMDGPU_INFO_HW_IP_INFO`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
pub struct AmdgpuHwIpInfo {
    pub hw_ip_version_major: u32,
    pub hw_ip_version_minor: u32,
    pub capabilities_flags: u64,
    pub ib_start_alignment: u32,
    pub ib_size_alignment: u32,
    pub available_rings: u32,
    pub ip_discovery_version: u32,
}

/// libdrm_amdgpu's `struct amdgpu_gpu_info`: what `amdgpu_query_gpu_info` derives from the
/// device info and, on chips older than Vega, a handful of registers. Not a kernel struct, but the
/// capset carries it verbatim, so its layout is the wire's.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct AmdgpuGpuInfo {
    pub asic_id: u32,
    pub chip_rev: u32,
    pub chip_external_rev: u32,
    pub family_id: u32,
    pub ids_flags: u64,
    pub max_engine_clk: u64,
    pub max_memory_clk: u64,
    pub num_shader_engines: u32,
    pub num_shader_arrays_per_engine: u32,
    pub avail_quad_shader_pipes: u32,
    pub max_quad_shader_pipes: u32,
    pub cache_entries_per_quad_pipe: u32,
    pub num_hw_gfx_contexts: u32,
    pub rb_pipes: u32,
    pub enabled_rb_pipes_mask: u32,
    pub gpu_counter_freq: u32,
    pub backend_disable: [u32; 4],
    pub mc_arb_ramcfg: u32,
    pub gb_addr_cfg: u32,
    pub gb_tile_mode: [u32; 32],
    pub gb_macro_tile_mode: [u32; 16],
    pub pa_sc_raster_cfg: [u32; 4],
    pub pa_sc_raster_cfg1: [u32; 4],
    pub cu_active_number: u32,
    pub cu_ao_mask: u32,
    pub cu_bitmap: [[u32; 4]; 4],
    pub vram_type: u32,
    pub vram_bit_width: u32,
    pub ce_ram_size: u32,
    pub vce_harvest_config: u32,
    pub pci_rev_id: u32,
}

/// libdrm_amdgpu's `struct amdgpu_buffer_size_alignments`, carried in the capset.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default, PartialEq, Eq)]
pub struct AmdgpuBufferSizeAlignments {
    pub size_local: u64,
    pub size_remote: u64,
}

/// libdrm_amdgpu's `struct amdgpu_heap_info`, three of which the response region's header
/// carries.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default, PartialEq, Eq)]
pub struct AmdgpuHeapInfo {
    pub heap_size: u64,
    pub heap_usage: u64,
    pub max_allocation: u64,
}

// Sizes from the headers, computed by hand from their field lists and pinned here: a field added
// or reordered without the header saying so is caught before anything reaches the kernel.
const _: () = {
    assert!(size_of::<AmdgpuInfo>() == 32);
    assert!(size_of::<AmdgpuDeviceInfo>() == 440);
    assert!(std::mem::offset_of!(AmdgpuDeviceInfo, ids_flags) == 136);
    assert!(std::mem::offset_of!(AmdgpuDeviceInfo, high_va_offset) == 336);
    assert!(std::mem::offset_of!(AmdgpuDeviceInfo, tcc_disabled_mask) == 360);
    assert!(size_of::<AmdgpuHwIpInfo>() == 32);
    assert!(size_of::<AmdgpuGpuInfo>() == 416);
    assert!(std::mem::offset_of!(AmdgpuGpuInfo, gb_tile_mode) == 100);
    assert!(std::mem::offset_of!(AmdgpuGpuInfo, pci_rev_id) == 412);
    assert!(size_of::<AmdgpuBufferSizeAlignments>() == 16);
    assert!(size_of::<AmdgpuHeapInfo>() == 24);
};
