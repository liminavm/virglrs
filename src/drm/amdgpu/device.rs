// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! The amdgpu backend's side of the kernel: finding the device, describing it in the capset the
//! way libdrm_amdgpu describes it, and the per-context device a guest's commands run against.

use super::ids;
use super::proto::{self, QueryInfoReq, cmd};
use crate::drm::Refused;
use crate::drm::capset::{Capset, DriverVersion};
use crate::drm::context::{Backend, Reply};
use crate::drm::region::ResponseRegion;
use crate::drm::sys::{self, Device};
use crate::drm::uapi::{
    AMDGPU_FAMILY_AI, AMDGPU_FAMILY_CI, AMDGPU_HW_IP_NUM, AMDGPU_IDS_FLAGS_FUSION,
    AmdgpuBufferSizeAlignments, AmdgpuDeviceInfo, AmdgpuGpuInfo, AmdgpuHeapInfo, AmdgpuHwIpInfo,
    query,
};
use bytemuck::Zeroable;
use std::io;
use std::path::PathBuf;

/// The amdgpu device this host has, as probed once at init.
pub struct Probed {
    node: PathBuf,
    pub capset: Capset,
}

impl Probed {
    /// The first render node an `amdgpu` driver answers on, described.
    ///
    /// `None` when there is none, or when the first one cannot be described -- the C fails its
    /// whole init then, and takes classic and venus down with it; here the host simply offers no
    /// native context, and says why.
    pub fn find() -> Option<Probed> {
        for node in sys::render_nodes() {
            let Ok(dev) = Device::open(&node) else { continue };
            let Ok(version) = dev.version() else { continue };
            if version.name != "amdgpu" {
                continue;
            }
            return match describe(&dev, &version) {
                Ok(capset) => Some(Probed { node, capset }),
                Err(e) => {
                    eprintln!("[virglrs] native: {} cannot be described: {e}", node.display());
                    None
                }
            };
        }
        None
    }

    pub fn describe(&self) -> String {
        let name = &self.capset.amdgpu.marketing_name;
        let end = name.iter().position(|&b| b == 0).unwrap_or(name.len());
        format!("amdgpu on {} ({})", self.node.display(), String::from_utf8_lossy(&name[..end]))
    }

    /// A device of the context's own, named after it for the kernel's accounting.
    pub fn open(&self, name: &str) -> Result<Amdgpu, Refused> {
        let errno = |e: io::Error| Refused::NoDevice(e.raw_os_error().unwrap_or(libc::EIO));
        let device = Device::open(&self.node).map_err(errno)?;
        // Best effort, as in the C, which ignores the result too: the name only labels the file
        // in the kernel's per-client statistics, and the kernel refuses names it does not like.
        if !name.is_empty()
            && let Err(e) = device.set_client_name(name)
        {
            eprintln!("[virglrs] native: the kernel would not take the name {name:?}: {e}");
        }
        let rings = rings(&device);
        if rings == 0 {
            return Err(Refused::NoDevice(libc::ENODEV));
        }
        Ok(Amdgpu { device, rings, async_error: 0 })
    }
}

/// The capset for `dev`, derived as libdrm_amdgpu derives each field when the C asks it.
fn describe(dev: &Device, version: &sys::Version) -> io::Result<Capset> {
    // `amdgpu_device_initialize` refuses these two, and the C's probe with it.
    if version.major != 3 {
        return Err(io::Error::other(format!("amdgpu {}.x is not 3.x", version.major)));
    }
    let mut accel = [0u8; 4];
    dev.amdgpu_info(query::ACCEL_WORKING, [0; 4], &mut accel)?;
    if u32::from_le_bytes(accel) == 0 {
        return Err(io::Error::other("acceleration is not working"));
    }
    let mut info = AmdgpuDeviceInfo::zeroed();
    dev.amdgpu_info(query::DEV_INFO, [0; 4], bytemuck::bytes_of_mut(&mut info))?;
    let gpu_info = gpu_info(dev, &info)?;
    let alignments = AmdgpuBufferSizeAlignments {
        size_local: info.pte_fragment_size as u64,
        size_remote: info.gart_page_size as u64,
    };
    let table = ids::table_path(std::env::var("AMDGPU_ASIC_ID_TABLE_PATHS").ok().as_deref(), |p| {
        p.exists()
    });
    let name = ids::marketing_name(
        std::fs::read_to_string(table).ok().as_deref(),
        std::fs::read_to_string("/proc/cpuinfo").ok().as_deref(),
        info.device_id,
        info.pci_rev,
        info.ids_flags & AMDGPU_IDS_FLAGS_FUSION != 0,
    );
    Ok(Capset::amdgpu(
        DriverVersion {
            major: version.major,
            minor: version.minor,
            patchlevel: version.patchlevel,
        },
        address32_hi(&info),
        alignments,
        gpu_info,
        &name,
    ))
}

/// `amdgpu_query_sw_info(amdgpu_sw_info_address32_hi)`: the top half of the addresses the 32-bit
/// virtual address range lies in -- the high range's first 4 GiB where there is a high range,
/// else the low range's.
fn address32_hi(info: &AmdgpuDeviceInfo) -> u32 {
    const GIB4: u64 = 1 << 32;
    let high_32_max = info.high_va_max.min((info.high_va_offset & !0xffff_ffff) + GIB4);
    let max = if high_32_max != 0 { high_32_max } else { info.virtual_address_max.min(GIB4) };
    (max.wrapping_sub(1) >> 32) as u32
}

/// `amdgpu_query_gpu_info_init`: the device info rearranged, and on chips older than Vega the
/// tiling and raster registers that describe what the device info does not.
fn gpu_info(dev: &Device, d: &AmdgpuDeviceInfo) -> io::Result<AmdgpuGpuInfo> {
    let mut g = AmdgpuGpuInfo::zeroed();
    g.asic_id = d.device_id;
    g.chip_rev = d.chip_rev;
    g.chip_external_rev = d.external_rev;
    g.family_id = d.family;
    g.max_engine_clk = d.max_engine_clock;
    g.max_memory_clk = d.max_memory_clock;
    g.gpu_counter_freq = d.gpu_counter_freq;
    g.enabled_rb_pipes_mask = d.enabled_rb_pipes_mask;
    g.rb_pipes = d.num_rb_pipes;
    g.ids_flags = d.ids_flags;
    g.num_hw_gfx_contexts = d.num_hw_gfx_contexts;
    g.num_shader_engines = d.num_shader_engines;
    g.num_shader_arrays_per_engine = d.num_shader_arrays_per_engine;
    g.vram_type = d.vram_type;
    g.vram_bit_width = d.vram_bit_width;
    g.ce_ram_size = d.ce_ram_size;
    g.vce_harvest_config = d.vce_harvest_config;
    g.pci_rev_id = d.pci_rev;

    let read = |offset: u32, instance: u32, out: &mut [u32]| -> io::Result<()> {
        let params = [offset, out.len() as u32, instance, 0];
        dev.amdgpu_info(query::READ_MMR_REG, params, bytemuck::cast_slice_mut(out))
    };
    // Every shader engine's own registers, then the broadcast ones.
    const BROADCAST: u32 = 0xffff_ffff;
    if d.family < AMDGPU_FAMILY_AI {
        // Pre-Vega parts have at most four engines, which is all the struct has room for.
        for se in 0..(d.num_shader_engines as usize).min(4) {
            let instance = (se as u32) | (0xff << 8);
            let mut v = [0u32];
            read(0x263d, instance, &mut v)?;
            g.backend_disable[se] = (v[0] >> 16) & 0xff;
            read(0xa0d4, instance, std::slice::from_mut(&mut g.pa_sc_raster_cfg[se]))?;
            if d.family >= AMDGPU_FAMILY_CI {
                read(0xa0d5, instance, std::slice::from_mut(&mut g.pa_sc_raster_cfg1[se]))?;
            }
        }
    }
    read(0x263e, BROADCAST, std::slice::from_mut(&mut g.gb_addr_cfg))?;
    if d.family < AMDGPU_FAMILY_AI {
        read(0x2644, BROADCAST, &mut g.gb_tile_mode)?;
        if d.family >= AMDGPU_FAMILY_CI {
            read(0x2664, BROADCAST, &mut g.gb_macro_tile_mode)?;
        }
        read(0x9d8, BROADCAST, std::slice::from_mut(&mut g.mc_arb_ramcfg))?;
    }
    g.cu_active_number = d.cu_active_number;
    g.cu_ao_mask = d.cu_ao_mask;
    g.cu_bitmap = d.cu_bitmap;
    Ok(g)
}

/// How many hardware rings the device has, across every IP type -- the guest's ring indices 1..,
/// ring 0 being the CPU.
fn rings(dev: &Device) -> u32 {
    (0..AMDGPU_HW_IP_NUM)
        .filter_map(|ip| {
            let mut info = AmdgpuHwIpInfo::zeroed();
            dev.amdgpu_info(query::HW_IP_INFO, [ip, 0, 0, 0], bytemuck::bytes_of_mut(&mut info))
                .ok()?;
            Some(info.available_rings.count_ones())
        })
        .sum()
}

/// One guest context's amdgpu device.
pub struct Amdgpu {
    device: Device,
    /// The guest's ring indices run from 1 to this.
    #[allow(dead_code, reason = "fences arrive with command submission")]
    rings: u32,
    /// The count of failures the guest learns of through the response region's header. The host
    /// owns it: the guest can write that word, so it is stored there and never read back.
    #[allow(dead_code, reason = "the first failing commands arrive with buffer creation")]
    async_error: u32,
}

impl Amdgpu {
    /// `amdgpu_query_heap_info` for the three heaps the response region's header carries. A heap
    /// whose query fails is left as it was, as the C leaves it.
    fn heaps(&self) -> [Option<(usize, AmdgpuHeapInfo)>; 3] {
        let q = |query: u32| -> Option<u64> {
            let mut v = [0u8; 8];
            self.device.amdgpu_info(query, [0; 4], &mut v).ok()?;
            Some(u64::from_le_bytes(v))
        };
        let mut vram_gtt = [0u64; 3];
        let Ok(()) = self.device.amdgpu_info(
            query::VRAM_GTT,
            [0; 4],
            bytemuck::cast_slice_mut(&mut vram_gtt),
        ) else {
            return [None, None, None];
        };
        let [vram_size, vram_cpu_accessible, gtt_size] = vram_gtt;
        let heap = |size: u64, usage: Option<u64>| {
            usage.map(|heap_usage| AmdgpuHeapInfo {
                heap_size: size,
                heap_usage,
                max_allocation: vram_cpu_accessible,
            })
        };
        [
            heap(vram_cpu_accessible, q(query::VIS_VRAM_USAGE))
                .map(|h| (proto::shmem::VIS_VRAM, h)),
            heap(vram_size, q(query::VRAM_USAGE)).map(|h| (proto::shmem::VRAM, h)),
            heap(gtt_size, q(query::GTT_USAGE)).map(|h| (proto::shmem::GTT, h)),
        ]
    }

    fn write_heaps(&self, region: &ResponseRegion) {
        for (at, heap) in self.heaps().into_iter().flatten() {
            region.write_header(at, bytemuck::bytes_of(&heap));
        }
    }

    /// `AMDGPU_CCMD_QUERY_INFO`: the guest's `DRM_AMDGPU_INFO`, run on its own device and answered
    /// with the kernel's result and up to `return_size` bytes.
    fn query_info(&self, req: &[u8], reply: &mut Reply<'_>) -> Result<(), Refused> {
        let req: QueryInfoReq = crate::drm::proto::decode(req);
        let size = req.info.return_size;
        if size as usize > sys::MAX_INFO - proto::RSP_HEADER {
            return Err(Refused::ResponseTooLarge(size));
        }
        let rsp = reply.stage(proto::RSP_HEADER + size as usize)?;
        let (head, payload) = rsp.split_at_mut(proto::RSP_HEADER);
        // The guest's whole request goes to the kernel but for where the answer lands, which is
        // ours. The kernel's error is the guest's answer, not a refusal: the guest asks for things
        // a given device does not have, and is told so.
        let ret = match self.device.amdgpu_info(req.info.query, req.info.params, payload) {
            Ok(()) => 0,
            Err(e) => -e.raw_os_error().unwrap_or(libc::EIO),
        };
        head[proto::RSP_RET..].copy_from_slice(&ret.to_le_bytes());
        Ok(())
    }
}

impl Backend for Amdgpu {
    fn alignment(&self) -> usize {
        proto::ALIGNMENT
    }

    fn header_len(&self) -> usize {
        proto::shmem::LEN
    }

    fn region_attached(&mut self, region: &ResponseRegion) {
        self.write_heaps(region);
    }

    fn dispatch(
        &mut self,
        command: u32,
        req: &[u8],
        reply: &mut Reply<'_>,
        _region: Option<&ResponseRegion>,
    ) -> Result<(), Refused> {
        match command {
            cmd::QUERY_INFO => self.query_info(req, reply),
            cmd::GEM_NEW
            | cmd::BO_VA_OP
            | cmd::CS_SUBMIT
            | cmd::SET_METADATA
            | cmd::BO_QUERY_INFO
            | cmd::CREATE_CTX
            | cmd::RESERVE_VMID
            | cmd::SET_PSTATE
            | cmd::CS_QUERY_FENCE_STATUS => Err(Refused::Unserved(command)),
            other => Err(Refused::UnknownCommand(other)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(low_max: u64, high_offset: u64, high_max: u64) -> AmdgpuDeviceInfo {
        let mut d = AmdgpuDeviceInfo::zeroed();
        d.virtual_address_max = low_max;
        d.high_va_offset = high_offset;
        d.high_va_max = high_max;
        d
    }

    #[test]
    fn address32_hi_comes_from_the_high_range_when_there_is_one() {
        // A gfx11 APU: a low range under 128 TiB and a high range at the top of the address space.
        let d = info(0x7fff_ffff_f000, 0xffff_8000_0000_0000, 0xffff_ffff_ffff_f000);
        assert_eq!(address32_hi(&d), 0xffff_8000);
    }

    #[test]
    fn address32_hi_falls_back_to_the_low_range() {
        assert_eq!(address32_hi(&info(0x7fff_ffff_f000, 0, 0)), 0);
        assert_eq!(address32_hi(&info(0, 0, 0)), 0xffff_ffff, "libdrm's wrap, kept");
    }
}
