// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! The amdgpu backend's side of the kernel: finding the device, describing it in the capset the
//! way libdrm_amdgpu describes it, and the per-context device a guest's commands run against.

use super::ids;
use super::proto::{self, BoVaOpReq, GemNewReq, QueryInfoReq, cmd};
use crate::drm::capset::{Capset, DriverVersion};
use crate::drm::context::{Backend, Reply};
use crate::drm::region::ResponseRegion;
use crate::drm::sys::{self, Device};
use crate::drm::uapi::{
    AMDGPU_FAMILY_AI, AMDGPU_FAMILY_CI, AMDGPU_GEM_CREATE_CPU_GTT_USWC, AMDGPU_GEM_DOMAIN_GTT,
    AMDGPU_HW_IP_NUM, AMDGPU_IDS_FLAGS_FUSION, AMDGPU_VA_OP_MAP, AMDGPU_VA_OP_REPLACE,
    AmdgpuBufferSizeAlignments, AmdgpuDeviceInfo, AmdgpuGemCreate, AmdgpuGemVa, AmdgpuGpuInfo,
    AmdgpuHeapInfo, AmdgpuHwIpInfo, query,
};
use crate::drm::{Buffer, Object, Refused};
use crate::ids::ResourceHandle;
use bytemuck::Zeroable;
use std::io;
use std::path::PathBuf;
use std::sync::{Arc, Weak};

/// `VIRTGPU_BLOB_FLAG_USE_SHAREABLE`: the guest will share the blob, so it is exported now.
const BLOB_FLAG_USE_SHAREABLE: u32 = 2;

fn errno(e: &io::Error) -> i32 {
    e.raw_os_error().unwrap_or(libc::EIO)
}

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
    pub fn open(&self, name: &str, budget: &Arc<crate::budget::Budget>) -> Result<Amdgpu, Refused> {
        let device = Arc::new(Device::open(&self.node).map_err(|e| Refused::NoDevice(errno(&e)))?);
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
        Ok(Amdgpu {
            device,
            rings,
            async_error: 0,
            pending: Default::default(),
            bound: Default::default(),
            budget: crate::budget::Classic::open(budget),
        })
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
    device: Arc<Device>,
    /// The guest's ring indices run from 1 to this.
    #[allow(dead_code, reason = "fences arrive with command submission")]
    rings: u32,
    /// The count of failures the guest learns of through the response region's header. The host
    /// owns it: the guest can write that word, so it is stored there and never read back.
    async_error: u32,
    /// Buffers `GEM_NEW` made that no resource has bound yet, by blob id. The context is their
    /// only owner.
    pending: crate::Map<u32, Object>,
    /// The buffers bound to resources in this context, which is how the guest names a buffer in
    /// every later command. The resource owns each: this names it, so a buffer whose resource is
    /// gone stops resolving with no purge at the unref, and a detach is what removes the name.
    bound: crate::Map<ResourceHandle, Weak<Object>>,
    /// What a buffer is counted against. Counted, never refused, as classic's driver memory is:
    /// the kernel has already allocated it by the time there is a size to count.
    budget: crate::budget::Classic,
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

    /// Tell the guest a command with no answer failed. Stored, never incremented in place: the
    /// count is the host's, and the region's word is the guest's to scribble on.
    fn async_failure(&mut self, region: Option<&ResponseRegion>) {
        self.async_error = self.async_error.wrapping_add(1);
        if let Some(region) = region {
            region.store_header_u32(proto::shmem::ASYNC_ERROR, self.async_error);
        }
    }

    /// The buffer bound to `res_id` in this context, while its resource stands.
    fn buffer(&self, res_id: u32) -> Option<Arc<Object>> {
        self.bound.get(&ResourceHandle::new(res_id)?)?.upgrade()
    }

    /// `AMDGPU_CCMD_GEM_NEW`: a buffer, held under the guest's blob id until a resource binds it.
    /// It has no answer; a failure is counted in `async_error`, and ends the submission.
    fn gem_new(&mut self, req: &[u8], region: Option<&ResponseRegion>) -> Result<(), Refused> {
        let req: GemNewReq = crate::drm::proto::decode(req);
        let made = self.make(&req);
        if made.is_err() {
            self.async_failure(region);
        }
        made
    }

    fn make(&mut self, req: &GemNewReq) -> Result<(), Refused> {
        if req.pad != 0 {
            return Err(Refused::Reserved(req.pad as u64));
        }
        let blob = u32::try_from(req.blob_id)
            .ok()
            .filter(|&id| id != 0)
            .ok_or(Refused::BlobId(req.blob_id))?;
        if self.pending.contains_key(&blob) {
            return Err(Refused::BlobIdInUse(req.blob_id));
        }
        let gem = self
            .device
            .amdgpu_gem_create(AmdgpuGemCreate {
                bo_size: req.alloc_size,
                alignment: req.phys_alignment,
                domains: req.preferred_heap as u64,
                domain_flags: req.flags,
            })
            .map_err(|e| Refused::Kernel(errno(&e)))?;
        // Write-combined unless it is GTT the guest did not ask to be, as the C decides it.
        let write_combining = req.preferred_heap != AMDGPU_GEM_DOMAIN_GTT
            || req.flags & AMDGPU_GEM_CREATE_CPU_GTT_USWC != 0;
        let charge = self.budget.charge("native buffer", req.alloc_size);
        self.pending
            .insert(blob, Object { gem, size: req.alloc_size, write_combining, _charge: charge });
        Ok(())
    }

    /// `AMDGPU_CCMD_BO_VA_OP`: map a range of the context's GPU address space to a resource's
    /// buffer, or unmap it. The kernel's refusal is the guest's answer, and is also counted.
    fn bo_va_op(
        &mut self,
        req: &[u8],
        reply: &mut Reply<'_>,
        region: Option<&ResponseRegion>,
    ) -> Result<(), Refused> {
        let req: BoVaOpReq = crate::drm::proto::decode(req);
        let rsp = reply.stage(proto::RSP_HEADER)?;
        if req.flags2 & !proto::BO_VA_OP_SPARSE_BO != 0 {
            return Err(Refused::Reserved(req.flags2));
        }
        // The kernel's flags are 32 bits; libdrm drops the rest without a word, and a guest that
        // set them asked for something no kernel offers.
        let flags = u32::try_from(req.flags).map_err(|_| Refused::Reserved(req.flags))?;
        let object = if req.flags2 & proto::BO_VA_OP_SPARSE_BO != 0 {
            None
        } else {
            // The C's answer too: the guest may already have closed it.
            Some(self.buffer(req.res_id).ok_or(Refused::NoBuffer(req.res_id))?)
        };
        // libdrm's own check, made before it asks.
        let ret = if !(AMDGPU_VA_OP_MAP..=AMDGPU_VA_OP_REPLACE).contains(&req.op) {
            -libc::EINVAL
        } else {
            let va = AmdgpuGemVa {
                operation: req.op,
                flags,
                va_address: req.va,
                offset_in_bo: req.offset,
                map_size: req.vm_map_size,
                ..Default::default()
            };
            match self.device.amdgpu_gem_va(object.as_ref().map(|o| &o.gem), va) {
                Ok(()) => 0,
                Err(e) => -errno(&e),
            }
        };
        rsp[proto::RSP_RET..].copy_from_slice(&ret.to_le_bytes());
        if ret != 0 {
            eprintln!(
                "[virglrs] native: va op {} on resource {} at {:#x}+{:#x} refused: {}",
                req.op,
                req.res_id,
                req.va,
                req.vm_map_size,
                io::Error::from_raw_os_error(-ret),
            );
            self.async_failure(region);
        }
        Ok(())
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
        region: Option<&ResponseRegion>,
    ) -> Result<(), Refused> {
        match command {
            cmd::QUERY_INFO => self.query_info(req, reply),
            cmd::GEM_NEW => self.gem_new(req, region),
            cmd::BO_VA_OP => self.bo_va_op(req, reply, region),
            cmd::CS_SUBMIT
            | cmd::SET_METADATA
            | cmd::BO_QUERY_INFO
            | cmd::CREATE_CTX
            | cmd::RESERVE_VMID
            | cmd::SET_PSTATE
            | cmd::CS_QUERY_FENCE_STATUS => Err(Refused::Unserved(command)),
            other => Err(Refused::UnknownCommand(other)),
        }
    }

    /// The C's order: the resource must not already have a buffer here, the blob id must name a
    /// pending one, and a shareable blob is exported now -- a buffer the kernel will not export
    /// is refused before it leaves the pending set, so the guest can still bind it unshared.
    fn get_blob(
        &mut self,
        handle: ResourceHandle,
        blob: u32,
        size: u64,
        flags: u32,
        region: Option<&ResponseRegion>,
    ) -> Result<Buffer, Refused> {
        if self.bound.get(&handle).is_some_and(|w| w.strong_count() > 0) {
            return Err(Refused::ResourceBound(handle.get()));
        }
        let object = self.pending.get(&blob).ok_or(Refused::NoSuchBlob(blob as u64))?;
        if size > object.size {
            return Err(Refused::BlobTooLarge { asked: size, have: object.size });
        }
        let dmabuf = if flags & BLOB_FLAG_USE_SHAREABLE != 0 {
            Some(object.gem.export().map_err(|e| Refused::Kernel(errno(&e)))?)
        } else {
            None
        };
        let object = Arc::new(self.pending.remove(&blob).expect("found above"));
        self.bound.insert(handle, Arc::downgrade(&object));
        if let Some(region) = region {
            self.write_heaps(region);
        }
        Ok(Buffer::new(object, dmabuf))
    }

    fn forget(&mut self, handle: ResourceHandle) {
        self.bound.remove(&handle);
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

    /// A buffer made, bound, mapped into the GPU address space and the host's, then detached --
    /// against this host's amdgpu device. It says so when there is none, because a test that
    /// passes by skipping reads exactly like one that ran.
    #[test]
    fn a_buffer_is_made_bound_mapped_and_forgotten_on_the_device() {
        use crate::drm::context::Context;
        use crate::drm::proto::ReqHeader;
        use crate::guest_mem::GuestMap;
        use crate::ids::BlobId;

        let Some(probed) = Probed::find() else {
            eprintln!("SKIPPED: no amdgpu device on this host");
            return;
        };
        let budget = crate::budget::Budget::with_cap(None, false);
        let mut ctx = Context::new("t".into(), Box::new(probed.open("t", &budget).unwrap()));
        let region = Arc::new(GuestMap::anonymous(4096).unwrap());
        ctx.attach_region(Arc::clone(&region), 4096, 1).unwrap();
        let hdr =
            |cmd, len: usize, seqno, rsp_off| ReqHeader { cmd, len: len as u32, seqno, rsp_off };

        let gem_new = GemNewReq {
            hdr: hdr(cmd::GEM_NEW, size_of::<GemNewReq>(), 1, 0),
            blob_id: 7,
            alloc_size: 64 << 10,
            phys_alignment: 4096,
            preferred_heap: AMDGPU_GEM_DOMAIN_GTT,
            pad: 0,
            flags: 0,
        };
        ctx.submit(bytemuck::bytes_of(&gem_new)).unwrap();
        assert_eq!(region.load_u32(proto::shmem::ASYNC_ERROR), Some(0));
        assert_eq!(
            ctx.submit(bytemuck::bytes_of(&gem_new)).err(),
            Some(Refused::BlobIdInUse(7)),
            "a blob id is pending once"
        );
        assert_eq!(region.load_u32(proto::shmem::ASYNC_ERROR), Some(1), "and the guest is told");

        let res = ResourceHandle::new(3).unwrap();
        assert_eq!(
            ctx.get_blob(res, BlobId(7), 128 << 10, 1).err(),
            Some(Refused::BlobTooLarge { asked: 128 << 10, have: 64 << 10 })
        );
        let buffer = ctx.get_blob(res, BlobId(7), 64 << 10, 1).unwrap();
        assert_eq!(
            ctx.get_blob(res, BlobId(7), 64 << 10, 1).err(),
            Some(Refused::ResourceBound(3))
        );
        let (addr, len) = buffer.host_mapping().expect("GTT maps");
        assert!(addr != 0 && len == 64 << 10);
        assert!(!buffer.write_combining(), "GTT without USWC is cached");

        let rsp_off = 256u32;
        assert!(region.store_u32(proto::shmem::LEN + rsp_off as usize, 8));
        let va_op = |res_id, seqno| BoVaOpReq {
            hdr: hdr(cmd::BO_VA_OP, size_of::<BoVaOpReq>(), seqno, rsp_off),
            va: 1 << 40,
            vm_map_size: 64 << 10,
            flags: 0xe, // readable, writeable, executable
            flags2: 0,
            offset: 0,
            res_id,
            op: AMDGPU_VA_OP_MAP,
        };
        let ret_at = proto::shmem::LEN + rsp_off as usize + proto::RSP_RET;
        ctx.submit(bytemuck::bytes_of(&va_op(3, 2))).unwrap();
        assert_eq!(region.load_u32(ret_at), Some(0), "the kernel mapped it");
        assert_eq!(region.load_u32(crate::drm::proto::shmem::SEQNO), Some(2));
        ctx.submit(bytemuck::bytes_of(&va_op(3, 3))).unwrap();
        assert_eq!(region.load_u32(ret_at), Some(-libc::EINVAL as u32), "a second map overlaps");
        assert_eq!(region.load_u32(proto::shmem::ASYNC_ERROR), Some(2));

        ctx.forget(res);
        assert_eq!(ctx.submit(bytemuck::bytes_of(&va_op(3, 4))).err(), Some(Refused::NoBuffer(3)));
        drop(buffer);
        eprintln!("RAN: on {}", probed.describe());
    }

    #[test]
    fn address32_hi_falls_back_to_the_low_range() {
        assert_eq!(address32_hi(&info(0x7fff_ffff_f000, 0, 0)), 0);
        assert_eq!(address32_hi(&info(0, 0, 0)), 0xffff_ffff, "libdrm's wrap, kept");
    }
}
