// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! The amdgpu backend's side of the kernel: finding the device, describing it in the capset the
//! way libdrm_amdgpu describes it, and the per-context device a guest's commands run against.

use super::ids;
use super::proto::{
    self, BoQueryInfoReq, BoQueryInfoRsp, BoVaOpReq, ChunkDesc, CreateCtxReq, CsSubmitReq,
    GemNewReq, QueryFenceStatusReq, QueryInfoReq, ReserveVmidReq, SetMetadataReq, SetPstateReq,
    cmd,
};
use crate::drm::capset::{Capset, DriverVersion};
use crate::drm::context::{Backend, Reply};
use crate::drm::region::ResponseRegion;
use crate::drm::sys::{self, AmdgpuCtx, Chunk, Device};
use crate::drm::uapi::{
    AMDGPU_FAMILY_AI, AMDGPU_FAMILY_CI, AMDGPU_GEM_CREATE_CPU_GTT_USWC, AMDGPU_GEM_DOMAIN_GTT,
    AMDGPU_HW_IP_NUM, AMDGPU_IDS_FLAGS_FUSION, AMDGPU_VA_OP_MAP, AMDGPU_VA_OP_REPLACE,
    AmdgpuBoListEntry, AmdgpuBufferSizeAlignments, AmdgpuCsChunkDep, AmdgpuCsChunkFence,
    AmdgpuCsChunkIb, AmdgpuDeviceInfo, AmdgpuGemCreate, AmdgpuGemVa, AmdgpuGpuInfo, AmdgpuHeapInfo,
    AmdgpuHwIpInfo, chunk, query,
};
use crate::drm::{Buffer, Object, Refused, Source};
use crate::ids::{ResourceHandle, RingIdx};
use bytemuck::Zeroable;
use std::io;
use std::os::fd::OwnedFd;
use std::path::PathBuf;
use std::sync::{Arc, Weak};

/// The longest a guest's fence query may block its context. See
/// [`Amdgpu::query_fence_status`].
const FENCE_WAIT_BOUND: std::time::Duration = std::time::Duration::from_secs(1);

/// libdrm's `AMDGPU_CS_MAX_RINGS`: the rings of one IP a fence query may name.
const MAX_RINGS_PER_IP: u32 = 8;

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
            gpu_contexts: Default::default(),
            last_submit: (0..rings).map(|_| None).collect(),
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
    rings: u32,
    /// The GPU contexts the guest made, by the id the kernel gave each.
    gpu_contexts: crate::Map<u32, AmdgpuCtx>,
    /// What each ring's newest submission signals, until a fence takes it. Index `i` is ring
    /// `i + 1`.
    last_submit: Vec<Option<OwnedFd>>,
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
        let charge = Some(self.budget.charge("native buffer", req.alloc_size));
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

    /// `AMDGPU_CCMD_CREATE_CTX`: a GPU context, answered with its id, or the end of one. The
    /// kernel's refusal is the guest's answer.
    fn create_ctx(&mut self, req: &[u8], reply: &mut Reply<'_>) -> Result<(), Refused> {
        let req: CreateCtxReq = crate::drm::proto::decode(req);
        let rsp = reply.stage(proto::CREATE_CTX_RSP)?;
        if req.flags & !proto::CREATE_CTX_DESTROY != 0 {
            return Err(Refused::Reserved(req.flags as u64));
        }
        let ret = if req.flags & proto::CREATE_CTX_DESTROY != 0 {
            // Dropping it frees it. An id with no context is the guest's mistake, and is answered.
            match self.gpu_contexts.remove(&req.priority_or_id) {
                Some(_) => 0,
                None => -libc::EINVAL,
            }
        } else {
            match self.device.amdgpu_ctx_create(req.priority_or_id as i32) {
                Ok(ctx) => {
                    let id = ctx.id();
                    rsp[proto::CREATE_CTX_RSP_ID..][..4].copy_from_slice(&id.to_le_bytes());
                    self.gpu_contexts.insert(id, ctx);
                    0
                }
                Err(e) => -errno(&e),
            }
        };
        rsp[proto::RSP_RET..][..4].copy_from_slice(&ret.to_le_bytes());
        Ok(())
    }

    /// `AMDGPU_CCMD_CS_SUBMIT`: run the guest's chunks on one of its GPU contexts, and keep what
    /// the submission signals for the ring's next fence.
    ///
    /// Every resource id is resolved to this context's buffer and every chunk is checked before
    /// anything reaches the kernel. Indirect buffers and dependencies go through as the guest
    /// wrote them: they name GPU addresses and sequence numbers in this context's own file, and
    /// the kernel checks those.
    fn cs_submit(
        &mut self,
        req: &[u8],
        reply: &mut Reply<'_>,
        region: Option<&ResponseRegion>,
    ) -> Result<(), Refused> {
        let head: CsSubmitReq = crate::drm::proto::decode(req);
        reply.stage(proto::RSP_HEADER)?;
        if head.num_chunks > proto::MAX_CHUNKS {
            return Err(Refused::TooManyChunks(head.num_chunks));
        }
        if head.ring_idx == 0 || head.ring_idx > self.rings {
            return Err(Refused::Ring(head.ring_idx));
        }
        let ctx = self.gpu_contexts.get(&head.ctx_id).ok_or(Refused::NoGpuContext(head.ctx_id))?;
        let descs_at = size_of::<CsSubmitReq>();
        let data_at = descs_at + head.num_chunks as usize * size_of::<ChunkDesc>();
        let descs = req.get(descs_at..data_at).ok_or(Refused::Chunk(0))?;
        let descs: Vec<ChunkDesc> = descs
            .as_chunks::<{ size_of::<ChunkDesc>() }>()
            .0
            .iter()
            .map(|d| bytemuck::pod_read_unaligned(d))
            .collect();

        // What the chunks name, held here so the borrows below outlive the call.
        let mut ibs = Vec::new();
        let mut deps: Vec<Vec<AmdgpuCsChunkDep>> = Vec::new();
        let mut fence: Option<(Arc<Object>, u32)> = None;
        let mut bo_list: Option<Vec<(Arc<Object>, u32)>> = None;
        for d in &descs {
            let id = d.chunk_id as u32;
            let bad = || Refused::Chunk(id);
            let at = data_at.checked_add(d.offset as usize).ok_or_else(bad)?;
            let bytes = req.get(at..at + d.length_dw as usize * 4).ok_or_else(bad)?;
            match id {
                chunk::IB => {
                    if bytes.len() != size_of::<AmdgpuCsChunkIb>() {
                        return Err(bad());
                    }
                    ibs.push(bytemuck::pod_read_unaligned::<AmdgpuCsChunkIb>(bytes));
                }
                chunk::DEPENDENCIES => {
                    let (whole, rest) = bytes.as_chunks::<{ size_of::<AmdgpuCsChunkDep>() }>();
                    if !rest.is_empty() {
                        return Err(bad());
                    }
                    deps.push(whole.iter().map(|d| bytemuck::pod_read_unaligned(d)).collect());
                }
                chunk::FENCE => {
                    let size = size_of::<AmdgpuCsChunkFence>();
                    // The C's checks: aligned for the struct, room for one, the offset a whole
                    // number of fence values, and one fence per submission.
                    if !at.is_multiple_of(4) || bytes.len() < size || fence.is_some() {
                        return Err(bad());
                    }
                    let f: AmdgpuCsChunkFence = bytemuck::pod_read_unaligned(&bytes[..size]);
                    if !f.offset.is_multiple_of(8) {
                        return Err(bad());
                    }
                    let object = self.buffer(f.handle).ok_or(Refused::NoBuffer(f.handle))?;
                    fence = Some((object, f.offset));
                }
                chunk::BO_HANDLES => {
                    if !at.is_multiple_of(4) || bo_list.is_some() {
                        return Err(bad());
                    }
                    let entries = bytes.as_chunks::<{ size_of::<AmdgpuBoListEntry>() }>().0;
                    let list = entries
                        .iter()
                        .map(|e| {
                            let e: AmdgpuBoListEntry = bytemuck::pod_read_unaligned(e);
                            let object = self.buffer(e.bo_handle);
                            Ok((object.ok_or(Refused::NoBuffer(e.bo_handle))?, e.bo_priority))
                        })
                        .collect::<Result<_, Refused>>()?;
                    bo_list = Some(list);
                }
                _ => return Err(bad()),
            }
        }

        let signalled = self.device.syncobj_create().map_err(|e| Refused::Kernel(errno(&e)))?;
        let gems: Option<Vec<(&sys::Gem, u32)>> =
            bo_list.as_ref().map(|l| l.iter().map(|(o, p)| (&o.gem, *p)).collect());
        // In the guest's order, as the C passes them; the output syncobj last.
        let mut chunks = Vec::with_capacity(descs.len() + 1);
        let (mut ib, mut dep) = (ibs.iter(), deps.iter());
        for d in &descs {
            chunks.push(match d.chunk_id as u32 {
                chunk::IB => Chunk::Ib(*ib.next().expect("one per IB descriptor")),
                chunk::DEPENDENCIES => Chunk::Dependencies(dep.next().expect("one per descriptor")),
                chunk::FENCE => {
                    let (object, offset) = fence.as_ref().expect("the one FENCE descriptor");
                    Chunk::Fence { gem: &object.gem, offset: *offset }
                }
                chunk::BO_HANDLES => Chunk::BoList(gems.as_deref().expect("the one list")),
                _ => unreachable!("refused above"),
            });
        }
        chunks.push(Chunk::SyncobjOut(&signalled));

        let submitted =
            self.device.amdgpu_cs(ctx, &chunks).and_then(|_seqno| signalled.export_sync_file());
        match submitted {
            Ok(sync) => {
                self.last_submit[head.ring_idx as usize - 1] = Some(sync);
                Ok(())
            }
            Err(e) => {
                eprintln!(
                    "[virglrs] native: submission to ring {} of GPU context {} refused: {e}",
                    head.ring_idx, head.ctx_id
                );
                self.async_failure(region);
                Err(Refused::Kernel(errno(&e)))
            }
        }
    }

    /// `AMDGPU_CCMD_SET_METADATA`: the buffer's tiling and the guest driver's description of it,
    /// stored by the kernel for whoever imports the buffer next. The kernel's refusal is the
    /// guest's answer.
    fn set_metadata(&mut self, req: &[u8], reply: &mut Reply<'_>) -> Result<(), Refused> {
        let head: SetMetadataReq = crate::drm::proto::decode(req);
        let rsp = reply.stage(proto::RSP_HEADER)?;
        let object = self.buffer(head.res_id).ok_or(Refused::NoBuffer(head.res_id))?;
        if head.size_metadata > proto::MAX_METADATA {
            return Err(Refused::Reserved(head.size_metadata as u64));
        }
        let at = size_of::<SetMetadataReq>();
        let data = req
            .get(at..at + head.size_metadata as usize)
            .ok_or(Refused::RequestLength(head.hdr.len))?;
        let ret = match object.gem.amdgpu_set_metadata(head.flags, head.tiling_info, data) {
            Ok(()) => 0,
            Err(e) => -errno(&e),
        };
        rsp[proto::RSP_RET..].copy_from_slice(&ret.to_le_bytes());
        Ok(())
    }

    /// `AMDGPU_CCMD_BO_QUERY_INFO`: how a resource's buffer was made, and its metadata. A
    /// resource with no buffer here is answered, not refused, as the C answers it: a guest's KMS
    /// may name a classic resource.
    fn bo_query_info(&mut self, req: &[u8], reply: &mut Reply<'_>) -> Result<(), Refused> {
        let req: BoQueryInfoReq = crate::drm::proto::decode(req);
        let rsp = reply.stage(size_of::<BoQueryInfoRsp>())?;
        if req.pad != 0 {
            return Err(Refused::Reserved(req.pad as u64));
        }
        let mut out: BoQueryInfoRsp = bytemuck::Zeroable::zeroed();
        out.ret = match self.buffer(req.res_id).map(|o| o.gem.amdgpu_query()) {
            None => -libc::EINVAL,
            Some(Err(e)) => -errno(&e),
            // libdrm's own check on the kernel's answer.
            Some(Ok((_, meta))) if meta.data_size_bytes > proto::MAX_METADATA => -libc::EINVAL,
            Some(Ok((info, meta))) => {
                out.alloc_size = info.bo_size;
                out.phys_alignment = info.alignment;
                out.preferred_heap = info.domains as u32;
                out.alloc_flags = info.domain_flags;
                out.flags = meta.flags;
                out.tiling_info = meta.tiling_info;
                out.size_metadata = meta.data_size_bytes;
                out.umd_metadata = meta.data;
                0
            }
        };
        // The length word is the region's to write; the rest is the answer.
        rsp[proto::RSP_RET..].copy_from_slice(&bytemuck::bytes_of(&out)[proto::RSP_RET..]);
        Ok(())
    }

    /// `AMDGPU_CCMD_RESERVE_VMID`: reserve the context's device file a VMID, or give it back.
    fn reserve_vmid(&mut self, req: &[u8], reply: &mut Reply<'_>) -> Result<(), Refused> {
        let req: ReserveVmidReq = crate::drm::proto::decode(req);
        let rsp = reply.stage(proto::RSP_HEADER)?;
        if req.flags & !proto::RESERVE_VMID_UNRESERVE != 0 {
            return Err(Refused::Reserved(req.flags));
        }
        let reserve = req.flags & proto::RESERVE_VMID_UNRESERVE == 0;
        let ret = match self.device.amdgpu_reserve_vmid(reserve) {
            Ok(()) => 0,
            Err(e) => -errno(&e),
        };
        rsp[proto::RSP_RET..].copy_from_slice(&ret.to_le_bytes());
        Ok(())
    }

    /// `AMDGPU_CCMD_SET_PSTATE`: read or set a GPU context's stable power state.
    ///
    /// The C passes the guest's operation to `DRM_AMDGPU_CTX` as it came, so a guest naming
    /// `FREE_CTX` there frees a context the host still holds. Only the two power-state operations
    /// reach the kernel here; any other is answered `-EINVAL`.
    fn set_pstate(&mut self, req: &[u8], reply: &mut Reply<'_>) -> Result<(), Refused> {
        let req: SetPstateReq = crate::drm::proto::decode(req);
        let rsp = reply.stage(proto::RSP_WITH_WORD)?;
        if req.pad != 0 {
            return Err(Refused::Reserved(req.pad as u64));
        }
        let ctx = self.gpu_contexts.get(&req.ctx_id).ok_or(Refused::NoGpuContext(req.ctx_id))?;
        let set = match req.op {
            crate::drm::uapi::AMDGPU_CTX_OP_GET_STABLE_PSTATE => Some(false),
            crate::drm::uapi::AMDGPU_CTX_OP_SET_STABLE_PSTATE => Some(true),
            _ => None,
        };
        let (ret, flags) = match set.map(|set| ctx.stable_pstate(set, req.flags)) {
            None => (-libc::EINVAL, 0),
            Some(Ok(flags)) => (0, flags),
            Some(Err(e)) => (-errno(&e), 0),
        };
        rsp[proto::RSP_RET..][..4].copy_from_slice(&ret.to_le_bytes());
        rsp[proto::RSP_WORD..][..4].copy_from_slice(&flags.to_le_bytes());
        Ok(())
    }

    /// `AMDGPU_CCMD_CS_QUERY_FENCE_STATUS`: whether a submission has finished, waiting a while
    /// for it if the guest asked to.
    ///
    /// The wait blocks this context's submissions, so it is bounded at [`FENCE_WAIT_BOUND`]: as
    /// long as any guest Mesa asks for, RADV's idle wait included. A guest that asked for longer,
    /// or forever, is told the work is still running and asks again. A deadline given in the
    /// guest's own clock is read the same way, since the host's clock does not share its epoch.
    fn query_fence_status(&mut self, req: &[u8], reply: &mut Reply<'_>) -> Result<(), Refused> {
        let req: QueryFenceStatusReq = crate::drm::proto::decode(req);
        let rsp = reply.stage(proto::RSP_WITH_WORD)?;
        let ctx = self.gpu_contexts.get(&req.ctx_id).ok_or(Refused::NoGpuContext(req.ctx_id))?;
        let timeout = if req.flags & proto::FENCE_TIMEOUT_IS_ABSOLUTE != 0 {
            FENCE_WAIT_BOUND
        } else {
            std::time::Duration::from_nanos(req.timeout_ns).min(FENCE_WAIT_BOUND)
        };
        // libdrm's checks, then its shortcut for a submission that never was.
        let (ret, expired) = if req.ip_type >= AMDGPU_HW_IP_NUM || req.ring >= MAX_RINGS_PER_IP {
            (-libc::EINVAL, false)
        } else if req.fence == 0 {
            (0, true)
        } else {
            match ctx.wait_cs(req.ip_type, req.ip_instance, req.ring, req.fence, timeout) {
                Ok(busy) => (0, !busy),
                Err(e) => (-errno(&e), false),
            }
        };
        rsp[proto::RSP_RET..][..4].copy_from_slice(&ret.to_le_bytes());
        rsp[proto::RSP_WORD..][..4].copy_from_slice(&(expired as u32).to_le_bytes());
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
            cmd::CREATE_CTX => self.create_ctx(req, reply),
            cmd::CS_SUBMIT => self.cs_submit(req, reply, region),
            cmd::SET_METADATA => self.set_metadata(req, reply),
            cmd::BO_QUERY_INFO => self.bo_query_info(req, reply),
            cmd::RESERVE_VMID => self.reserve_vmid(req, reply),
            cmd::SET_PSTATE => self.set_pstate(req, reply),
            cmd::CS_QUERY_FENCE_STATUS => self.query_fence_status(req, reply),
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
        let have = object.extent().map_or(0, |e| e as u64);
        if size > have {
            return Err(Refused::BlobTooLarge { asked: size, have });
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

    /// The C imports the buffer's dma-buf whenever the context has no name for it. Here the copy
    /// already on this file is reused first -- the original, when the guest closed and reopened
    /// the buffer in the context that made it, or an earlier import -- because importing a buffer
    /// a file already holds hands back the handle it already has, and two owners would close it
    /// twice.
    ///
    /// Storage another renderer holds -- a classic texture's buffer, a venus allocation -- arrives
    /// as its dma-buf and is imported the same way.
    fn attach(&mut self, handle: ResourceHandle, source: Source<'_>) -> Result<(), Refused> {
        use std::os::fd::AsFd;
        if self.bound.get(&handle).is_some_and(|w| w.strong_count() > 0) {
            return Ok(());
        }
        let kernel = |e: io::Error| Refused::Kernel(errno(&e));
        let object = match source {
            Source::Native(buffer) => {
                let here = std::iter::once(&buffer.object)
                    .chain(&buffer.imports.copies)
                    .find(|o| o.gem.is_in(&self.device))
                    .cloned();
                match here {
                    Some(o) => o,
                    None => {
                        let gem = match buffer.dmabuf.as_ref() {
                            Some(fd) => self.device.import_dmabuf(fd.as_fd()),
                            None => buffer
                                .object
                                .gem
                                .export()
                                .and_then(|fd| self.device.import_dmabuf(fd.as_fd())),
                        }
                        .map_err(kernel)?;
                        let object = Arc::new(Object {
                            gem,
                            size: buffer.object.size,
                            write_combining: buffer.object.write_combining,
                            _charge: None,
                        });
                        buffer.imports.copies.push(Arc::clone(&object));
                        object
                    }
                }
            }
            Source::Foreign { imports, dmabuf } => {
                match imports.copies.iter().find(|o| o.gem.is_in(&self.device)).cloned() {
                    Some(o) => o,
                    None => {
                        // A dma-buf's size is where its end is.
                        let size = std::io::Seek::seek(
                            &mut std::fs::File::from(dmabuf.try_clone().map_err(kernel)?),
                            std::io::SeekFrom::End(0),
                        )
                        .map_err(kernel)?;
                        let gem = self.device.import_dmabuf(dmabuf.as_fd()).map_err(kernel)?;
                        let object =
                            Arc::new(Object { gem, size, write_combining: false, _charge: None });
                        imports.copies.push(Arc::clone(&object));
                        object
                    }
                }
            }
        };
        self.bound.insert(handle, Arc::downgrade(&object));
        Ok(())
    }

    fn forget(&mut self, handle: ResourceHandle) {
        self.bound.remove(&handle);
    }

    fn rings(&self) -> u32 {
        self.rings
    }

    fn take_last_submit(&mut self, ring: RingIdx) -> Option<OwnedFd> {
        self.last_submit[ring.0 as usize - 1].take()
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
        let backend = probed.open("t", &budget).unwrap();
        let retirement = crate::fence::Retirement::start(
            Box::new(NoFences),
            crate::vrend::debug::Switches::default(),
        );
        let timelines = crate::drm::timeline::Timelines::start(
            crate::ids::ContextId::new(1).unwrap(),
            backend.rings,
            retirement.handle(),
        )
        .unwrap();
        let mut ctx = Context::new("t".into(), Box::new(backend), Some(timelines));
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

        // RADV makes buffers smaller than a page, and the guest's kernel rounds the blob up to
        // one: the blob covers the page the kernel allocated, and no more.
        let small = GemNewReq { blob_id: 8, alloc_size: 960, ..gem_new };
        ctx.submit(bytemuck::bytes_of(&GemNewReq { hdr: hdr(cmd::GEM_NEW, 56, 2, 0), ..small }))
            .unwrap();
        let page = crate::guest_mem::page_round(1).unwrap() as u64;
        let res8 = ResourceHandle::new(8).unwrap();
        assert_eq!(
            ctx.get_blob(res8, BlobId(8), 2 * page, 1).err(),
            Some(Refused::BlobTooLarge { asked: 2 * page, have: page })
        );
        let small = ctx.get_blob(res8, BlobId(8), page, 1).expect("a page of a sub-page buffer");
        assert_eq!(small.host_mapping().map(|(_, len)| len), Some(page), "the whole page maps");

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

        // A GPU context, answered with its id; a submission naming it with no IB is the kernel's
        // to refuse, which ends the submission and is counted; one naming a resource the context
        // has no buffer for never reaches the kernel.
        let create = CreateCtxReq {
            hdr: hdr(cmd::CREATE_CTX, size_of::<CreateCtxReq>(), 5, rsp_off),
            priority_or_id: 0,
            flags: 0,
        };
        assert!(region.store_u32(proto::shmem::LEN + rsp_off as usize, 16));
        ctx.submit(bytemuck::bytes_of(&create)).unwrap();
        assert_eq!(region.load_u32(ret_at), Some(0));
        let gpu_ctx = region.load_u32(ret_at + 4).unwrap();
        assert!(gpu_ctx != 0, "the kernel numbers contexts from 1");
        let submit = |bo: u32, seqno| {
            let mut req = bytemuck::bytes_of(&CsSubmitReq {
                hdr: hdr(cmd::CS_SUBMIT, 48, seqno, rsp_off),
                ctx_id: gpu_ctx,
                num_chunks: 1,
                pad: 0,
                ring_idx: 1,
            })
            .to_vec();
            req.extend(bytemuck::bytes_of(&ChunkDesc {
                chunk_id: chunk::BO_HANDLES as u16,
                length_dw: 2,
                offset: 0,
            }));
            req.extend(bytemuck::bytes_of(&AmdgpuBoListEntry { bo_handle: bo, bo_priority: 0 }));
            req
        };
        assert_eq!(ctx.submit(&submit(9, 6)).err(), Some(Refused::NoBuffer(9)));
        assert_eq!(ctx.submit(&submit(3, 7)).err(), Some(Refused::Kernel(libc::EINVAL)));
        assert_eq!(region.load_u32(proto::shmem::ASYNC_ERROR), Some(3));
        assert_eq!(ctx.submit_fence(RingIdx(1), crate::ids::FenceId(1)), Ok(true));
        // The remaining commands, each answered by the kernel or by libdrm's own checks.
        assert!(region.store_u32(proto::shmem::LEN + rsp_off as usize, 16));
        let word_at = proto::shmem::LEN + rsp_off as usize + proto::RSP_WORD;
        let pstate = SetPstateReq {
            hdr: hdr(cmd::SET_PSTATE, size_of::<SetPstateReq>(), 10, rsp_off),
            ctx_id: gpu_ctx,
            op: crate::drm::uapi::AMDGPU_CTX_OP_GET_STABLE_PSTATE,
            flags: 0,
            pad: 0,
        };
        ctx.submit(bytemuck::bytes_of(&pstate)).unwrap();
        assert_eq!(region.load_u32(ret_at), Some(0), "the power state reads");
        let free_ctx = SetPstateReq { op: crate::drm::uapi::AMDGPU_CTX_OP_FREE_CTX, ..pstate };
        ctx.submit(bytemuck::bytes_of(&free_ctx)).unwrap();
        assert_eq!(region.load_u32(ret_at), Some(-libc::EINVAL as u32), "no other op reaches it");
        let fence = |fence, ring, seqno| QueryFenceStatusReq {
            hdr: hdr(cmd::CS_QUERY_FENCE_STATUS, size_of::<QueryFenceStatusReq>(), seqno, rsp_off),
            ctx_id: gpu_ctx,
            ip_type: 0,
            ip_instance: 0,
            ring,
            fence,
            timeout_ns: u64::MAX,
            flags: 0,
        };
        ctx.submit(bytemuck::bytes_of(&fence(0, 0, 11))).unwrap();
        assert_eq!((region.load_u32(ret_at), region.load_u32(word_at)), (Some(0), Some(1)));
        ctx.submit(bytemuck::bytes_of(&fence(1, 8, 12))).unwrap();
        assert_eq!(region.load_u32(ret_at), Some(-libc::EINVAL as u32), "libdrm's ring check");
        // A sequence number the ring never reached is the kernel's to refuse, and it does at
        // once, so this does not reach the bound on the wait.
        let started = std::time::Instant::now();
        ctx.submit(bytemuck::bytes_of(&fence(1, 0, 13))).unwrap();
        assert_eq!(region.load_u32(ret_at), Some(-libc::EINVAL as u32), "a fence never submitted");
        assert!(started.elapsed() < FENCE_WAIT_BOUND, "refused, not waited for");
        assert_eq!(
            ctx.submit(bytemuck::bytes_of(&QueryFenceStatusReq { ctx_id: 99, ..fence(1, 0, 14) }))
                .err(),
            Some(Refused::NoGpuContext(99))
        );
        let vmid = |flags, seqno| ReserveVmidReq {
            hdr: hdr(cmd::RESERVE_VMID, size_of::<ReserveVmidReq>(), seqno, rsp_off),
            flags,
        };
        ctx.submit(bytemuck::bytes_of(&vmid(0, 15))).unwrap();
        let reserved = region.load_u32(ret_at);
        ctx.submit(bytemuck::bytes_of(&vmid(proto::RESERVE_VMID_UNRESERVE, 16))).unwrap();
        eprintln!("reserve vmid answered {reserved:?}, unreserve {:?}", region.load_u32(ret_at));
        assert_eq!(ctx.submit(bytemuck::bytes_of(&vmid(2, 17))).err(), Some(Refused::Reserved(2)));

        let destroy =
            CreateCtxReq { priority_or_id: gpu_ctx, flags: proto::CREATE_CTX_DESTROY, ..create };
        ctx.submit(bytemuck::bytes_of(&destroy)).unwrap();
        assert_eq!(region.load_u32(ret_at), Some(0));
        ctx.submit(bytemuck::bytes_of(&destroy)).unwrap();
        assert_eq!(region.load_u32(ret_at), Some(-libc::EINVAL as u32), "it is gone");

        ctx.forget(res);
        assert_eq!(ctx.submit(bytemuck::bytes_of(&va_op(3, 9))).err(), Some(Refused::NoBuffer(3)));
        drop(buffer);
        eprintln!("RAN: on {}", probed.describe());
    }

    /// The kernel answers a re-import of a buffer this file already holds with the handle it
    /// already has; the import is refused, not handed a second owner that would close it twice.
    #[test]
    fn a_buffer_reimported_into_its_own_file_is_refused_not_owned_twice() {
        let Some(probed) = Probed::find() else {
            eprintln!("SKIPPED: no amdgpu device on this host");
            return;
        };
        let device = Arc::new(Device::open(&probed.node).unwrap());
        let gem = device
            .amdgpu_gem_create(AmdgpuGemCreate {
                bo_size: 4096,
                alignment: 4096,
                domains: AMDGPU_GEM_DOMAIN_GTT as u64,
                domain_flags: 0,
            })
            .unwrap();
        let fd = gem.export().unwrap();
        let again = device.import_dmabuf(std::os::fd::AsFd::as_fd(&fd));
        assert_eq!(again.err().and_then(|e| e.raw_os_error()), Some(libc::EEXIST));
        let other = Arc::new(Device::open(&probed.node).unwrap());
        let imported = other.import_dmabuf(std::os::fd::AsFd::as_fd(&fd)).unwrap();
        drop((gem, imported));
        eprintln!("RAN: re-import on {}", probed.describe());
    }

    /// Storage another renderer holds reaches a native context as a dma-buf: it is imported at
    /// its whole size, owned by the resource, and imported once per device file however many
    /// handles name it there.
    #[test]
    fn a_foreign_buffer_is_imported_once_per_file_at_its_own_size() {
        use crate::drm::Imports;
        use crate::drm::context::Backend;

        let Some(probed) = Probed::find() else {
            eprintln!("SKIPPED: no amdgpu device on this host");
            return;
        };
        let budget = crate::budget::Budget::with_cap(None, false);
        let mut a = probed.open("a", &budget).unwrap();
        let mut b = probed.open("b", &budget).unwrap();
        // The renderer that made it: a file of its own, as vrend's or venus's would be.
        let maker = Arc::new(Device::open(&probed.node).unwrap());
        let gem = maker
            .amdgpu_gem_create(AmdgpuGemCreate {
                bo_size: 960,
                alignment: 4096,
                domains: AMDGPU_GEM_DOMAIN_GTT as u64,
                domain_flags: 0,
            })
            .unwrap();
        let mut imports = Imports::default();
        fn foreign<'a>(imports: &'a mut Imports, gem: &sys::Gem) -> Source<'a> {
            Source::Foreign { imports, dmabuf: gem.export().unwrap() }
        }
        let (one, two) = (ResourceHandle::new(1).unwrap(), ResourceHandle::new(2).unwrap());

        a.attach(one, foreign(&mut imports, &gem)).unwrap();
        let in_a = a.bound[&one].upgrade().expect("the resource owns what a names");
        let page = crate::guest_mem::page_round(1).unwrap() as u64;
        assert_eq!(in_a.size, page, "the size is the dma-buf's, not what was asked of it");
        assert!(in_a.gem.is_in(&a.device));
        b.attach(one, foreign(&mut imports, &gem)).unwrap();
        assert_eq!(imports.copies.len(), 2, "one copy per file");
        // A second handle on the same storage in a file that has it reuses the copy, where a
        // second import would be refused as a second owner.
        a.attach(two, foreign(&mut imports, &gem)).unwrap();
        assert_eq!(imports.copies.len(), 2);
        assert!(Arc::ptr_eq(&a.bound[&two].upgrade().unwrap(), &in_a));
        drop(in_a);
        drop(imports);
        assert!(a.bound[&one].upgrade().is_none(), "the context's name dies with the resource");
        eprintln!("RAN: foreign import on {}", probed.describe());
    }

    struct NoFences;

    impl crate::fence::FenceSink for NoFences {
        fn context_fence(&mut self, _: crate::ids::ContextId, _: RingIdx, _: crate::ids::FenceId) {}
        fn global_fence(&mut self, _: crate::ids::ClientFenceId) {}
        fn present_fence(&mut self, _: crate::ids::FenceId) {}
    }

    #[test]
    fn address32_hi_falls_back_to_the_low_range() {
        assert_eq!(address32_hi(&info(0x7fff_ffff_f000, 0, 0)), 0);
        assert_eq!(address32_hi(&info(0, 0, 0)), 0xffff_ffff, "libdrm's wrap, kept");
    }
}
