// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! The backend for a VA-API driver: the guest's picture descriptor goes to the driver as the
//! picture and slice parameters it already is, and every decode target is a surface of ours.
//!
//! This is the shape the guest's own driver was written for -- mesa's VA frontend turned the
//! application's parameter buffers into the descriptor, and this turns them back -- so nothing is
//! re-synthesized. What VA-API needs that the descriptor cannot carry is storage: a reference
//! picture is a surface, so each decode target the guest names gets one, kept on the decode
//! thread for as long as the target lives, and the descriptor's reference handles are resolved
//! to those surfaces.
//!
//! **Who owns a surface.** The decode thread does, and it finds one by the target's
//! [`Identity`], never by the guest's handle. A unit carries a share of the identity of its
//! target and of every reference, so a target the guest destroys mid-decode stays decodable until
//! the unit is done; between units the thread keeps only a `Weak`, and drops a surface when its
//! target is gone. No destroy path on the render thread has to know any of this.
//!
//! Only VP9 is served so far. [`crate::decode::Codec::va_profile`] is the list, and the probe
//! asks the driver about nothing else, so no other codec is advertised and none reaches here.

use std::rc::Rc;
use std::sync::{Arc, Weak};

use cros_libva as va;

use super::{Backend, Delivery, Destination, Identity, Layout, Lookup, Shape, TargetFormat};
use super::{HevcSource, h264_slice, h265_slice, pending, vp9};
use crate::decode::{Imaged, Picture, PixelFormat};
use crate::surface::{Held, PlaneLayout, PlaneLayouts};
use crate::vrend::egl::{Importer, Plane};
use crate::vrend::gl::gles::{GL_R8, GL_RG8};
use crate::vrend::proto::{VideoBufferHandle, VideoCodecHandle};

/// `VA_INVALID_SURFACE`: an empty reference slot.
const NO_SURFACE: va::VASurfaceID = va::VA_INVALID_SURFACE;

/// One unit for the decode thread.
pub struct Unit {
    picture: Params,
    /// The target the picture is decoded into, or `None` for a unit with no target -- which VP9
    /// never has, and which is refused at decode.
    target: Option<Target>,
    pixels: PixelFormat,
    bytes: Vec<u8>,
    /// How the picture reaches the target. On the GPU the decode thread images the surface and
    /// delivery copies from the images; on any other route it is read back into memory.
    route: Route<Importer>,
}

/// How a picture reaches its target, and why when it is not on the GPU. Said by the decode
/// thread whenever it changes, which is the positive control that the GPU copy is in use: a
/// silent fallback reads back the same pixels and scores the same.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Route<T> {
    Gpu(T),
    /// The host has no dma-buf import, no EGL-image textures or no copy-image.
    NoImporter,
    NoTarget,
    /// A composite target, whose surface is not a pair of textures.
    Composite,
    /// A layout other than NV12, which a surface's two planes are not.
    Layout(Layout),
    /// Plane textures in formats a plane import cannot be copied into.
    PlaneFormats {
        luma: u32,
        chroma: u32,
        count: usize,
    },
}

impl<T> Route<T> {
    fn map<U>(self, f: impl FnOnce(T) -> U) -> Route<U> {
        match self {
            Route::Gpu(t) => Route::Gpu(f(t)),
            Route::NoImporter => Route::NoImporter,
            Route::NoTarget => Route::NoTarget,
            Route::Composite => Route::Composite,
            Route::Layout(layout) => Route::Layout(layout),
            Route::PlaneFormats { luma, chroma, count } => {
                Route::PlaneFormats { luma, chroma, count }
            }
        }
    }
}

/// What the descriptor says, per codec.
enum Params {
    Vp9 {
        frame: Box<vp9::Frame>,
        /// The descriptor's eight reference slots, resolved on the render thread. `None` is an
        /// empty slot, or a handle naming no target.
        refs: [Option<Arc<Identity>>; 8],
    },
    H264 {
        picture: Arc<h264_slice::Picture>,
        /// The DPB's targets, resolved the same way, one per DPB entry.
        dpb: Vec<Option<Arc<Identity>>>,
        /// The codec's extent: the wire has no `pic_width_in_mbs`.
        width: u32,
        height: u32,
    },
    Hevc {
        picture: HevcInput,
        /// The DPB's targets, slot for slot, resolved the same way.
        refs: [Option<Arc<Identity>>; 16],
        width: u32,
        height: u32,
    },
}

/// What an HEVC frame is decoded from here: the descriptor, read whole.
pub(super) type HevcInput = Arc<h265_slice::Picture>;

/// Read the descriptor. Never `None`: the slices are read when the frame is decoded, so there is
/// nothing to wait for, and nothing a session needs that the wire leaves out.
pub(super) fn hevc_input(source: HevcSource<'_>) -> Result<Option<HevcInput>, String> {
    let decoding = h265_slice::Decoding::read(source.descriptor, &source.desc);
    Ok(Some(Arc::new(h265_slice::Picture { desc: source.desc, decoding })))
}

/// VA-API takes the slices as they were sent, start codes and all: each NAL goes to the driver in
/// a slice-data buffer of its own. Only the framing is checked -- an access unit that is not
/// Annex-B has no slices to find.
pub(super) fn reframe(bitstream: Vec<u8>) -> Option<Vec<u8>> {
    super::bitstream::NalUnits::new(&bitstream).is_some().then_some(bitstream)
}

/// A decode target, as the decode thread needs it.
struct Target {
    identity: Arc<Identity>,
    width: u32,
    height: u32,
}

/// What a codec's decode thread keeps between units.
#[derive(Default)]
pub struct Host {
    /// Opened at the first unit. `None` after that only if it would not open, which is said
    /// once and leaves every frame undecoded.
    va: Option<Va>,
    unavailable: bool,
    /// The route the last picture took, so a change is said once rather than every frame.
    route: Option<Route<()>>,
}

struct Va {
    display: Rc<va::Display>,
    /// The driver's NV12 image format, which every picture is read back through.
    nv12: va::VAImageFormat,
    decoder: Option<Decoder>,
    surfaces: Vec<Stored>,
}

/// A VA context, and the frame shape it was made for.
struct Decoder {
    profile: va::VAProfile::Type,
    width: u32,
    height: u32,
    /// Declared before the config: a context is destroyed before the config it was made from.
    context: Rc<va::Context>,
    _config: va::Config,
}

/// A decode target's surface.
struct Stored {
    identity: Weak<Identity>,
    surface: va::Surface<()>,
    /// The surface's planes as EGL images, made the first time a picture in it goes out on the
    /// GPU. A surface is decoded into in place, so one export serves every frame it holds.
    imaging: Imaging,
}

enum Imaging {
    Untried,
    Ready(Arc<Imaged>),
    /// The export or the import was refused, which was said; the surface is read back instead.
    Refused,
}

impl Backend for Host {
    type Unit = Unit;

    fn unit(
        lookup: &Lookup<'_>,
        shape: &Shape,
        bytes: &[u8],
        delivery: Delivery<'_>,
        pixels: Option<PixelFormat>,
    ) -> Unit {
        let resolve = |handle: u32| {
            (handle != 0)
                .then(|| lookup.buffers.get(&VideoBufferHandle(handle)))
                .flatten()
                .map(|buffer| Arc::clone(&buffer.identity))
        };
        let picture = match shape {
            Shape::Vp9(frame) => Params::Vp9 {
                frame: frame.clone(),
                refs: std::array::from_fn(|slot| resolve(frame.refs[slot])),
            },
            Shape::H264 { picture, width, height, .. } => Params::H264 {
                picture: Arc::clone(picture),
                dpb: picture.decoding.dpb.iter().map(|r| resolve(r.buffer)).collect(),
                width: *width,
                height: *height,
            },
            Shape::Hevc { input, width, height, .. } => Params::Hevc {
                picture: Arc::clone(input),
                refs: input.decoding.refs.map(|r| r.and_then(|r| resolve(r.buffer))),
                width: *width,
                height: *height,
            },
            Shape::Av1 { .. } => {
                unreachable!("VA-API does not advertise AV1, so no such codec is created")
            }
        };
        Unit {
            picture,
            target: delivery.target().map(|buffer| Target {
                identity: Arc::clone(&buffer.identity),
                width: buffer.width,
                height: buffer.height,
            }),
            pixels: pixels.unwrap_or(PixelFormat::BiPlanar420),
            bytes: bytes.to_vec(),
            route: match (delivery.target().map(route_of), lookup.importer) {
                (Some(Route::Gpu(())), Some(importer)) => Route::Gpu(importer.clone()),
                (Some(Route::Gpu(())), None) => Route::NoImporter,
                (Some(other), _) => other.map(|()| unreachable!()),
                (None, _) => Route::NoTarget,
            },
        }
    }

    fn decode(
        &mut self,
        handle: VideoCodecHandle,
        unit: &Unit,
        phases: &mut pending::Phases,
    ) -> Option<Picture> {
        if self.va.is_none() && !self.unavailable {
            self.va = Va::open();
            if self.va.is_none() {
                eprintln!(
                    "[virglrs] video codec {handle}: the VA-API display would not open on the \
                     decode thread; nothing this codec decodes reaches its targets"
                );
                self.unavailable = true;
            }
        }
        let va = self.va.as_mut()?;
        let Some(target) = &unit.target else {
            eprintln!(
                "[virglrs] video codec {handle}: a VA-API frame with no target to decode into"
            );
            return None;
        };
        // Every surface whose target is gone, gone with it.
        va.surfaces.retain(|stored| stored.identity.strong_count() > 0);

        let began = std::time::Instant::now();
        let (profile, width, height) = match &unit.picture {
            Params::Vp9 { frame, .. } => {
                (va::VAProfile::VAProfileVP9Profile0, frame.width, frame.height)
            }
            Params::H264 { width, height, .. } => {
                (va::VAProfile::VAProfileH264High, *width, *height)
            }
            Params::Hevc { width, height, .. } => {
                (va::VAProfile::VAProfileHEVCMain, *width, *height)
            }
        };
        let made = va.decoder_for(profile, width, height);
        let Some(target_at) = va.surface_for(target) else {
            eprintln!("[virglrs] video codec {handle}: no VA surface for the decode target");
            return None;
        };
        if made {
            phases.create = Some(began.elapsed());
        }
        let Some(decoder) = &va.decoder else {
            eprintln!("[virglrs] video codec {handle}: no VA decode context");
            return None;
        };
        let id = |slot: &Option<Arc<Identity>>| slot.as_ref().and_then(|i| va.surface_id(i));

        let began = std::time::Instant::now();
        let surface = &va.surfaces[target_at].surface;
        let decoded = match &unit.picture {
            Params::Vp9 { frame, refs } => {
                let reference_frames = refs.each_ref().map(|slot| id(slot).unwrap_or(NO_SURFACE));
                decode_vp9(decoder, surface, frame, reference_frames, &unit.bytes)
            }
            Params::H264 { picture, dpb, width, height } => {
                let dpb: Vec<Option<va::VASurfaceID>> = dpb.iter().map(id).collect();
                let extent = (*width, *height);
                decode_h264(decoder, surface, picture, &dpb, extent, &unit.bytes)
            }
            Params::Hevc { picture, refs, .. } => {
                let refs = refs.each_ref().map(id);
                decode_hevc(decoder, surface, picture, &refs, &unit.bytes)
            }
        };
        phases.session = Some(began.elapsed());
        if let Err(why) = decoded {
            eprintln!("[virglrs] video codec {handle}: the VA driver decoded no picture ({why})");
            return None;
        }

        let began = std::time::Instant::now();
        let route = unit.route.clone().map(|_| ());
        if self.route.as_ref() != Some(&route) {
            eprintln!("[virglrs] video codec {handle}: VA pictures reach their targets {route:?}");
            self.route = Some(route);
        }
        let importer = match &unit.route {
            Route::Gpu(importer) => Some(importer),
            _ => None,
        };
        let imaged = importer.and_then(|importer| va.imaged(target_at, importer));
        let surface = &va.surfaces[target_at].surface;
        let picture = match imaged {
            Some(planes) => Some(Picture::imaged(planes, width, height)),
            None => read_back(surface, va.nv12, width, height, unit.pixels),
        };
        phases.write = Some(began.elapsed());
        if picture.is_none() {
            eprintln!("[virglrs] video codec {handle}: the decoded VA surface could not be read");
        }
        picture
    }
}

impl Va {
    fn open() -> Option<Va> {
        let display = va::Display::open()?;
        let nv12 = display
            .query_image_formats()
            .ok()?
            .into_iter()
            .find(|format| format.fourcc == va::VA_FOURCC_NV12)?;
        Some(Va { display, nv12, decoder: None, surfaces: Vec::new() })
    }

    /// Make the context fit the frame, returning whether it had to be made.
    ///
    /// Rebuilt when the profile or the extent changes. That loses nothing a later frame needs: a
    /// reference picture is a surface, and surfaces belong to the display, not to the context.
    fn decoder_for(&mut self, profile: va::VAProfile::Type, width: u32, height: u32) -> bool {
        if self
            .decoder
            .as_ref()
            .is_some_and(|d| (d.profile, d.width, d.height) == (profile, width, height))
        {
            return false;
        }
        self.decoder = None;
        let made = self
            .display
            .create_config(vec![], profile, va::VAEntrypoint::VAEntrypointVLD)
            .and_then(|config| {
                self.display
                    .create_context::<()>(&config, width, height, None, true)
                    .map(|context| (config, context))
            });
        match made {
            Ok((config, context)) => {
                self.decoder = Some(Decoder { profile, width, height, context, _config: config });
            }
            Err(why) => eprintln!("[virglrs] video: VA context for {width}x{height}: {why}"),
        }
        true
    }

    /// The index of the target's surface, made if the target has none yet.
    fn surface_for(&mut self, target: &Target) -> Option<usize> {
        if let Some(at) = self.position(&target.identity) {
            return Some(at);
        }
        let surface = self
            .display
            .create_surfaces(
                va::VA_RT_FORMAT_YUV420,
                Some(va::VA_FOURCC_NV12),
                target.width,
                target.height,
                Some(va::UsageHint::USAGE_HINT_DECODER),
                vec![()],
            )
            .map_err(|why| eprintln!("[virglrs] video: VA surface: {why}"))
            .ok()?
            .pop()?;
        let identity = Arc::downgrade(&target.identity);
        self.surfaces.push(Stored { identity, surface, imaging: Imaging::Untried });
        Some(self.surfaces.len() - 1)
    }

    /// The images over the surface at `at`, made the first time they are asked for.
    fn imaged(&mut self, at: usize, importer: &Importer) -> Option<Arc<Imaged>> {
        let stored = &mut self.surfaces[at];
        if let Imaging::Untried = stored.imaging {
            stored.imaging = match image(&stored.surface, importer) {
                Ok(planes) => Imaging::Ready(Arc::new(planes)),
                Err(why) => {
                    eprintln!(
                        "[virglrs] video: a VA surface could not be imaged ({why}); its pictures \
                         are read back through memory instead"
                    );
                    Imaging::Refused
                }
            };
        }
        match &stored.imaging {
            Imaging::Ready(planes) => Some(Arc::clone(planes)),
            Imaging::Untried | Imaging::Refused => None,
        }
    }

    /// The surface a reference names, if its target has ever been decoded into here.
    fn surface_id(&self, identity: &Arc<Identity>) -> Option<va::VASurfaceID> {
        self.position(identity).map(|at| self.surfaces[at].surface.id())
    }

    fn position(&self, identity: &Arc<Identity>) -> Option<usize> {
        self.surfaces
            .iter()
            .position(|stored| std::ptr::eq(stored.identity.as_ptr(), Arc::as_ptr(identity)))
    }
}

/// Why a frame was not decoded.
enum Failure {
    /// The descriptor's slice lies outside the bytes the guest sent.
    Slice {
        offset: u32,
        size: u32,
        held: usize,
    },
    /// The frame is wider or taller than VP9 can say.
    Extent {
        width: u32,
        height: u32,
    },
    /// An H.264 slice that could not be described to the driver.
    H264(h264_slice::Refused),
    /// An HEVC slice segment that could not be described to the driver.
    Hevc(h265_slice::Refused),
    Va(&'static str, va::VaError),
}

impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Failure::Slice { offset, size, held } => write!(
                f,
                "the descriptor's slice, {size} bytes at {offset}, lies outside the {held} bytes \
                 sent"
            ),
            Failure::Extent { width, height } => {
                write!(f, "a {width}x{height} frame is outside what VP9 can declare")
            }
            Failure::H264(why) => write!(f, "H.264: {why}"),
            Failure::Hevc(why) => write!(f, "HEVC: {why}"),
            Failure::Va(step, why) => write!(f, "{step}: {why}"),
        }
    }
}

/// Decode one VP9 frame into `surface`, and wait for it.
///
/// The descriptor's fields go across one for one, which is what the C's `vp9_fill_picture_param`
/// and `vp9_fill_slice_param` do.
fn decode_vp9(
    decoder: &Decoder,
    surface: &va::Surface<()>,
    frame: &vp9::Frame,
    reference_frames: [va::VASurfaceID; 8],
    bytes: &[u8],
) -> Result<(), Failure> {
    // The slice the descriptor names and the bytes that arrived are two statements of one
    // length, and the driver reads by the first. Reconciled here, where both are in sight: a
    // slice outside what was sent is refused rather than handed to a driver to read past.
    let (offset, size) = (frame.slice_data_offset, frame.slice_data_size);
    let end = u64::from(offset) + u64::from(size);
    if size == 0 || end > bytes.len() as u64 {
        return Err(Failure::Slice { offset, size, held: bytes.len() });
    }

    // The descriptor's extent is sixteen bits; one that fell back to the codec's may not fit.
    let (Ok(width), Ok(height)) = (u16::try_from(frame.width), u16::try_from(frame.height)) else {
        return Err(Failure::Extent { width: frame.width, height: frame.height });
    };

    let p = frame.pic_fields;
    let fields = va::VP9PicFields::new(
        p.subsampling_x(),
        p.subsampling_y(),
        p.frame_type(),
        p.show_frame(),
        p.error_resilient_mode(),
        p.intra_only(),
        p.allow_high_precision_mv(),
        p.mcomp_filter_type(),
        p.frame_parallel_decoding_mode(),
        p.reset_frame_context(),
        p.refresh_frame_context(),
        p.frame_context_idx(),
        p.segmentation_enabled(),
        p.segmentation_temporal_update(),
        p.segmentation_update_map(),
        p.last_ref_frame(),
        p.last_ref_frame_sign_bias(),
        p.golden_ref_frame(),
        p.golden_ref_frame_sign_bias(),
        p.alt_ref_frame(),
        p.alt_ref_frame_sign_bias(),
        p.lossless_flag(),
    );
    let picture = va::PictureParameterBufferVP9::new(
        width,
        height,
        reference_frames,
        &fields,
        frame.filter_level,
        frame.sharpness_level,
        frame.log2_tile_rows,
        frame.log2_tile_columns,
        frame.frame_header_length_in_bytes,
        frame.first_partition_size,
        frame.mb_segment_tree_probs,
        frame.segment_pred_probs,
        frame.profile,
        frame.bit_depth,
    );
    let segments = frame.segments.map(|segment| {
        va::SegmentParameterVP9::new(
            &va::VP9SegmentFlags::new(
                segment.reference_enabled(),
                segment.reference(),
                segment.reference_skipped(),
            ),
            segment.filter_level,
            segment.luma_ac_quant_scale,
            segment.luma_dc_quant_scale,
            segment.chroma_ac_quant_scale,
            segment.chroma_dc_quant_scale,
        )
    });
    let slice = va::SliceParameterBufferVP9::new(size, offset, frame.slice_data_flag, segments);

    render(
        decoder,
        surface,
        vec![
            va::BufferType::PictureParameter(va::PictureParameter::VP9(picture)),
            va::BufferType::SliceParameter(va::SliceParameter::VP9(slice)),
            va::BufferType::SliceData(bytes.to_vec()),
        ],
    )
}

/// Hand the driver a picture's buffers, and wait for the picture.
fn render(
    decoder: &Decoder,
    surface: &va::Surface<()>,
    buffers: Vec<va::BufferType>,
) -> Result<(), Failure> {
    let context = &decoder.context;
    let mut pending = va::Picture::new(0, Rc::clone(context), surface);
    for kind in buffers {
        let buffer =
            context.create_buffer(kind).map_err(|why| Failure::Va("vaCreateBuffer", why))?;
        pending.add_buffer(buffer);
    }
    pending
        .begin::<()>()
        .map_err(|why| Failure::Va("vaBeginPicture", why))?
        .render()
        .map_err(|why| Failure::Va("vaRenderPicture", why))?
        .end()
        .map_err(|why| Failure::Va("vaEndPicture", why))?
        .sync::<()>()
        .map_err(|(why, _)| Failure::Va("vaSyncSurface", why))?;
    Ok(())
}

/// Decode one H.264 frame into `surface`, and wait for it.
///
/// The picture parameters and the scaling lists are the descriptor's, field for field, as the C's
/// `h264_fill_picture_param` writes them. The slice parameters are not on the wire, and the C sends
/// them empty; they are rebuilt here from each slice's own header and the descriptor's DPB, which is
/// what lets a driver other than Mesa's decode the stream (see [`h264_slice`]).
fn decode_h264(
    decoder: &Decoder,
    surface: &va::Surface<()>,
    picture: &h264_slice::Picture,
    dpb_surfaces: &[Option<va::VASurfaceID>],
    (width, height): (u32, u32),
    bytes: &[u8],
) -> Result<(), Failure> {
    use va::{
        VA_PICTURE_H264_BOTTOM_FIELD as BOTTOM, VA_PICTURE_H264_INVALID as INVALID,
        VA_PICTURE_H264_LONG_TERM_REFERENCE as LONG, VA_PICTURE_H264_SHORT_TERM_REFERENCE as SHORT,
        VA_PICTURE_H264_TOP_FIELD as TOP,
    };
    let desc = &picture.desc;
    let dec = &picture.decoding;
    let slices = h264_slice::slices(bytes, desc).map_err(Failure::H264)?;
    if slices.is_empty() {
        return Err(Failure::H264(h264_slice::Refused::Unsupported(
            "an access unit with no slice",
        )));
    }

    let invalid = || va::PictureH264::new(NO_SURFACE, 0, INVALID, 0, 0);
    // A DPB entry as VA-API names it, or `None` for one that is not a reference: an entry
    // neither of whose fields is marked -- the guest zero-fills the slots up to `num_ref_frames`,
    // and handle 0 of a zeroed slot can name a live target -- or one whose target has no surface
    // here, a picture this codec never decoded, which no list may point at.
    let reference = |i: usize| {
        let r = &dec.dpb[i];
        let mut flags = if r.long_term { LONG } else { SHORT };
        match (r.top, r.bottom) {
            (true, true) => {}
            (true, false) => flags |= TOP,
            (false, true) => flags |= BOTTOM,
            (false, false) => return None,
        }
        let id = dpb_surfaces.get(i).copied().flatten()?;
        Some(va::PictureH264::new(
            id,
            r.frame_idx,
            flags,
            r.field_order_cnt[0],
            r.field_order_cnt[1],
        ))
    };
    let reference_frames: [va::PictureH264; 16] = std::array::from_fn(|i| {
        if i < dec.dpb.len() { reference(i) } else { None }.unwrap_or_else(invalid)
    });
    let current = va::PictureH264::new(
        surface.id(),
        dec.frame_num,
        if dec.is_reference { SHORT } else { 0 },
        dec.field_order_cnt[0],
        dec.field_order_cnt[1],
    );

    let mbs = |n: u32| n.div_ceil(16);
    let mut height_mbs = mbs(height);
    if !desc.frame_mbs_only {
        // A field-capable stream counts its height in pairs of macroblock rows.
        height_mbs = height_mbs.next_multiple_of(2);
    }
    let seq = va::H264SeqFields::new(
        u32::from(desc.chroma_format_idc),
        0,
        0,
        u32::from(desc.frame_mbs_only),
        u32::from(dec.mb_adaptive_frame_field),
        u32::from(desc.direct_8x8_inference),
        u32::from(dec.min_luma_bi_pred_8x8),
        u32::from(desc.log2_max_frame_num_minus4),
        u32::from(desc.pic_order_cnt_type),
        u32::from(desc.log2_max_pic_order_cnt_lsb_minus4),
        u32::from(desc.delta_pic_order_always_zero),
    );
    let fields = va::H264PicFields::new(
        u32::from(desc.entropy_coding_mode),
        u32::from(desc.weighted_pred),
        u32::from(desc.weighted_bipred_idc),
        u32::from(desc.transform_8x8_mode),
        u32::from(desc.field_pic),
        u32::from(desc.constrained_intra_pred),
        u32::from(desc.bottom_field_pic_order_in_frame_present),
        u32::from(desc.deblocking_filter_control_present),
        u32::from(desc.redundant_pic_cnt_present),
        u32::from(dec.is_reference),
    );
    let too_big = || Failure::H264(h264_slice::Refused::OutOfRange("the picture's extent"));
    let parameters = va::PictureParameterBufferH264::new(
        current,
        reference_frames,
        u16::try_from(mbs(width).saturating_sub(1)).map_err(|_| too_big())?,
        u16::try_from(height_mbs.saturating_sub(1)).map_err(|_| too_big())?,
        desc.bit_depth_luma_minus8,
        desc.bit_depth_chroma_minus8,
        desc.num_ref_frames,
        &seq,
        desc.num_slice_groups_minus1,
        dec.slice_group_map_type,
        u16::from(dec.slice_group_change_rate_minus1),
        desc.pic_init_qp_minus26,
        desc.pic_init_qs_minus26,
        desc.chroma_qp_index_offset,
        desc.second_chroma_qp_index_offset,
        &fields,
        dec.frame_num as u16,
    );
    let mut buffers = vec![
        va::BufferType::PictureParameter(va::PictureParameter::H264(parameters)),
        va::BufferType::IQMatrix(va::IQMatrix::H264(va::IQMatrixBufferH264::new(
            dec.scaling_4x4,
            dec.scaling_8x8,
        ))),
    ];

    for slice in &slices {
        let h = &slice.header;
        let lists = h264_slice::ref_lists(h, dec, desc).map_err(Failure::H264)?;
        let list =
            |l: usize| lists[l].map(|entry| entry.and_then(reference).unwrap_or_else(invalid));
        let blank = h264_slice::ListWeights {
            luma_flag: false,
            luma_weight: [0; 32],
            luma_offset: [0; 32],
            chroma_flag: false,
            chroma_weight: [[0; 2]; 32],
            chroma_offset: [[0; 2]; 32],
        };
        let (luma_denom, chroma_denom, [w0, w1]) = match &h.weights {
            Some(w) => (w.luma_log2_denom, w.chroma_log2_denom, w.lists.clone()),
            None => (0, 0, [blank.clone(), blank]),
        };
        let too_long = || Failure::H264(h264_slice::Refused::OutOfRange("a slice's length"));
        let params = va::SliceParameterBufferH264::new(
            u32::try_from(slice.nal.len()).map_err(|_| too_long())?,
            0,
            va::VA_SLICE_DATA_FLAG_ALL,
            u16::try_from(h.data_bit_offset).map_err(|_| too_long())?,
            u16::try_from(h.first_mb_in_slice).map_err(|_| too_long())?,
            h.slice_type as u8,
            u8::from(h.direct_spatial_mv_pred),
            h.num_ref_idx_active_minus1[0],
            h.num_ref_idx_active_minus1[1],
            h.cabac_init_idc,
            h.slice_qp_delta,
            h.disable_deblocking_filter_idc,
            h.slice_alpha_c0_offset_div2,
            h.slice_beta_offset_div2,
            list(0),
            list(1),
            luma_denom,
            chroma_denom,
            u8::from(w0.luma_flag),
            w0.luma_weight,
            w0.luma_offset,
            u8::from(w0.chroma_flag),
            w0.chroma_weight,
            w0.chroma_offset,
            u8::from(w1.luma_flag),
            w1.luma_weight,
            w1.luma_offset,
            u8::from(w1.chroma_flag),
            w1.chroma_weight,
            w1.chroma_offset,
        );
        buffers.push(va::BufferType::SliceParameter(va::SliceParameter::H264(params)));
        buffers.push(va::BufferType::SliceData(slice.nal.to_vec()));
    }
    render(decoder, surface, buffers)
}

/// Decode one HEVC frame into `surface`, and wait for it.
///
/// The picture parameters and scaling lists are the descriptor's, as the C's
/// `h265_fill_picture_param` writes them, with the DPB's slots marked by which part of the current
/// reference picture set they are in. The slice parameters are rebuilt from each slice segment's
/// own header (see [`h265_slice`]), as H.264's are.
fn decode_hevc(
    decoder: &Decoder,
    surface: &va::Surface<()>,
    picture: &h265_slice::Picture,
    ref_surfaces: &[Option<va::VASurfaceID>; 16],
    bytes: &[u8],
) -> Result<(), Failure> {
    use va::{
        VA_PICTURE_HEVC_INVALID as INVALID, VA_PICTURE_HEVC_LONG_TERM_REFERENCE as LONG,
        VA_PICTURE_HEVC_RPS_LT_CURR as LT_CURR, VA_PICTURE_HEVC_RPS_ST_CURR_AFTER as AFTER,
        VA_PICTURE_HEVC_RPS_ST_CURR_BEFORE as BEFORE,
    };
    let desc = &picture.desc;
    let dec = &picture.decoding;
    let refused = |why| Failure::Hevc(why);
    let slices = h265_slice::slices(bytes, picture).map_err(refused)?;
    let Some(last) = slices.len().checked_sub(1) else {
        return Err(refused(h265_slice::Refused::Unsupported("an access unit with no slice")));
    };
    // VA-API's DPB has fifteen slots, the descriptor's sixteen: a set naming the last one has no
    // index to be given.
    let sets = [&dec.st_curr_before, &dec.st_curr_after, &dec.lt_curr];
    if sets.iter().any(|set| set.iter().any(|&slot| slot >= 15)) {
        return Err(refused(h265_slice::Refused::OutOfRange("a reference picture set slot")));
    }

    let invalid = va::PictureHEVC::new(NO_SURFACE, 0, INVALID);
    let reference_frames: [va::PictureHEVC; 15] = std::array::from_fn(|i| {
        let (Some(r), Some(id)) = (dec.refs[i], ref_surfaces[i]) else {
            return invalid;
        };
        let mut flags = if r.long_term { LONG } else { 0 };
        if dec.st_curr_before.contains(&i) {
            flags |= BEFORE;
        }
        if dec.st_curr_after.contains(&i) {
            flags |= AFTER;
        }
        if dec.lt_curr.contains(&i) {
            flags |= LT_CURR;
        }
        va::PictureHEVC::new(id, r.poc, flags)
    });
    // A set slot whose target this codec never decoded into would point the driver at nothing.
    if sets
        .iter()
        .any(|set| set.iter().any(|&slot| reference_frames[slot].picture_id() == NO_SURFACE))
    {
        return Err(refused(h265_slice::Refused::NoSuchReference));
    }

    let flag = u32::from;
    let pic_fields = va::HevcPicFields::new(
        u32::from(desc.chroma_format_idc),
        flag(desc.separate_colour_plane),
        flag(desc.pcm_enabled),
        flag(desc.scaling_list_enabled),
        flag(desc.transform_skip_enabled),
        flag(desc.amp_enabled),
        flag(desc.strong_intra_smoothing_enabled),
        flag(desc.sign_data_hiding_enabled),
        flag(desc.constrained_intra_pred),
        flag(desc.cu_qp_delta_enabled),
        flag(desc.weighted_pred),
        flag(desc.weighted_bipred),
        flag(desc.transquant_bypass_enabled),
        flag(desc.tiles.is_some()),
        flag(desc.entropy_coding_sync_enabled),
        flag(desc.pps_loop_filter_across_slices_enabled),
        flag(desc.loop_filter_across_tiles_enabled),
        flag(desc.pcm_loop_filter_disabled),
        0,
        0,
    );
    let intra = slices.iter().all(|s| s.header.slice_type == h265_slice::SliceType::I);
    let parsing = va::HevcSliceParsingFields::new(
        flag(desc.lists_modification_present),
        flag(desc.long_term_ref_pics_present),
        flag(desc.sps_temporal_mvp_enabled),
        flag(desc.cabac_init_present),
        flag(desc.output_flag_present),
        flag(desc.dependent_slice_segments_enabled),
        flag(desc.pps_slice_chroma_qp_offsets_present),
        flag(desc.sample_adaptive_offset_enabled),
        flag(desc.deblocking_filter_override_enabled),
        flag(desc.pps_deblocking_filter_disabled),
        flag(desc.slice_segment_header_extension_present),
        flag(dec.rap),
        flag(dec.idr),
        flag(intra),
    );
    let too_big = || refused(h265_slice::Refused::OutOfRange("the picture's extent"));
    let tiles = &dec.tiles;
    let parameters = va::PictureParameterBufferHEVC::new(
        va::PictureHEVC::new(surface.id(), dec.curr_poc, 0),
        reference_frames,
        u16::try_from(desc.pic_width_in_luma_samples).map_err(|_| too_big())?,
        u16::try_from(desc.pic_height_in_luma_samples).map_err(|_| too_big())?,
        &pic_fields,
        desc.sps_max_dec_pic_buffering_minus1,
        desc.bit_depth_luma_minus8,
        desc.bit_depth_chroma_minus8,
        desc.pcm_sample_bit_depth_luma_minus1,
        desc.pcm_sample_bit_depth_chroma_minus1,
        desc.log2_min_luma_coding_block_size_minus3,
        desc.log2_diff_max_min_luma_coding_block_size,
        desc.log2_min_transform_block_size_minus2,
        desc.log2_diff_max_min_transform_block_size,
        desc.log2_min_pcm_luma_coding_block_size_minus3,
        desc.log2_diff_max_min_pcm_luma_coding_block_size,
        desc.max_transform_hierarchy_depth_intra,
        desc.max_transform_hierarchy_depth_inter,
        desc.init_qp_minus26,
        desc.diff_cu_qp_delta_depth,
        desc.pps_cb_qp_offset,
        desc.pps_cr_qp_offset,
        desc.log2_parallel_merge_level_minus2,
        tiles.columns_minus1,
        tiles.rows_minus1,
        tiles.column_widths_minus1,
        tiles.row_heights_minus1,
        &parsing,
        desc.log2_max_pic_order_cnt_lsb_minus4,
        desc.num_short_term_ref_pic_sets,
        desc.num_long_term_ref_pics_sps,
        desc.num_ref_idx_l0_default_active_minus1,
        desc.num_ref_idx_l1_default_active_minus1,
        desc.pps_beta_offset_div2,
        desc.pps_tc_offset_div2,
        desc.num_extra_slice_header_bits,
        dec.st_rps_bits,
    );
    let mut buffers =
        vec![va::BufferType::PictureParameter(va::PictureParameter::HEVC(parameters))];
    if desc.scaling_list_enabled {
        let lists = &dec.scaling;
        buffers.push(va::BufferType::IQMatrix(va::IQMatrix::HEVC(va::IQMatrixBufferHEVC::new(
            lists.list_4x4,
            lists.list_8x8,
            lists.list_16x16,
            lists.list_32x32,
            lists.dc_16x16,
            lists.dc_32x32,
        ))));
    }

    for (n, slice) in slices.iter().enumerate() {
        let h = &slice.header;
        let lists = h265_slice::ref_lists(h, dec).map_err(refused)?;
        // Indices into ReferenceFrames, which holds the DPB slot for slot; 0xff is no entry.
        let ref_pic_list = lists.map(|list| list.map(|slot| slot.map_or(0xff, |s| s as u8)));
        let flags = va::HevcLongSliceFlags::new(
            u32::from(n == last),
            flag(slice.dependent),
            h.slice_type as u32,
            u32::from(h.colour_plane_id),
            flag(h.sao_luma),
            flag(h.sao_chroma),
            flag(h.mvd_l1_zero),
            flag(h.cabac_init),
            flag(h.temporal_mvp),
            flag(h.deblocking_filter_disabled),
            flag(h.collocated_from_l0),
            flag(h.loop_filter_across_slices),
        );
        let weights = h.weights.clone().unwrap_or_default();
        let [w0, w1] = &weights.lists;
        let too_long = || refused(h265_slice::Refused::OutOfRange("a slice's length"));
        let params = va::SliceParameterBufferHEVC::new(
            u32::try_from(slice.nal.len()).map_err(|_| too_long())?,
            0,
            va::VA_SLICE_DATA_FLAG_ALL,
            slice.data_byte_offset,
            slice.segment_address,
            ref_pic_list,
            &flags,
            if h.temporal_mvp { h.collocated_ref_idx } else { 0xff },
            h.num_ref_idx_active_minus1[0],
            h.num_ref_idx_active_minus1[1],
            h.slice_qp_delta,
            h.cb_qp_offset,
            h.cr_qp_offset,
            h.beta_offset_div2,
            h.tc_offset_div2,
            weights.luma_log2_denom,
            weights.delta_chroma_log2_denom,
            w0.delta_luma_weight,
            w0.luma_offset,
            w0.delta_chroma_weight,
            w0.chroma_offset,
            w1.delta_luma_weight,
            w1.luma_offset,
            w1.delta_chroma_weight,
            w1.chroma_offset,
            h.five_minus_max_num_merge_cand,
            slice.num_entry_point_offsets,
            0,
            slice.header_escapes,
        );
        buffers.push(va::BufferType::SliceParameter(va::SliceParameter::HEVC(params)));
        buffers.push(va::BufferType::SliceData(slice.nal.to_vec()));
    }
    render(decoder, surface, buffers)
}

/// Whether a target's planes take a GPU copy from a surface's: two per-plane textures, NV12, in
/// the formats a plane import produces -- `R8` for luma and `RG8` for the interleaved chroma --
/// since `glCopyImageSubData` copies only between compatible formats and reports a mismatch as a
/// GL error nothing would read.
fn route_of(buffer: &Arc<super::Buffer>) -> Route<()> {
    let Destination::PerPlane(planes) = &buffer.destination else {
        return Route::Composite;
    };
    if !matches!(buffer.format, Layout::Served(TargetFormat::Nv12)) {
        return Route::Layout(buffer.format);
    }
    let format = |i: usize| planes.get(i).map_or(0, |p: &super::Plane| p.gl.internalformat);
    if planes.len() == 2 && format(0) == GL_R8 && format(1) == GL_RG8 {
        Route::Gpu(())
    } else {
        Route::PlaneFormats { luma: format(0), chroma: format(1), count: planes.len() }
    }
}

/// Export a surface and image its two planes.
///
/// The export is the composed form cros-libva asks for, which on every driver measured is one
/// object with both planes in it -- what the per-plane import is written for. Anything else is
/// refused here rather than imported as a guess.
fn image(surface: &va::Surface<()>, importer: &Importer) -> Result<Imaged, String> {
    let mut exported = surface.export_prime().map_err(|why| format!("export: {why}"))?;
    let [layer] = exported.layers.as_slice() else {
        return Err(format!("{} layers", exported.layers.len()));
    };
    if exported.objects.len() != 1 || layer.num_planes != 2 {
        let objects = exported.objects.len();
        return Err(format!("{objects} objects, {} planes", layer.num_planes));
    }
    let planes: Vec<PlaneLayout> = (0..2)
        .map(|i| PlaneLayout { offset: u64::from(layer.offset[i]), pitch: layer.pitch[i] })
        .collect();
    let object = exported.objects.remove(0);
    let layout = crate::surface::Layout {
        width: exported.width,
        height: exported.height,
        fourcc: exported.fourcc,
        modifier: object.drm_format_modifier,
        planes: PlaneLayouts::new(&planes).map_err(|why| format!("{why:?}"))?,
        alloc_size: u64::from(object.size),
    };
    let held: Arc<dyn Held> = Arc::new(crate::dmabuf::Surface::exported(object.fd, layout));
    let import = |plane| {
        importer
            .image_from_surface_plane(Arc::clone(&held), plane)
            .map_err(|why| format!("import of {plane:?}: {why:?}"))
    };
    Ok(Imaged { luma: import(Plane::Luma)?, chroma: import(Plane::ChromaPair)? })
}

/// Copy the decoded picture out of its surface, at the frame's extent.
fn read_back(
    surface: &va::Surface<()>,
    nv12: va::VAImageFormat,
    width: u32,
    height: u32,
    pixels: PixelFormat,
) -> Option<Picture> {
    // An NV12 image covers whole chroma samples, so an odd extent is read at the next even one.
    let (surface_width, surface_height) = surface.size();
    let coded = (
        width.next_multiple_of(2).min(surface_width),
        height.next_multiple_of(2).min(surface_height),
    );
    let image = va::Image::create_from(surface, nv12, coded, coded)
        .map_err(|why| eprintln!("[virglrs] video: vaGetImage: {why}"))
        .ok()?;
    Picture::from_nv12(&image, width.min(coded.0), height.min(coded.1), pixels)
}
