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

use super::{Backend, Buffers, Delivery, Identity, Shape, pending, vp9};
use crate::decode::{Picture, PixelFormat};
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
}

/// What the descriptor says, per codec.
enum Params {
    Vp9 {
        frame: Box<vp9::Frame>,
        /// The descriptor's eight reference slots, resolved on the render thread. `None` is an
        /// empty slot, or a handle naming no target.
        refs: [Option<Arc<Identity>>; 8],
    },
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
}

impl Backend for Host {
    type Unit = Unit;

    fn unit(
        _name: &'static str,
        shape: &Shape,
        bytes: &[u8],
        delivery: Delivery<'_>,
        pixels: Option<PixelFormat>,
        buffers: &Buffers,
    ) -> Unit {
        let resolve = |handle: u32| {
            (handle != 0)
                .then(|| buffers.get(&VideoBufferHandle(handle)))
                .flatten()
                .map(|buffer| Arc::clone(&buffer.identity))
        };
        let picture = match shape {
            Shape::Vp9(frame) => Params::Vp9 {
                frame: frame.clone(),
                refs: std::array::from_fn(|slot| resolve(frame.refs[slot])),
            },
            Shape::H264 { .. } | Shape::Hevc { .. } | Shape::Av1 { .. } => {
                unreachable!("VA-API advertises VP9 only, so no other codec is created")
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
        let Params::Vp9 { frame, refs } = &unit.picture;
        let made = va.decoder_for(va::VAProfile::VAProfileVP9Profile0, frame.width, frame.height);
        let Some(target_at) = va.surface_for(target) else {
            eprintln!("[virglrs] video codec {handle}: no VA surface for the decode target");
            return None;
        };
        if made {
            phases.create = Some(began.elapsed());
        }
        let Some(decoder) = &va.decoder else {
            eprintln!("[virglrs] video codec {handle}: no VA decode context for VP9");
            return None;
        };
        let reference_frames = refs.each_ref().map(|slot| {
            slot.as_ref().and_then(|identity| va.surface_id(identity)).unwrap_or(NO_SURFACE)
        });

        let began = std::time::Instant::now();
        let surface = &va.surfaces[target_at].surface;
        let decoded = decode_vp9(decoder, surface, frame, reference_frames, &unit.bytes);
        phases.session = Some(began.elapsed());
        if let Err(why) = decoded {
            eprintln!("[virglrs] video codec {handle}: the VA driver decoded no picture ({why})");
            return None;
        }

        let began = std::time::Instant::now();
        let picture = read_back(surface, va.nv12, frame.width, frame.height, unit.pixels);
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
        self.surfaces.push(Stored { identity: Arc::downgrade(&target.identity), surface });
        Some(self.surfaces.len() - 1)
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

    let context = &decoder.context;
    let buffer = |what, kind| context.create_buffer(kind).map_err(|why| Failure::Va(what, why));
    let mut pending = va::Picture::new(0, Rc::clone(context), surface);
    pending.add_buffer(buffer(
        "picture parameters",
        va::BufferType::PictureParameter(va::PictureParameter::VP9(picture)),
    )?);
    pending.add_buffer(buffer(
        "slice parameters",
        va::BufferType::SliceParameter(va::SliceParameter::VP9(slice)),
    )?);
    pending.add_buffer(buffer("slice data", va::BufferType::SliceData(bytes.to_vec()))?);
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
