// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! The backend for a host decoder that takes whole access units into a session: VideoToolbox on
//! macOS, and the decoder-less host elsewhere, whose session cannot be built.
//!
//! What such a decoder wants is a bitstream and the configuration record a container would
//! declare for it, so everything here is in those terms: the unit is the re-framed access unit,
//! and the session is keyed on the parameter sets the descriptors were turned back into.

use super::{Backend, Delivery, HevcSource, Lookup, Shape, h265, pending};
use crate::decode::{Configuration, Picture, PixelFormat, Session, SessionKey};
use crate::vrend::proto::VideoCodecHandle;

/// One unit for the decode thread.
pub struct Unit {
    /// Which codec the host was asked for, for a message about the host.
    name: &'static str,
    /// The extent the picture must come back at, and the configuration record the session is
    /// keyed on.
    width: u32,
    height: u32,
    config: Configuration,
    /// The layout the target wants, or `None` for a unit with no target: that expresses no
    /// opinion about the layout, so the session keeps the one it has. Rebuilding it around a
    /// default would tear a live session down mid-stream on any layout but NV12.
    pixels: Option<PixelFormat>,
    bytes: Vec<u8>,
    /// The frame's picture is known to come back wrong on this host; see [`Shape::misreturned`].
    misreturned: bool,
    /// Whether that picture was meant for a target, which is the only case worth saying so.
    withheld: bool,
}

/// VideoToolbox takes length-prefixed NALs only, so an Annex-B access unit's framing is rewritten
/// and nothing else: the emulation-prevention bytes inside each NAL stay exactly as the encoder
/// wrote them. One rewrite for H.264 and HEVC, because NAL framing is the one thing the two did not
/// change between them -- which is why the C reaches for its H.264 function here too.
pub(super) fn reframe(bitstream: Vec<u8>) -> Option<Vec<u8>> {
    super::h264::annexb_to_avcc(&bitstream)
}

/// What an HEVC frame is decoded from here: the three parameter sets the session is keyed on.
pub(super) type HevcInput = h265::ParameterSets;

/// Write the frame's parameter sets, or `None` while no slice header has arrived to say which
/// PPS the frame uses.
///
/// The inspection is not only for the id: it establishes that no slice predicts from a reference
/// picture set declared in the SPS, whose contents are absent from the wire and are therefore
/// written empty. A slice that does is refused here rather than decoded into quietly wrong pixels.
pub(super) fn hevc_input(source: HevcSource<'_>) -> Result<Option<HevcInput>, String> {
    let HevcSource { desc, accumulated, ref_pic_sets, extent: (width, height), profile, .. } =
        source;
    match desc.slice_inspect(accumulated, ref_pic_sets) {
        Ok(None) => return Ok(None),
        Ok(Some(_id)) => {}
        Err(why) => return Err(format!("slice refused: {why}")),
    }
    desc.parameter_sets(width, height, profile)
        .map(Some)
        .map_err(|why| format!("no parameter set: {why}"))
}

impl Shape {
    /// The codec configuration record a session for this frame is built around.
    pub(super) fn configuration(&self) -> Configuration {
        match self {
            Shape::Vp9(frame) => {
                Configuration::vp9(frame.profile, frame.bit_depth, frame.subsampling())
            }
            Shape::H264 { sets, .. } => Configuration::h264(sets.sps.clone(), sets.pps.clone()),
            Shape::Hevc { input: sets, .. } => {
                Configuration::hevc(sets.vps.clone(), sets.sps.clone(), sets.pps.clone())
            }
            Shape::Av1 { config, .. } => Configuration::av1c(config.clone()),
        }
    }
}

/// What a codec's decode thread keeps between units: the decompression session.
///
/// Built on the first unit, not at creation: it is keyed on the shape of the frame it will
/// decode, and that arrives with the descriptor rather than with the creation arguments -- a VP9
/// stream may change resolution or bit depth at a key frame.
#[derive(Default)]
pub struct Host {
    session: Option<Session>,
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
        let (width, height) = shape.extent();
        Unit {
            name: lookup.name,
            width,
            height,
            config: shape.configuration(),
            pixels,
            bytes: bytes.to_vec(),
            misreturned: shape.misreturned(),
            withheld: matches!(delivery, Delivery::Withheld(_)),
        }
    }

    fn decode(
        &mut self,
        handle: VideoCodecHandle,
        unit: &Unit,
        phases: &mut pending::Phases,
    ) -> Option<Picture> {
        let session = &mut self.session;
        let pixels = unit
            .pixels
            .unwrap_or_else(|| session.as_ref().map_or(PixelFormat::BiPlanar420, Session::pixels));
        let key = SessionKey {
            width: unit.width,
            height: unit.height,
            pixels,
            config: unit.config.clone(),
        };
        // Rebuilt only when the frame's shape actually changes: a rebuild takes the reference
        // pictures with it, and every frame after one that did not need it then predicts from an
        // empty buffer -- which decodes "successfully" and looks like slightly wrong colour.
        let began = std::time::Instant::now();
        let rebuilt = !session.as_ref().is_some_and(|s| s.serves(&key));
        if rebuilt && !adopt(session, &key, handle) {
            match Session::create(key) {
                Ok(created) => *session = Some(created),
                Err(status) => {
                    // The probe advertised this codec, so a host that now says it has no such
                    // decoder is contradicting itself and every later frame will fail the same
                    // way.
                    assert!(
                        !status.is_no_such_decoder(),
                        "VideoToolbox advertised {} and then had no decoder for it",
                        unit.name,
                    );
                    eprintln!("[virglrs] video codec {handle}: no decode session ({status:?})");
                    return None;
                }
            }
        }
        if rebuilt {
            phases.create = Some(began.elapsed());
        }
        let live = session.as_mut().expect("a session was just built or kept");

        let began = std::time::Instant::now();
        let decoded = live.decode(&unit.bytes);
        phases.session = Some(began.elapsed());
        let picture = match decoded {
            Ok(picture) => picture,
            Err(why) => {
                eprintln!("[virglrs] video codec {handle}: the host decoded no picture ({why:?})");
                return None;
            }
        };
        // Decoded, which is all a frame the host returns wrong was submitted for. Ahead of the
        // width check: such a frame comes back at some other width, and whatever it comes back
        // at, it is not refused -- the decoder has it, and later frames predict from it.
        if unit.misreturned {
            // Only a withheld delivery is news. A re-emission claiming its slot never had a
            // picture to deliver, and its first emission already said this.
            if unit.withheld {
                eprintln!(
                    "[virglrs] video codec {handle}: this host does not return AV1 \
                     super-resolution frames correctly; the frame is decoded for later frames to \
                     predict from, but its picture is withheld and the target keeps what it held"
                );
            }
            return None;
        }
        // The picture comes back at its coded width. A host returning some other width has
        // returned something that is not this frame, and delivering it puts visibly wrong
        // content on screen with nothing anywhere reporting a problem.
        if picture.width() != unit.width {
            eprintln!(
                "[virglrs] video codec {handle}: the host returned a {}-wide picture for a frame \
                 that declares {}; refusing it",
                picture.width(),
                unit.width,
            );
            return None;
        }
        Some(picture)
    }
}

/// Try to carry the live session across a change in the frame's shape.
///
/// **H.264's parameter sets are not constant across a stream, and tearing the session down when
/// they change is not survivable.** `num_ref_idx_lX_active_minus1` reaches us as the effective
/// *per-slice* count, so a slice that overrides the PPS default changes the PPS written for it by
/// a byte or two mid-GOP. Keying the session on those bytes rebuilds the decompression session
/// there and takes the reference pictures with it: every frame after the first override predicts
/// from an empty buffer, which decodes "successfully" and puts quietly wrong pixels on screen.
///
/// So the parameter sets drive the format description, and the session is asked whether it will
/// take the new one. Falling through to a rebuild stays correct, just lossy -- and says so,
/// because a stream that does it every frame is worth knowing about.
fn adopt(session: &mut Option<Session>, key: &SessionKey, handle: VideoCodecHandle) -> bool {
    let Some(live) = session.as_mut() else {
        return false;
    };
    if live.adopt(key) {
        return true;
    }
    if key.config.is_parameter_sets() {
        eprintln!(
            "[virglrs] video codec {handle}: the parameter sets changed in a way the live session \
             would not take; its reference pictures are lost across the rebuild"
        );
    }
    false
}
