// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! AV1 decoded in software, by dav1d.
//!
//! Two kinds of host need it: one with no AV1 silicon, where it decodes every unit, and one whose
//! hardware returns super-resolution pictures wrongly, where a codec switches to it at the first
//! such frame. Both are fed the serializer's own temporal units, so the two decoders see the same
//! bytes and the serializer stays the one source of frame headers.
//!
//! AV1 only, deliberately: it is the one codec whose hardware path has a hole in it.

use crate::decode::{PixelFormat, Plane};

/// A dav1d decoder, configured for one unit in and one picture out.
pub struct Decoder(dav1d::Decoder);

impl Decoder {
    pub fn open() -> Result<Decoder, dav1d::Error> {
        let mut settings = dav1d::Settings::new();
        // In order and one at a time: max_frame_delay 1 is what keeps output in coding order,
        // one picture per unit, which the one-unit-one-target model rests on. Frame threading
        // would buy throughput at the cost of a picture arriving after the frame that asked for it
        // has been answered.
        settings.set_max_frame_delay(1);
        // Tile and row threads within a frame reorder nothing, but not dav1d's default of every
        // core: this runs beside the guest's vCPU threads, and taking all of them to decode would
        // stall the guest it decodes for. Half, bounded, is far more than a desktop's resolutions
        // need.
        let cores = std::thread::available_parallelism().map_or(4, |n| n.get()) as u32;
        settings.set_n_threads((cores / 2).clamp(2, 8));
        // The guest allocates a surface for every decoded frame, hidden ones included, and a
        // later show_existing_frame displays one without any of it reaching the host. dav1d
        // outputs only shown pictures by default, which would leave those surfaces unwritten and
        // break one picture per unit.
        settings.set_output_invisible_frames(true);
        dav1d::Decoder::with_settings(&settings).map(Decoder)
    }

    /// Decode one temporal unit. `None` is a unit that produced no picture.
    ///
    /// More than one picture from one unit is the serializer splitting frames wrongly; the last
    /// is the one kept, which is the one a player would have shown.
    pub fn decode(&mut self, unit: &[u8]) -> Result<Option<dav1d::Picture>, dav1d::Error> {
        let mut picture = None;
        // dav1d keeps a reference to what it is given, so it gets its own copy.
        let mut sent = self.0.send_data(unit.to_vec(), None, None, None);
        // Again means a picture must be taken before the rest of the unit fits. Sending anything
        // else before the pending data has gone is a panic in the binding, so it is drained here
        // and nowhere else.
        while matches!(sent, Err(dav1d::Error::Again)) {
            match self.0.get_picture() {
                Ok(taken) => picture = Some(taken),
                Err(dav1d::Error::Again) => {}
                Err(why) => return Err(why),
            }
            sent = self.0.send_pending_data();
        }
        sent?;
        loop {
            match self.0.get_picture() {
                Ok(taken) => picture = Some(taken),
                Err(dav1d::Error::Again) => return Ok(picture),
                Err(why) => return Err(why),
            }
        }
    }
}

/// One plane of a picture, copied out of dav1d's pool.
struct Owned {
    width: u32,
    height: u32,
    pitch: usize,
    bytes: Vec<u8>,
}

/// A decoded picture, laid out the way its target wants it.
///
/// Owned rather than borrowed from dav1d: the picture leaves the decode thread for the render
/// thread's per-plane uploads, and a biplanar target needs its chroma interleaved, which dav1d
/// cannot produce, so the bytes are rewritten anyway.
pub struct Picture {
    planes: Vec<Owned>,
}

/// Why a decoded picture cannot be delivered.
#[derive(Debug, PartialEq, Eq)]
pub enum Undeliverable {
    /// The targets are 8-bit.
    BitDepth(usize),
    /// The targets are 4:2:0.
    Layout(dav1d::PixelLayout),
}

impl std::fmt::Display for Undeliverable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Undeliverable::BitDepth(bits) => write!(f, "a {bits}-bit picture"),
            Undeliverable::Layout(layout) => write!(f, "a {layout:?} picture"),
        }
    }
}

impl Picture {
    /// Copy a decoded picture out in `pixels`' layout.
    pub fn new(decoded: &dav1d::Picture, pixels: PixelFormat) -> Result<Picture, Undeliverable> {
        use dav1d::PlanarImageComponent::{U, V, Y};
        if decoded.bit_depth() != 8 {
            return Err(Undeliverable::BitDepth(decoded.bit_depth()));
        }
        if decoded.pixel_layout() != dav1d::PixelLayout::I420 {
            return Err(Undeliverable::Layout(decoded.pixel_layout()));
        }
        let (width, height) = (decoded.width(), decoded.height());
        let (chroma_width, chroma_height) = (width.div_ceil(2), height.div_ceil(2));
        let plane = |component| {
            let bytes = decoded.plane(component);
            (bytes.to_vec(), decoded.stride(component) as usize)
        };
        let (luma, luma_pitch) = plane(Y);
        let luma = Owned { width, height, pitch: luma_pitch, bytes: luma };
        let (u, u_pitch) = plane(U);
        let (v, v_pitch) = plane(V);
        let planes = match pixels {
            PixelFormat::Planar420 => {
                let chroma = |bytes, pitch| Owned {
                    width: chroma_width,
                    height: chroma_height,
                    pitch,
                    bytes,
                };
                vec![luma, chroma(u, u_pitch), chroma(v, v_pitch)]
            }
            PixelFormat::BiPlanar420 => {
                let interleaved = interleave(
                    (&u, u_pitch),
                    (&v, v_pitch),
                    chroma_width as usize,
                    chroma_height as usize,
                );
                let pitch = chroma_width as usize * 2;
                vec![
                    luma,
                    Owned { width: chroma_width, height: chroma_height, pitch, bytes: interleaved },
                ]
            }
        };
        Ok(Picture { planes })
    }

    pub fn plane_count(&self) -> usize {
        self.planes.len()
    }

    /// One plane, or `None` past the end.
    pub fn plane(&self, index: usize) -> Option<Plane<'_>> {
        let Owned { width, height, pitch, bytes } = self.planes.get(index)?;
        Some(Plane { width: *width, height: *height, pitch: *pitch, bytes })
    }
}

/// Two chroma planes interleaved into one, U first: NV12's second plane, tightly pitched.
fn interleave(
    (u, u_pitch): (&[u8], usize),
    (v, v_pitch): (&[u8], usize),
    width: usize,
    height: usize,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(width * 2 * height);
    for (u_row, v_row) in u.chunks(u_pitch).zip(v.chunks(v_pitch)).take(height) {
        for (u, v) in u_row[..width].iter().zip(&v_row[..width]) {
            out.extend_from_slice(&[*u, *v]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Four 64x48 frames of ffmpeg's `testsrc`, encoded by aomenc with no lag, so one frame per
    /// temporal unit and every one shown.
    const STREAM: &[u8] = include_bytes!("testdata/testsrc-64x48.ivf");

    /// FNV-1a of each frame's tight I420, as libaom's `aomdec --rawvideo` decodes the stream.
    const LIBAOM: [u64; 4] = [
        0x78c8_f8ae_1b5d_2db9,
        0x1cca_14f2_11da_148f,
        0xbc04_0c3b_8175_bf43,
        0xead0_0a11_5c30_eef8,
    ];

    /// The temporal units of an IVF file: a 32-byte file header, then per frame a 4-byte length
    /// and an 8-byte timestamp before the unit.
    fn units(ivf: &[u8]) -> Vec<&[u8]> {
        let mut rest = &ivf[32..];
        let mut out = Vec::new();
        while rest.len() >= 12 {
            let len = u32::from_le_bytes(rest[..4].try_into().unwrap()) as usize;
            out.push(&rest[12..12 + len]);
            rest = &rest[12 + len..];
        }
        out
    }

    fn fnv(planes: &[Plane<'_>]) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for plane in planes {
            let row_bytes = plane.bytes.len() / plane.height as usize;
            assert_eq!(row_bytes, plane.pitch);
            for y in 0..plane.height {
                let row = &plane.row(y).unwrap()[..plane.width as usize];
                for &b in row {
                    hash = (hash ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
                }
            }
        }
        hash
    }

    /// Every unit gives exactly one picture, and each matches libaom's decode byte for byte.
    #[test]
    fn each_unit_decodes_to_the_picture_libaom_decodes() {
        let mut decoder = Decoder::open().expect("dav1d opens");
        let units = units(STREAM);
        assert_eq!(units.len(), LIBAOM.len());
        for (unit, expected) in units.into_iter().zip(LIBAOM) {
            let decoded = decoder.decode(unit).expect("decodes").expect("a picture per unit");
            assert_eq!((decoded.width(), decoded.height()), (64, 48));
            let picture = Picture::new(&decoded, PixelFormat::Planar420).expect("8-bit 4:2:0");
            let planes: Vec<_> = (0..3).map(|i| picture.plane(i).unwrap()).collect();
            assert_eq!(fnv(&planes), expected);
        }
    }

    /// A biplanar target gets the same luma and the chroma interleaved U first.
    #[test]
    fn a_biplanar_picture_interleaves_the_chroma_planes() {
        let mut decoder = Decoder::open().expect("dav1d opens");
        let decoded = decoder.decode(units(STREAM)[0]).expect("decodes").expect("a picture");
        let planar = Picture::new(&decoded, PixelFormat::Planar420).unwrap();
        let biplanar = Picture::new(&decoded, PixelFormat::BiPlanar420).unwrap();
        assert_eq!(biplanar.plane_count(), 2);
        assert_eq!(biplanar.plane(0).unwrap().bytes, planar.plane(0).unwrap().bytes);
        let (u, v, uv) =
            (planar.plane(1).unwrap(), planar.plane(2).unwrap(), biplanar.plane(1).unwrap());
        assert_eq!((uv.width, uv.height, uv.pitch), (32, 24, 64));
        for y in 0..24 {
            let (u, v, uv) = (u.row(y).unwrap(), v.row(y).unwrap(), uv.row(y).unwrap());
            for x in 0..32 {
                assert_eq!((uv[2 * x], uv[2 * x + 1]), (u[x], v[x]), "chroma at ({x}, {y})");
            }
        }
    }

    /// Nine 64x48 frames of `testsrc`, encoded by aomenc with lag and alt-refs: four frames are
    /// hidden, and three units show an earlier one again with `show_existing_frame`.
    const HIDDEN: &[u8] = include_bytes!("testdata/testsrc-64x48-hidden.ivf");

    /// FNV-1a of each frame libaom's `aomdec --rawvideo` outputs for [`HIDDEN`]: the shown
    /// ones, in display order.
    const LIBAOM_SHOWN: [u64; 9] = [
        0x3480_dba3_9378_5dbf,
        0x88f6_b005_fa8c_9a17,
        0x347f_ca83_5e6b_9791,
        0x5542_510c_1d3b_6434,
        0x7d89_7cce_e388_86a2,
        0x9863_a276_617d_52fa,
        0xa46f_f125_7ecb_6a6b,
        0x5008_56c6_8eda_9ed3,
        0x0fdf_6945_bb74_3f68,
    ];

    /// An AV1 `leb128`, and how many bytes it took.
    fn leb128(bytes: &[u8]) -> (usize, usize) {
        let mut value = 0;
        for (i, b) in bytes.iter().enumerate() {
            value |= usize::from(b & 0x7f) << (7 * i);
            if b & 0x80 == 0 {
                return (value, i + 1);
            }
        }
        panic!("a leb128 runs off the end");
    }

    /// Split each temporal unit into one unit per frame, the way the serializer emits them: a
    /// temporal delimiter, the sequence header where the unit had one, and one frame. Each comes
    /// with whether that frame is shown.
    fn frames(ivf: &[u8]) -> Vec<(Vec<u8>, bool)> {
        const TEMPORAL_DELIMITER: u8 = 2;
        const FRAME_HEADER: u8 = 3;
        const FRAME: u8 = 6;
        let mut out = Vec::new();
        for unit in units(ivf) {
            let mut prefix = Vec::new();
            let mut at = 0;
            while at < unit.len() {
                let kind = (unit[at] >> 3) & 0xf;
                let extension = usize::from((unit[at] >> 2) & 1);
                let (size, leb) = leb128(&unit[at + 1 + extension..]);
                let body = at + 1 + extension + leb;
                let obu = &unit[at..body + size];
                match kind {
                    FRAME_HEADER | FRAME => {
                        let first = unit[body];
                        let shown = first & 0x80 != 0 || first & 0x10 != 0;
                        let mut one = vec![TEMPORAL_DELIMITER << 3 | 2, 0];
                        one.extend_from_slice(&prefix);
                        one.extend_from_slice(obu);
                        out.push((one, shown));
                        prefix.clear();
                    }
                    TEMPORAL_DELIMITER => {}
                    _ => prefix.extend_from_slice(obu),
                }
                at = body + size;
            }
        }
        out
    }

    /// A unit holding a hidden frame still gives its picture, because the guest allocated a
    /// surface for it and a later `show_existing_frame` displays that surface without anything
    /// reaching the host. And the shown pictures are libaom's, frame for frame.
    #[test]
    fn a_hidden_frame_gives_its_picture_too() {
        let mut decoder = Decoder::open().expect("dav1d opens");
        let frames = frames(HIDDEN);
        assert_eq!(frames.iter().filter(|(_, shown)| !shown).count(), 4, "four hidden frames");
        let mut shown = Vec::new();
        for (index, (unit, is_shown)) in frames.iter().enumerate() {
            let decoded = decoder.decode(unit).expect("decodes");
            let decoded = decoded.unwrap_or_else(|| panic!("unit {index} gave no picture"));
            if *is_shown {
                let picture = Picture::new(&decoded, PixelFormat::Planar420).unwrap();
                let planes: Vec<_> = (0..3).map(|i| picture.plane(i).unwrap()).collect();
                shown.push(fnv(&planes));
            }
        }
        assert_eq!(shown, LIBAOM_SHOWN);
    }

    /// A decoder started mid-stream on a unit that carries the history behind it gives that
    /// unit the picture a decoder that saw the whole stream gives -- and one started without the
    /// history does not.
    #[test]
    fn a_decoder_started_with_the_history_catches_up() {
        use crate::vrend::proto::VideoCodecHandle;
        use crate::vrend::video::{Picture as Decoded, Software, SoftwareUnit};
        let frames: Vec<Vec<u8>> = frames(HIDDEN).into_iter().map(|(unit, _)| unit).collect();
        let at = frames.len() - 1;
        let hash = |picture: Option<Decoded>| {
            let Some(Decoded::Software(picture)) = picture else {
                return None;
            };
            Some(fnv(&(0..3).map(|i| picture.plane(i).unwrap()).collect::<Vec<_>>()))
        };
        let unit = |replay| SoftwareUnit {
            bytes: frames[at].clone(),
            replay,
            pixels: Some(PixelFormat::Planar420),
        };
        let handle = VideoCodecHandle(1);

        let mut whole = Software::default();
        for frame in &frames[..at] {
            whole
                .decode(handle, &SoftwareUnit { bytes: frame.clone(), replay: None, pixels: None });
        }
        let expected = hash(whole.decode(handle, &unit(None))).expect("the last picture");

        let mut caught_up = Software::default();
        let replay = Some(frames[..at].to_vec());
        assert_eq!(hash(caught_up.decode(handle, &unit(replay))), Some(expected));

        let mut cold = Software::default();
        assert_ne!(
            hash(cold.decode(handle, &unit(None))),
            Some(expected),
            "no history, no picture"
        );
    }

    /// Bytes that are not AV1 are an error from the decoder, not a panic and not a picture.
    #[test]
    fn a_unit_that_is_not_av1_is_an_error() {
        let mut decoder = Decoder::open().expect("dav1d opens");
        assert!(decoder.decode(&[0xff; 64]).is_err());
    }
}
