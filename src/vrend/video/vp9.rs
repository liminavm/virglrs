// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! The VP9 picture descriptor, `struct virgl_vp9_picture_desc`, read whole.
//!
//! The two backends want different amounts of it. VideoToolbox parses the real bitstream and
//! keeps its own reference pictures, so it consults only what a container declares -- extent,
//! profile, depth, subsampling, key-ness. VA-API parses nothing: every field of
//! `VADecPictureParameterBufferVP9` and `VASliceParameterBufferVP9` comes from here, and so does
//! the reference list. One read serves both, so the two cannot disagree about what a frame was.

/// Where each field sits in `struct virgl_vp9_picture_desc`, measured with `offsetof` against
/// `virgl_video_hw.h` rather than counted by hand.
mod at {
    pub const REF: usize = 264;
    pub const FRAME_WIDTH: usize = 328;
    pub const FRAME_HEIGHT: usize = 330;
    pub const PIC_FIELDS: usize = 332;
    pub const FILTER_LEVEL: usize = 336;
    pub const SHARPNESS_LEVEL: usize = 337;
    pub const LOG2_TILE_ROWS: usize = 338;
    pub const LOG2_TILE_COLUMNS: usize = 339;
    pub const FRAME_HEADER_LENGTH_IN_BYTES: usize = 340;
    pub const FIRST_PARTITION_SIZE: usize = 342;
    pub const MB_SEGMENT_TREE_PROBS: usize = 344;
    pub const SEGMENT_PRED_PROBS: usize = 351;
    pub const PROFILE: usize = 354;
    pub const BIT_DEPTH: usize = 355;
    pub const SLICE_DATA_SIZE: usize = 372;
    pub const SLICE_DATA_OFFSET: usize = 376;
    pub const SLICE_DATA_FLAG: usize = 380;
    pub const SEG_PARAM: usize = 384;
    /// `sizeof(struct virgl_vp9_segment_parameter)`, and its fields within one.
    pub const SEG_PARAM_BYTES: usize = 18;
    pub const SEG_FILTER_LEVEL: usize = 2;
    pub const SEG_LUMA_AC: usize = 10;
    pub const SEG_LUMA_DC: usize = 12;
    pub const SEG_CHROMA_AC: usize = 14;
    pub const SEG_CHROMA_DC: usize = 16;
}

/// `sizeof(struct virgl_vp9_picture_desc)`: the whole descriptor is read.
pub const DESCRIPTOR_BYTES: usize = 528;

const _: () = assert!(at::SEG_PARAM + 8 * at::SEG_PARAM_BYTES <= DESCRIPTOR_BYTES);

/// `pic_fields`, one `uint32_t` of bitfields. Positions and widths measured the same way, against
/// the compiler's own layout of the struct.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct PicFields(pub u32);

impl PicFields {
    fn bits(self, shift: u32, width: u32) -> u32 {
        (self.0 >> shift) & ((1 << width) - 1)
    }

    pub fn subsampling_x(self) -> u32 {
        self.bits(0, 1)
    }
    pub fn subsampling_y(self) -> u32 {
        self.bits(1, 1)
    }
    /// 0 is a key frame.
    pub fn frame_type(self) -> u32 {
        self.bits(2, 1)
    }
    pub fn show_frame(self) -> u32 {
        self.bits(3, 1)
    }
    pub fn error_resilient_mode(self) -> u32 {
        self.bits(4, 1)
    }
    pub fn intra_only(self) -> u32 {
        self.bits(5, 1)
    }
    pub fn allow_high_precision_mv(self) -> u32 {
        self.bits(6, 1)
    }
    pub fn mcomp_filter_type(self) -> u32 {
        self.bits(7, 3)
    }
    pub fn frame_parallel_decoding_mode(self) -> u32 {
        self.bits(10, 1)
    }
    pub fn reset_frame_context(self) -> u32 {
        self.bits(11, 2)
    }
    pub fn refresh_frame_context(self) -> u32 {
        self.bits(13, 1)
    }
    pub fn frame_context_idx(self) -> u32 {
        self.bits(14, 2)
    }
    pub fn segmentation_enabled(self) -> u32 {
        self.bits(16, 1)
    }
    pub fn segmentation_temporal_update(self) -> u32 {
        self.bits(17, 1)
    }
    pub fn segmentation_update_map(self) -> u32 {
        self.bits(18, 1)
    }
    pub fn last_ref_frame(self) -> u32 {
        self.bits(19, 3)
    }
    pub fn last_ref_frame_sign_bias(self) -> u32 {
        self.bits(22, 1)
    }
    pub fn golden_ref_frame(self) -> u32 {
        self.bits(23, 3)
    }
    pub fn golden_ref_frame_sign_bias(self) -> u32 {
        self.bits(26, 1)
    }
    pub fn alt_ref_frame(self) -> u32 {
        self.bits(27, 3)
    }
    pub fn alt_ref_frame_sign_bias(self) -> u32 {
        self.bits(30, 1)
    }
    pub fn lossless_flag(self) -> u32 {
        self.bits(31, 1)
    }
}

/// One `struct virgl_vp9_segment_parameter`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Segment {
    /// The `segment_flags` bitfield word: `segment_reference_enabled` at bit 0,
    /// `segment_reference` at bits 1-2, `segment_reference_skipped` at bit 3.
    pub flags: u16,
    pub filter_level: [[u8; 2]; 4],
    pub luma_ac_quant_scale: i16,
    pub luma_dc_quant_scale: i16,
    pub chroma_ac_quant_scale: i16,
    pub chroma_dc_quant_scale: i16,
}

impl Segment {
    pub fn reference_enabled(&self) -> u16 {
        self.flags & 1
    }
    pub fn reference(&self) -> u16 {
        (self.flags >> 1) & 3
    }
    pub fn reference_skipped(&self) -> u16 {
        (self.flags >> 3) & 1
    }
}

/// A VP9 picture descriptor.
///
/// The few fields both backends read are named for what they mean -- [`Frame::key`],
/// [`Frame::subsampling`], an extent that falls back to the codec's -- and the rest are kept as
/// the guest wrote them, for a backend that hands them to the driver as they are.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    /// The eight reference slots, as guest decode-target handles. Zero is an empty slot. Only
    /// the first eight of the descriptor's sixteen are VP9's.
    pub refs: [u32; 8],
    pub width: u32,
    pub height: u32,
    pub pic_fields: PicFields,
    pub filter_level: u8,
    pub sharpness_level: u8,
    pub log2_tile_rows: u8,
    pub log2_tile_columns: u8,
    pub frame_header_length_in_bytes: u8,
    pub first_partition_size: u16,
    pub mb_segment_tree_probs: [u8; 7],
    pub segment_pred_probs: [u8; 3],
    pub profile: u8,
    pub bit_depth: u8,
    pub slice_data_size: u32,
    pub slice_data_offset: u32,
    pub slice_data_flag: u32,
    pub segments: [Segment; 8],
}

impl Frame {
    /// Read a descriptor the guest wrote.
    ///
    /// Total on purpose: a short descriptor reads as zeros rather than failing, which is what the
    /// C does and what the protocol allows -- the resource carries whatever the guest's driver
    /// wrote and its size is the guest's choice. An extent that reads zero falls back to the
    /// codec's own creation arguments, which is the only place a missing extent can come from.
    pub fn read(blob: &[u8], codec_width: u32, codec_height: u32) -> Frame {
        let byte = |at: usize| blob.get(at).copied().unwrap_or(0);
        let short = |at: usize| u16::from_le_bytes([byte(at), byte(at + 1)]);
        let word =
            |at: usize| u32::from_le_bytes([byte(at), byte(at + 1), byte(at + 2), byte(at + 3)]);
        let bytes = |at: usize| -> [u8; 7] { std::array::from_fn(|i| byte(at + i)) };

        let width = u32::from(short(at::FRAME_WIDTH));
        let height = u32::from(short(at::FRAME_HEIGHT));
        let tree = bytes(at::MB_SEGMENT_TREE_PROBS);
        let pred = bytes(at::SEGMENT_PRED_PROBS);
        Frame {
            refs: std::array::from_fn(|i| word(at::REF + 4 * i)),
            width: if width == 0 { codec_width } else { width },
            height: if height == 0 { codec_height } else { height },
            pic_fields: PicFields(word(at::PIC_FIELDS)),
            filter_level: byte(at::FILTER_LEVEL),
            sharpness_level: byte(at::SHARPNESS_LEVEL),
            log2_tile_rows: byte(at::LOG2_TILE_ROWS),
            log2_tile_columns: byte(at::LOG2_TILE_COLUMNS),
            frame_header_length_in_bytes: byte(at::FRAME_HEADER_LENGTH_IN_BYTES),
            first_partition_size: short(at::FIRST_PARTITION_SIZE),
            mb_segment_tree_probs: tree,
            segment_pred_probs: [pred[0], pred[1], pred[2]],
            profile: byte(at::PROFILE),
            // A descriptor that declares no depth means the only one profile 0 has.
            bit_depth: match byte(at::BIT_DEPTH) {
                0 => 8,
                depth => depth,
            },
            slice_data_size: word(at::SLICE_DATA_SIZE),
            slice_data_offset: word(at::SLICE_DATA_OFFSET),
            slice_data_flag: word(at::SLICE_DATA_FLAG),
            segments: std::array::from_fn(|i| {
                let at = at::SEG_PARAM + i * at::SEG_PARAM_BYTES;
                let level = |n: usize| byte(at + at::SEG_FILTER_LEVEL + n);
                Segment {
                    flags: short(at),
                    filter_level: std::array::from_fn(|r| [level(2 * r), level(2 * r + 1)]),
                    luma_ac_quant_scale: short(at + at::SEG_LUMA_AC) as i16,
                    luma_dc_quant_scale: short(at + at::SEG_LUMA_DC) as i16,
                    chroma_ac_quant_scale: short(at + at::SEG_CHROMA_AC) as i16,
                    chroma_dc_quant_scale: short(at + at::SEG_CHROMA_DC) as i16,
                }
            }),
        }
    }

    /// `frame_type == 0`: a key frame, which re-seeds every reference slot. An intra-only frame
    /// is not one -- it does not refresh them all.
    pub fn key(&self) -> bool {
        self.pic_fields.frame_type() == 0
    }

    /// The vpcC encoding of the chroma subsampling, not the stream's two flags: 1 is 4:2:0, 3 is
    /// 4:4:4.
    pub fn subsampling(&self) -> u8 {
        if self.pic_fields.subsampling_x() != 0 && self.pic_fields.subsampling_y() != 0 {
            1
        } else {
            3
        }
    }
}

#[cfg(test)]
pub fn test_frame(key: bool, width: u32, height: u32) -> Frame {
    let mut blob = vec![0u8; DESCRIPTOR_BYTES];
    let fields: u32 = 0b011 | if key { 0 } else { 0b100 };
    blob[at::PIC_FIELDS..][..4].copy_from_slice(&fields.to_le_bytes());
    Frame::read(&blob, width, height)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The descriptor offsets are the load-bearing numbers in this file: read one wrong and the
    /// frame is decoded with a field nobody sent. They were measured with `offsetof`, so the test
    /// builds a descriptor the same way -- by planting each field at its offset -- and checks it
    /// reads back.
    #[test]
    fn the_descriptor_fields_are_where_offsetof_put_them() {
        let mut blob = vec![0u8; DESCRIPTOR_BYTES];
        for slot in 0..16 {
            blob[at::REF + 4 * slot..][..4].copy_from_slice(&(100 + slot as u32).to_le_bytes());
        }
        blob[at::FRAME_WIDTH..][..2].copy_from_slice(&640u16.to_le_bytes());
        blob[at::FRAME_HEIGHT..][..2].copy_from_slice(&480u16.to_le_bytes());
        // subsampling_x and subsampling_y set, frame_type clear, mcomp_filter_type 5, alt_ref 6,
        // lossless set: a 4:2:0 key frame with the outermost fields at both ends of the word.
        let fields: u32 = 0b011 | (5 << 7) | (6 << 27) | (1 << 31);
        blob[at::PIC_FIELDS..][..4].copy_from_slice(&fields.to_le_bytes());
        blob[at::FILTER_LEVEL] = 7;
        blob[at::LOG2_TILE_COLUMNS] = 2;
        blob[at::FIRST_PARTITION_SIZE..][..2].copy_from_slice(&0x1234u16.to_le_bytes());
        blob[at::MB_SEGMENT_TREE_PROBS..][..7].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7]);
        blob[at::SEGMENT_PRED_PROBS..][..3].copy_from_slice(&[8, 9, 10]);
        blob[at::PROFILE] = 2;
        blob[at::BIT_DEPTH] = 10;
        blob[at::SLICE_DATA_SIZE..][..4].copy_from_slice(&4096u32.to_le_bytes());
        let last = at::SEG_PARAM + 7 * at::SEG_PARAM_BYTES;
        blob[last..][..2].copy_from_slice(&0b1101u16.to_le_bytes());
        blob[last + at::SEG_FILTER_LEVEL + 7] = 63;
        blob[last + at::SEG_CHROMA_DC..][..2].copy_from_slice(&(-3i16).to_le_bytes());

        let frame = Frame::read(&blob, 1, 1);
        assert_eq!(frame.refs, [100, 101, 102, 103, 104, 105, 106, 107], "only VP9's eight");
        assert_eq!((frame.width, frame.height), (640, 480));
        assert!(frame.key());
        assert_eq!(frame.subsampling(), 1);
        assert_eq!(frame.pic_fields.mcomp_filter_type(), 5);
        assert_eq!(frame.pic_fields.alt_ref_frame(), 6);
        assert_eq!(frame.pic_fields.lossless_flag(), 1);
        assert_eq!(frame.pic_fields.show_frame(), 0);
        assert_eq!((frame.filter_level, frame.log2_tile_columns), (7, 2));
        assert_eq!(frame.first_partition_size, 0x1234);
        assert_eq!(frame.mb_segment_tree_probs, [1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(frame.segment_pred_probs, [8, 9, 10]);
        assert_eq!((frame.profile, frame.bit_depth), (2, 10));
        assert_eq!(frame.slice_data_size, 4096);
        let seg = frame.segments[7];
        assert_eq!((seg.reference_enabled(), seg.reference(), seg.reference_skipped()), (1, 2, 1));
        assert_eq!(seg.filter_level[3][1], 63);
        assert_eq!(seg.chroma_dc_quant_scale, -3);
        assert_eq!(frame.segments[0], Segment::default());

        // frame_type set is an inter frame, and it is the bit the keyframe gate turns on.
        blob[at::PIC_FIELDS] |= 0b100;
        assert!(!Frame::read(&blob, 1, 1).key());
    }

    /// A guest need not fill the descriptor, and one shorter than the fields we read must not
    /// panic -- it reads as zeros, and zero means "take it from the codec".
    #[test]
    fn a_short_descriptor_falls_back_to_the_codec() {
        for len in [0, 1, at::PIC_FIELDS, DESCRIPTOR_BYTES - 1] {
            let frame = Frame::read(&vec![0u8; len], 352, 240);
            assert_eq!(frame.width, 352, "len {len}");
            assert_eq!(frame.height, 240, "len {len}");
            assert_eq!(frame.bit_depth, 8, "len {len}: profile 0 has only one depth");
            assert!(frame.key(), "len {len}: frame_type zero is a key frame");
        }
    }
}
