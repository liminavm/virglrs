// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! HEVC slice segment headers and reference picture lists, for a decoder that is handed
//! parameters rather than a bitstream.
//!
//! The guest's descriptor carries the picture's parameter sets and its reference picture set
//! already resolved -- which DPB slots are before, after and long-term -- but nothing per slice.
//! VA-API wants each slice's header fields and its final lists (8.3.4), so they are rebuilt here
//! from the slice segment headers the guest also sent, as [`super::h264_slice`] does for H.264.
//! The short-term set a slice header carries is skipped by the length the descriptor gives,
//! because its result is already in the descriptor.

use std::fmt;

use super::bitstream::{NalUnits, Reader};
use super::h265::PictureDesc;

/// The last VCL NAL unit type: every slice segment is 0..=31, and the reserved ones are refused
/// by name below rather than parsed as a slice.
const NAL_LAST_VCL: u8 = 31;
const NAL_IDR_W_RADL: u8 = 19;
const NAL_IDR_N_LP: u8 = 20;
const NAL_BLA_W_LP: u8 = 16;
const NAL_RSV_IRAP_23: u8 = 23;

/// Where the decode-side fields sit in `struct virgl_h265_picture_desc`, measured with
/// `offsetof`.
mod at {
    pub const SPS_SCALINGLIST4X4: usize = 284;
    pub const SPS_SCALINGLIST8X8: usize = 380;
    pub const SPS_SCALINGLIST16X16: usize = 764;
    pub const SPS_SCALINGLIST32X32: usize = 1148;
    pub const SPS_SCALINGLISTDCCOEFF16X16: usize = 1276;
    pub const SPS_SCALINGLISTDCCOEFF32X32: usize = 1282;
    pub const PPS_COLUMN_WIDTH_MINUS1: usize = 1320;
    pub const PPS_ROW_HEIGHT_MINUS1: usize = 1360;
    pub const PPS_NUM_TILE_COLUMNS_MINUS1: usize = 1404;
    pub const PPS_NUM_TILE_ROWS_MINUS1: usize = 1405;
    pub const PPS_UNIFORM_SPACING_FLAG: usize = 1406;
    pub const CURR_PIC_ORDER_CNT_VAL: usize = 1420;
    pub const REF: usize = 1424;
    pub const PIC_ORDER_CNT_VAL: usize = 1488;
    pub const NUM_SHORT_TERM_PICTURE_SLICE_HEADER_BITS: usize = 1560;
    pub const IS_LONG_TERM: usize = 1568;
    pub const IDR_PIC_FLAG: usize = 1584;
    pub const RAP_PIC_FLAG: usize = 1585;
    pub const NUM_POC_ST_CURR_BEFORE: usize = 1587;
    pub const NUM_POC_ST_CURR_AFTER: usize = 1588;
    pub const NUM_POC_LT_CURR: usize = 1589;
    pub const REF_PIC_SET_ST_CURR_BEFORE: usize = 1592;
    pub const REF_PIC_SET_ST_CURR_AFTER: usize = 1600;
    pub const REF_PIC_SET_LT_CURR: usize = 1608;
}

/// A whole HEVC picture descriptor: what a parameter set is written out of, and the rest.
pub struct Picture {
    pub desc: PictureDesc,
    pub decoding: Decoding,
}

/// One picture in the DPB.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reference {
    /// The guest's decode-target handle for it.
    pub buffer: u32,
    pub poc: i32,
    pub long_term: bool,
}

/// The effective scaling lists, in VA-API's layout.
///
/// Not the wire's: mesa's VA frontend reorders every list as it copies it from the application,
/// `desc[j] = va[UP_RIGHT_DIAGONAL[j]]`, so each is put back here by the inverse. The DC terms
/// are copied as they are.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScalingLists {
    pub list_4x4: [[u8; 16]; 6],
    pub list_8x8: [[u8; 64]; 6],
    pub list_16x16: [[u8; 64]; 6],
    pub list_32x32: [[u8; 64]; 2],
    pub dc_16x16: [u8; 6],
    pub dc_32x32: [u8; 2],
}

/// The PPS's tiles, as VA-API states them: a width and a height per tile, with uniform spacing
/// worked out here (6.5.1), since VA-API carries no flag for it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileSizes {
    pub columns_minus1: u8,
    pub rows_minus1: u8,
    pub column_widths_minus1: [u16; 19],
    pub row_heights_minus1: [u16; 21],
}

/// What the descriptor says about the picture being decoded and the ones it may predict from,
/// beyond what a parameter set is written out of ([`PictureDesc`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decoding {
    pub curr_poc: i32,
    /// The DPB, slot by slot. `None` is an empty slot.
    pub refs: [Option<Reference>; 16],
    /// The current reference picture set, as DPB slots: `RefPicSetStCurrBefore`, nearest first;
    /// `RefPicSetStCurrAfter`, nearest first; `RefPicSetLtCurr`.
    pub st_curr_before: Vec<usize>,
    pub st_curr_after: Vec<usize>,
    pub lt_curr: Vec<usize>,
    /// How many bits the short-term set in each slice header takes, when it is there.
    pub st_rps_bits: u32,
    pub idr: bool,
    pub rap: bool,
    pub scaling: ScalingLists,
    pub tiles: TileSizes,
}

impl Decoding {
    /// Read the decode-side fields of a descriptor. Total, as [`PictureDesc::read`] is: a short
    /// descriptor reads as zeros.
    pub fn read(blob: &[u8], desc: &PictureDesc) -> Decoding {
        let byte = |at: usize| blob.get(at).copied().unwrap_or(0);
        let word = |at: usize| [byte(at), byte(at + 1), byte(at + 2), byte(at + 3)];
        let unsigned = |at: usize| u32::from_le_bytes(word(at));
        let signed = |at: usize| i32::from_le_bytes(word(at));
        let half = |at: usize| u16::from_le_bytes([byte(at), byte(at + 1)]);
        // A set's entries, clipped to the eight the wire has room for.
        let set = |count: usize, at: usize| {
            (0..usize::from(byte(count)).min(8)).map(|i| usize::from(byte(at + i))).collect()
        };

        let columns_minus1 = byte(at::PPS_NUM_TILE_COLUMNS_MINUS1);
        let rows_minus1 = byte(at::PPS_NUM_TILE_ROWS_MINUS1);
        let mut tiles = TileSizes {
            columns_minus1,
            rows_minus1,
            column_widths_minus1: std::array::from_fn(|i| {
                half(at::PPS_COLUMN_WIDTH_MINUS1 + 2 * i)
            }),
            row_heights_minus1: std::array::from_fn(|i| half(at::PPS_ROW_HEIGHT_MINUS1 + 2 * i)),
        };
        if byte(at::PPS_UNIFORM_SPACING_FLAG) != 0 {
            let (width, height) = desc.size_in_ctbs();
            uniform(&mut tiles.column_widths_minus1, columns_minus1, width);
            uniform(&mut tiles.row_heights_minus1, rows_minus1, height);
        }

        Decoding {
            curr_poc: signed(at::CURR_PIC_ORDER_CNT_VAL),
            refs: std::array::from_fn(|i| {
                let buffer = unsigned(at::REF + 4 * i);
                // Handles are assigned from 1, so 0 is a slot with no picture in it.
                (buffer != 0).then(|| Reference {
                    buffer,
                    poc: signed(at::PIC_ORDER_CNT_VAL + 4 * i),
                    long_term: byte(at::IS_LONG_TERM + i) != 0,
                })
            }),
            st_curr_before: set(at::NUM_POC_ST_CURR_BEFORE, at::REF_PIC_SET_ST_CURR_BEFORE),
            st_curr_after: set(at::NUM_POC_ST_CURR_AFTER, at::REF_PIC_SET_ST_CURR_AFTER),
            lt_curr: set(at::NUM_POC_LT_CURR, at::REF_PIC_SET_LT_CURR),
            st_rps_bits: unsigned(at::NUM_SHORT_TERM_PICTURE_SLICE_HEADER_BITS),
            idr: byte(at::IDR_PIC_FLAG) != 0,
            rap: byte(at::RAP_PIC_FLAG) != 0,
            scaling: ScalingLists {
                list_4x4: std::array::from_fn(|l| {
                    unscan(|j| byte(at::SPS_SCALINGLIST4X4 + 16 * l + j), &UP_RIGHT_DIAGONAL_4X4)
                }),
                list_8x8: std::array::from_fn(|l| {
                    unscan(|j| byte(at::SPS_SCALINGLIST8X8 + 64 * l + j), &UP_RIGHT_DIAGONAL_8X8)
                }),
                list_16x16: std::array::from_fn(|l| {
                    unscan(|j| byte(at::SPS_SCALINGLIST16X16 + 64 * l + j), &UP_RIGHT_DIAGONAL_8X8)
                }),
                list_32x32: std::array::from_fn(|l| {
                    unscan(|j| byte(at::SPS_SCALINGLIST32X32 + 64 * l + j), &UP_RIGHT_DIAGONAL_8X8)
                }),
                dc_16x16: std::array::from_fn(|i| byte(at::SPS_SCALINGLISTDCCOEFF16X16 + i)),
                dc_32x32: std::array::from_fn(|i| byte(at::SPS_SCALINGLISTDCCOEFF32X32 + i)),
            },
            tiles,
        }
    }

    /// `NumPicTotalCurr` (7-55), for a stream with no current-picture referencing.
    fn num_pic_total_curr(&self) -> usize {
        self.st_curr_before.len() + self.st_curr_after.len() + self.lt_curr.len()
    }
}

/// The up-right diagonal scan of a 4x4 block (6.5.3), as mesa's `vl_zscan_h265_up_right_diagonal_16`.
const UP_RIGHT_DIAGONAL_4X4: [usize; 16] = [0, 4, 1, 8, 5, 2, 12, 9, 6, 3, 13, 10, 7, 14, 11, 15];

/// The same for an 8x8 block, as mesa's `vl_zscan_h265_up_right_diagonal`.
const UP_RIGHT_DIAGONAL_8X8: [usize; 64] = [
    0, 8, 1, 16, 9, 2, 24, 17, 10, 3, 32, 25, 18, 11, 4, 40, 33, 26, 19, 12, 5, 48, 41, 34, 27, 20,
    13, 6, 56, 49, 42, 35, 28, 21, 14, 7, 57, 50, 43, 36, 29, 22, 15, 58, 51, 44, 37, 30, 23, 59,
    52, 45, 38, 31, 60, 53, 46, 39, 61, 54, 47, 62, 55, 63,
];

/// Undo mesa's reordering of one list: what it stored at `j` came from `scan[j]`.
fn unscan<const N: usize>(stored: impl Fn(usize) -> u8, scan: &[usize; N]) -> [u8; N] {
    let mut out = [0; N];
    for (j, &from) in scan.iter().enumerate() {
        out[from] = stored(j);
    }
    out
}

/// Uniformly spaced tile sizes (6-3, 6-4): `count_minus1 + 1` tiles across `ctbs`.
fn uniform(sizes: &mut [u16], count_minus1: u8, ctbs: u32) {
    let n = u32::from(count_minus1) + 1;
    for (i, size) in sizes.iter_mut().enumerate().take(n as usize) {
        let i = i as u32;
        let width = ((i + 1) * ctbs) / n - (i * ctbs) / n;
        *size = width.saturating_sub(1).min(u32::from(u16::MAX)) as u16;
    }
}

/// `slice_type` (Table 7-7).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SliceType {
    B = 0,
    P = 1,
    I = 2,
}

impl SliceType {
    /// How many reference lists a slice of this type has.
    fn lists(self) -> usize {
        match self {
            SliceType::B => 2,
            SliceType::P => 1,
            SliceType::I => 0,
        }
    }
}

/// One list's explicit weights (7.3.6.3), VA-API's shape: deltas as sent, and the chroma offset
/// already derived (7-56), since that is what VA-API asks for.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ListWeights {
    pub delta_luma_weight: [i8; 15],
    pub luma_offset: [i8; 15],
    pub delta_chroma_weight: [[i8; 2]; 15],
    pub chroma_offset: [[i8; 2]; 15],
}

/// `pred_weight_table()`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PredWeights {
    pub luma_log2_denom: u8,
    pub delta_chroma_log2_denom: i8,
    pub lists: [ListWeights; 2],
}

/// The fields of an independent slice segment header (7.3.6.1) that a parameter-buffer decoder
/// is told. A dependent segment takes these from the independent one before it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SliceHeader {
    pub slice_type: SliceType,
    pub colour_plane_id: u8,
    pub sao_luma: bool,
    pub sao_chroma: bool,
    pub temporal_mvp: bool,
    pub num_ref_idx_active_minus1: [u8; 2],
    /// `list_entry_lX`, when the slice modifies the list.
    pub list_entries: [Option<Vec<u8>>; 2],
    pub mvd_l1_zero: bool,
    pub cabac_init: bool,
    pub collocated_from_l0: bool,
    pub collocated_ref_idx: u8,
    pub weights: Option<PredWeights>,
    pub five_minus_max_num_merge_cand: u8,
    pub slice_qp_delta: i8,
    pub cb_qp_offset: i8,
    pub cr_qp_offset: i8,
    pub deblocking_filter_disabled: bool,
    pub beta_offset_div2: i8,
    pub tc_offset_div2: i8,
    pub loop_filter_across_slices: bool,
}

/// One slice segment of an access unit: the NAL as sent, and its header.
pub struct Slice<'a> {
    pub nal: &'a [u8],
    pub dependent: bool,
    pub segment_address: u32,
    /// The independent header this segment decodes under: its own, or for a dependent segment
    /// the one before it.
    pub header: SliceHeader,
    pub num_entry_point_offsets: u16,
    /// Where `slice_data()` begins, in RBSP bytes from the start of the NAL, its two header
    /// bytes included -- what VA-API's `slice_data_byte_offset` says.
    pub data_byte_offset: u32,
    /// The emulation-prevention bytes before that point.
    pub header_escapes: u16,
}

/// Why a slice could not be described.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refused {
    /// The header ends before the field named.
    Truncated(&'static str),
    /// A value the syntax bounds, outside the bound.
    OutOfRange(&'static str),
    /// Syntax this reader does not handle.
    Unsupported(&'static str),
    /// A dependent segment with no independent one before it in the access unit.
    Orphan,
    /// The reference picture set names a slot the DPB does not hold.
    NoSuchReference,
}

impl fmt::Display for Refused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refused::Truncated(what) => write!(f, "the slice header ends before {what}"),
            Refused::OutOfRange(what) => write!(f, "{what} is out of range"),
            Refused::Unsupported(what) => write!(f, "{what} is not supported"),
            Refused::Orphan => f.write_str("a dependent slice segment has no slice to depend on"),
            Refused::NoSuchReference => {
                f.write_str("the reference picture set names a picture the DPB does not hold")
            }
        }
    }
}

/// Every slice segment of an Annex-B access unit, with its header read.
pub fn slices<'a>(annexb: &'a [u8], picture: &Picture) -> Result<Vec<Slice<'a>>, Refused> {
    let units = NalUnits::new(annexb).ok_or(Refused::Unsupported("a slice not in Annex-B"))?;
    let mut out: Vec<Slice<'a>> = Vec::new();
    for nal in units {
        let Some(&first) = nal.unit.first() else {
            continue;
        };
        let kind = (first >> 1) & 0x3f;
        if kind > NAL_LAST_VCL {
            continue;
        }
        let previous = out.last().map(|s| &s.header);
        let slice = read_segment(nal.unit, picture, previous)?;
        out.push(slice);
    }
    Ok(out)
}

/// Read one slice segment header, from the NAL header on.
fn read_segment<'a>(
    nal: &'a [u8],
    picture: &Picture,
    previous: Option<&SliceHeader>,
) -> Result<Slice<'a>, Refused> {
    let desc = &picture.desc;
    let dec = &picture.decoding;
    let mut r = Reader::new(nal);

    macro_rules! read {
        ($e:expr, $what:literal) => {
            $e.ok_or(Refused::Truncated($what))?
        };
    }
    macro_rules! flag {
        ($what:literal) => {
            read!(r.u(1), $what) != 0
        };
    }

    let nal_header = read!(r.u(16), "the NAL header");
    let kind = ((nal_header >> 9) & 0x3f) as u8;
    if (22..=31).contains(&kind) || (10..=15).contains(&kind) {
        return Err(Refused::Unsupported("a reserved NAL unit type"));
    }

    let first_slice_segment_in_pic = flag!("first_slice_segment_in_pic_flag");
    if (NAL_BLA_W_LP..=NAL_RSV_IRAP_23).contains(&kind) {
        read!(r.u(1), "no_output_of_prior_pics_flag");
    }
    read!(r.ue(), "slice_pic_parameter_set_id");
    let mut dependent = false;
    let mut segment_address = 0;
    if !first_slice_segment_in_pic {
        if desc.dependent_slice_segments_enabled {
            dependent = flag!("dependent_slice_segment_flag");
        }
        segment_address = read!(r.u(desc.segment_address_bits()), "slice_segment_address");
    }

    let header = if dependent {
        previous.cloned().ok_or(Refused::Orphan)?
    } else {
        read_independent(&mut r, desc, dec, kind)?
    };

    let mut num_entry_point_offsets = 0;
    if desc.tiles.is_some() || desc.entropy_coding_sync_enabled {
        let n = read!(r.ue(), "num_entry_point_offsets");
        num_entry_point_offsets =
            u16::try_from(n).map_err(|_| Refused::OutOfRange("num_entry_point_offsets"))?;
        if n > 0 {
            let bits = read!(r.ue(), "offset_len_minus1") + 1;
            if bits > 32 {
                return Err(Refused::OutOfRange("offset_len_minus1"));
            }
            for _ in 0..n {
                read!(r.u(bits), "entry_point_offset_minus1");
            }
        }
    }
    if desc.slice_segment_header_extension_present {
        let length = read!(r.ue(), "slice_segment_header_extension_length");
        if length > 256 {
            return Err(Refused::OutOfRange("slice_segment_header_extension_length"));
        }
        for _ in 0..length {
            read!(r.u(8), "slice_segment_header_extension_data_byte");
        }
    }
    // byte_alignment(): a one, then zeros to the byte boundary.
    if !flag!("alignment_bit_equal_to_one") {
        return Err(Refused::OutOfRange("alignment_bit_equal_to_one"));
    }
    while !r.rbsp_bits().is_multiple_of(8) {
        read!(r.u(1), "alignment_bit_equal_to_zero");
    }

    Ok(Slice {
        nal,
        dependent,
        segment_address,
        header,
        num_entry_point_offsets,
        data_byte_offset: u32::try_from(r.rbsp_bits() / 8)
            .map_err(|_| Refused::OutOfRange("the slice header's length"))?,
        header_escapes: u16::try_from(r.escapes())
            .map_err(|_| Refused::OutOfRange("the slice header's escapes"))?,
    })
}

/// The part of a slice segment header only an independent segment carries.
fn read_independent(
    r: &mut Reader<'_>,
    desc: &PictureDesc,
    dec: &Decoding,
    kind: u8,
) -> Result<SliceHeader, Refused> {
    macro_rules! read {
        ($e:expr, $what:literal) => {
            $e.ok_or(Refused::Truncated($what))?
        };
    }
    macro_rules! flag {
        ($what:literal) => {
            read!(r.u(1), $what) != 0
        };
    }

    for _ in 0..desc.num_extra_slice_header_bits {
        read!(r.u(1), "slice_reserved_flag");
    }
    let slice_type = match read!(r.ue(), "slice_type") {
        0 => SliceType::B,
        1 => SliceType::P,
        2 => SliceType::I,
        _ => return Err(Refused::OutOfRange("slice_type")),
    };
    if desc.output_flag_present {
        read!(r.u(1), "pic_output_flag");
    }
    let mut colour_plane_id = 0;
    if desc.separate_colour_plane {
        colour_plane_id = read!(r.u(2), "colour_plane_id") as u8;
    }
    let mut temporal_mvp = false;
    if kind != NAL_IDR_W_RADL && kind != NAL_IDR_N_LP {
        let lsb_bits = u32::from(desc.log2_max_pic_order_cnt_lsb_minus4) + 4;
        read!(r.u(lsb_bits), "slice_pic_order_cnt_lsb");
        if !flag!("short_term_ref_pic_set_sps_flag") {
            // The set is already resolved into the descriptor; only its length matters here.
            for _ in 0..dec.st_rps_bits {
                read!(r.u(1), "st_ref_pic_set");
            }
        } else if desc.num_short_term_ref_pic_sets > 1 {
            read!(
                r.u(ceil_log2(desc.num_short_term_ref_pic_sets.into())),
                "short_term_ref_pic_set_idx"
            );
        }
        if desc.long_term_ref_pics_present {
            let mut num_long_term_sps = 0;
            if desc.num_long_term_ref_pics_sps > 0 {
                num_long_term_sps = read!(r.ue(), "num_long_term_sps");
            }
            let num_long_term_pics = read!(r.ue(), "num_long_term_pics");
            let total = num_long_term_sps.saturating_add(num_long_term_pics);
            if total > 32 {
                return Err(Refused::OutOfRange("the number of long-term pictures"));
            }
            for i in 0..total {
                if i < num_long_term_sps {
                    if desc.num_long_term_ref_pics_sps > 1 {
                        read!(r.u(ceil_log2(desc.num_long_term_ref_pics_sps.into())), "lt_idx_sps");
                    }
                } else {
                    read!(r.u(lsb_bits), "poc_lsb_lt");
                    read!(r.u(1), "used_by_curr_pic_lt_flag");
                }
                if flag!("delta_poc_msb_present_flag") {
                    read!(r.ue(), "delta_poc_msb_cycle_lt");
                }
            }
        }
        if desc.sps_temporal_mvp_enabled {
            temporal_mvp = flag!("slice_temporal_mvp_enabled_flag");
        }
    }

    let chroma = !desc.separate_colour_plane && desc.chroma_format_idc != 0;
    let (mut sao_luma, mut sao_chroma) = (false, false);
    if desc.sample_adaptive_offset_enabled {
        sao_luma = flag!("slice_sao_luma_flag");
        if chroma {
            sao_chroma = flag!("slice_sao_chroma_flag");
        }
    }

    let mut active = [0u8; 2];
    let mut list_entries = [None, None];
    let (mut mvd_l1_zero, mut cabac_init) = (false, false);
    let (mut collocated_from_l0, mut collocated_ref_idx) = (true, 0u8);
    let mut weights = None;
    let mut five_minus_max_num_merge_cand = 0;
    if slice_type != SliceType::I {
        active =
            [desc.num_ref_idx_l0_default_active_minus1, desc.num_ref_idx_l1_default_active_minus1];
        if flag!("num_ref_idx_active_override_flag") {
            active[0] = bounded(read!(r.ue(), "num_ref_idx_l0_active_minus1"), 14, "l0 count")?;
            if slice_type == SliceType::B {
                active[1] = bounded(read!(r.ue(), "num_ref_idx_l1_active_minus1"), 14, "l1 count")?;
            }
        }
        if active.iter().any(|&n| n > 14) {
            return Err(Refused::OutOfRange("a default reference count"));
        }
        let total = dec.num_pic_total_curr();
        if desc.lists_modification_present && total > 1 {
            let bits = ceil_log2(total as u32);
            for (list, entries) in list_entries.iter_mut().enumerate().take(slice_type.lists()) {
                if flag!("ref_pic_list_modification_flag") {
                    let mut out = Vec::with_capacity(usize::from(active[list]) + 1);
                    for _ in 0..=active[list] {
                        out.push(read!(r.u(bits), "list_entry") as u8);
                    }
                    *entries = Some(out);
                }
            }
        }
        if slice_type == SliceType::B {
            mvd_l1_zero = flag!("mvd_l1_zero_flag");
        }
        if desc.cabac_init_present {
            cabac_init = flag!("cabac_init_flag");
        }
        if temporal_mvp {
            if slice_type == SliceType::B {
                collocated_from_l0 = flag!("collocated_from_l0_flag");
            }
            let list = if collocated_from_l0 { 0 } else { 1 };
            if active[list] > 0 {
                collocated_ref_idx =
                    bounded(read!(r.ue(), "collocated_ref_idx"), 14, "collocated_ref_idx")?;
            }
        }
        if (desc.weighted_pred && slice_type == SliceType::P)
            || (desc.weighted_bipred && slice_type == SliceType::B)
        {
            weights = Some(pred_weight_table(r, slice_type.lists(), active, chroma)?);
        }
        five_minus_max_num_merge_cand = bounded(
            read!(r.ue(), "five_minus_max_num_merge_cand"),
            4,
            "five_minus_max_num_merge_cand",
        )?;
    }

    let slice_qp_delta = to_i8(read!(r.se(), "slice_qp_delta"), "slice_qp_delta")?;
    let (mut cb_qp_offset, mut cr_qp_offset) = (0, 0);
    if desc.pps_slice_chroma_qp_offsets_present {
        cb_qp_offset = to_i8(read!(r.se(), "slice_cb_qp_offset"), "slice_cb_qp_offset")?;
        cr_qp_offset = to_i8(read!(r.se(), "slice_cr_qp_offset"), "slice_cr_qp_offset")?;
    }
    let mut override_deblocking = false;
    if desc.deblocking_filter_override_enabled {
        override_deblocking = flag!("deblocking_filter_override_flag");
    }
    let mut deblocking_filter_disabled = desc.pps_deblocking_filter_disabled;
    let (mut beta_offset_div2, mut tc_offset_div2) =
        (desc.pps_beta_offset_div2, desc.pps_tc_offset_div2);
    if override_deblocking {
        deblocking_filter_disabled = flag!("slice_deblocking_filter_disabled_flag");
        if !deblocking_filter_disabled {
            beta_offset_div2 = to_i8(read!(r.se(), "slice_beta_offset_div2"), "beta offset")?;
            tc_offset_div2 = to_i8(read!(r.se(), "slice_tc_offset_div2"), "tc offset")?;
        }
    }
    let mut loop_filter_across_slices = desc.pps_loop_filter_across_slices_enabled;
    if desc.pps_loop_filter_across_slices_enabled
        && (sao_luma || sao_chroma || !deblocking_filter_disabled)
    {
        loop_filter_across_slices = flag!("slice_loop_filter_across_slices_enabled_flag");
    }

    Ok(SliceHeader {
        slice_type,
        colour_plane_id,
        sao_luma,
        sao_chroma,
        temporal_mvp,
        num_ref_idx_active_minus1: active,
        list_entries,
        mvd_l1_zero,
        cabac_init,
        collocated_from_l0,
        collocated_ref_idx,
        weights,
        five_minus_max_num_merge_cand,
        slice_qp_delta,
        cb_qp_offset,
        cr_qp_offset,
        deblocking_filter_disabled,
        beta_offset_div2,
        tc_offset_div2,
        loop_filter_across_slices,
    })
}

/// `Ceil(Log2(n))`, which sizes an index into `n` things.
fn ceil_log2(n: u32) -> u32 {
    match n {
        0 | 1 => 0,
        n => u32::BITS - (n - 1).leading_zeros(),
    }
}

fn bounded(v: u32, max: u32, what: &'static str) -> Result<u8, Refused> {
    if v > max {
        return Err(Refused::OutOfRange(what));
    }
    Ok(v as u8)
}

fn to_i8(v: i32, what: &'static str) -> Result<i8, Refused> {
    i8::try_from(v).map_err(|_| Refused::OutOfRange(what))
}

fn pred_weight_table(
    r: &mut Reader<'_>,
    lists: usize,
    active: [u8; 2],
    chroma: bool,
) -> Result<PredWeights, Refused> {
    let truncated = Refused::Truncated("pred_weight_table");
    let luma_log2_denom = bounded(r.ue().ok_or(truncated)?, 7, "luma_log2_weight_denom")?;
    let mut delta_chroma_log2_denom = 0;
    if chroma {
        delta_chroma_log2_denom =
            to_i8(r.se().ok_or(truncated)?, "delta_chroma_log2_weight_denom")?;
    }
    let chroma_log2_denom = i32::from(luma_log2_denom) + i32::from(delta_chroma_log2_denom);
    if !(0..=7).contains(&chroma_log2_denom) {
        return Err(Refused::OutOfRange("ChromaLog2WeightDenom"));
    }
    // wpOffsetHalfRangeC for a stream without high-precision offsets: the Main profile's.
    let half_range = 1i32 << 7;

    let mut out = PredWeights { luma_log2_denom, delta_chroma_log2_denom, ..Default::default() };
    for (list, weights) in out.lists.iter_mut().enumerate().take(lists) {
        let n = usize::from(active[list]) + 1;
        let mut luma = [false; 15];
        let mut chroma_flags = [false; 15];
        for flag in luma.iter_mut().take(n) {
            *flag = r.u(1).ok_or(truncated)? != 0;
        }
        if chroma {
            for flag in chroma_flags.iter_mut().take(n) {
                *flag = r.u(1).ok_or(truncated)? != 0;
            }
        }
        for i in 0..n {
            if luma[i] {
                weights.delta_luma_weight[i] =
                    to_i8(r.se().ok_or(truncated)?, "delta_luma_weight")?;
                weights.luma_offset[i] = to_i8(r.se().ok_or(truncated)?, "luma_offset")?;
            }
            if chroma_flags[i] {
                for j in 0..2 {
                    let delta_weight = to_i8(r.se().ok_or(truncated)?, "delta_chroma_weight")?;
                    let delta_offset = r.se().ok_or(truncated)?;
                    if !(-4 * half_range..4 * half_range).contains(&delta_offset) {
                        return Err(Refused::OutOfRange("delta_chroma_offset"));
                    }
                    let weight = (1 << chroma_log2_denom) + i32::from(delta_weight);
                    // ChromaOffset (7-56).
                    let offset = (half_range + delta_offset
                        - ((half_range * weight) >> chroma_log2_denom))
                        .clamp(-half_range, half_range - 1);
                    weights.delta_chroma_weight[i][j] = delta_weight;
                    weights.chroma_offset[i][j] = offset as i8;
                }
            }
        }
    }
    Ok(out)
}

/// The two reference lists for one slice (8.3.4), as DPB slots. `None` is an entry past the
/// slice's active count.
pub fn ref_lists(
    header: &SliceHeader,
    decoding: &Decoding,
) -> Result<[[Option<usize>; 15]; 2], Refused> {
    let present = |slot: &usize| decoding.refs.get(*slot).is_some_and(Option::is_some);
    let sets = [&decoding.st_curr_before, &decoding.st_curr_after, &decoding.lt_curr];
    if !sets.iter().all(|set| set.iter().all(present)) {
        return Err(Refused::NoSuchReference);
    }
    let total = decoding.num_pic_total_curr();
    let mut out = [[None; 15]; 2];
    if header.slice_type == SliceType::I {
        return Ok(out);
    }
    if total == 0 {
        return Err(Refused::NoSuchReference);
    }
    for (list, slots) in out.iter_mut().enumerate().take(header.slice_type.lists()) {
        let active = usize::from(header.num_ref_idx_active_minus1[list]) + 1;
        // RefPicListTemp: the sets in order, repeated until the list is long enough (8-8, 8-10).
        let order = if list == 0 {
            [&decoding.st_curr_before, &decoding.st_curr_after, &decoding.lt_curr]
        } else {
            [&decoding.st_curr_after, &decoding.st_curr_before, &decoding.lt_curr]
        };
        let wanted = active.max(total);
        let temp: Vec<usize> =
            order.iter().flat_map(|set| set.iter().copied()).cycle().take(wanted).collect();
        for (i, slot) in slots.iter_mut().enumerate().take(active) {
            let index = match &header.list_entries[list] {
                Some(entries) => usize::from(entries[i]),
                None => i,
            };
            *slot = Some(*temp.get(index).ok_or(Refused::OutOfRange("list_entry"))?);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decoding(before: Vec<usize>, after: Vec<usize>, lt: Vec<usize>) -> Decoding {
        let mut d = Decoding::read(&[], &PictureDesc::read(&[]));
        for slot in before.iter().chain(&after).chain(&lt) {
            d.refs[*slot] = Some(Reference { buffer: 1 + *slot as u32, poc: 0, long_term: false });
        }
        (d.st_curr_before, d.st_curr_after, d.lt_curr) = (before, after, lt);
        d
    }

    fn header(slice_type: SliceType, active: [u8; 2]) -> SliceHeader {
        SliceHeader {
            slice_type,
            colour_plane_id: 0,
            sao_luma: false,
            sao_chroma: false,
            temporal_mvp: false,
            num_ref_idx_active_minus1: active,
            list_entries: [None, None],
            mvd_l1_zero: false,
            cabac_init: false,
            collocated_from_l0: true,
            collocated_ref_idx: 0,
            weights: None,
            five_minus_max_num_merge_cand: 0,
            slice_qp_delta: 0,
            cb_qp_offset: 0,
            cr_qp_offset: 0,
            deblocking_filter_disabled: false,
            beta_offset_div2: 0,
            tc_offset_div2: 0,
            loop_filter_across_slices: false,
        }
    }

    /// 8.3.4: list 0 is before, after, long-term; list 1 is after, before, long-term; each is
    /// repeated to the active count when the sets are shorter.
    #[test]
    fn lists_follow_the_reference_picture_set() {
        let d = decoding(vec![3, 1], vec![5], vec![7]);
        let lists = ref_lists(&header(SliceType::B, [5, 2]), &d).unwrap();
        assert_eq!(&lists[0][..7], &[Some(3), Some(1), Some(5), Some(7), Some(3), Some(1), None]);
        assert_eq!(&lists[1][..4], &[Some(5), Some(3), Some(1), None]);
    }

    /// `list_entry` picks from the full temporary list, and an I slice has no lists.
    #[test]
    fn a_modification_indexes_the_temporary_list() {
        let d = decoding(vec![2], vec![4], vec![]);
        let mut h = header(SliceType::P, [1, 0]);
        h.list_entries[0] = Some(vec![1, 1]);
        let lists = ref_lists(&h, &d).unwrap();
        assert_eq!(&lists[0][..3], &[Some(4), Some(4), None]);
        assert!(
            ref_lists(&header(SliceType::I, [0, 0]), &d)
                .unwrap()
                .iter()
                .flatten()
                .all(Option::is_none)
        );

        h.list_entries[0] = Some(vec![5, 0]);
        assert_eq!(ref_lists(&h, &d), Err(Refused::OutOfRange("list_entry")));
    }

    /// A set that names an empty slot is refused rather than handed to the driver.
    #[test]
    fn a_set_naming_an_empty_slot_is_refused() {
        let mut d = decoding(vec![2], vec![], vec![]);
        d.refs[2] = None;
        assert_eq!(ref_lists(&header(SliceType::P, [0, 0]), &d), Err(Refused::NoSuchReference));
    }

    /// Uniform spacing divides the CTBs as 6-3 does: the remainder goes to the later tiles.
    #[test]
    fn uniform_tiles_split_the_picture_as_the_spec_does() {
        let mut sizes = [0u16; 19];
        uniform(&mut sizes, 2, 10);
        assert_eq!(&sizes[..3], &[2, 2, 3]);
    }

    /// A P slice segment reads back field for field, skipping the short-term set by the
    /// descriptor's length, and reports where its data starts in bytes.
    #[test]
    fn a_p_slice_segment_reads_back_with_its_data_offset() {
        use crate::vrend::video::bitstream::{Escape, Writer};
        let mut desc = PictureDesc::read(&[]);
        desc.log2_max_pic_order_cnt_lsb_minus4 = 4;
        let mut decoding = Decoding::read(&[], &desc);
        decoding.st_rps_bits = 5;
        let picture = Picture { desc, decoding };

        let mut w = Writer::new(Escape::Rbsp);
        w.u(16, 1 << 9); // TRAIL_R
        w.u(1, 1); // first_slice_segment_in_pic_flag
        w.ue(0); // slice_pic_parameter_set_id
        w.ue(1); // slice_type P
        w.u(8, 0x5a); // slice_pic_order_cnt_lsb
        w.u(1, 0); // short_term_ref_pic_set_sps_flag
        w.u(5, 0b10110); // st_ref_pic_set, skipped by length
        w.u(1, 1); // num_ref_idx_active_override_flag
        w.ue(1); // num_ref_idx_l0_active_minus1
        w.ue(2); // five_minus_max_num_merge_cand
        w.se(-3); // slice_qp_delta
        w.u(1, 1); // alignment_bit_equal_to_one
        w.align();
        let nal = w.finish();

        let slice = read_segment(&nal, &picture, None).unwrap();
        assert!(!slice.dependent);
        assert_eq!(slice.header.slice_type, SliceType::P);
        assert_eq!(slice.header.num_ref_idx_active_minus1, [1, 0]);
        assert_eq!(slice.header.five_minus_max_num_merge_cand, 2);
        assert_eq!(slice.header.slice_qp_delta, -3);
        // 16 + 1 + 1 + 3 + 8 + 1 + 5 + 1 + 3 + 3 + 5 bits of header, then the alignment one.
        assert_eq!(slice.data_byte_offset, 48 / 8);
        assert_eq!(slice.header_escapes, 0);
    }

    /// Each scan is a permutation, and undoing it puts an entry back where mesa took it from.
    #[test]
    fn unscanning_inverts_the_frontend_reorder() {
        for scan in [&UP_RIGHT_DIAGONAL_8X8[..], &UP_RIGHT_DIAGONAL_4X4[..]] {
            let mut seen = vec![false; scan.len()];
            scan.iter().for_each(|&i| seen[i] = true);
            assert!(seen.iter().all(|&s| s), "a scan visits every position once");
        }
        let va: [u8; 16] = std::array::from_fn(|i| i as u8 * 3);
        let stored: [u8; 16] = std::array::from_fn(|j| va[UP_RIGHT_DIAGONAL_4X4[j]]);
        assert_eq!(unscan(|j| stored[j], &UP_RIGHT_DIAGONAL_4X4), va);
    }
}
