// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! H.264 slice headers and reference picture lists, for a decoder that is handed parameters
//! rather than a bitstream.
//!
//! The guest's descriptor is the gallium hardware-decoder shape: sequence, picture and DPB, and
//! nothing per slice. VA-API wants each slice's header fields and its *final* reference lists --
//! initialised and then modified, as 8.2.4 builds them -- and the C fills none of that, which is
//! why it works on Mesa's driver alone: Mesa's frontend parses the slices itself and ignores the
//! parameters. A driver that follows the spec reads them, so they are rebuilt here from the slice
//! headers the guest also sent and from the descriptor's DPB.
//!
//! Frame pictures only. A field picture builds its lists from fields of alternating parity
//! (8.2.4.2.5), which is not written, and is refused by name rather than decoded as a frame.

use std::fmt;

use super::bitstream::{NalUnits, Reader};
use super::h264::PictureDesc;

const NAL_SLICE_NON_IDR: u8 = 1;
const NAL_SLICE_IDR: u8 = 5;

/// Where the decode-side fields sit in `struct virgl_h264_picture_desc`, measured with `offsetof`.
mod at {
    pub const SEQ_SCALING_MATRIX_PRESENT_FLAG: usize = 269;
    pub const MB_ADAPTIVE_FRAME_FIELD_FLAG: usize = 1791;
    pub const MIN_LUMA_BI_PRED_SIZE_8X8: usize = 1793;
    pub const SLICE_GROUP_MAP_TYPE: usize = 1799;
    pub const SLICE_GROUP_CHANGE_RATE_MINUS1: usize = 1800;
    pub const PPS_SCALING_LIST_4X4: usize = 1812;
    pub const PPS_SCALING_LIST_8X8: usize = 1908;
    pub const FRAME_NUM: usize = 2296;
    pub const BOTTOM_FIELD_FLAG: usize = 2301;
    pub const FIELD_ORDER_CNT: usize = 2308;
    pub const IS_LONG_TERM: usize = 2316;
    pub const TOP_IS_REFERENCE: usize = 2332;
    pub const BOTTOM_IS_REFERENCE: usize = 2348;
    pub const FIELD_ORDER_CNT_LIST: usize = 2364;
    pub const FRAME_NUM_LIST: usize = 2492;
    pub const BUFFER_ID: usize = 2556;
    pub const IS_REFERENCE: usize = 2620;
    pub const NUM_REF_FRAMES: usize = 2621;
}

/// A whole H.264 picture descriptor: what a parameter set is written out of, and the rest.
pub struct Picture {
    pub desc: PictureDesc,
    pub decoding: Decoding,
}

/// What the descriptor says about the picture being decoded and the ones it may predict from,
/// beyond what a parameter set is written out of ([`PictureDesc`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decoding {
    pub frame_num: u32,
    pub bottom_field: bool,
    pub field_order_cnt: [i32; 2],
    /// Whether this picture is itself a reference.
    pub is_reference: bool,
    /// The DPB, `num_ref_frames` long.
    pub dpb: Vec<Reference>,
    pub mb_adaptive_frame_field: bool,
    pub min_luma_bi_pred_8x8: bool,
    pub seq_scaling_matrix_present: bool,
    pub slice_group_map_type: u8,
    pub slice_group_change_rate_minus1: u8,
    /// The PPS scaling lists, as mesa's VA frontend copied them from the application -- which is
    /// VA-API's own layout, so they go back out unchanged.
    pub scaling_4x4: [[u8; 16]; 6],
    pub scaling_8x8: [[u8; 64]; 2],
}

/// One picture in the DPB.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reference {
    /// The guest's decode-target handle for it.
    pub buffer: u32,
    /// `FrameNum` for a short-term reference, `LongTermFrameIdx` for a long-term one.
    pub frame_idx: u32,
    pub long_term: bool,
    pub top: bool,
    pub bottom: bool,
    pub field_order_cnt: [i32; 2],
}

impl Reference {
    /// Whether both fields are references, which is what a frame picture may predict from.
    fn is_frame(&self) -> bool {
        self.top && self.bottom
    }

    /// A frame's picture order count: the lesser of its fields' (8-1).
    fn poc(&self) -> i32 {
        self.field_order_cnt[0].min(self.field_order_cnt[1])
    }
}

impl Decoding {
    /// Read the decode-side fields of a descriptor. Total, as [`PictureDesc::read`] is.
    pub fn read(blob: &[u8]) -> Decoding {
        let byte = |at: usize| blob.get(at).copied().unwrap_or(0);
        let flag = |at: usize| byte(at) != 0;
        let word = |at: usize| [byte(at), byte(at + 1), byte(at + 2), byte(at + 3)];
        let unsigned = |at: usize| u32::from_le_bytes(word(at));
        let signed = |at: usize| i32::from_le_bytes(word(at));

        let count = usize::from(byte(at::NUM_REF_FRAMES)).min(16);
        Decoding {
            frame_num: unsigned(at::FRAME_NUM),
            bottom_field: flag(at::BOTTOM_FIELD_FLAG),
            field_order_cnt: [signed(at::FIELD_ORDER_CNT), signed(at::FIELD_ORDER_CNT + 4)],
            is_reference: flag(at::IS_REFERENCE),
            dpb: (0..count)
                .map(|i| Reference {
                    buffer: unsigned(at::BUFFER_ID + 4 * i),
                    frame_idx: unsigned(at::FRAME_NUM_LIST + 4 * i),
                    long_term: flag(at::IS_LONG_TERM + i),
                    top: flag(at::TOP_IS_REFERENCE + i),
                    bottom: flag(at::BOTTOM_IS_REFERENCE + i),
                    field_order_cnt: [
                        signed(at::FIELD_ORDER_CNT_LIST + 8 * i),
                        signed(at::FIELD_ORDER_CNT_LIST + 8 * i + 4),
                    ],
                })
                .collect(),
            mb_adaptive_frame_field: flag(at::MB_ADAPTIVE_FRAME_FIELD_FLAG),
            min_luma_bi_pred_8x8: flag(at::MIN_LUMA_BI_PRED_SIZE_8X8),
            seq_scaling_matrix_present: flag(at::SEQ_SCALING_MATRIX_PRESENT_FLAG),
            slice_group_map_type: byte(at::SLICE_GROUP_MAP_TYPE),
            slice_group_change_rate_minus1: byte(at::SLICE_GROUP_CHANGE_RATE_MINUS1),
            scaling_4x4: std::array::from_fn(|l| {
                std::array::from_fn(|i| byte(at::PPS_SCALING_LIST_4X4 + 16 * l + i))
            }),
            scaling_8x8: std::array::from_fn(|l| {
                std::array::from_fn(|i| byte(at::PPS_SCALING_LIST_8X8 + 64 * l + i))
            }),
        }
    }
}

/// `slice_type` modulo 5 (Table 7-6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SliceType {
    P = 0,
    B = 1,
    I = 2,
    Sp = 3,
    Si = 4,
}

impl SliceType {
    fn of(raw: u32) -> SliceType {
        match raw % 5 {
            0 => SliceType::P,
            1 => SliceType::B,
            2 => SliceType::I,
            3 => SliceType::Sp,
            _ => SliceType::Si,
        }
    }

    fn predicts(self) -> bool {
        matches!(self, SliceType::P | SliceType::Sp | SliceType::B)
    }
}

/// One `modification_of_pic_nums_idc` and its argument (7.3.3.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Modification {
    /// 0: `abs_diff_pic_num_minus1`, subtracted from the prediction.
    Subtract(u32),
    /// 1: `abs_diff_pic_num_minus1`, added to it.
    Add(u32),
    /// 2: `long_term_pic_num`.
    LongTerm(u32),
}

/// One list's explicit weights, VA-API's shape: every entry filled, the default where the stream
/// sent none, and a flag saying whether any entry was sent at all.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ListWeights {
    pub luma_flag: bool,
    pub luma_weight: [i16; 32],
    pub luma_offset: [i16; 32],
    pub chroma_flag: bool,
    pub chroma_weight: [[i16; 2]; 32],
    pub chroma_offset: [[i16; 2]; 32],
}

/// `pred_weight_table()` (7.3.3.2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PredWeights {
    pub luma_log2_denom: u8,
    pub chroma_log2_denom: u8,
    pub lists: [ListWeights; 2],
}

/// The fields of a slice header (7.3.3) a parameter-buffer decoder is told.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SliceHeader {
    pub first_mb_in_slice: u32,
    pub slice_type: SliceType,
    pub frame_num: u32,
    pub field_pic: bool,
    pub direct_spatial_mv_pred: bool,
    pub num_ref_idx_active_minus1: [u8; 2],
    pub modifications: [Vec<Modification>; 2],
    pub weights: Option<PredWeights>,
    pub cabac_init_idc: u8,
    pub slice_qp_delta: i8,
    pub disable_deblocking_filter_idc: u8,
    pub slice_alpha_c0_offset_div2: i8,
    pub slice_beta_offset_div2: i8,
    /// Where `slice_data()` begins, in RBSP bits from the start of the NAL, its header byte
    /// included: emulation-prevention bytes are not counted, as VA-API's
    /// `slice_data_bit_offset` says.
    pub data_bit_offset: u32,
}

/// One slice of an access unit: the NAL as sent, and its header.
pub struct Slice<'a> {
    pub nal: &'a [u8],
    pub header: SliceHeader,
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
    /// A modification names a picture the DPB does not hold.
    NoSuchReference,
}

impl fmt::Display for Refused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refused::Truncated(what) => write!(f, "the slice header ends before {what}"),
            Refused::OutOfRange(what) => write!(f, "{what} is out of range"),
            Refused::Unsupported(what) => write!(f, "{what} is not supported"),
            Refused::NoSuchReference => {
                f.write_str("a list modification names a picture the DPB does not hold")
            }
        }
    }
}

/// Every slice NAL of an Annex-B access unit, with its header read.
pub fn slices<'a>(annexb: &'a [u8], desc: &PictureDesc) -> Result<Vec<Slice<'a>>, Refused> {
    let units = NalUnits::new(annexb).ok_or(Refused::Unsupported("a slice not in Annex-B"))?;
    let mut out = Vec::new();
    for nal in units {
        let kind = nal.unit[0] & 0x1f;
        if kind != NAL_SLICE_NON_IDR && kind != NAL_SLICE_IDR {
            continue;
        }
        let header = read_header(nal.unit, desc)?;
        out.push(Slice { nal: nal.unit, header });
    }
    Ok(out)
}

/// Read one slice header, from the NAL header byte on.
fn read_header(nal: &[u8], desc: &PictureDesc) -> Result<SliceHeader, Refused> {
    let mut r = Reader::new(nal);
    let header = r.u(8).ok_or(Refused::Truncated("the NAL header"))?;
    let nal_ref_idc = (header >> 5) & 3;
    let idr = header & 0x1f == u32::from(NAL_SLICE_IDR);

    macro_rules! read {
        ($e:expr, $what:literal) => {
            $e.ok_or(Refused::Truncated($what))?
        };
    }

    let first_mb_in_slice = read!(r.ue(), "first_mb_in_slice");
    let slice_type = SliceType::of(read!(r.ue(), "slice_type"));
    let _pps_id = read!(r.ue(), "pic_parameter_set_id");
    if desc.separate_colour_plane {
        read!(r.u(2), "colour_plane_id");
    }
    let frame_num = read!(r.u(u32::from(desc.log2_max_frame_num_minus4) + 4), "frame_num");
    let mut field_pic = false;
    if !desc.frame_mbs_only {
        field_pic = read!(r.u(1), "field_pic_flag") != 0;
        if field_pic {
            read!(r.u(1), "bottom_field_flag");
        }
    }
    if idr {
        read!(r.ue(), "idr_pic_id");
    }
    match desc.pic_order_cnt_type {
        0 => {
            let bits = u32::from(desc.log2_max_pic_order_cnt_lsb_minus4) + 4;
            read!(r.u(bits), "pic_order_cnt_lsb");
            if desc.bottom_field_pic_order_in_frame_present && !field_pic {
                read!(r.se(), "delta_pic_order_cnt_bottom");
            }
        }
        1 if !desc.delta_pic_order_always_zero => {
            read!(r.se(), "delta_pic_order_cnt[0]");
            if desc.bottom_field_pic_order_in_frame_present && !field_pic {
                read!(r.se(), "delta_pic_order_cnt[1]");
            }
        }
        _ => {}
    }
    if desc.redundant_pic_cnt_present {
        read!(r.ue(), "redundant_pic_cnt");
    }
    let mut direct_spatial_mv_pred = false;
    if slice_type == SliceType::B {
        direct_spatial_mv_pred = read!(r.u(1), "direct_spatial_mv_pred_flag") != 0;
    }
    // The descriptor's counts, not the PPS defaults: mesa writes the per-slice values there and
    // leaves the PPS ones dead (see `PictureDesc::num_ref_frames`). A slice that overrides them
    // says so below, and wins.
    let mut active = [desc.num_ref_idx_l0_active_minus1, desc.num_ref_idx_l1_active_minus1];
    if slice_type.predicts() && read!(r.u(1), "num_ref_idx_active_override_flag") != 0 {
        active[0] = bounded(read!(r.ue(), "num_ref_idx_l0_active_minus1"), 31, "l0 count")?;
        if slice_type == SliceType::B {
            active[1] = bounded(read!(r.ue(), "num_ref_idx_l1_active_minus1"), 31, "l1 count")?;
        }
    }
    let lists = match slice_type {
        SliceType::B => 2,
        SliceType::P | SliceType::Sp => 1,
        SliceType::I | SliceType::Si => 0,
    };
    let mut modifications = [Vec::new(), Vec::new()];
    for list in modifications.iter_mut().take(lists) {
        if read!(r.u(1), "ref_pic_list_modification_flag") == 0 {
            continue;
        }
        loop {
            let m = match read!(r.ue(), "modification_of_pic_nums_idc") {
                0 => Modification::Subtract(read!(r.ue(), "abs_diff_pic_num_minus1")),
                1 => Modification::Add(read!(r.ue(), "abs_diff_pic_num_minus1")),
                2 => Modification::LongTerm(read!(r.ue(), "long_term_pic_num")),
                3 => break,
                _ => return Err(Refused::OutOfRange("modification_of_pic_nums_idc")),
            };
            if list.len() > 32 {
                return Err(Refused::OutOfRange("the number of list modifications"));
            }
            list.push(m);
        }
    }

    let explicit = (desc.weighted_pred && matches!(slice_type, SliceType::P | SliceType::Sp))
        || (desc.weighted_bipred_idc == 1 && slice_type == SliceType::B);
    let weights = if explicit {
        let chroma = !desc.separate_colour_plane && desc.chroma_format_idc != 0;
        Some(pred_weight_table(&mut r, lists, active, chroma)?)
    } else {
        None
    };

    if nal_ref_idc != 0 {
        skip_dec_ref_pic_marking(&mut r, idr)?;
    }
    let mut cabac_init_idc = 0;
    if desc.entropy_coding_mode && !matches!(slice_type, SliceType::I | SliceType::Si) {
        cabac_init_idc = bounded(read!(r.ue(), "cabac_init_idc"), 2, "cabac_init_idc")?;
    }
    let slice_qp_delta = clamp_i8(read!(r.se(), "slice_qp_delta"), "slice_qp_delta")?;
    if matches!(slice_type, SliceType::Sp | SliceType::Si) {
        if slice_type == SliceType::Sp {
            read!(r.u(1), "sp_for_switch_flag");
        }
        read!(r.se(), "slice_qs_delta");
    }
    let (mut idc, mut alpha, mut beta) = (0, 0, 0);
    if desc.deblocking_filter_control_present {
        idc = bounded(read!(r.ue(), "disable_deblocking_filter_idc"), 2, "deblocking idc")?;
        if idc != 1 {
            alpha = clamp_i8(read!(r.se(), "slice_alpha_c0_offset_div2"), "alpha offset")?;
            beta = clamp_i8(read!(r.se(), "slice_beta_offset_div2"), "beta offset")?;
        }
    }
    if desc.num_slice_groups_minus1 > 0 {
        // `slice_group_change_cycle` would follow, sized from the picture's map units.
        return Err(Refused::Unsupported("slice groups"));
    }

    Ok(SliceHeader {
        first_mb_in_slice,
        slice_type,
        frame_num,
        field_pic,
        direct_spatial_mv_pred,
        num_ref_idx_active_minus1: active,
        modifications,
        weights,
        cabac_init_idc,
        slice_qp_delta,
        disable_deblocking_filter_idc: idc,
        slice_alpha_c0_offset_div2: alpha,
        slice_beta_offset_div2: beta,
        data_bit_offset: u32::try_from(r.rbsp_bits())
            .map_err(|_| Refused::OutOfRange("the slice header's length"))?,
    })
}

fn bounded(v: u32, max: u32, what: &'static str) -> Result<u8, Refused> {
    if v > max {
        return Err(Refused::OutOfRange(what));
    }
    Ok(v as u8)
}

fn clamp_i8(v: i32, what: &'static str) -> Result<i8, Refused> {
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
    let chroma_log2_denom =
        if chroma { bounded(r.ue().ok_or(truncated)?, 7, "chroma_log2_weight_denom")? } else { 0 };
    let default = |denom: u8| 1i16 << denom;
    let blank = ListWeights {
        luma_flag: false,
        luma_weight: [default(luma_log2_denom); 32],
        luma_offset: [0; 32],
        chroma_flag: false,
        chroma_weight: [[default(chroma_log2_denom); 2]; 32],
        chroma_offset: [[0; 2]; 32],
    };
    let mut out = [blank.clone(), blank];
    for (list, weights) in out.iter_mut().enumerate().take(lists) {
        for i in 0..=usize::from(active[list]) {
            if r.u(1).ok_or(truncated)? != 0 {
                weights.luma_flag = true;
                weights.luma_weight[i] = clamp_i16(r.se().ok_or(truncated)?)?;
                weights.luma_offset[i] = clamp_i16(r.se().ok_or(truncated)?)?;
            }
            if chroma && r.u(1).ok_or(truncated)? != 0 {
                weights.chroma_flag = true;
                for c in 0..2 {
                    weights.chroma_weight[i][c] = clamp_i16(r.se().ok_or(truncated)?)?;
                    weights.chroma_offset[i][c] = clamp_i16(r.se().ok_or(truncated)?)?;
                }
            }
        }
    }
    Ok(PredWeights { luma_log2_denom, chroma_log2_denom, lists: out })
}

fn clamp_i16(v: i32) -> Result<i16, Refused> {
    i16::try_from(v).map_err(|_| Refused::OutOfRange("a prediction weight"))
}

/// Read past `dec_ref_pic_marking()` (7.3.3.3). The marking is the DPB's business, and the guest
/// sends the DPB it produced; only the header's length depends on it here.
fn skip_dec_ref_pic_marking(r: &mut Reader<'_>, idr: bool) -> Result<(), Refused> {
    let truncated = Refused::Truncated("dec_ref_pic_marking");
    if idr {
        r.u(2).ok_or(truncated)?;
        return Ok(());
    }
    if r.u(1).ok_or(truncated)? == 0 {
        return Ok(());
    }
    for _ in 0..=66 {
        // How many `ue(v)` arguments each operation carries (7.3.3.3).
        let arguments = match r.ue().ok_or(truncated)? {
            0 => return Ok(()),
            5 => 0,
            1 | 2 | 4 | 6 => 1,
            3 => 2,
            _ => return Err(Refused::OutOfRange("memory_management_control_operation")),
        };
        for _ in 0..arguments {
            r.ue().ok_or(truncated)?;
        }
    }
    Err(Refused::OutOfRange("the number of memory management operations"))
}

/// The two reference lists for one slice of a frame picture, as indices into the DPB: the
/// initial lists of 8.2.4.2, cut to the slice's active counts, then modified as 8.2.4.3 says.
/// `None` is an entry with no reference picture.
pub fn ref_lists(
    header: &SliceHeader,
    decoding: &Decoding,
    desc: &PictureDesc,
) -> Result<[[Option<usize>; 32]; 2], Refused> {
    if header.field_pic {
        return Err(Refused::Unsupported("field pictures"));
    }
    let max_frame_num = 1i64 << (u32::from(desc.log2_max_frame_num_minus4) + 4);
    let curr = i64::from(header.frame_num);
    // FrameNumWrap (8-27), which is PicNum for a frame.
    let pic_num = |r: &Reference| {
        let n = i64::from(r.frame_idx);
        if n > curr { n - max_frame_num } else { n }
    };
    let dpb = &decoding.dpb;
    let usable = |i: &usize| dpb[*i].is_frame();
    let mut short: Vec<usize> =
        (0..dpb.len()).filter(usable).filter(|&i| !dpb[i].long_term).collect();
    let mut long: Vec<usize> =
        (0..dpb.len()).filter(usable).filter(|&i| dpb[i].long_term).collect();
    long.sort_by_key(|&i| dpb[i].frame_idx);

    let initial: [Vec<usize>; 2] = match header.slice_type {
        SliceType::P | SliceType::Sp => {
            short.sort_by_key(|&i| std::cmp::Reverse(pic_num(&dpb[i])));
            [short.into_iter().chain(long.iter().copied()).collect(), Vec::new()]
        }
        SliceType::B => {
            let poc = decoding.field_order_cnt[0].min(decoding.field_order_cnt[1]);
            let mut before: Vec<usize> =
                short.iter().copied().filter(|&i| dpb[i].poc() < poc).collect();
            let mut after: Vec<usize> =
                short.iter().copied().filter(|&i| dpb[i].poc() > poc).collect();
            before.sort_by_key(|&i| std::cmp::Reverse(dpb[i].poc()));
            after.sort_by_key(|&i| dpb[i].poc());
            let l0: Vec<usize> = before.iter().chain(&after).chain(&long).copied().collect();
            let mut l1: Vec<usize> = after.iter().chain(&before).chain(&long).copied().collect();
            if l1.len() > 1 && l1 == l0 {
                l1.swap(0, 1);
            }
            [l0, l1]
        }
        SliceType::I | SliceType::Si => [Vec::new(), Vec::new()],
    };

    let mut out = [[None; 32]; 2];
    let lists = match header.slice_type {
        SliceType::B => 2,
        SliceType::P | SliceType::Sp => 1,
        SliceType::I | SliceType::Si => 0,
    };
    for list in 0..lists {
        let n = usize::from(header.num_ref_idx_active_minus1[list]) + 1;
        // One longer than the list while it is modified (8.2.4.3.1), then cut back.
        let mut entries: Vec<Option<usize>> =
            (0..=n).map(|i| initial[list].get(i).copied()).collect();
        entries[n] = None;
        let mut pred = curr;
        for (ref_idx, m) in header.modifications[list].iter().enumerate() {
            if ref_idx >= n {
                return Err(Refused::OutOfRange("the number of list modifications"));
            }
            let target = match *m {
                Modification::Subtract(diff) | Modification::Add(diff) => {
                    let delta = i64::from(diff) + 1;
                    let mut no_wrap = if matches!(m, Modification::Subtract(_)) {
                        pred - delta
                    } else {
                        pred + delta
                    };
                    if no_wrap < 0 {
                        no_wrap += max_frame_num;
                    } else if no_wrap >= max_frame_num {
                        no_wrap -= max_frame_num;
                    }
                    pred = no_wrap;
                    let wanted = if no_wrap > curr { no_wrap - max_frame_num } else { no_wrap };
                    (0..dpb.len())
                        .find(|&i| usable(&i) && !dpb[i].long_term && pic_num(&dpb[i]) == wanted)
                }
                Modification::LongTerm(num) => (0..dpb.len())
                    .find(|&i| usable(&i) && dpb[i].long_term && dpb[i].frame_idx == num),
            }
            .ok_or(Refused::NoSuchReference)?;
            entries.insert(ref_idx, Some(target));
            entries.pop();
            // Every later copy of the picture goes: it now sits at `ref_idx`.
            let mut kept = entries[..=ref_idx].to_vec();
            kept.extend(entries[ref_idx + 1..].iter().filter(|e| **e != Some(target)));
            kept.resize(n + 1, None);
            entries = kept;
        }
        for (slot, entry) in out[list].iter_mut().zip(entries.into_iter().take(n)) {
            *slot = entry;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vrend::video::bitstream::{Escape, Writer};

    fn frame(buffer: u32, frame_idx: u32, poc: i32, long_term: bool) -> Reference {
        Reference {
            buffer,
            frame_idx,
            long_term,
            top: true,
            bottom: true,
            field_order_cnt: [poc, poc + 1],
        }
    }

    fn decoding(poc: i32, dpb: Vec<Reference>) -> Decoding {
        let mut d = Decoding::read(&[]);
        d.field_order_cnt = [poc, poc + 1];
        d.dpb = dpb;
        d
    }

    fn header(slice_type: SliceType, frame_num: u32, active: [u8; 2]) -> SliceHeader {
        SliceHeader {
            first_mb_in_slice: 0,
            slice_type,
            frame_num,
            field_pic: false,
            direct_spatial_mv_pred: false,
            num_ref_idx_active_minus1: active,
            modifications: [Vec::new(), Vec::new()],
            weights: None,
            cabac_init_idc: 0,
            slice_qp_delta: 0,
            disable_deblocking_filter_idc: 0,
            slice_alpha_c0_offset_div2: 0,
            slice_beta_offset_div2: 0,
            data_bit_offset: 0,
        }
    }

    fn desc() -> PictureDesc {
        PictureDesc::read(&[])
    }

    /// 8.2.4.2.1: a P slice's list is the short-term frames by descending PicNum -- frame_num
    /// wrapped below the current one -- then the long-term ones by ascending LongTermPicNum.
    #[test]
    fn a_p_list_orders_by_pic_num_then_long_term() {
        // max_frame_num is 16 for log2_max_frame_num_minus4 = 0. Current frame_num 2: frame 14
        // is from before the wrap, so its PicNum is -2 and it sorts last of the short-term.
        let dpb = vec![
            frame(10, 1, 0, false),
            frame(11, 14, 0, false),
            frame(12, 0, 0, false),
            frame(13, 3, 0, true),
            frame(14, 1, 0, true),
        ];
        let lists =
            ref_lists(&header(SliceType::P, 2, [4, 0]), &decoding(0, dpb), &desc()).unwrap();
        assert_eq!(&lists[0][..6], &[Some(0), Some(2), Some(1), Some(4), Some(3), None]);
        assert!(lists[1].iter().all(Option::is_none));
    }

    /// 8.2.4.2.3: a B slice's lists go by POC around the current picture, nearest first, with
    /// list 1 starting after it -- and a list 1 identical to list 0 has its first two swapped.
    #[test]
    fn b_lists_order_by_poc_around_the_current_picture() {
        let dpb = vec![frame(1, 0, 0, false), frame(2, 1, 8, false), frame(3, 2, 4, false)];
        let lists = ref_lists(&header(SliceType::B, 3, [2, 2]), &decoding(6, dpb.clone()), &desc())
            .unwrap();
        assert_eq!(&lists[0][..3], &[Some(2), Some(0), Some(1)]);
        assert_eq!(&lists[1][..3], &[Some(1), Some(2), Some(0)]);

        // Everything before the current picture: the lists come out equal, and list 1 swaps.
        let lists =
            ref_lists(&header(SliceType::B, 3, [1, 1]), &decoding(10, dpb), &desc()).unwrap();
        assert_eq!(&lists[0][..2], &[Some(1), Some(2)]);
        assert_eq!(&lists[1][..2], &[Some(2), Some(1)]);
    }

    /// 8.2.4.3.1: a modification moves the named picture to the front and removes its later
    /// copy, and the prediction carries from one modification to the next.
    #[test]
    fn a_modification_moves_a_picture_forward() {
        let dpb = vec![frame(1, 3, 0, false), frame(2, 2, 0, false), frame(3, 1, 0, false)];
        let mut h = header(SliceType::P, 4, [2, 0]);
        // 4 - (2 + 1) = 1: frame_num 1 first. Then 1 + (0 + 1) = 2: frame_num 2 second.
        h.modifications[0] = vec![Modification::Subtract(2), Modification::Add(0)];
        let lists = ref_lists(&h, &decoding(0, dpb.clone()), &desc()).unwrap();
        assert_eq!(&lists[0][..3], &[Some(2), Some(1), Some(0)]);

        h.modifications[0] = vec![Modification::Subtract(9)];
        assert_eq!(ref_lists(&h, &decoding(0, dpb), &desc()), Err(Refused::NoSuchReference));
    }

    /// The header reader stops where slice data starts, and says where in RBSP bits.
    #[test]
    fn a_p_slice_header_reads_back_with_its_data_offset() {
        let mut d = desc();
        d.frame_mbs_only = true;
        d.pic_order_cnt_type = 2;
        d.deblocking_filter_control_present = true;
        d.num_ref_idx_l0_active_minus1 = 0;
        let mut w = Writer::new(Escape::Rbsp);
        w.u(8, 0x41); // nal_ref_idc 2, non-IDR slice
        w.ue(0); // first_mb_in_slice
        w.ue(5); // slice_type P (all slices P)
        w.ue(0); // pps id
        w.u(4, 3); // frame_num
        w.u(1, 1); // num_ref_idx_active_override_flag
        w.ue(1); // l0 active minus1
        w.u(1, 1); // ref_pic_list_modification_flag_l0
        w.ue(0);
        w.ue(1); // Subtract(1)
        w.ue(3); // end
        w.u(1, 0); // adaptive_ref_pic_marking_mode_flag
        w.se(-2); // slice_qp_delta
        w.ue(0); // disable_deblocking_filter_idc
        w.se(1);
        w.se(-1);
        w.align();
        let nal = w.finish();
        let h = read_header(&nal, &d).unwrap();
        assert_eq!(h.slice_type, SliceType::P);
        assert_eq!(h.frame_num, 3);
        assert_eq!(h.num_ref_idx_active_minus1, [1, 0]);
        assert_eq!(h.modifications[0], vec![Modification::Subtract(1)]);
        assert_eq!(h.slice_qp_delta, -2);
        assert_eq!((h.slice_alpha_c0_offset_div2, h.slice_beta_offset_div2), (1, -1));
        // 8 header bits, then 1+5+1+4 +1+3 +1+1+3+5 +1 +5 +1+3+3 for the fields above.
        assert_eq!(h.data_bit_offset, 46);
    }

    /// A header that holds an emulation-prevention byte reports the same offset as its RBSP:
    /// VA-API counts the slice data's position without the escapes.
    #[test]
    fn an_escaped_header_reports_its_offset_in_rbsp_bits() {
        let mut d = desc();
        d.frame_mbs_only = true;
        d.pic_order_cnt_type = 2;
        // A 16-bit frame_num of 0 puts two zero bytes in the header, and the next byte is small
        // enough that an encoder must escape it.
        d.log2_max_frame_num_minus4 = 12;
        let header = |escape| {
            let mut w = Writer::new(escape);
            w.u(8, 0x41);
            w.ue(0);
            w.ue(5);
            w.ue(0);
            w.u(16, 0);
            w.u(1, 0); // num_ref_idx_active_override_flag
            w.u(1, 0); // ref_pic_list_modification_flag_l0
            w.u(1, 0); // adaptive_ref_pic_marking_mode_flag
            w.se(-8); // slice_qp_delta: 000010001, which leaves 0x02 after the zero bytes
            w.align();
            w.finish()
        };
        let (escaped, rbsp) = (header(Escape::Rbsp), header(Escape::Raw));
        assert_eq!(escaped.len(), rbsp.len() + 1, "the escaped header carries one 0x03");
        let offset = |nal: &[u8]| read_header(nal, &d).unwrap().data_bit_offset;
        assert_eq!(offset(&escaped), offset(&rbsp));
        assert_eq!(offset(&rbsp), 8 + 1 + 5 + 1 + 16 + 1 + 1 + 1 + 9);
    }
}
