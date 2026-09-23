// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! An HEVC decode as a guest submits it: a picture descriptor, then the slices. The first two
//! bytes say where the one ends and the other begins, so both halves are the fuzzer's to shape.
#![no_main]

use libfuzzer_sys::fuzz_target;
use virglrenderer::vrend::video::h265::{HevcProfile, PictureDesc, RefPicSets};

fuzz_target!(|data: &[u8]| {
    let Some((split, rest)) = data.split_first_chunk::<2>() else { return };
    let at = (u16::from_le_bytes(*split) as usize).min(rest.len());
    let (desc, slices) = rest.split_at(at);
    let picture = PictureDesc::read(desc);
    let _ = picture.parameter_sets(1920, 1080, HevcProfile::Main);
    let _ = picture.slice_inspect(slices, &mut RefPicSets::Exact);
});
