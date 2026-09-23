// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! An AV1 decode as a guest submits it: a sequence and a frame descriptor, then the tile data the
//! renderer wraps into a temporal unit. The first two bytes say where the descriptor ends.
#![no_main]

use libfuzzer_sys::fuzz_target;
use virglrenderer::vrend::video::av1::{FrameDesc, ObuState, SeqParams};

fuzz_target!(|data: &[u8]| {
    let Some((split, rest)) = data.split_first_chunk::<2>() else { return };
    let at = (u16::from_le_bytes(*split) as usize).min(rest.len());
    let (desc, tiles) = rest.split_at(at);
    if let Ok(seq) = SeqParams::read(desc) {
        let _ = seq.sequence_header();
    }
    let Ok(frame) = FrameDesc::read(desc) else { return };
    let mut state = ObuState::<()>::new();
    let _ = state.build_temporal_unit(&frame, tiles, ());
    let _ = state.flush_held(&frame);
});
