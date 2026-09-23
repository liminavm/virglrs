// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! An H.264 bitstream as a guest submits it for decode: Annex B, every byte the guest's.
#![no_main]

use libfuzzer_sys::fuzz_target;
use virglrenderer::vrend::video::h264;

fuzz_target!(|annexb: &[u8]| {
    let _ = h264::annexb_to_avcc(annexb);
    let _ = h264::slice_pps_id(annexb);
    let _ = h264::has_idr(annexb);
});
