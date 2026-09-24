// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! The warm-up takes the process's first-session cost off the first real decode.
//!
//! Its own test binary, so its own process: the first session any test in the library suite
//! builds warms VideoToolbox for every test after it, and a warm process cannot show a cold cost.

#![cfg(target_os = "macos")]

use std::time::{Duration, Instant};
use virglrenderer::decode::{Codec, Configuration, PixelFormat, Session, SessionKey, Support};

/// A cold process builds its first session in ~50 ms and a warm one in ~3. The bound sits well
/// between them, with room for a loaded host.
const WARM: Duration = Duration::from_millis(25);

/// After the warm-up has run, the first session a guest's frame asks for -- H.264 here, a codec the
/// warm-up did not use -- is built at the warm cost.
#[test]
fn after_the_warm_up_a_first_session_of_another_codec_is_cheap() {
    let support = Support::probe();
    assert!(support.decodes(Codec::Vp9) && support.decodes(Codec::H264), "no decode silicon");
    if let Some(warming) = virglrenderer::decode::warm_up(&support) {
        warming.join().expect("the warm-up thread finishes");
    }
    let key = SessionKey {
        width: 1280,
        height: 720,
        pixels: PixelFormat::BiPlanar420,
        config: Configuration::H264 { sps: SPS.to_vec(), pps: PPS.to_vec() },
    };
    let began = Instant::now();
    Session::create(key).expect("VideoToolbox builds an H.264 session");
    let took = began.elapsed();
    assert!(took < WARM, "the first H.264 session took {took:?}; the process was not warm");
}

/// The SPS and PPS of a 1280x720 High-profile x264 stream, as NAL units without start codes.
const SPS: &[u8] = &[
    0x67, 0x64, 0x00, 0x1f, 0xac, 0xd9, 0x40, 0x50, 0x05, 0xbb, 0x01, 0x10, 0x00, 0x00, 0x03, 0x00,
    0x10, 0x00, 0x00, 0x03, 0x03, 0xc0, 0xf1, 0x83, 0x19, 0x60,
];
const PPS: &[u8] = &[0x68, 0xeb, 0xe3, 0xcb, 0x22, 0xc0];
