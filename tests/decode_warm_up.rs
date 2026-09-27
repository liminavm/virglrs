// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! The warm-up takes the process's first-session cost off the first real decode.
//!
//! Its own test binary, so no library test has warmed the process it runs in. Each sample is a
//! fresh process besides: the cost is paid once per process, so a process can be measured once.

#![cfg(target_os = "macos")]

use std::time::{Duration, Instant};
use virglrenderer::decode::{Codec, Configuration, PixelFormat, Session, SessionKey, Support};

/// A cold process builds its first session in 50-60 ms and a warm one in 5-7, idle. The bound
/// sits well between them.
const WARM: Duration = Duration::from_millis(25);

/// How many processes are measured; the verdict is their median. With every core busy a warm
/// session still costs ~5 ms, but about one in ten stalls for 20-55 ms, which one sample cannot
/// tell from a cold process. The median of five is over the bound only if three stall, under 1% of
/// runs at that load; a process that was not warmed costs 50 ms or more every time, and fails every
/// sample. Each process costs ~1.5 s, nearly all of it `Support::probe`, which is that slow only
/// from an executable in `target/debug/deps`.
const PROCESSES: usize = 5;

/// Set in a sampled process's environment; its test then measures instead of returning.
const SAMPLE: &str = "VIRGLRS_WARM_UP_SAMPLE";

/// After the warm-up has run, the first session a guest's frame asks for -- H.264 here, a codec the
/// warm-up did not use -- is built at the warm cost.
#[test]
fn after_the_warm_up_a_first_session_of_another_codec_is_cheap() {
    let support = Support::probe();
    assert!(support.decodes(Codec::Vp9) && support.decodes(Codec::H264), "no decode silicon");
    let mut took: Vec<Duration> = (0..PROCESSES).map(|_| sampled()).collect();
    took.sort();
    let median = took[PROCESSES / 2];
    assert!(median < WARM, "the first H.264 sessions took {took:?}; the processes were not warm");
}

/// One fresh process's first H.264 session, run as [`sample_role`].
fn sampled() -> Duration {
    let exe = std::env::current_exe().expect("the test binary");
    let out = std::process::Command::new(exe)
        .args(["--exact", "sample_role", "--nocapture"])
        .env(SAMPLE, "1")
        .output()
        .expect("a sampled process");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "the sampled process failed: {stdout}");
    let micros = stdout
        .lines()
        .find_map(|l| l.strip_prefix("first-h264-us "))
        .unwrap_or_else(|| panic!("the sampled process reported no time: {stdout}"));
    Duration::from_micros(micros.parse().expect("a count of microseconds"))
}

/// The sampled process: warm up as the renderer does, then time the first H.264 session. A normal
/// run has nothing to sample and passes.
#[test]
fn sample_role() {
    if std::env::var_os(SAMPLE).is_none() {
        return;
    }
    let support = Support::probe();
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
    println!("first-h264-us {}", began.elapsed().as_micros());
    std::process::exit(0);
}

/// The SPS and PPS of a 1280x720 High-profile x264 stream, as NAL units without start codes.
const SPS: &[u8] = &[
    0x67, 0x64, 0x00, 0x1f, 0xac, 0xd9, 0x40, 0x50, 0x05, 0xbb, 0x01, 0x10, 0x00, 0x00, 0x03, 0x00,
    0x10, 0x00, 0x00, 0x03, 0x03, 0xc0, 0xf1, 0x83, 0x19, 0x60,
];
const PPS: &[u8] = &[0x68, 0xeb, 0xe3, 0xcb, 0x22, 0xc0];
