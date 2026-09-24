// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! Time VideoToolbox session creation, in order, in one process.
//!
//! The first session a process builds costs tens of milliseconds and every later one a few, so
//! the question is what the first one warms: VideoToolbox as a whole, or only the codec it was
//! for. Each argument is one session: `vp9`, or `h264:<annex-b file>` / `hevc:<annex-b file>`,
//! whose parameter sets are taken from the stream. Run each order as its own process.
//!
//! `cargo run --example first-session -- vp9 h264:t.h264 hevc:t.hevc vp9`

use std::time::Instant;
use virglrenderer::decode::{Configuration, PixelFormat, Session, SessionKey, Support};

/// The NAL units of an Annex B stream, without their start codes.
fn nals(stream: &[u8]) -> Vec<&[u8]> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 3 <= stream.len() {
        if stream[i..i + 3] == [0, 0, 1] {
            starts.push(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut out = Vec::new();
    for (n, &s) in starts.iter().enumerate() {
        let mut end = starts.get(n + 1).map_or(stream.len(), |&next| next - 3);
        while end > s && stream[end - 1] == 0 {
            end -= 1;
        }
        out.push(&stream[s..end]);
    }
    out
}

/// The first NAL unit of a type, by the codec's own reading of the header.
fn first(units: &[&[u8]], kind: impl Fn(u8) -> u8, want: u8) -> Vec<u8> {
    units
        .iter()
        .find(|u| kind(u[0]) == want)
        .expect("the stream carries the parameter set")
        .to_vec()
}

fn main() {
    // Registers the supplemental decoders, without which there is no VP9 session to build.
    let _support = Support::probe();
    let started = Instant::now();
    for arg in std::env::args().skip(1) {
        let (name, path) = arg.split_once(':').unwrap_or((arg.as_str(), ""));
        let config = match name {
            "vp9" => Configuration::vp9(0, 8, 1),
            "h264" => {
                let bytes = std::fs::read(path).expect("reading the H.264 stream");
                let units = nals(&bytes);
                let t = |b: u8| b & 0x1f;
                Configuration::H264 { sps: first(&units, t, 7), pps: first(&units, t, 8) }
            }
            "hevc" => {
                let bytes = std::fs::read(path).expect("reading the HEVC stream");
                let units = nals(&bytes);
                let t = |b: u8| (b >> 1) & 0x3f;
                Configuration::Hevc {
                    vps: first(&units, t, 32),
                    sps: first(&units, t, 33),
                    pps: first(&units, t, 34),
                }
            }
            other => panic!("unknown codec {other}"),
        };
        let key = SessionKey { width: 1280, height: 720, pixels: PixelFormat::BiPlanar420, config };
        let began = Instant::now();
        let session = Session::create(key);
        let took = began.elapsed();
        println!(
            "{name:5} {:7.2} ms  {}  (at {:.0} ms)",
            took.as_secs_f64() * 1e3,
            if session.is_ok() { "ok" } else { "FAILED" },
            began.duration_since(started).as_secs_f64() * 1e3,
        );
    }
}
