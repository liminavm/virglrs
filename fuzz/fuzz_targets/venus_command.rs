// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! One venus command, as a guest could send it: decoded, and if it decodes, encoded back.
//!
//! The decoder is the trust boundary, so anything that panics here is a guest aborting the
//! renderer. Beyond that, two properties hold of any command that decodes cleanly: `sizeof` agrees
//! with what the encoder wrote, and the round trip is a fixed point -- decoding what was written
//! and writing it again changes nothing. Byte identity with the input is not one of them: a guest
//! may send a boolean that is not 0 or 1, or a chained struct the renderer skips, and both come
//! back canonical.
#![no_main]

use std::sync::atomic::AtomicBool;

use bumpalo::Bump;
use libfuzzer_sys::fuzz_target;
use virglrenderer::venus::cs::{AllOfIt, Decoder, Encoder, IdentityObjects};
use virglrenderer::venus::proto::serialize::vn_round_trip_args;
use virglrenderer::venus::proto::types::{VkCommandTypeEXT, VkFlags};

/// Round-trip one command. `None` if it did not decode cleanly, else what was written and what
/// `sizeof` said it would be.
fn round_trip(wire: &[u8]) -> Option<(Vec<u8>, usize)> {
    let temp = Bump::new();
    let hard = AtomicBool::new(false);
    let mut dec = Decoder::new(wire, &temp, &IdentityObjects, &hard);
    let cmd = dec.decode_scalar::<VkCommandTypeEXT>();
    let flags = dec.decode_scalar::<VkFlags>();
    if dec.fatal() {
        return None;
    }
    let mut out = Vec::new();
    let mut enc = Encoder::growing(&mut out, &AllOfIt);
    let size = vn_round_trip_args(&mut dec, &mut enc, cmd, flags)?;
    if dec.fatal() {
        return None;
    }
    let written = enc.written().len();
    assert!(!enc.fatal(), "the encoder ran out of room it grows itself");
    drop(enc);
    out.truncate(written);
    Some((out, size))
}

fuzz_target!(|wire: &[u8]| {
    let Some((once, size)) = round_trip(wire) else { return };
    assert_eq!(size, once.len(), "sizeof disagrees with what the encoder wrote");
    let (twice, _) = round_trip(&once).expect("a command this encoder wrote decodes");
    assert_eq!(once, twice, "the round trip is not a fixed point");
});
