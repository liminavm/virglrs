// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! A snapshot's sync blob, which crosses a save and a restore and so is not ours by the time it is
//! read back. Decoding it must refuse what it cannot read rather than panic, and what it does read
//! must survive another encode and decode unchanged.
#![no_main]

use libfuzzer_sys::fuzz_target;
use virglrenderer::venus::sync::{decode, encode};

fuzz_target!(|blob: &[u8]| {
    let Ok(captured) = decode(blob) else { return };
    let again = decode(&encode(captured.clone())).expect("a blob this encoder wrote decodes");
    let mut want = captured;
    let mut got = again;
    want.sort();
    got.sort();
    assert_eq!(want, got, "a decoded blob does not survive another round trip");
});
