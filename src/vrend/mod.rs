// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! vrend -- the classic virgl renderer: gallium state and TGSI shaders over the wire, GL on the
//! host.
//!
//! `proto` is the protocol as types, `decode` the boundary that parses the guest's dwords into
//! them and refuses the rest, `encode` the way back -- which exists so that a recorded stream
//! decoded and re-encoded is a differential test against the guest's own encoder, with no C dump
//! to diff against. `pipe` is gallium's vocabulary, the enums those types are made of.
//!
//! `formats` is what each format *is* (gallium's description, which every transfer size comes
//! from) and what GL calls it. `tgsi` is the shader language the guest sends, as typed tokens.
//!
//! `egl` and `gl` are the host side: the winsys and the driver's entry points, the two named
//! unsafe modules of this renderer (CLAUDE.md). Everything above them is safe Rust.

pub mod blitter;
pub mod caps;
pub mod content;
pub mod context;
pub mod debug;
pub mod decode;
pub mod dirty;
pub mod egl;
pub mod encode;
pub mod features;
pub mod formats;
pub mod gl;
pub mod journal;
pub mod pipe;
pub mod proto;
pub mod resource;
pub mod shader;
pub mod tally;
pub mod tgsi;
pub mod transfer;
pub mod video;
#[allow(clippy::module_inception)]
pub mod vrend;
pub mod waiter;

/// Serialises the tests that open a display of their own.
///
/// They are ordinary tests and run with the rest, but they cannot run *beside* each other: two
/// displays open at once in one process leave KosmicKrisp unable to make a shared context, and
/// the tests then fail each other rather than the thing they are about. `cargo test` gives every
/// test its own thread, so the constraint has to be held here -- a note telling a reader to pass
/// a filter is not one, and was how these sat unrun long enough for one of them to rot.
///
/// It orders them; it does not make them cheap. Each still opens and terminates a display.
#[cfg(test)]
pub(crate) fn one_display_at_a_time() -> std::sync::MutexGuard<'static, ()> {
    static DISPLAY: std::sync::Mutex<()> = std::sync::Mutex::new(());
    // Take the guard back out of a poisoned lock. `panic = "abort"` does not reach here: cargo
    // ignores the setting for the test profile, so a failing test unwinds and poisons this. And
    // there is nothing to recover -- the guard serialises displays, it does not protect state a
    // panicking test could have left half-written, so the next test may have it. Propagating the
    // poison instead turned one real `eglInitialize` failure into six red tests, five of which
    // named the lock rather than the thing they were about.
    DISPLAY.lock().unwrap_or_else(|e| e.into_inner())
}
