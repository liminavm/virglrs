// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! virglrs -- the Rust virglrenderer.
//!
//! The plan is `docs/design.md`; the test floor is `harness/README.md`. This crate builds a
//! `cdylib` exporting the same C ABI as `libvirglrenderer.1.dylib`, so it is swapped in by pointing
//! `VIRGL_PREFIX` at its prefix.
//!
//! What is real here: the ABI types, the resource table, the context table, fence tracking and
//! asynchronous fence retirement. What is not: both renderers. Every entry point belonging to a
//! later phase returns `-ENOTSUP` rather than a plausible success -- see `ffi::todo_phase`.

/// The prefix every refusal is printed under, whichever renderer refused.
///
/// A poisoned context is this renderer's one fatal outcome: the guest's client keeps running,
/// nothing crashes, and the only trace is a line on stderr. That makes the line an interface,
/// and it has to be greppable by exactly one needle -- vrend and venus each used to poison in
/// their own words, so a harness checking for one spelling read the other's fatal as silence,
/// and a compositor's dead context sat in a passing test's diagnostics for days.
///
/// Whatever follows this prefix, a line carrying it means work was refused and the context it
/// names is finished. Do not print it for anything a guest can recover from.
pub const REFUSED: &str = "[virglrs] refused:";

pub mod abi;
pub mod budget;
pub mod config;
pub mod fence;
pub mod ffi;
pub mod guest_mem;
pub mod ids;
pub mod metal;
pub mod renderer;
pub mod venus;
pub mod videotoolbox;
pub mod vrend;
pub mod vulkan;
