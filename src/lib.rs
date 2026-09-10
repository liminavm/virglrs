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

/// A lookup table keyed by a handle, hashed rather than ordered.
///
/// These are the tables the command path walks -- a resource handle, an object handle, a
/// sub-context id -- and they are looked up several times per guest command. A `BTreeMap` answers
/// each one with a tree descent and a chain of key comparisons; measured under a live desktop
/// (15 000-fish aquarium, GNOME, vkmark) `BTreeMap` operations were **23.9%** of all time spent
/// processing guest GL commands, against 14% for the GL driver those commands exist to drive. The
/// C reaches for `_mesa_hash_table` for exactly these tables, and the choice of container is the
/// whole difference.
///
/// `FxHasher` and not the default: these keys are small integers, and SipHash's DoS resistance
/// buys nothing for a table whose keys never leave this process. Nothing outside the renderer
/// chooses them, so there is no adversary to be resistant to.
///
/// **Use this only where iteration order cannot be observed.** A `BTreeMap` here was also an
/// implicit sort, and two places leaned on it: the snapshot journal (which now sorts by `Seq`
/// explicitly in `vrend::journal::order`, as it always should have) and sub-context teardown
/// (which names its reverse-id order at the site). A table that must be walked in key order keeps
/// its `BTreeMap`, and says why.
pub type Map<K, V> = rustc_hash::FxHashMap<K, V>;

pub mod abi;
pub mod budget;
pub mod config;
#[cfg(target_os = "macos")]
#[path = "videotoolbox.rs"]
pub mod decode;
#[cfg(not(target_os = "macos"))]
#[path = "decode_unbacked.rs"]
pub mod decode;
#[cfg(not(target_os = "macos"))]
pub mod dmabuf;
pub mod fence;
pub mod ffi;
pub mod guest_mem;
pub mod ids;
#[cfg(target_os = "macos")]
pub mod metal;
pub mod renderer;
pub mod surface;
pub mod venus;
pub mod vrend;
pub mod vulkan;
