// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! venus -- Vulkan passthrough.
//!
//! `cs` is the command stream: the trust boundary every guest byte crosses. The decoder generated
//! against it lives in `virglrs/venus-gen/`, which forks venus-protocol's generator to emit Rust
//! rather than C, and is built into `OUT_DIR` by `build.rs` -- never checked in, because generated
//! code that is edited by hand is a bug in a place no one will look.

pub mod capset;
pub mod context;
#[allow(unsafe_code)]
pub mod cs;
#[allow(unsafe_code)]
pub mod driver;
pub mod journal;
pub mod ledger;
pub mod monitor;
pub mod objects;
#[allow(unsafe_code)]
pub mod proto;
pub mod ring;
pub mod ring_thread;
pub mod sync;
pub mod tally;
pub mod vkr;
