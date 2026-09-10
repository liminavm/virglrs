// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! Storage a rendered frame can be handed to someone else in, and the platform that mints it.
//!
//! Every host has one currency for this and no host has two. On macOS it is the IOSurface
//! [`crate::metal`] mints, because KosmicKrisp will import one but cannot produce one, so the
//! host is the only party that can create the storage a guest renders into. On Linux the
//! direction is the other way round: the driver allocates and this renderer exports a descriptor
//! of what it allocated, which is [`crate::dmabuf`].
//!
//! That inversion is why this module exists and why it holds so little. What the renderer needs
//! from storage it does not own is one question -- *what keeps these pixels alive* -- and the
//! answer differs per host only in what the keepalive is made of. [`Held`] is that question, and
//! it is the same trait on both. Everything below it is the platform's, including which
//! direction storage travels in.
//!
//! No unsafe here: this is the neutral contract, and the foreign calls that satisfy it live in
//! the platform module behind it.

#[cfg(target_os = "macos")]
pub use crate::metal::{PixelFormat, PlanarFormat, Surface};

#[cfg(not(target_os = "macos"))]
pub use crate::dmabuf::{PixelFormat, PlanarFormat, Surface};

/// One plane of a planar surface, as this side lays it out.
///
/// The pitch and offset are dictated, never discovered. The guest is told this layout and
/// addresses the planes by it, so a surface laid out any other way shears every plane after the
/// first -- which is why [`Surface::planar`] sends all five numbers and refuses a surface that
/// came back with different ones.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PlaneShape {
    pub width: u32,
    pub height: u32,
    pub bytes_per_element: u32,
    /// The tight row, aligned up to what a linear Metal texture of this plane's *sampled*
    /// format demands.
    pub bytes_per_row: u32,
    /// Where the plane starts in the surface's one allocation: the planes before it, tightly.
    pub offset: u32,
}

/// Whatever keeps a [`Surface`] alive, seen as the surface.
///
/// A surface is adopted as GL storage by whoever renders through it, and that is not always
/// whoever minted it: a venus allocation's surface is imported by a classic context compositing
/// the client that owns it. What the importer holds has to be the *owner's* share, because the
/// owner's share carries more than the pixels -- venus's also carries the memory charge, whose
/// release is the share going away. A fresh share over the same surface would keep the pixels
/// and drop the charge, and the ledger would report as free memory that is still held.
///
/// So the importer names what it holds by what it can do with it -- yield the surface -- and the
/// owner decides what holding means.
pub trait Held: Send + Sync {
    fn surface(&self) -> &Surface;
}
