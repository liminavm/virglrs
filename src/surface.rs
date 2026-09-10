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

// How storage could be handed to another process, as every importer spells it.
//
// Here and not in `crate::dmabuf` because it is the *question*, not the answer: "how would this
// resource be exported" is asked of the Rust API on both hosts, and a caller must be able to ask
// without first knowing which host it is on. A host that exports nothing answers that it cannot,
// which is a different thing from the question being unaskable.
//
// Plain data throughout -- no descriptor, no lifetime, nothing a platform owns. What owns the
// descriptor is `Held`; what fills these numbers in is the platform module behind it.

/// `DRM_FORMAT_MOD_LINEAR`: rows one after another, no tiling.
pub const DRM_FORMAT_MOD_LINEAR: u64 = 0;
/// `DRM_FORMAT_MOD_INVALID`: the importer should not assume any particular layout. What a driver
/// that cannot report a modifier leaves behind, and never something to pass on as if it were one.
pub const DRM_FORMAT_MOD_INVALID: u64 = 0x00ff_ffff_ffff_ffff;

/// The most planes any format here has. NV12 is two; the array is sized for what DRM allows so
/// that a format added later does not silently truncate.
pub const MAX_PLANES: usize = 4;

/// Where one plane sits in the exported allocation.
///
/// Offset and pitch together, because neither locates a plane on its own and a caller holding one
/// without the other has to guess the second -- which is how a plane ends up sheared. Read from
/// the driver, never computed here: what this side would compute is what the layout *ought* to
/// be, and the export is worth having precisely because the driver may disagree.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PlaneLayout {
    pub offset: u64,
    pub pitch: u32,
}

/// What the exporting driver said about the allocation, as an importer needs it.
///
/// One value rather than six arguments threaded through the export path: an importer needs every
/// field or none of them, and a `fourcc` that arrived without its modifier describes a buffer
/// nobody can read. See [`crate::surface::Held`] for why the descriptor and the keepalive travel
/// together rather than as a pair.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Layout {
    pub width: u32,
    pub height: u32,
    pub fourcc: u32,
    /// `DRM_FORMAT_MOD_INVALID` when the driver would not say. An importer must then be told the
    /// modifier is unknown rather than handed `LINEAR`, which is a different claim.
    pub modifier: u64,
    pub planes: [PlaneLayout; MAX_PLANES],
    pub plane_count: u32,
    /// The whole allocation, which is what a mapping covers and what the budget was charged.
    pub alloc_size: u64,
}

impl Layout {
    /// The first plane's pitch, for the many callers that only ever have one plane.
    pub fn bytes_per_row(&self) -> u32 {
        self.planes[0].pitch
    }
}

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
