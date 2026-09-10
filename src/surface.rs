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

/// Why a layout does not describe a buffer.
///
/// Every variant is the guest's error, not the host's: these are the checks a layout that arrived
/// over the wire has to pass before anything maps or images the descriptor under it. So they are
/// reported and refused, never asserted -- see CLAUDE.md on the trust boundary.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BadLayout {
    /// An empty image. Nothing downstream has a meaning for a zero extent.
    Extent { width: u32, height: u32 },
    /// A FourCC this renderer has no plane rule for, so it cannot check the rest.
    Fourcc(u32),
    /// A layout claim this side cannot check. See [`Describable::describe`].
    Modifier(u64),
    /// The plane count disagrees with what the FourCC has.
    PlaneCount { said: u32, wants: u32 },
    /// A row narrower than the pixels it must hold.
    Pitch { plane: u32, pitch: u32, tight: u32 },
    /// A plane that reaches past the end of the buffer.
    Overrun { plane: u32, end: u64, size: u64 },
    /// The descriptor could not be duplicated -- out of file descriptors, and the host's fault
    /// rather than the guest's.
    NoDescriptor,
}

impl core::fmt::Display for BadLayout {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BadLayout::Extent { width, height } => write!(f, "an empty {width}x{height} image"),
            BadLayout::Fourcc(c) => write!(f, "fourcc {c:#010x} has no plane rule here"),
            BadLayout::Modifier(m) => {
                write!(f, "modifier {m:#018x} is not one this side can bound")
            }
            BadLayout::PlaneCount { said, wants } => {
                write!(f, "{said} planes for a format that has {wants}")
            }
            BadLayout::Pitch { plane, pitch, tight } => {
                write!(f, "plane {plane}'s pitch {pitch} is narrower than its {tight}-byte row")
            }
            BadLayout::Overrun { plane, end, size } => {
                write!(f, "plane {plane} ends at {end} of a {size}-byte buffer")
            }
            BadLayout::NoDescriptor => f.write_str("the descriptor could not be duplicated"),
        }
    }
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

/// Storage that has no layout of its own, and can only be read under one someone else supplies.
///
/// The exporting host's other shape. Most storage arrives with the layout the driver laid it out
/// in, and is a [`Surface`] from the moment it exists. Some never can be: a `VkBuffer` a guest
/// blits its frame into has no format, no tiling and no layout query, so the driver has no answer
/// to give and the only description that will ever exist is the guest's own, which arrives later
/// with the command that says what the resource is.
///
/// Reading is not filling in. Each call mints a *separate* [`Held`] over its own reference to the
/// same storage, so two importers may read one buffer under different layouts without either
/// being able to disturb the other -- and a guest that describes one resource two different ways
/// cannot make either description read the other's numbers.
///
/// Host-neutral because the question is: "can this be adopted as a texture's storage" is asked by
/// vrend on both hosts. A host whose storage always carries its own layout implements this
/// nowhere, and [`Adoptable`] then only ever holds its other arm.
pub trait Describable: Send + Sync {
    /// Taken by `Arc<Self>` rather than by reference: what comes back has to hold a share of the
    /// storage it reads, and a share cannot be recovered from a borrow of one. Passing the share
    /// in is what makes "the reading outlives nothing it depends on" a property of the signature.
    fn describe(
        self: std::sync::Arc<Self>,
        layout: Layout,
    ) -> Result<std::sync::Arc<dyn Held>, BadLayout>;
}

/// Storage a blob resource offers a context that is about to say what it is.
///
/// The two shapes above, as one value, because the choice between them is the storage's and not
/// the importer's -- a caller that had to test which it held would be re-deciding something
/// already settled at the export.
pub enum Adoptable {
    /// Storage that knows its own layout: the driver laid it out and said so, or this host minted
    /// it. Adopted exactly as it stands.
    Ready(std::sync::Arc<dyn Held>),
    /// Storage with no layout of its own. The description the guest sends is the only one there
    /// will ever be, and it is checked against the storage before anything reads through it.
    Unread(std::sync::Arc<dyn Describable>),
}
