// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! Storage a rendered frame can be handed to someone else in, and the platform that mints it.
//!
//! Every host has one currency for this and no host has two. On macOS it is the IOSurface
//! [`crate::metal`] mints, because KosmicKrisp will import one but cannot produce one, so the
//! host is the only party that can create the storage a guest renders into. On Linux the
//! direction is the other way round -- the driver allocates and the host exports a descriptor of
//! what it allocated -- so nothing is minted here at all and [`Surface`] is uninhabited.
//!
//! That inversion is why this module exists and why it holds so little. What the renderer needs
//! from storage it does not own is one question -- *what keeps these pixels alive* -- and the
//! answer differs per host only in what the keepalive is made of. [`Held`] is that question, and
//! it is the same trait on both. Everything below it is the platform's.
//!
//! No unsafe here: this is the neutral contract, and the foreign calls that satisfy it live in
//! the platform module behind it.

#[cfg(target_os = "macos")]
pub use crate::metal::{PixelFormat, PlanarFormat, Surface};

#[cfg(not(target_os = "macos"))]
pub use unbacked::{PixelFormat, PlanarFormat, Surface};

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

/// A host with no storage of its own to mint.
///
/// [`Surface`] here is uninhabited, which is the whole design: the plumbing that carries one --
/// `Option<Arc<dyn Held>>` through vrend's resources, the renderer's `Classic` backing, venus's
/// exported allocations -- compiles unchanged and is structurally `None`. A host that cannot
/// mint storage does not need a flag saying so, and no call site needs a branch for a case the
/// type system has already ruled out.
///
/// Every method is `match *self {}`: total, because there is no value to answer for. A caller
/// that reaches one has already proved it holds a surface, which on this host it cannot.
/// Minting has no counterpart at all, so the constructors are simply absent and asking for one
/// is a build error rather than a runtime refusal -- see `docs/linux-port.md` for what replaces
/// them.
#[cfg(not(target_os = "macos"))]
mod unbacked {
    use crate::ids::SurfaceId;
    use crate::surface::PlaneShape;

    /// A pixel format storage can be minted in. See [`crate::metal::PixelFormat`] for what the
    /// variants mean; they are the wire's, not a platform's, so they are the same on every host.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum PixelFormat {
        /// 32-bit BGRA, what a compositor presents.
        Bgra,
        /// 32-bit RGBA. The same bytes in the other order.
        Rgba,
    }

    /// The layout of planar storage. See [`crate::metal::PlanarFormat`].
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum PlanarFormat {
        /// A full-resolution 8-bit luma plane, then a half-resolution plane of interleaved
        /// two-byte chroma. What NV12 names on the wire.
        BiPlanar420,
    }

    impl PlanarFormat {
        /// How many bytes one sample of a plane is. A property of the layout, not of a host.
        pub fn bytes_per_element(self, plane: usize) -> u32 {
            match (self, plane) {
                (PlanarFormat::BiPlanar420, 0) => 1,
                (PlanarFormat::BiPlanar420, _) => 2,
            }
        }
    }

    /// Storage this host cannot mint. Uninhabited on purpose -- see the module docs.
    pub enum Surface {}

    impl Surface {
        pub fn id(&self) -> SurfaceId {
            match *self {}
        }

        pub fn bytes_per_row(&self) -> u32 {
            match *self {}
        }

        pub fn alloc_size(&self) -> u64 {
            match *self {}
        }

        pub fn host_addr(&self) -> usize {
            match *self {}
        }

        pub fn read_into(&self, _dst: &mut [u8]) -> usize {
            match *self {}
        }

        pub fn write_from(&self, _src: &[u8]) -> usize {
            match *self {}
        }

        pub fn read_rows(&self, _dst: &mut [u8], _stride: usize, _height: u32) -> u32 {
            match *self {}
        }

        pub fn plane_count(&self) -> u32 {
            match *self {}
        }

        pub fn plane(&self, _plane: u32) -> Option<(PlaneShape, u32)> {
            match *self {}
        }

        pub fn write_plane(
            &self,
            _plane: u32,
            _src: &[u8],
            _src_pitch: usize,
            _rows: u32,
            _row_bytes: usize,
        ) -> bool {
            match *self {}
        }

        pub fn copy_plane_from(&self, _src: &Surface, _plane: u32) -> bool {
            match *self {}
        }
    }

    impl super::Held for Surface {
        fn surface(&self) -> &Surface {
            match *self {}
        }
    }

    impl std::fmt::Debug for Surface {
        fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match *self {}
        }
    }
}
