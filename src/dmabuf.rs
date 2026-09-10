// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! Storage a driver allocated and this renderer exported a descriptor of.
//!
//! The counterpart of [`crate::metal`], and its mirror image. On macOS the host mints an
//! IOSurface and the driver imports it, because KosmicKrisp will import storage and cannot
//! produce it. Here the driver allocates and the host asks for a descriptor of what it
//! allocated: `VkExportMemoryAllocateInfo` at the allocate, `vkGetMemoryFdKHR` after. So nothing
//! in this module mints anything -- there is no constructor that makes storage, only one that
//! takes ownership of a descriptor of storage that already exists.
//!
//! That inversion is the whole reason the two modules cannot be one. It also decides what a
//! [`Surface`] here *is*: a file descriptor plus the layout needed to interpret it, which is what
//! a compositor imports and what a GL context turns into an `EGLImage`. The pixels are the
//! driver's; this owns the right to name them and the obligation to close the descriptor.
//!
//! # CPU access is a favour, not a property
//!
//! An IOSurface is CPU-addressable by construction. A dma-buf is not: whether the fd can be
//! mapped is the exporting driver's choice, and a tiled buffer's bytes would not be pixels even
//! if it could. So every CPU-side method here goes through [`Surface::mapped`], which takes the
//! mapping lazily and answers `None` when there is none -- and each of those methods then reports
//! nothing read rather than a buffer of zeros. A zero-filled read is indistinguishable from a
//! black frame, which is the failure this renderer keeps finding in its own diagnostics; saying
//! "no bytes" is worse to receive and impossible to misread.
//!
//! # Unsafe
//!
//! This module is on CLAUDE.md's list by decision, not by accident, and `docs/linux-port.md` says
//! why: mapping a descriptor is a foreign call with a lifetime the type system cannot see, and
//! the alternative was to spread it through `driver.rs` and `egl.rs`. The unsafe here is the
//! `mmap`/`munmap` pair and nothing else -- the layout arithmetic above it is ordinary Rust, and
//! the export call itself belongs to the driver module that owns the device.

use std::os::fd::{AsRawFd, OwnedFd};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use crate::ids::SurfaceId;
use crate::surface::PlaneShape;

/// A DRM `fourcc`, as the kernel and every importer spell a pixel format.
///
/// `a | b << 8 | c << 16 | d << 24` -- the four characters in memory order, which is why the
/// names read backwards from the byte order they describe.
const fn fourcc(a: u8, b: u8, c: u8, d: u8) -> u32 {
    (a as u32) | ((b as u32) << 8) | ((c as u32) << 16) | ((d as u32) << 24)
}

/// `DRM_FORMAT_ARGB8888`: `A:R:G:B` in a little-endian word, so `B,G,R,A` in memory.
const DRM_FORMAT_ARGB8888: u32 = fourcc(b'A', b'R', b'2', b'4');
/// `DRM_FORMAT_ABGR8888`: `A:B:G:R` in a little-endian word, so `R,G,B,A` in memory.
const DRM_FORMAT_ABGR8888: u32 = fourcc(b'A', b'B', b'2', b'4');
/// `DRM_FORMAT_NV12`: a luma plane then an interleaved half-resolution chroma plane.
const DRM_FORMAT_NV12: u32 = fourcc(b'N', b'V', b'1', b'2');

/// `DRM_FORMAT_MOD_LINEAR`: rows one after another, no tiling.
pub const DRM_FORMAT_MOD_LINEAR: u64 = 0;
/// `DRM_FORMAT_MOD_INVALID`: the importer should not assume any particular layout. What a driver
/// that cannot report a modifier leaves behind, and never something to pass on as if it were one.
pub const DRM_FORMAT_MOD_INVALID: u64 = 0x00ff_ffff_ffff_ffff;

/// The most planes any format here has. NV12 is two; the array is sized for what DRM allows so
/// that a format added later does not silently truncate.
pub const MAX_PLANES: usize = 4;

/// A pixel format storage can be exported in.
///
/// The same two variants [`crate::metal::PixelFormat`] has, and for the same reason: these are
/// the wire's formats, not a platform's. What differs is only what each one is called downstream
/// -- a `MTLPixelFormat` there, a DRM `fourcc` here.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PixelFormat {
    /// 32-bit BGRA, what a compositor presents.
    Bgra,
    /// 32-bit RGBA. The same bytes in the other order.
    Rgba,
}

impl PixelFormat {
    /// What an importer is told this format is.
    pub fn fourcc(self) -> u32 {
        match self {
            PixelFormat::Bgra => DRM_FORMAT_ARGB8888,
            PixelFormat::Rgba => DRM_FORMAT_ABGR8888,
        }
    }

    /// Bytes one pixel occupies. Both variants are four; the method exists so a format added
    /// later cannot be assumed to be.
    pub fn bytes_per_pixel(self) -> u32 {
        match self {
            PixelFormat::Bgra | PixelFormat::Rgba => 4,
        }
    }

    /// The tightest row this format can have at that width, for a caller checking a driver's
    /// reported pitch is at least plausible.
    pub fn linear_pitch(self, width: u32) -> Option<u32> {
        width.checked_mul(self.bytes_per_pixel())
    }
}

/// The layout of planar storage. See [`crate::metal::PlanarFormat`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PlanarFormat {
    /// A full-resolution 8-bit luma plane, then a half-resolution plane of interleaved two-byte
    /// chroma. What NV12 names on the wire.
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

    /// What an importer is told this layout is.
    pub fn fourcc(self) -> u32 {
        match self {
            PlanarFormat::BiPlanar420 => DRM_FORMAT_NV12,
        }
    }

    /// How many planes it has.
    pub fn plane_count(self) -> u32 {
        match self {
            PlanarFormat::BiPlanar420 => 2,
        }
    }
}

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

/// A process-local name for an exported surface.
///
/// **Not an IOSurface id, and not a handle anything outside this process can resolve.** A
/// descriptor is what crosses a process boundary here; this exists because the renderer's own
/// plumbing -- scores, journal entries, the census -- names a surface by a small integer, and
/// that plumbing is shared with the host where the integer is globally meaningful. Monotonic and
/// never reused, so a stale id names nothing rather than something else, which the global
/// IOSurface ids cannot promise.
fn next_id() -> SurfaceId {
    static NEXT: AtomicU32 = AtomicU32::new(1);
    SurfaceId(NEXT.fetch_add(1, Ordering::Relaxed))
}

/// A read-write mapping of an exported descriptor, and the only thing that unmaps one.
struct Mapping {
    ptr: *mut core::ffi::c_void,
    len: usize,
}

// SAFETY: the mapping is a plain shared memory window with no thread affinity, and every access
// through it below goes via a raw pointer read or write of a `Copy` type. What makes it sound to
// share is the same thing that makes the IOSurface path sound: the guest's own fences order the
// GPU's writes against the host's reads, and nothing here caches a derived pointer.
unsafe impl Send for Mapping {}
// SAFETY: as above.
unsafe impl Sync for Mapping {}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: a mapping this type made, of this length, unmapped once -- owning it is what
        // makes it once.
        let rc = unsafe { libc::munmap(self.ptr, self.len) };
        assert!(
            rc == 0,
            "munmap of an exported dma-buf failed: {}",
            std::io::Error::last_os_error()
        );
    }
}

/// Storage the driver owns, named by a descriptor this renderer holds.
///
/// The descriptor is the surface. Dropping this closes it, which is what releases the renderer's
/// claim on the driver's allocation -- and because a [`crate::surface::Held`] share is what
/// travels, the claim outlives the context that exported it exactly as an IOSurface share does.
pub struct Surface {
    fd: OwnedFd,
    id: SurfaceId,
    layout: Layout,
    /// Taken on first CPU access and never retaken. `None` inside the `OnceLock` is a driver
    /// that would not let this fd be mapped, which is a fact about the buffer and does not
    /// change -- so it is asked once and remembered, rather than failing sixty times a second.
    map: OnceLock<Option<Mapping>>,
    /// Whether the refusal above has been reported. Latched, for the same reason
    /// [`crate::venus::driver::Pages`] latches its own: a compositor asks every frame.
    said: AtomicBool,
}

impl Surface {
    /// Take ownership of a descriptor the driver just exported, and what it describes.
    ///
    /// Deliberately not called `new` or `create`: nothing is created here. The caller has already
    /// made the allocation and asked the driver for both halves, and this is where the two stop
    /// being separable.
    pub fn exported(fd: OwnedFd, layout: Layout) -> Surface {
        Surface { fd, id: next_id(), layout, map: OnceLock::new(), said: AtomicBool::new(false) }
    }

    /// The descriptor, borrowed. An importer dups it; nothing takes it, because this owns it for
    /// as long as anyone holds a share.
    pub fn fd(&self) -> std::os::fd::BorrowedFd<'_> {
        use std::os::fd::AsFd;
        self.fd.as_fd()
    }

    /// What the driver said the allocation is.
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn id(&self) -> SurfaceId {
        self.id
    }

    pub fn bytes_per_row(&self) -> u32 {
        self.layout.bytes_per_row()
    }

    pub fn alloc_size(&self) -> u64 {
        self.layout.alloc_size
    }

    /// Where the host reaches the bytes, or `0` for a descriptor that cannot be mapped.
    ///
    /// Zero and not a panic, because "the host cannot address this" is a legitimate state for a
    /// dma-buf and the callers that ask are deciding whether to publish an address -- but a
    /// caller that publishes `0` would be handing the VMM a null mapping, so every one of them
    /// must test it. See [`Surface::mapped`].
    pub fn host_addr(&self) -> usize {
        self.mapped().map_or(0, |m| m.ptr as usize)
    }

    /// The mapping, taken on first use.
    ///
    /// `MAP_SHARED` and read-write: an importer that could only read could not be composited
    /// into, and a private mapping would take a copy-on-write snapshot whose writes the driver
    /// never sees -- which is the quietest possible way to render into nothing.
    fn mapped(&self) -> Option<&Mapping> {
        self.map
            .get_or_init(|| {
                let len = usize::try_from(self.layout.alloc_size).ok().filter(|n| *n != 0)?;
                // SAFETY: a descriptor this type owns, mapped whole at a length the driver
                // reported for it. The pointer is checked against `MAP_FAILED` before it is
                // wrapped, and the wrapper's `Drop` is the only unmap.
                let ptr = unsafe {
                    libc::mmap(
                        core::ptr::null_mut(),
                        len,
                        libc::PROT_READ | libc::PROT_WRITE,
                        libc::MAP_SHARED,
                        self.fd.as_raw_fd(),
                        0,
                    )
                };
                if ptr == libc::MAP_FAILED {
                    return None;
                }
                Some(Mapping { ptr, len })
            })
            .as_ref()
    }

    /// Say once that these bytes cannot be reached, and what was being asked of them.
    ///
    /// The one diagnostic this module prints. It exists because the alternative -- a read that
    /// returns zero bytes -- is silent at the call site and looks like an empty surface, and
    /// because [`crate::surface`]'s callers were written against storage that is always
    /// addressable and have no vocabulary for storage that is not.
    fn refuse(&self, what: &str) {
        if !self.said.swap(true, Ordering::Relaxed) {
            eprintln!(
                "[virglrs] dma-buf surface {}: {what} needs a CPU mapping and this descriptor \
                 cannot be mapped ({}); nothing was read or written",
                self.id.0,
                std::io::Error::last_os_error()
            );
        }
    }

    /// Copy the whole allocation out. Returns how many bytes were copied, which is `0` for a
    /// descriptor the host cannot map -- never a zero-filled `dst`.
    pub fn read_into(&self, dst: &mut [u8]) -> usize {
        let Some(map) = self.mapped() else {
            self.refuse("reading a surface");
            return 0;
        };
        let n = dst.len().min(map.len);
        // SAFETY: `map` covers `map.len` bytes and `n` is within both it and `dst`; the two
        // cannot overlap, because `dst` is a Rust slice and this mapping is not aliased by one.
        unsafe { core::ptr::copy_nonoverlapping(map.ptr.cast::<u8>(), dst.as_mut_ptr(), n) };
        n
    }

    /// Copy into the whole allocation. Returns how many bytes were written.
    pub fn write_from(&self, src: &[u8]) -> usize {
        let Some(map) = self.mapped() else {
            self.refuse("writing a surface");
            return 0;
        };
        let n = src.len().min(map.len);
        // SAFETY: as `read_into`, in the other direction.
        unsafe { core::ptr::copy_nonoverlapping(src.as_ptr(), map.ptr.cast::<u8>(), n) };
        n
    }

    /// Copy `height` rows out, re-pitching from the surface's stride to `stride`.
    ///
    /// Returns the number of rows copied, which is `0` rather than a partial picture when the
    /// mapping is missing. A short `dst` stops the copy at the row that would not fit, so the
    /// count is what a caller can trust it to have.
    pub fn read_rows(&self, dst: &mut [u8], stride: usize, height: u32) -> u32 {
        let Some(map) = self.mapped() else {
            self.refuse("reading a surface's rows");
            return 0;
        };
        let pitch = self.layout.bytes_per_row() as usize;
        let row = pitch.min(stride);
        let mut done = 0;
        for y in 0..height as usize {
            let (src_at, dst_at) = (y * pitch, y * stride);
            if src_at + row > map.len || dst_at + row > dst.len() {
                break;
            }
            // SAFETY: both ranges were just bounds-checked against their own extents, and the
            // mapping is not aliased by the destination slice.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    map.ptr.cast::<u8>().add(src_at),
                    dst.as_mut_ptr().add(dst_at),
                    row,
                );
            }
            done += 1;
        }
        done
    }

    pub fn plane_count(&self) -> u32 {
        self.layout.plane_count
    }

    /// One plane's shape and where it starts.
    ///
    /// The shape is derived from what the driver reported, not from what this side would have
    /// laid out -- see [`PlaneLayout`]. `bytes_per_element` is the only field that is a property
    /// of the format rather than of the allocation.
    pub fn plane(&self, plane: u32) -> Option<(PlaneShape, u32)> {
        if plane >= self.layout.plane_count {
            return None;
        }
        let p = self.layout.planes[plane as usize];
        // A subsampled chroma plane is half the luma plane in both directions. Two planes means
        // NV12 here, which is the only planar layout this renderer exports.
        let sub = u32::from(plane > 0 && self.layout.plane_count == 2);
        let shape = PlaneShape {
            width: self.layout.width >> sub,
            height: self.layout.height >> sub,
            bytes_per_element: if plane == 0 { 1 } else { 2 },
            bytes_per_row: p.pitch,
            offset: u32::try_from(p.offset).unwrap_or(u32::MAX),
        };
        Some((shape, p.pitch))
    }

    /// Write `rows` rows into one plane, re-pitching from `src_pitch`.
    pub fn write_plane(
        &self,
        plane: u32,
        src: &[u8],
        src_pitch: usize,
        rows: u32,
        row_bytes: usize,
    ) -> bool {
        let Some(map) = self.mapped() else {
            self.refuse("writing a surface plane");
            return false;
        };
        let Some((_, pitch)) = self.plane(plane) else { return false };
        let base = self.layout.planes[plane as usize].offset as usize;
        let (pitch, row) = (pitch as usize, row_bytes.min(pitch as usize));
        for y in 0..rows as usize {
            let (dst_at, src_at) = (base + y * pitch, y * src_pitch);
            if dst_at + row > map.len || src_at + row > src.len() {
                return false;
            }
            // SAFETY: both ranges bounds-checked immediately above against the mapping and the
            // source slice; the two cannot overlap.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    src.as_ptr().add(src_at),
                    map.ptr.cast::<u8>().add(dst_at),
                    row,
                );
            }
        }
        true
    }

    /// Copy one plane straight out of another surface of the same shape.
    pub fn copy_plane_from(&self, src: &Surface, plane: u32) -> bool {
        let (Some((shape, _)), Some((src_shape, src_pitch))) =
            (self.plane(plane), src.plane(plane))
        else {
            return false;
        };
        if shape.width != src_shape.width || shape.height != src_shape.height {
            return false;
        }
        let Some(map) = src.mapped() else {
            src.refuse("copying a surface plane");
            return false;
        };
        let base = src.layout.planes[plane as usize].offset as usize;
        let row = (shape.width * shape.bytes_per_element) as usize;
        let mut buf = vec![0u8; row * shape.height as usize];
        for y in 0..shape.height as usize {
            let at = base + y * src_pitch as usize;
            if at + row > map.len {
                return false;
            }
            // SAFETY: bounds-checked against the source mapping immediately above; `buf` is a
            // fresh allocation this call owns.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    map.ptr.cast::<u8>().add(at),
                    buf.as_mut_ptr().add(y * row),
                    row,
                );
            }
        }
        self.write_plane(plane, &buf, row, shape.height, row)
    }
}

impl crate::surface::Held for Surface {
    fn surface(&self) -> &Surface {
        self
    }
}

impl std::fmt::Debug for Surface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DmaBuf")
            .field("id", &self.id.0)
            .field("fd", &self.fd.as_raw_fd())
            .field("width", &self.layout.width)
            .field("height", &self.layout.height)
            .field("fourcc", &format_args!("{:#010x}", self.layout.fourcc))
            .field("modifier", &format_args!("{:#018x}", self.layout.modifier))
            .field("planes", &self.layout.plane_count)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A descriptor over ordinary shared memory, so the layout and CPU paths can be exercised
    /// without a Vulkan driver. It is a real dma-buf in every respect this module can observe --
    /// an fd that maps -- which is the point: the parts under test are the arithmetic and the
    /// refusals, and neither is about where the fd came from.
    fn memfd(len: usize) -> OwnedFd {
        use std::os::fd::FromRawFd;
        // SAFETY: a NUL-terminated literal and no flags; the descriptor returned is fresh and
        // owned by nothing else, so wrapping it is its only owner.
        let fd = unsafe { libc::memfd_create(c"virglrs-test".as_ptr(), 0) };
        assert!(fd >= 0, "memfd_create: {}", std::io::Error::last_os_error());
        // SAFETY: a descriptor this call just made.
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };
        // SAFETY: sizing a fresh memfd nothing has mapped yet.
        let rc = unsafe { libc::ftruncate(fd.as_raw_fd(), len as libc::off_t) };
        assert!(rc == 0, "ftruncate: {}", std::io::Error::last_os_error());
        fd
    }

    fn plain(width: u32, height: u32, pitch: u32) -> Surface {
        let size = u64::from(pitch) * u64::from(height);
        Surface::exported(
            memfd(size as usize),
            Layout {
                width,
                height,
                fourcc: PixelFormat::Bgra.fourcc(),
                modifier: DRM_FORMAT_MOD_LINEAR,
                planes: [PlaneLayout { offset: 0, pitch }; MAX_PLANES],
                plane_count: 1,
                alloc_size: size,
            },
        )
    }

    /// The fourcc names the byte order in memory, and the two the wire has are each other
    /// reversed -- so getting them the wrong way round swaps red and blue on every frame and
    /// nothing else changes. Pinned against the literal codes rather than against each other.
    #[test]
    fn a_fourcc_names_the_byte_order_in_memory() {
        assert_eq!(PixelFormat::Bgra.fourcc(), 0x3432_5241, "DRM_FORMAT_ARGB8888");
        assert_eq!(PixelFormat::Rgba.fourcc(), 0x3432_4241, "DRM_FORMAT_ABGR8888");
        assert_eq!(PlanarFormat::BiPlanar420.fourcc(), 0x3231_564e, "DRM_FORMAT_NV12");
        assert_ne!(PixelFormat::Bgra.fourcc(), PixelFormat::Rgba.fourcc());
    }

    /// Rows come back re-pitched, and a destination that cannot hold them stops the copy rather
    /// than truncating the last row into whatever follows it.
    #[test]
    fn reading_rows_repitches_and_stops_where_the_destination_ends() {
        let s = plain(4, 3, 32);
        assert_eq!(s.write_from(&(0u8..96).collect::<Vec<_>>()), 96);

        let mut dst = vec![0u8; 16 * 3];
        assert_eq!(s.read_rows(&mut dst, 16, 3), 3);
        assert_eq!(&dst[0..16], &(0u8..16).collect::<Vec<_>>()[..]);
        assert_eq!(&dst[16..32], &(32u8..48).collect::<Vec<_>>()[..]);

        // Room for two rows, asked for three.
        let mut short = vec![0u8; 16 * 2];
        assert_eq!(s.read_rows(&mut short, 16, 3), 2);
    }

    /// A read of a surface bigger than the buffer offered copies what fits and says how much,
    /// so a caller can never mistake a truncated read for a whole one.
    #[test]
    fn a_read_says_how_much_it_copied() {
        let s = plain(4, 4, 16);
        assert_eq!(s.write_from(&[0xab; 64]), 64);
        let mut small = [0u8; 10];
        assert_eq!(s.read_into(&mut small), 10);
        assert_eq!(small, [0xab; 10]);
        let mut big = [0u8; 128];
        assert_eq!(s.read_into(&mut big), 64, "the surface's own extent, not the buffer's");
        assert_eq!(big[64], 0, "nothing beyond the surface was written");
    }

    /// The planar shape is the driver's numbers, not this side's arithmetic: a chroma plane at an
    /// offset and pitch that no tight layout would produce still reads back exactly as reported.
    #[test]
    fn a_plane_reports_what_the_driver_laid_out_not_what_would_be_tight() {
        let size = 64 * 8 + 64 * 4;
        let s = Surface::exported(
            memfd(size),
            Layout {
                width: 32,
                height: 8,
                fourcc: PlanarFormat::BiPlanar420.fourcc(),
                modifier: DRM_FORMAT_MOD_LINEAR,
                planes: [
                    PlaneLayout { offset: 0, pitch: 64 },
                    // Neither tight (32) nor at the tight offset (256).
                    PlaneLayout { offset: 512, pitch: 64 },
                    PlaneLayout { offset: 0, pitch: 0 },
                    PlaneLayout { offset: 0, pitch: 0 },
                ],
                plane_count: 2,
                alloc_size: size as u64,
            },
        );
        assert_eq!(s.plane_count(), 2);
        let (luma, luma_pitch) = s.plane(0).expect("plane 0");
        assert_eq!((luma.width, luma.height, luma_pitch), (32, 8, 64));
        assert_eq!(luma.bytes_per_element, 1);
        let (chroma, chroma_pitch) = s.plane(1).expect("plane 1");
        assert_eq!((chroma.width, chroma.height, chroma_pitch), (16, 4, 64));
        assert_eq!((chroma.bytes_per_element, chroma.offset), (2, 512));
        assert!(s.plane(2).is_none(), "a plane the layout does not have");
    }

    /// Writing a plane lands at the driver's offset, and leaves the plane before it alone.
    #[test]
    fn writing_a_plane_lands_at_the_offset_the_driver_gave() {
        let size = 64 * 8 + 64 * 4;
        let s = Surface::exported(
            memfd(size),
            Layout {
                width: 32,
                height: 8,
                fourcc: PlanarFormat::BiPlanar420.fourcc(),
                modifier: DRM_FORMAT_MOD_LINEAR,
                planes: [
                    PlaneLayout { offset: 0, pitch: 64 },
                    PlaneLayout { offset: 512, pitch: 64 },
                    PlaneLayout { offset: 0, pitch: 0 },
                    PlaneLayout { offset: 0, pitch: 0 },
                ],
                plane_count: 2,
                alloc_size: size as u64,
            },
        );
        assert!(s.write_plane(1, &[0x5a; 32 * 4], 32, 4, 32));
        let mut all = vec![0u8; size];
        assert_eq!(s.read_into(&mut all), size);
        assert!(all[..512].iter().all(|b| *b == 0), "the luma plane was not touched");
        assert_eq!(&all[512..544], &[0x5a; 32][..]);
        assert!(all[544..576].iter().all(|b| *b == 0), "only row_bytes of the row was written");
    }

    /// A descriptor that cannot be mapped reports nothing read, and says so once rather than
    /// per frame. Zero bytes and not a zero-filled buffer: the two are indistinguishable at the
    /// call site and only one of them is honest.
    #[test]
    fn an_unmappable_descriptor_reads_nothing_and_says_so_once() {
        // A zero-length allocation cannot be mapped, which is the same refusal a driver that
        // declines to map its buffer produces, reached without needing such a driver.
        let s = Surface::exported(
            memfd(0),
            Layout {
                width: 0,
                height: 0,
                fourcc: PixelFormat::Bgra.fourcc(),
                modifier: DRM_FORMAT_MOD_INVALID,
                planes: [PlaneLayout { offset: 0, pitch: 0 }; MAX_PLANES],
                plane_count: 1,
                alloc_size: 0,
            },
        );
        let mut dst = [0xffu8; 8];
        assert_eq!(s.read_into(&mut dst), 0);
        assert_eq!(dst, [0xff; 8], "the buffer was left as it was, not zero-filled");
        assert_eq!(s.host_addr(), 0);
        assert!(s.said.load(Ordering::Relaxed), "the refusal was reported");
        // Latched: a second read does not report again.
        s.said.store(false, Ordering::Relaxed);
        assert_eq!(s.read_into(&mut dst), 0);
        assert!(s.said.load(Ordering::Relaxed));
    }

    /// Every id is its own, and none is reused. A recycled id would name a live surface with a
    /// stale reference, which is the lifetime bug the whole share scheme exists to prevent.
    #[test]
    fn ids_are_never_reused() {
        let a = plain(2, 2, 8);
        let b = plain(2, 2, 8);
        assert_ne!(a.id(), b.id());
        let a_id = a.id();
        drop(a);
        let c = plain(2, 2, 8);
        assert_ne!(c.id(), a_id, "a dead surface's id was handed out again");
    }
}
