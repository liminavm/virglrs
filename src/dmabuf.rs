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
pub use crate::surface::{
    BadLayout, DRM_FORMAT_MOD_INVALID, DRM_FORMAT_MOD_LINEAR, Layout, MAX_PLANES, PlaneLayout,
};

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

    /// A second reference to the same storage, and how to read it.
    ///
    /// Duplicated and not handed over: the surface goes on naming its storage after an export,
    /// and a caller closing what it was given must not close what this still holds. `dup` is what
    /// makes the two independent -- both name one dma-buf, and the kernel frees it when the last
    /// one goes.
    ///
    /// The layout travels with it because a descriptor alone is bytes nobody can interpret, and a
    /// caller that had to fetch the two separately could hold a layout belonging to a different
    /// export of the same resource.
    pub fn export(&self) -> Option<(std::os::fd::OwnedFd, Layout)> {
        Some((self.fd.try_clone().ok()?, self.layout))
    }

    /// What the driver said the allocation is.
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn id(&self) -> SurfaceId {
        self.id
    }

    /// Never this host: `dmabuf.rs` has no constructor that makes storage, so every surface
    /// here is one the guest's driver laid out. See [`crate::surface::Layouter`].
    pub fn layouter(&self) -> crate::surface::Layouter {
        crate::surface::Layouter::TheExportingDriver
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

    /// Never a host allocation, whether or not it is mapped.
    ///
    /// The distinction [`Surface::host_addr`] does not make. That address is a mapping of the
    /// *exporting driver's* buffer -- a GEM mmap -- and it is a fine thing to publish to the VMM,
    /// which is what a mappable blob does with it. It is not a fine thing to hand a second driver
    /// as `VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT`: that handle type means memory
    /// the *host* allocated, and importing another driver's pages under it aliases storage the
    /// importer knows nothing about. The route for these bytes is the descriptor, and until a
    /// dma-buf handle type is passed at `vkAllocateMemory` there is no route at all.
    ///
    /// So `None` unconditionally, and not "`None` unless mapped": whether a mapping has been
    /// taken yet is a fact about this process, and the question is about whose pages they are.
    pub fn as_host_allocation(&self) -> Option<usize> {
        None
    }

    /// Whether the bytes behind this descriptor are pixels the CPU can read in row order.
    ///
    /// **Only a linear buffer is.** A tiled one is a perfectly good dma-buf -- an importer hands
    /// it to a GPU, which knows the modifier and detiles as it samples -- and its bytes read in
    /// row order are not the picture. Measured on this host: an Intel scanout exports with
    /// modifier `0x0100000000000001` (`I915_FORMAT_MOD_X_TILED`), so this is the common case and
    /// not an exotic one.
    ///
    /// The distinction matters because the caller of a CPU read has a slow path and needs to be
    /// told to take it. A read that returned tiled bytes would be a picture-shaped answer that is
    /// not the picture, which is worse than no answer -- it would hash, it would differ from the
    /// reference leg, and nothing in the difference would say why.
    pub fn readable(&self) -> bool {
        self.layout.modifier == DRM_FORMAT_MOD_LINEAR
    }

    /// The mapping, taken on first use.
    ///
    /// `MAP_SHARED` and read-write: an importer that could only read could not be composited
    /// into, and a private mapping would take a copy-on-write snapshot whose writes the driver
    /// never sees -- which is the quietest possible way to render into nothing.
    ///
    /// `None` for a tiled buffer even though the descriptor would map perfectly well: see
    /// [`Surface::readable`]. Refusing here rather than at each call site is what keeps every CPU
    /// path honest at once.
    fn mapped(&self) -> Option<&Mapping> {
        if !self.readable() {
            return None;
        }
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
        if self.said.swap(true, Ordering::Relaxed) {
            return;
        }
        // Two different refusals, and they send a reader to different places. A tiled buffer is
        // working exactly as intended and simply cannot be read this way; a descriptor that will
        // not map is a driver saying no. Reporting both as "cannot be mapped" would have the
        // reader hunting for a mapping failure that never happened.
        if !self.readable() {
            eprintln!(
                "[virglrs] dma-buf surface {}: {what} cannot read a tiled buffer (modifier \
                 {:#018x}); its bytes in row order are not the picture, so nothing was read and \
                 the caller must take its slow path",
                self.id.0, self.layout.modifier,
            );
        } else {
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
        // How a plane is *sampled* is a property of the FourCC, never of how many planes the
        // allocation has. A second plane means half-resolution two-byte chroma in NV12 and
        // something else entirely under a compressed modifier, which carries an auxiliary plane
        // for a single-plane format -- this host's ICL exports `I915_FORMAT_MOD_Y_TILED_CCS` as
        // two planes of ARGB8888. Counting planes to decide the layout would hand that buffer's
        // compression plane out as chroma at half the width.
        //
        // So a plane past the first exists to be *named* in an import and not to be read as a
        // picture, and the FourCC's own rule is what says how many planes of pixels there are.
        // `plane_rule` is that rule and this asks it rather than carrying a second copy: the
        // copy that was here said four bytes an element for everything that is not NV12, which
        // is wrong for R8, RG16 and the four 16-bit-float codes the rule knows.
        let (pixel_planes, element) = plane_rule(self.layout.fourcc)?;
        if plane >= pixel_planes {
            return None;
        }
        // The only multi-plane rule is NV12's, whose chroma is half-resolution in both axes.
        let sub = u32::from(plane > 0);
        let shape = PlaneShape {
            width: self.layout.width >> sub,
            height: self.layout.height >> sub,
            bytes_per_element: element[plane as usize],
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

/// How a FourCC's pixels sit in memory: how many planes, and how many bytes one element of each
/// plane is. What a layout has to be checked against before anything reads the buffer under it.
///
/// Matched on the four characters rather than on computed codes, because that is how
/// `drm_fourcc.h` writes them and it is what makes a wrong entry visible on the page.
///
/// The set is every code [`crate::vrend::formats::scanout_fourcc`] can produce, plus `NV12` for
/// the planar decode path -- and the two are pinned together by a test, so a code added to the
/// generated table without a rule here fails this tree's own suite rather than one guest's
/// window. `None` is a FourCC this renderer will not bound, and is refused by name.
fn plane_rule(fourcc: u32) -> Option<(u32, [u32; MAX_PLANES])> {
    let flat = |bytes: u32| Some((1, [bytes, 0, 0, 0]));
    match &fourcc.to_le_bytes() {
        b"R8  " => flat(1),
        b"RG16" => flat(2),
        b"AR24" | b"XR24" | b"AB24" | b"XB24" => flat(4),
        b"AR30" | b"XR30" | b"AB30" | b"XB30" => flat(4),
        b"AB4H" | b"XB4H" | b"AB48" | b"XB48" => flat(8),
        // The one planar layout: full-resolution 8-bit luma, then half-resolution two-byte
        // interleaved chroma.
        b"NV12" => Some((2, [1, 2, 0, 0])),
        _ => None,
    }
}

/// Storage the driver owns, named by a descriptor and nothing else.
///
/// A [`Surface`] is a descriptor *plus* the layout needed to read it. Some storage never gets the
/// second half. Mesa's Wayland WSI shares a compositor a linear buffer it blits its rendered
/// image into, and dedicates that allocation to a `VkBuffer` -- which has no format, no tiling
/// and no `vkGetImageSubresourceLayout`, so there is no question to ask the driver. Measured on
/// this host: `vkcube`'s swapchain is three 2048000-byte allocations, each dedicated to a buffer
/// and declared for export as DMA_BUF, beside an OPTIMAL image that is never exported. What the
/// buffer holds is decided by whoever blits into it, and the only party that knows is the guest.
///
/// So this is the descriptor on its own, and it stays one for its whole life. An importer that
/// learns a layout does not fill it in: [`Descriptor::describe`] mints a separate [`Surface`]
/// over a second reference to the same buffer. Two importers may read one descriptor under
/// different layouts and neither can see the other's, which is what stops a guest that sends two
/// disagreeing descriptions of one resource from making either of them read the other's numbers.
pub struct Descriptor {
    fd: OwnedFd,
    id: SurfaceId,
    /// The kernel's own figure for the buffer, and what every bound below is measured against.
    /// `dma_buf_llseek` exists to answer exactly this. Deliberately not the allocation size the
    /// guest asked for: that is a number the guest chooses, and this is the one it cannot.
    size: u64,
    /// Whether it has already been said that this storage has no layout of its own. Latched, for
    /// the same reason [`Surface`]'s own refusal is: a compositor asks every frame.
    said: AtomicBool,
}

impl Descriptor {
    /// Take ownership of a descriptor the driver just exported, and ask the kernel how big it is.
    ///
    /// `None` for a buffer of no size, which is a descriptor nothing can be read through and
    /// which would make every bound below vacuous.
    pub fn exported(fd: OwnedFd) -> Option<Descriptor> {
        // SAFETY: a descriptor this call owns; `lseek` on a dma-buf reads its size and moves a
        // file offset nothing here uses.
        let end = unsafe { libc::lseek(fd.as_raw_fd(), 0, libc::SEEK_END) };
        let size = u64::try_from(end).ok().filter(|n| *n != 0)?;
        Some(Descriptor { fd, id: next_id(), size, said: AtomicBool::new(false) })
    }

    /// Whether this is the first time the descriptor has been asked for a layout it does not
    /// have. `true` once, so a caller says why exactly once rather than per frame.
    pub fn first_refusal(&self) -> bool {
        !self.said.swap(true, Ordering::Relaxed)
    }

    /// The kernel's size for the buffer.
    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn id(&self) -> SurfaceId {
        self.id
    }

    /// The descriptor, borrowed. An importer dups it; nothing takes it.
    pub fn fd(&self) -> std::os::fd::BorrowedFd<'_> {
        use std::os::fd::AsFd;
        self.fd.as_fd()
    }

    /// A second reference to the same buffer, for a caller that publishes descriptors.
    pub fn export(&self) -> Option<OwnedFd> {
        self.fd.try_clone().ok()
    }

    /// Read this buffer under a layout someone else chose.
    ///
    /// The layout does not come from the driver here -- there is no driver answer to be had -- so
    /// it comes from the guest, and this is the trust boundary it crosses. Every field is checked
    /// against the kernel's size for the buffer, in checked arithmetic, before a [`Surface`]
    /// exists to be mapped or imaged: a plane that reaches past the end would otherwise be an
    /// `mmap` the host reads out of, or a GPU fetch outside the BO, on the guest's say-so.
    ///
    /// The modifier is an allowlist rather than a range check because a modifier is not a number
    /// to bound: a compressed one carries an auxiliary plane whose size follows a rule of its own,
    /// and "the planes the guest declared fit" is not the same claim for it. `LINEAR` and
    /// `INVALID` are what a guest compositing over virgl can actually produce -- virgl advertises
    /// no modifiers -- so anything else is refused by name until one is measured.
    ///
    /// `alloc_size` is taken from the kernel and never from the caller's `layout`: the buffer's
    /// extent is one fact, and the descriptor is the only party that holds it.
    pub fn describe(&self, layout: Layout) -> Result<Surface, BadLayout> {
        if layout.width == 0 || layout.height == 0 {
            return Err(BadLayout::Extent { width: layout.width, height: layout.height });
        }
        if layout.modifier != DRM_FORMAT_MOD_LINEAR && layout.modifier != DRM_FORMAT_MOD_INVALID {
            return Err(BadLayout::Modifier(layout.modifier));
        }
        let (wants, bytes) = plane_rule(layout.fourcc).ok_or(BadLayout::Fourcc(layout.fourcc))?;
        if layout.plane_count != wants {
            return Err(BadLayout::PlaneCount { said: layout.plane_count, wants });
        }
        for at in 0..wants {
            let p = layout.planes[at as usize];
            // A subsampled plane rounds *up*: an odd-sized NV12 image still has a chroma row for
            // its last luma row, and rounding down would bound the buffer one row short. Only a
            // planar layout has a plane past the first, so the rule and the subsampling are the
            // same fact.
            let sub = u32::from(at > 0);
            let (w, h) = (layout.width.div_ceil(1 << sub), layout.height.div_ceil(1 << sub));
            let tight = w
                .checked_mul(bytes[at as usize])
                .ok_or(BadLayout::Extent { width: layout.width, height: layout.height })?;
            if p.pitch < tight {
                return Err(BadLayout::Pitch { plane: at, pitch: p.pitch, tight });
            }
            let end = u64::from(p.pitch)
                .checked_mul(u64::from(h))
                .and_then(|rows| rows.checked_add(p.offset))
                .ok_or(BadLayout::Overrun { plane: at, end: u64::MAX, size: self.size })?;
            if end > self.size {
                return Err(BadLayout::Overrun { plane: at, end, size: self.size });
            }
        }
        let fd = self.fd.try_clone().map_err(|_| BadLayout::NoDescriptor)?;
        Ok(Surface::exported(fd, Layout { alloc_size: self.size, ..layout }))
    }
}

impl std::fmt::Debug for Descriptor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Descriptor")
            .field("id", &self.id.0)
            .field("fd", &self.fd.as_raw_fd())
            .field("size", &self.size)
            .finish()
    }
}

/// One importer's reading of a [`Descriptor`], as the thing that keeps it alive.
///
/// The [`crate::surface::Held`] share that travels for storage whose layout was never the
/// driver's to give. It holds two things that must not come apart: a share of the descriptor its
/// owner exported -- which carries the memory charge, so releasing it is what credits the budget
/// -- and this importer's own [`Surface`] over a second reference to the same buffer.
///
/// The reading is per-importer by construction. Nothing here is written into the shared
/// descriptor, so a second importer describing the same buffer differently gets its own `Surface`
/// and cannot disturb this one.
pub struct Described {
    /// Held, never read: what it carries is the right to keep the buffer and its charge alive for
    /// as long as this reading of them exists.
    _share: std::sync::Arc<dyn Send + Sync>,
    surface: Surface,
}

impl Described {
    pub fn new(share: std::sync::Arc<dyn Send + Sync>, surface: Surface) -> Described {
        Described { _share: share, surface }
    }
}

impl crate::surface::Held for Described {
    fn surface(&self) -> &Surface {
        &self.surface
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

    /// A tiled buffer is a good descriptor and a bad picture, and every CPU path says so.
    ///
    /// The case this exists for was measured, not imagined: an Intel scanout on this host exports
    /// with modifier `0x0100000000000001` (`I915_FORMAT_MOD_X_TILED`). The descriptor is exactly
    /// what a compositor wants -- it hands it to a GPU that knows the modifier -- and the same bytes
    /// read in row order are not the frame. Before this, they were read, hashed, and compared
    /// against the reference leg, where the difference would have looked like a renderer bug.
    ///
    /// Refused, and not "read as zero": the two are told apart by the caller, which has a slow
    /// path to fall back to and needs to be sent to it.
    #[test]
    fn a_tiled_buffer_refuses_every_cpu_path() {
        const X_TILED: u64 = 0x0100_0000_0000_0001;
        let size = 64 * 16;
        let tiled = Surface::exported(
            memfd(size),
            Layout {
                width: 16,
                height: 16,
                fourcc: PixelFormat::Bgra.fourcc(),
                modifier: X_TILED,
                planes: [PlaneLayout { offset: 0, pitch: 64 }; MAX_PLANES],
                plane_count: 1,
                alloc_size: size as u64,
            },
        );
        assert!(!tiled.readable(), "a modifier that is not LINEAR is not rows");

        let mut dst = [0xabu8; 64];
        assert_eq!(tiled.read_into(&mut dst), 0);
        assert_eq!(dst, [0xab; 64], "and the buffer is left as it was, not filled with tiles");
        assert_eq!(tiled.read_rows(&mut dst, 16, 4), 0);
        assert_eq!(tiled.write_from(&[0; 8]), 0, "nor written through");
        assert_eq!(tiled.host_addr(), 0, "and there is no address to publish");

        // The descriptor itself is untouched by any of that: exporting it is the whole point.
        let (fd, layout) = tiled.export().expect("a tiled buffer still exports");
        assert_eq!(layout.modifier, X_TILED, "and the importer is told how to read it");
        drop(fd);

        // The same surface laid out linearly reads back, so the refusal is about the modifier and
        // not about anything else this test happens to have set up.
        let linear = plain(16, 16, 64);
        assert!(linear.readable());
        assert_eq!(linear.write_from(&[0x5a; 64]), 64);
        assert_eq!(linear.read_into(&mut dst), 64);
        assert_eq!(dst, [0x5a; 64]);
    }

    fn bgra(width: u32, height: u32, pitch: u32) -> Layout {
        Layout {
            width,
            height,
            fourcc: PixelFormat::Bgra.fourcc(),
            modifier: DRM_FORMAT_MOD_LINEAR,
            planes: [PlaneLayout { offset: 0, pitch }; MAX_PLANES],
            plane_count: 1,
            alloc_size: 0,
        }
    }

    /// The buffer's extent is the kernel's, and a description of it can only ever be a reading.
    ///
    /// A guest that sends a layout claiming a bigger allocation than the buffer has would, if the
    /// claim were taken, get an `mmap` of that length and a GPU fetch bounded by it. The size a
    /// `Surface` ends up with comes from `lseek` on the descriptor, so the claim cannot travel.
    #[test]
    fn a_description_takes_its_extent_from_the_buffer_not_from_the_caller() {
        let d = Descriptor::exported(memfd(4096)).expect("a sized buffer");
        assert_eq!(d.size(), 4096);
        let lying = Layout { alloc_size: 1 << 30, ..bgra(16, 16, 64) };
        let s = d.describe(lying).expect("a layout that fits");
        assert_eq!(s.alloc_size(), 4096, "the kernel's figure, not the caller's");
    }

    /// A buffer with no size is a descriptor nothing can be read through, and every bound over it
    /// would be vacuous -- so there is no `Descriptor` for one to be checked against.
    #[test]
    fn an_empty_buffer_is_not_a_descriptor() {
        assert!(Descriptor::exported(memfd(0)).is_none());
    }

    /// The checks a guest-supplied layout has to pass, one refusal each.
    ///
    /// These are the trust boundary: past `describe` the layout is used to `mmap` and to build an
    /// EGL image, so a plane reaching past the buffer is a host read outside the allocation on the
    /// guest's say-so. Each case is the smallest change to a layout that does fit.
    #[test]
    fn a_layout_that_does_not_fit_the_buffer_is_refused() {
        let d = Descriptor::exported(memfd(4096)).expect("a sized buffer");
        d.describe(bgra(16, 16, 64)).expect("16 rows of 64 bytes is exactly 1024");

        assert_eq!(
            d.describe(bgra(0, 16, 64)).err(),
            Some(BadLayout::Extent { width: 0, height: 16 })
        );
        assert_eq!(
            d.describe(bgra(16, 16, 63)).err(),
            Some(BadLayout::Pitch { plane: 0, pitch: 63, tight: 64 }),
            "a row narrower than its own pixels"
        );
        assert_eq!(
            d.describe(bgra(16, 65, 64)).err(),
            Some(BadLayout::Overrun { plane: 0, end: 4160, size: 4096 }),
            "one row more than the buffer holds"
        );
        // The same overrun reached by the offset rather than by the height.
        let mut shifted = bgra(16, 16, 64);
        shifted.planes[0].offset = 3200;
        assert_eq!(
            d.describe(shifted).err(),
            Some(BadLayout::Overrun { plane: 0, end: 4224, size: 4096 })
        );
        // Arithmetic that would wrap is an overrun, not a pass.
        let mut huge = bgra(16, u32::MAX, 64);
        huge.planes[0].offset = u64::MAX - 16;
        assert!(matches!(d.describe(huge), Err(BadLayout::Overrun { .. })));

        assert_eq!(
            d.describe(Layout { plane_count: 2, ..bgra(16, 16, 64) }).err(),
            Some(BadLayout::PlaneCount { said: 2, wants: 1 }),
            "a plane count the fourcc does not have"
        );
        assert_eq!(
            d.describe(Layout { fourcc: 0xdead_beef, ..bgra(16, 16, 64) }).err(),
            Some(BadLayout::Fourcc(0xdead_beef))
        );
    }

    /// Only a layout this side can bound is accepted.
    ///
    /// A compressed modifier carries an auxiliary plane whose size follows a rule of its own, so
    /// "the planes the guest declared fit" is not the same claim for it -- and a guest naming one
    /// would have the buffer read as something it is not. Refused by name until one is measured.
    #[test]
    fn a_modifier_this_side_cannot_bound_is_refused_by_name() {
        const Y_TILED_CCS: u64 = 0x0100_0000_0000_0004;
        let d = Descriptor::exported(memfd(4096)).expect("a sized buffer");
        assert_eq!(
            d.describe(Layout { modifier: Y_TILED_CCS, ..bgra(16, 16, 64) }).err(),
            Some(BadLayout::Modifier(Y_TILED_CCS))
        );
        d.describe(Layout { modifier: DRM_FORMAT_MOD_INVALID, ..bgra(16, 16, 64) })
            .expect("no claim about the layout is a claim this side can carry");
    }

    /// An NV12 chroma plane is bounded at its own rounded-up extent.
    ///
    /// Rounding the half-resolution plane *down* would bound an odd-sized image one row short and
    /// let a layout through that reaches past the buffer by that row.
    #[test]
    fn a_subsampled_plane_is_bounded_at_its_rounded_up_extent() {
        let planar = |height: u32, offset: u64, size: usize| {
            let d = Descriptor::exported(memfd(size)).expect("a sized buffer");
            let mut l = bgra(16, height, 16);
            l.fourcc = PlanarFormat::BiPlanar420.fourcc();
            l.plane_count = 2;
            l.planes[1] = PlaneLayout { offset, pitch: 16 };
            d.describe(l).map(|_| ())
        };
        // 5 luma rows of 16, then 3 chroma rows of 16 at offset 80: 128 bytes in all.
        assert!(planar(5, 80, 128).is_ok(), "ceil(5/2) is 3 chroma rows");
        assert!(
            matches!(planar(5, 80, 127), Err(BadLayout::Overrun { plane: 1, .. })),
            "and the third chroma row has to be there"
        );
    }

    /// Two importers read one descriptor independently.
    ///
    /// The layout is the importer's, not the buffer's, and a guest may describe one resource
    /// differently in two contexts. Nothing is written back into the descriptor, so the second
    /// description cannot make the first read the second's numbers.
    #[test]
    fn two_descriptions_of_one_buffer_do_not_disturb_each_other() {
        let d = Descriptor::exported(memfd(4096)).expect("a sized buffer");
        let a = d.describe(bgra(16, 16, 64)).expect("a");
        let b = d.describe(bgra(8, 8, 128)).expect("b");
        assert_eq!((a.layout().width, a.bytes_per_row()), (16, 64));
        assert_eq!((b.layout().width, b.bytes_per_row()), (8, 128));
        assert_ne!(a.id(), b.id(), "each reading is its own surface");

        // And each holds its own reference to the one buffer: writing through one is visible
        // through the other, which is what makes them readings rather than copies.
        assert_eq!(a.write_from(&[0x5a; 64]), 64);
        let mut seen = [0u8; 64];
        assert_eq!(b.read_into(&mut seen), 64);
        assert_eq!(seen, [0x5a; 64]);
    }

    /// Every FourCC a scanout can carry has a rule for bounding a layout in it.
    ///
    /// The two tables are one fact in two places: `scanout_fourcc` is generated from the C's
    /// format tables and says which codes a resource may be described as, and `plane_rule` says
    /// how to bound a description in one. A code in the first with no entry in the second is a
    /// guest window refused at `SET_TYPE` with nothing in this tree having decided that it should
    /// be -- which is how `XB4H` was found, in a boot rather than here.
    #[test]
    fn every_scanout_fourcc_can_be_bounded() {
        let mut seen = 0;
        for (format, code) in crate::vrend::formats::scanout_fourccs() {
            seen += 1;
            let name: String = code.get().to_le_bytes().iter().map(|b| *b as char).collect();
            let Some((planes, element)) = plane_rule(code.get()) else {
                panic!(
                    "{name} ({}) is offered as a scanout format and cannot be bounded",
                    format.name()
                );
            };
            // And what a plane says about itself is that same rule, not a second copy of it. The
            // copy that used to live in `Surface::plane` answered four bytes an element for
            // everything but NV12, so R8 read four times too wide and the 16-bit-float codes
            // half.
            let one = Surface::exported(
                memfd(4096),
                Layout {
                    width: 4,
                    height: 4,
                    fourcc: code.get(),
                    modifier: DRM_FORMAT_MOD_LINEAR,
                    planes: [PlaneLayout { offset: 0, pitch: 64 }; MAX_PLANES],
                    plane_count: planes,
                    alloc_size: 4096,
                },
            );
            for at in 0..planes {
                let (shape, _) = one.plane(at).unwrap_or_else(|| {
                    panic!("{name} has {planes} planes of pixels and would not describe {at}")
                });
                assert_eq!(
                    shape.bytes_per_element, element[at as usize],
                    "{name} plane {at} disagrees with its own rule"
                );
            }
        }
        assert_eq!(seen, 14, "the generated table changed size; check the rules above it");
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
