// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! Guest memory the host can reach.
//!
//! This is one of the named unsafe modules (see CLAUDE.md). It exists so that everything else can
//! be safe: it owns the mapping, it owns the bounds check, and it hands out no way to hold a
//! reference into guest memory.
//!
//! # Why there is no `&[u8]` here
//!
//! The obvious API -- deref to a slice -- cannot be written soundly. A Rust reference asserts that
//! the memory behind it is not concurrently modified, and this memory is shared with a guest that
//! modifies it whenever it likes, hostile or merely racy. A `&[u8]` over it is a data race by
//! construction, and the compiler is entitled to optimise on an assumption that is false.
//!
//! So the entire surface is copies and atomics, by value:
//!
//! * [`GuestMap::load_u32`] / [`GuestMap::store_u32`] / [`GuestMap::fetch_or_u32`] for the control
//!   words, which host and guest genuinely share and which are therefore atomic on both sides.
//! * [`GuestMap::copy_out`] / [`GuestMap::copy_in`] for byte ranges, which are copied to or from
//!   host memory and then worked on there.
//!
//! The C reaches the same shape from the other direction: `vkr_ring.h` notes that commands are
//! read out of the ring into a temporary buffer before they are dispatched. Here that is not a
//! note, it is the only thing the type permits.
//!
//! # What is and is not guaranteed
//!
//! Memory safety is guaranteed unconditionally: every access is bounds-checked against the length
//! that was mapped, so no offset a guest can name reaches outside the mapping.
//!
//! Freedom from *torn reads* is not, and cannot be. A guest that rewrites a command while the host
//! copies it gets a copy that is half old and half new. That is a protocol violation, not a
//! soundness problem -- the bytes still land in a host buffer of known size, and the decoder that
//! reads them already treats every byte as hostile.

use std::io;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, OwnedFd};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU32, Ordering};

/// A shared mapping of a guest-visible buffer.
///
/// Dropping it unmaps. Nothing else does, and nothing hands out a pointer that could outlive it.
pub struct GuestMap {
    ptr: NonNull<u8>,
    len: usize,
    /// What these pages cost, for pages this renderer minted at the guest's request; `None` for a
    /// mapping of memory someone else owns, such as a descriptor the VMM imported. Held here so it
    /// is credited when the pages go -- the last share of the mapping, not the resource that
    /// asked for it: a ring keeps its pages past the resource's unref.
    charge: Option<crate::budget::Charge>,
}

// SAFETY: `GuestMap` is a pointer and a length, and every method that touches the memory does so
// atomically or by copying bytes it owns on the host side. There is no interior `&mut` and no
// cached derived pointer, so moving one between threads or sharing it does not create an alias
// that was not already there -- the guest is a concurrent writer regardless of what the host does.
unsafe impl Send for GuestMap {}
// SAFETY: as above; `&GuestMap` grants only atomic access and copies.
unsafe impl Sync for GuestMap {}

impl GuestMap {
    /// Map `len` bytes of an shm descriptor shared with the guest.
    ///
    /// The descriptor is borrowed, not consumed: the resource that owns it goes on owning it, and
    /// the mapping stays valid after the descriptor is closed, which is what POSIX guarantees and
    /// what lets a ring outlive the resource handle it was created from.
    ///
    /// `len` is the VMM's word for how big the resource is, and the descriptor is asked too. A
    /// mapping may run past the end of its file -- `mmap` allows it -- but the pages past the
    /// end have nothing behind them, and the first read or write of one is a `SIGBUS` that takes
    /// the worker down. So a length the descriptor does not hold is refused here, the one place
    /// that has both numbers, as the dma-buf mapping refuses one past what `lseek` reports.
    pub fn shm(fd: BorrowedFd<'_>, len: usize) -> io::Result<GuestMap> {
        if len == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cannot map an empty resource",
            ));
        }
        // SAFETY: `stat` is plain old data, so all-zeroes is a valid value to hand `fstat`.
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        // SAFETY: a live borrowed descriptor, and a local for the answer.
        if unsafe { libc::fstat(fd.as_raw_fd(), &mut st) } < 0 {
            return Err(io::Error::last_os_error());
        }
        let held = u64::try_from(st.st_size).unwrap_or(0);
        if len as u64 > held {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{len} bytes described over a descriptor that holds {held}"),
            ));
        }
        // SAFETY: a null hint lets the kernel choose the address; `len` is non-zero; the
        // descriptor is live for the duration of this call because it is borrowed. The flags are
        // the C's, from `vkr_context.c`: shared, so the guest sees our writes and we see its.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: mmap returned something other than MAP_FAILED, so it is a valid mapping of
        // `len` bytes and is never null.
        Ok(GuestMap { ptr: unsafe { NonNull::new_unchecked(ptr.cast::<u8>()) }, len, charge: None })
    }

    /// Map `len` bytes of fresh anonymous memory, rounded up to whole pages.
    ///
    /// This is the host minting storage for a Vulkan allocation the guest means to share: the
    /// driver is handed these pages by host-pointer import instead of memory of its own, so that
    /// the pages -- and not a mapping the driver lends -- are what a resource can hold a share of.
    /// No descriptor, because nothing needs one: the share travels inside this process as an
    /// `Arc`, and on this platform nothing re-imports a buffer by descriptor across contexts.
    ///
    /// Page-rounded because a host-pointer import requires both the pointer and the size to be
    /// multiples of the driver's import alignment, and a whole number of pages satisfies every
    /// alignment a driver reports.
    pub fn anonymous(len: usize) -> io::Result<GuestMap> {
        if len == 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "cannot map zero bytes"));
        }
        let len = page_round(len).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "size overflows when paged")
        })?;
        // SAFETY: a null hint lets the kernel choose the address; `len` is non-zero and
        // page-rounded; an anonymous private mapping names no descriptor.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: mmap returned something other than MAP_FAILED, so it is a valid mapping of
        // `len` bytes and is never null.
        Ok(GuestMap { ptr: unsafe { NonNull::new_unchecked(ptr.cast::<u8>()) }, len, charge: None })
    }

    /// Where the mapping starts in this process.
    ///
    /// An address, not a pointer: this is for handing to a VMM that will publish it to a guest,
    /// and nothing in this process may dereference it without going through the accessors above.
    pub fn host_addr(&self) -> usize {
        self.ptr.as_ptr() as usize
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// These pages, holding what they cost.
    ///
    /// The charge is for exactly the mapped length -- one number, so a charge taken for the size
    /// the guest asked and a mapping rounded up to whole pages cannot both stand.
    pub fn charged(mut self, charge: crate::budget::Charge) -> GuestMap {
        assert_eq!(charge.size(), self.len as u64, "a mapping is charged for its whole length");
        assert!(self.charge.is_none(), "a mapping is charged once");
        self.charge = Some(charge);
        self
    }

    /// The address of a naturally aligned 32-bit word at `at`, or `None` if it does not lie
    /// wholly inside the mapping or is misaligned.
    ///
    /// Both conditions have to be checked here rather than trusted from the layout, because the
    /// guest also names `extra` offsets at write time, long after any layout was validated.
    fn word(&self, at: usize) -> Option<*const AtomicU32> {
        const WORD: usize = size_of::<u32>();
        if !at.is_multiple_of(WORD) || at.checked_add(WORD)? > self.len {
            return None;
        }
        // SAFETY: `at + 4 <= len`, so the offset stays inside the mapping, and `at` is a multiple
        // of 4 while the mapping itself is page-aligned -- so the result is aligned for AtomicU32.
        Some(unsafe { self.ptr.as_ptr().add(at) }.cast::<AtomicU32>())
    }

    /// Read a control word the guest may be writing concurrently.
    pub fn load_u32(&self, at: usize) -> Option<u32> {
        let p = self.word(at)?;
        // SAFETY: `word` returned a pointer that is in bounds and aligned, and the mapping
        // outlives this borrow. The access is atomic, which is what makes a concurrent guest
        // write defined rather than a race.
        Some(unsafe { &*p }.load(Ordering::Acquire))
    }

    /// Write a control word the guest may be reading concurrently.
    ///
    /// Returns whether the offset named one. A `false` is always the guest's mistake -- the host's
    /// own control words came from a validated layout.
    #[must_use]
    pub fn store_u32(&self, at: usize, value: u32) -> bool {
        let Some(p) = self.word(at) else { return false };
        // SAFETY: as `load_u32`.
        unsafe { &*p }.store(value, Ordering::Release);
        true
    }

    /// Set bits in a control word without disturbing the others.
    ///
    /// Sequentially consistent, matching the C's `vkr_ring_set_status_bits`: the status word is
    /// how the host tells the guest something changed, and it is ordered against everything.
    #[must_use]
    pub fn fetch_or_u32(&self, at: usize, bits: u32) -> bool {
        let Some(p) = self.word(at) else { return false };
        // SAFETY: as `load_u32`.
        unsafe { &*p }.fetch_or(bits, Ordering::SeqCst);
        true
    }

    /// Clear bits in a control word without disturbing the others.
    ///
    /// The counterpart to [`GuestMap::fetch_or_u32`] and sequentially consistent for the same
    /// reason: the C's `vkr_ring_unset_status_bits` clears the IDLE bit as the ring leaves its
    /// park, and the guest is reading that word to decide whether it must ring the doorbell.
    #[must_use]
    pub fn fetch_and_u32(&self, at: usize, keep: u32) -> bool {
        let Some(p) = self.word(at) else { return false };
        // SAFETY: as `load_u32`.
        unsafe { &*p }.fetch_and(keep, Ordering::SeqCst);
        true
    }

    /// Read a control word with sequential consistency rather than acquire.
    ///
    /// This exists for exactly one caller: the ring's park handshake. A parking ring stores the
    /// IDLE bit and then loads the tail to check nothing arrived in between; the guest stores the
    /// tail and then loads the status to decide whether to ring the doorbell. If either load may
    /// be reordered before the other side's store, both can miss, and the ring sleeps on work that
    /// is already there with no one left to wake it. Acquire does not prevent that -- only a pair
    /// of sequentially consistent operations orders the two stores against the two loads.
    ///
    /// The C names the same requirement in `vkr_ring_load_tail_seqcst`, whose comment records that
    /// the 2 ms poll it replaced existed only to survive this race.
    pub fn load_u32_seqcst(&self, at: usize) -> Option<u32> {
        let p = self.word(at)?;
        // SAFETY: as `load_u32`.
        Some(unsafe { &*p }.load(Ordering::SeqCst))
    }

    /// Copy bytes out of guest memory into a host buffer.
    ///
    /// Returns whether the range lay inside the mapping. The copy may tear if the guest writes the
    /// same bytes at the same time; see the module docs for why that is a protocol problem and not
    /// a safety one.
    #[must_use]
    pub fn copy_out(&self, at: usize, dst: &mut [u8]) -> bool {
        if !self.holds(at, dst.len()) {
            return false;
        }
        // SAFETY: `holds` proved `at + dst.len() <= self.len`, so the source range is inside the
        // mapping. `dst` is a live host slice of exactly that length, and the two cannot overlap
        // because one is a mapping this type owns and the other is a borrow the caller brought.
        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr.as_ptr().add(at), dst.as_mut_ptr(), dst.len())
        };
        true
    }

    /// Copy bytes from a host buffer into guest memory.
    #[must_use]
    pub fn copy_in(&self, at: usize, src: &[u8]) -> bool {
        if !self.holds(at, src.len()) {
            return false;
        }
        // SAFETY: as `copy_out`, with the direction reversed. Writing is sound for the same reason
        // reading is: the destination is inside a mapping this type owns for its whole life.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), self.ptr.as_ptr().add(at), src.len())
        };
        true
    }

    /// Whether `len` bytes at `at` lie wholly inside the mapping. A zero-length range at the very
    /// end is inside it, which matters because an empty `extra` region is legal.
    fn holds(&self, at: usize, len: usize) -> bool {
        at.checked_add(len).is_some_and(|end| end <= self.len)
    }
}

/// A classic resource's guest pages: the scatter list the VMM attached, as one byte-addressed
/// span.
///
/// The VMM owns the pages and the list describes them; this type holds neither. What it adds is
/// the walk -- an offset into the concatenation resolved to a page and a position -- and the
/// bounds check, so that no offset a guest names reaches outside the pages the VMM described.
/// The C's walker asserts on an offset past the end (`iov.c`), a host abort reachable from a
/// guest value; here it is a `false`.
///
/// Copies only, for the reason [`GuestMap`] gives: the guest writes these pages whenever it
/// likes, so a reference into them cannot be sound.
///
/// A guest attaches a resource as whatever scatter list its allocator produced -- a 3.6 MB
/// framebuffer has arrived as 225 entries -- and a transfer copies it a row at a time, so the
/// walk is per row *and* per entry. The total is summed once here, so the bounds check is not a
/// pass over the list per row, and a [`Cursor`] lets a row start where the last one ended.
pub struct Iov<'a> {
    entries: &'a [crate::abi::GuestIov],
    len: u64,
}

/// Where a walk ended: the entry it stopped in and how many bytes precede that entry. Rows
/// arrive in ascending order, so the next walk resumes there and the list is crossed once per
/// transfer rather than once per row; a walk that starts before the cursor, or on a list other
/// than the one that advanced it, restarts from the head, so a cursor is never wrong, only
/// sometimes unhelpful.
#[derive(Clone, Copy, Debug)]
pub struct Cursor {
    /// The list the cursor was advanced on, by identity: an `entry` is meaningless in any
    /// other, and could lie past its end.
    list: *const crate::abi::GuestIov,
    entry: usize,
    base: u64,
}

impl Default for Cursor {
    fn default() -> Self {
        Cursor { list: std::ptr::null(), entry: 0, base: 0 }
    }
}

impl<'a> Iov<'a> {
    pub fn new(entries: &'a [crate::abi::GuestIov]) -> Iov<'a> {
        Iov { entries, len: entries.iter().map(|e| e.len as u64).sum() }
    }

    /// These pages, to be read from and nothing else.
    pub fn source(&self) -> Source<'a> {
        Source { entries: self.entries, len: self.len }
    }

    /// How many entries the list has: the multiplier on every row a transfer walks.
    pub fn entries(&self) -> usize {
        self.entries.len()
    }

    /// The total bytes the list describes.
    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether the entries are the same pages, in the same order, as `other`'s -- a list, or a
    /// source read from one.
    pub fn same_pages(&self, other: &Source<'_>) -> bool {
        self.entries.len() == other.entries.len()
            && self
                .entries
                .iter()
                .zip(other.entries)
                .all(|(a, b)| a.base == b.base && a.len == b.len)
    }

    /// Walk `len` bytes from `at`, handing each contiguous piece to `f` as a host pointer and a
    /// length. `false`, with nothing visited, if the range is not wholly inside the list.
    /// Resumes from `cursor` when `at` is not before it, and leaves the cursor at the entry the
    /// range began in.
    fn walk_from(
        &self,
        cursor: &mut Cursor,
        at: u64,
        len: usize,
        mut f: impl FnMut(*mut u8, usize, usize),
    ) -> bool {
        let Some(end) = at.checked_add(len as u64) else {
            return false;
        };
        if end > self.len {
            return false;
        }
        if at < cursor.base || cursor.list != self.entries.as_ptr() {
            *cursor = Cursor::default();
        }
        // Skip the entries wholly before `at`, from wherever the last walk left the cursor.
        let mut i = cursor.entry;
        let mut base = cursor.base;
        while let Some(e) = self.entries.get(i)
            && base + e.len as u64 <= at
        {
            base += e.len as u64;
            i += 1;
        }
        *cursor = Cursor { list: self.entries.as_ptr(), entry: i, base };
        let mut skip = (at - base) as usize;
        let mut done = 0usize;
        for e in &self.entries[i..] {
            if done == len {
                break;
            }
            let take = (e.len - skip).min(len - done);
            // The entry's base is the VMM's host address for the page; `skip` is inside it.
            f(e.base.0.cast::<u8>().wrapping_add(skip), done, take);
            done += take;
            skip = 0;
        }
        done == len
    }

    /// Copy bytes out of the guest pages into `dst`. Returns whether the range was inside them.
    #[must_use]
    pub fn copy_out(&self, at: u64, dst: &mut [u8]) -> bool {
        self.copy_out_from(&mut Cursor::default(), at, dst)
    }

    /// [`Iov::copy_out`] for one of a run of ascending rows: `cursor` carries where the last
    /// row was, so the list is not walked from its head for each.
    #[must_use]
    pub fn copy_out_from(&self, cursor: &mut Cursor, at: u64, dst: &mut [u8]) -> bool {
        self.walk_from(cursor, at, dst.len(), |src, into, n| {
            // SAFETY: the VMM's contract for an attached iov is that every entry addresses `len`
            // bytes of live guest memory until it detaches the list, and this renderer holds no
            // list past a detach. `walk_from` proved the piece is inside its entry; `dst` is a
            // host slice the caller owns, so the two cannot overlap.
            unsafe { std::ptr::copy_nonoverlapping(src, dst.as_mut_ptr().add(into), n) };
        })
    }

    /// Copy `src` into the guest pages at `at`. Returns whether the range was inside them.
    #[must_use]
    pub fn copy_in(&self, at: u64, src: &[u8]) -> bool {
        self.copy_in_from(&mut Cursor::default(), at, src)
    }

    /// [`Iov::copy_in`] for one of a run of ascending rows; see [`Iov::copy_out_from`].
    #[must_use]
    pub fn copy_in_from(&self, cursor: &mut Cursor, at: u64, src: &[u8]) -> bool {
        self.walk_from(cursor, at, src.len(), |dst, from, n| {
            // SAFETY: as `copy_out_from`, with the direction reversed; the VMM maps the pages
            // writable because a transfer from the host is what they are for.
            unsafe { std::ptr::copy_nonoverlapping(src.as_ptr().add(from), dst, n) };
        })
    }
}

/// An [`Iov`] that can only be read: the pages a transfer to the host copies out of.
///
/// Its own type rather than a promise not to write, because not every source is the guest's.
/// [`HostSpan`] is bytes a command or a snapshot carried, borrowed shared, and a `copy_in` through
/// one would be a write through `&`. A source has no `copy_in` to call.
#[derive(Clone, Copy)]
pub struct Source<'a> {
    entries: &'a [crate::abi::GuestIov],
    len: u64,
}

impl<'a> Source<'a> {
    /// The pages as a list again, privately: every read below is [`Iov`]'s own.
    fn iov(&self) -> Iov<'a> {
        Iov { entries: self.entries, len: self.len }
    }

    /// The total bytes the source describes.
    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// See [`Iov::copy_out`].
    #[must_use]
    pub fn copy_out(&self, at: u64, dst: &mut [u8]) -> bool {
        self.iov().copy_out(at, dst)
    }

    /// See [`Iov::copy_out_from`].
    #[must_use]
    pub fn copy_out_from(&self, cursor: &mut Cursor, at: u64, dst: &mut [u8]) -> bool {
        self.iov().copy_out_from(cursor, at, dst)
    }
}

/// Host bytes presented as one guest span, for a transfer whose bytes arrive in the command
/// stream rather than in attached pages (`RESOURCE_INLINE_WRITE`).
///
/// Holds the one-entry list an [`Iov`] walks, and borrows the bytes for as long as it lives, so
/// the entry cannot outlive what it points at.
pub struct HostSpan<'a> {
    entry: [crate::abi::GuestIov; 1],
    bytes: std::marker::PhantomData<&'a [u8]>,
}

impl<'a> HostSpan<'a> {
    pub fn new(bytes: &'a [u8]) -> HostSpan<'a> {
        HostSpan {
            entry: [crate::abi::GuestIov {
                base: crate::abi::VmmPtr(bytes.as_ptr().cast_mut().cast()),
                len: bytes.len(),
            }],
            bytes: std::marker::PhantomData,
        }
    }

    /// The span as pages to read from, which is all a shared borrow allows.
    pub fn source(&self) -> Source<'_> {
        Iov::new(&self.entry).source()
    }
}

/// Host bytes a transfer may write into, presented as one guest span.
///
/// The read-only [`HostSpan`] cannot serve a readback: it hands out only a [`Source`], because
/// its pointer was taken from a shared borrow. This holds the
/// borrow exclusively instead, so the write is the only one there is, and the entry cannot
/// outlive the bytes it points at.
pub struct HostSpanMut<'a> {
    entry: [crate::abi::GuestIov; 1],
    bytes: std::marker::PhantomData<&'a mut [u8]>,
}

impl<'a> HostSpanMut<'a> {
    pub fn new(bytes: &'a mut [u8]) -> HostSpanMut<'a> {
        HostSpanMut {
            entry: [crate::abi::GuestIov {
                base: crate::abi::VmmPtr(bytes.as_mut_ptr().cast()),
                len: bytes.len(),
            }],
            bytes: std::marker::PhantomData,
        }
    }

    pub fn iov(&self) -> Iov<'_> {
        Iov::new(&self.entry)
    }
}

/// The host's page size, which is what a mapping's length has to be a multiple of.
pub fn page_size() -> usize {
    // SAFETY: a plain sysconf query with no pointers involved.
    let n = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    assert!(n > 0, "the host must report a page size");
    n as usize
}

/// Round a length up to a whole number of pages, or `None` if that overflows.
///
/// The VMM maps a resource with `MAP_FIXED`, which requires a page-aligned size, so the rounded
/// length is the resource's real size -- not a detail of the allocation. Everything that checks an
/// offset against the resource has to check it against this, or it refuses layouts that fit.
pub fn page_round(len: usize) -> Option<usize> {
    let page = page_size();
    len.checked_add(page - 1).map(|n| n & !(page - 1))
}

/// Create an anonymous shared file of `len` bytes and map it.
///
/// This is the host minting memory for the guest rather than receiving it -- the C's
/// `os_create_anonymous_file` followed by the `mmap` in `vkr_context_create_resource_from_shm`.
/// The descriptor comes back with the mapping because the VMM will eventually want it: it is what
/// the guest maps on its side, and closing it here would make that impossible to offer later.
///
/// The file never appears in a directory anyone can open: on Linux it has no name at all, and on
/// macOS its name is unlinked before this returns.
pub fn anonymous_shm(len: usize, debug_name: &str) -> io::Result<(OwnedFd, GuestMap)> {
    let len = page_round(len)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "size overflows when paged"))?;
    let fd = anonymous_fd(debug_name)?;
    // The file starts empty; this is what gives it the length the mapping needs.
    // SAFETY: `fd` is a live descriptor this function just created and still owns.
    if unsafe { libc::ftruncate(fd.as_raw_fd(), len as libc::off_t) } < 0 {
        return Err(io::Error::last_os_error());
    }
    let map = GuestMap::shm(fd.as_fd(), len)?;
    Ok((fd, map))
}

/// A descriptor for a file with no directory entry, by whatever route the host offers.
fn anonymous_fd(debug_name: &str) -> io::Result<OwnedFd> {
    #[cfg(target_os = "linux")]
    {
        let name = std::ffi::CString::new(debug_name)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "name has a NUL in it"))?;
        // SAFETY: `name` is a live NUL-terminated string for the duration of the call.
        let fd = unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: memfd_create returned a fresh descriptor that nothing else owns.
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }
    #[cfg(not(target_os = "linux"))]
    {
        // macOS has no memfd, so this is shm_open under a name that is unlinked immediately. The
        // name only has to survive until then, but it still has to be unique, because two workers
        // racing on the same one would share memory that is supposed to be private.
        //
        // shm_open rejects O_CLOEXEC with EINVAL here, so close-on-exec is set afterwards.
        for attempt in 0..32u32 {
            let name = format!("/{}-{}-{}\0", debug_name, std::process::id(), attempt);
            // SAFETY: `name` is NUL-terminated above and live for the duration of the call.
            let fd = unsafe {
                libc::shm_open(
                    name.as_ptr().cast::<libc::c_char>(),
                    libc::O_CREAT | libc::O_EXCL | libc::O_RDWR,
                    0o600 as libc::c_uint,
                )
            };
            if fd >= 0 {
                // SAFETY: shm_open returned a fresh descriptor that nothing else owns; taking
                // ownership here is what closes it on every path out.
                let owned = unsafe { OwnedFd::from_raw_fd(fd) };
                // SAFETY: `owned` is live; both calls take the descriptor and no pointers.
                unsafe {
                    libc::fcntl(owned.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC);
                    libc::shm_unlink(name.as_ptr().cast::<libc::c_char>());
                }
                return Ok(owned);
            }
            let err = io::Error::last_os_error();
            if err.raw_os_error() != Some(libc::EEXIST) {
                return Err(err);
            }
        }
        Err(io::Error::new(io::ErrorKind::AlreadyExists, "no unused shm name in 32 tries"))
    }
}

impl Drop for GuestMap {
    fn drop(&mut self) {
        // SAFETY: this pointer and length came from the `mmap` in `shm` and have not changed
        // since; nothing else unmaps them, because nothing else holds them.
        let rc = unsafe { libc::munmap(self.ptr.as_ptr().cast::<libc::c_void>(), self.len) };
        assert_eq!(
            rc,
            0,
            "munmap of a mapping we made must succeed: {}",
            io::Error::last_os_error()
        );
    }
}

impl std::fmt::Debug for GuestMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Deliberately not the address: it is noise that changes every run, and printing guest
        // memory's location into a log is not something a debug impl should decide to do.
        f.debug_struct("GuestMap").field("len", &self.len).finish()
    }
}

/// A mapping some other owner holds, borrowed for as long as that owner is borrowed.
///
/// The address alone would be exactly the stale reference this tree refuses to make
/// representable, so the lifetime is carried in the type and tied at construction to whatever
/// keeps the mapping alive. Read by copy, like every other source here, because the memory is a
/// GPU driver's and may be written while it is read.
pub struct HostMapping<'a> {
    addr: usize,
    len: u64,
    owner: core::marker::PhantomData<&'a ()>,
}

impl<'a> HostMapping<'a> {
    /// Describe a mapping at `addr` running `len` bytes.
    ///
    /// # Safety
    ///
    /// `addr` must name a readable mapping of at least `len` bytes that stays mapped for the
    /// whole of `'a`. Tie `'a` to the value that owns the mapping -- not to the caller's
    /// convenience -- so that the borrow ending is the mapping ending.
    pub unsafe fn new(addr: usize, len: u64) -> HostMapping<'a> {
        HostMapping { addr, len, owner: core::marker::PhantomData }
    }

    /// How far the mapping runs.
    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Fill `dst` from `at`, or `false` with `dst` untouched if that range is not wholly inside
    /// the mapping. The bounds check is here because this is the only place that knows the size.
    #[must_use]
    pub fn copy_out(&self, at: u64, dst: &mut [u8]) -> bool {
        let Ok(at) = usize::try_from(at) else {
            return false;
        };
        if !at.checked_add(dst.len()).is_some_and(|end| end as u64 <= self.len) {
            return false;
        }
        // SAFETY: the check above proved `at + dst.len()` is inside the mapping, and the
        // constructor's contract is that the mapping is live for `'a`, which this borrow is
        // within. `dst` is a live host slice of exactly that length; the two cannot overlap
        // because one is a driver's mapping and the other a buffer the caller brought.
        unsafe {
            std::ptr::copy_nonoverlapping(
                (self.addr as *const u8).add(at),
                dst.as_mut_ptr(),
                dst.len(),
            )
        };
        true
    }

    /// Copy `src` into the mapping at `at`, or `false` with nothing written if it would not fit.
    ///
    /// The counterpart to `copy_out`, for the restore that has to put a capture back by the same
    /// route it was read: a snapshot written somewhere the driver does not look is a restore that
    /// silently does nothing.
    #[must_use]
    pub fn copy_in(&self, at: u64, src: &[u8]) -> bool {
        let Ok(at) = usize::try_from(at) else {
            return false;
        };
        if !at.checked_add(src.len()).is_some_and(|end| end as u64 <= self.len) {
            return false;
        }
        // SAFETY: as `copy_out`, with the direction reversed and the same bound proved.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), (self.addr as *mut u8).add(at), src.len())
        };
        true
    }
}

/// Where a blob's pixels are, for the one caller that has to read them without caring which.
///
/// A blob's bytes reach the host by one of two routes -- the guest's own scatter list, or a
/// mapping this process holds -- and the texture fill needs the bytes, not the route. Both are
/// read-only and read by copy, for the reason [`Iov`] gives: the guest writes them whenever it
/// likes, so a reference into them cannot be sound.
///
/// Borrowed from the resource table and never held: the pages belong to the VMM and the mapping
/// to whoever minted it, so a source outliving the resource it came from is exactly the stale
/// reference this tree refuses to make representable. Ask again at the next read instead.
pub enum PixelSource<'a> {
    /// The guest pages a VMM attached, as `Backing::Blob` with `BlobStorage::Guest` has them.
    Scattered(Iov<'a>),
    /// A mapping this process holds: minted shm, or the linear pages a venus allocation was
    /// published from.
    Mapped(&'a GuestMap),
    /// A mapping this process holds but did not make -- a Vulkan driver's own allocation, mapped
    /// once by whoever owns it. The bytes are read the same way; only the owner differs.
    Foreign(HostMapping<'a>),
}

impl PixelSource<'_> {
    /// The bytes on offer. Not what any consumer asked for -- a caller wanting fewer must check.
    pub fn len(&self) -> u64 {
        match self {
            PixelSource::Scattered(iov) => iov.len(),
            PixelSource::Mapped(map) => map.len() as u64,
            PixelSource::Foreign(m) => m.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Fill `dst` from `at`. `false`, with `dst` untouched, if the range is not wholly inside
    /// the source -- the bounds check belongs here because this is the only place that knows
    /// how far the source runs.
    pub fn copy_out(&self, at: u64, dst: &mut [u8]) -> bool {
        self.copy_out_from(&mut Cursor::default(), at, dst)
    }

    /// [`PixelSource::copy_out`] for one of a run of ascending rows. Only a scattered source
    /// has a list to walk; the cursor is carried for it and ignored by the rest.
    pub fn copy_out_from(&self, cursor: &mut Cursor, at: u64, dst: &mut [u8]) -> bool {
        match self {
            PixelSource::Scattered(iov) => iov.copy_out_from(cursor, at, dst),
            PixelSource::Mapped(map) => usize::try_from(at).is_ok_and(|at| map.copy_out(at, dst)),
            PixelSource::Foreign(m) => m.copy_out(at, dst),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::fd::{AsFd, OwnedFd};

    /// An anonymous shared mapping standing in for a guest resource. It is the same kind of object
    /// -- an shm descriptor -- so the mapping path under test is the real one.
    fn shm_fd(len: usize) -> OwnedFd {
        let mut file = std::env::temp_dir();
        file.push(format!(
            "virglrs-guestmap-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&file)
            .expect("a temp file to back the mapping");
        f.set_len(len as u64).expect("sized");
        std::fs::remove_file(&file).expect("unlinked; the descriptor keeps it alive");
        OwnedFd::from(f)
    }

    /// A length past the end of the descriptor is refused rather than mapped: the pages past the
    /// end have nothing behind them, and touching one is a `SIGBUS`.
    #[test]
    fn a_mapping_longer_than_its_descriptor_is_refused() {
        let fd = shm_fd(0x1000);
        let refused = GuestMap::shm(fd.as_fd(), 0x2000);
        assert!(refused.is_err(), "two pages described over a one-page descriptor");
        assert!(GuestMap::shm(fd.as_fd(), 0x1000).is_ok(), "and what it holds maps");
    }

    #[test]
    fn bytes_written_to_the_mapping_come_back() {
        let fd = shm_fd(0x1000);
        let m = GuestMap::shm(fd.as_fd(), 0x1000).expect("mapped");
        assert_eq!(m.len(), 0x1000);

        assert!(m.copy_in(0x40, b"venus"), "an in-bounds write is accepted");
        let mut out = [0u8; 5];
        assert!(m.copy_out(0x40, &mut out), "and reads back");
        assert_eq!(&out, b"venus");
    }

    /// The whole reason this type exists: an offset the guest chose can never reach outside what
    /// was mapped, whatever it is.
    #[test]
    fn no_offset_a_guest_can_name_reaches_outside_the_mapping() {
        let fd = shm_fd(0x1000);
        let m = GuestMap::shm(fd.as_fd(), 0x1000).expect("mapped");
        let mut buf = [0u8; 8];

        assert!(!m.copy_out(0x1000, &mut buf), "starting at the end reads nothing");
        assert!(!m.copy_out(0xffc, &mut buf), "a range that straddles the end is refused whole");
        assert!(!m.copy_out(usize::MAX, &mut buf), "an offset that would overflow is refused");
        assert!(!m.copy_in(usize::MAX - 4, b"nope"), "and so is the same on the way in");
        assert!(!m.store_u32(0x1000, 1), "a word starting at the end is outside it");
        assert!(m.load_u32(0x1000).is_none(), "and cannot be read either");

        // The tightest case there is, and the one a loosened bound survives: a *single* byte at
        // exactly the end. Every range above is wide enough that an off-by-one still refuses it.
        let mut one = [0u8; 1];
        assert!(!m.copy_out(0xfff + 1, &mut one), "one byte starting at the end is outside it");
        assert!(!m.copy_in(0x1000, b"x"), "and cannot be written there either");
        assert!(m.copy_out(0xfff, &mut one), "the byte before it is the last one inside");

        assert!(m.copy_out(0xff8, &mut buf), "the last eight bytes are inside, and readable");
        assert!(m.copy_out(0x1000, &mut []), "an empty range at the very end is inside it");
    }

    /// The control words are read atomically, so an unaligned one is not a control word at all.
    /// The layout check rejects these too, but `extra` offsets arrive at write time with no layout
    /// left to check them against, so the mapping refuses them itself.
    #[test]
    fn a_misaligned_control_word_is_refused_rather_than_read() {
        let fd = shm_fd(0x1000);
        let m = GuestMap::shm(fd.as_fd(), 0x1000).expect("mapped");

        for at in [1, 2, 3, 0x101] {
            assert!(m.load_u32(at).is_none(), "offset {at} is not 32-bit aligned");
            assert!(!m.store_u32(at, 7), "and cannot be written either");
            assert!(!m.fetch_or_u32(at, 7), "nor have bits set in it");
        }
        assert!(m.store_u32(0x100, 7), "the aligned word next to them is fine");
        assert_eq!(m.load_u32(0x100), Some(7));
    }

    #[test]
    fn status_bits_accumulate_instead_of_replacing_each_other() {
        let fd = shm_fd(0x1000);
        let m = GuestMap::shm(fd.as_fd(), 0x1000).expect("mapped");

        assert!(m.fetch_or_u32(0x20, 0b001));
        assert!(m.fetch_or_u32(0x20, 0b100));
        assert_eq!(m.load_u32(0x20), Some(0b101), "the earlier bit survived the later one");
    }

    /// Two mappings of one descriptor see each other's writes -- which is the property the whole
    /// ring depends on, and the one a private mapping would silently not have.
    #[test]
    fn the_mapping_is_shared_and_not_a_private_copy() {
        let fd = shm_fd(0x1000);
        let a = GuestMap::shm(fd.as_fd(), 0x1000).expect("mapped once");
        let b = GuestMap::shm(fd.as_fd(), 0x1000).expect("mapped twice");

        assert!(a.store_u32(0x80, 0xdecafbad));
        assert_eq!(b.load_u32(0x80), Some(0xdecafbad), "the other mapping sees it");

        assert!(b.copy_in(0x200, b"shared"));
        let mut out = [0u8; 6];
        assert!(a.copy_out(0x200, &mut out));
        assert_eq!(&out, b"shared");
    }

    /// The mapping outliving the descriptor is what lets a ring outlive the resource handle it was
    /// created from, so it is asserted rather than assumed.
    #[test]
    fn the_mapping_survives_the_descriptor_being_closed() {
        let m = {
            let fd = shm_fd(0x1000);
            GuestMap::shm(fd.as_fd(), 0x1000).expect("mapped")
        };
        assert!(m.store_u32(0, 0x1234), "still writable with the descriptor long gone");
        assert_eq!(m.load_u32(0), Some(0x1234));
    }

    #[test]
    fn a_minted_mapping_is_page_rounded_and_usable() {
        let page = page_size();
        // The size the venus corpus asks for a ring resource, which is not a whole page.
        let (fd, m) = anonymous_shm(0x24000 - 1, "virglrs-test").expect("minted");
        assert_eq!(m.len() % page, 0, "the length is a whole number of pages");
        assert!(m.len() >= 0x24000 - 1, "and is at least what was asked for");
        assert!(m.len() < 0x24000 - 1 + page, "without rounding further than one page");

        assert!(m.copy_in(0, b"ring"), "the memory is there and writable");
        let mut out = [0u8; 4];
        assert!(m.copy_out(0, &mut out));
        assert_eq!(&out, b"ring");

        // The descriptor comes back with it, because the VMM will need it to map its own side.
        let second = GuestMap::shm(fd.as_fd(), m.len()).expect("the descriptor still names it");
        assert_eq!(second.load_u32(0), m.load_u32(0), "and it is the same memory");
    }

    /// Every minting must be its own memory. On macOS the name is reused across attempts only on
    /// EEXIST, and a bug there would hand two resources the same bytes.
    #[test]
    fn two_minted_mappings_are_not_the_same_memory() {
        let (_a_fd, a) = anonymous_shm(0x1000, "virglrs-test").expect("one");
        let (_b_fd, b) = anonymous_shm(0x1000, "virglrs-test").expect("two");
        assert!(a.store_u32(0, 0xaaaa_aaaa));
        assert!(b.store_u32(0, 0xbbbb_bbbb));
        assert_eq!(a.load_u32(0), Some(0xaaaa_aaaa), "the second did not overwrite the first");
        assert_eq!(b.load_u32(0), Some(0xbbbb_bbbb));
    }

    #[test]
    fn a_page_round_that_would_overflow_is_refused() {
        assert_eq!(page_round(0), Some(0));
        assert_eq!(page_round(1), Some(page_size()));
        assert_eq!(page_round(page_size()), Some(page_size()), "an exact page is left alone");
        assert_eq!(page_round(usize::MAX), None, "rather than wrapping to something small");
    }

    #[test]
    fn an_empty_resource_is_refused_rather_than_mapped() {
        let fd = shm_fd(0x1000);
        assert!(GuestMap::shm(fd.as_fd(), 0).is_err(), "mmap of zero bytes is an error, not a map");
    }

    /// A scatter list over host buffers, the way the VMM hands one over: one entry per piece.
    fn scattered(pieces: &mut [Vec<u8>]) -> Vec<crate::abi::GuestIov> {
        pieces
            .iter_mut()
            .map(|b| crate::abi::GuestIov {
                base: crate::abi::VmmPtr(b.as_mut_ptr().cast()),
                len: b.len(),
            })
            .collect()
    }

    /// Byte `i` of the concatenation holds `i`, so a copy can be checked by position.
    fn numbered(sizes: &[usize]) -> Vec<Vec<u8>> {
        let mut n = 0u8;
        sizes
            .iter()
            .map(|&size| {
                (0..size)
                    .map(|_| {
                        n = n.wrapping_add(1);
                        n
                    })
                    .collect()
            })
            .collect()
    }

    /// A range that straddles entries reads and writes the pieces in list order.
    #[test]
    fn a_walk_crosses_entries_in_order() {
        let mut pieces = numbered(&[5, 3, 12]);
        let entries = scattered(&mut pieces);
        let iov = Iov::new(&entries);
        assert_eq!((iov.entries(), iov.len()), (3, 20));
        let mut got = [0u8; 10];
        assert!(iov.copy_out(3, &mut got));
        assert_eq!(got, [4, 5, 6, 7, 8, 9, 10, 11, 12, 13], "bytes 3..13 of the concatenation");
        assert!(iov.copy_in(4, &[0xa0, 0xa1, 0xa2, 0xa3, 0xa4]));
        assert_eq!(&pieces[0][4..], [0xa0], "the tail of the first piece");
        assert_eq!(pieces[1], [0xa1, 0xa2, 0xa3], "the whole second");
        assert_eq!(pieces[2][0], 0xa4, "the head of the third");
    }

    /// Ascending rows resume from the entry the last row ended in rather than the head of the
    /// list, and a row that goes backwards is served correctly by restarting.
    #[test]
    fn ascending_rows_carry_the_cursor_and_a_backwards_row_restarts() {
        let mut pieces = numbered(&[4; 40]);
        let entries = scattered(&mut pieces);
        let iov = Iov::new(&entries);
        let flat: Vec<u8> = pieces.iter().flatten().copied().collect();
        let mut cursor = Cursor::default();
        let mut last_entry = 0;
        for row in 0..18u64 {
            let at = row * 8 + 1;
            let mut got = [0u8; 6];
            assert!(iov.copy_out_from(&mut cursor, at, &mut got), "row {row} is inside");
            assert_eq!(got, flat[at as usize..at as usize + 6], "row {row}");
            assert!(cursor.entry >= last_entry, "the cursor never goes back on an ascending row");
            assert_eq!(cursor.base, (cursor.entry as u64) * 4, "and names the bytes before it");
            last_entry = cursor.entry;
        }
        assert_eq!(last_entry, 34, "row 17 begins at byte 137, entry 34");
        let mut got = [0u8; 6];
        assert!(iov.copy_out_from(&mut cursor, 2, &mut got), "a backwards row is still inside");
        assert_eq!(got, flat[2..8], "and reads what is there, from the head again");
        assert_eq!(cursor.entry, 0);
        assert!(iov.copy_in_from(&mut cursor, 158, &[7, 7]), "a write at the very end");
        assert_eq!(pieces[39][2..], [7, 7]);
    }

    /// A range that leaves the list is refused whole with nothing visited, from a fresh walk
    /// and from a cursor alike; an empty range at the very end is inside.
    #[test]
    fn a_range_past_a_scattered_list_is_refused_whole() {
        let mut pieces = numbered(&[6, 6]);
        let entries = scattered(&mut pieces);
        let iov = Iov::new(&entries);
        let mut got = [0u8; 2];
        assert!(!iov.copy_out(11, &mut got), "straddles the end");
        assert_eq!(got, [0, 0], "and nothing was visited");
        assert!(!iov.copy_in(12, b"x"), "one byte past the end");
        assert!(iov.copy_out(12, &mut []), "an empty range at the very end is inside it");
        let mut cursor = Cursor::default();
        assert!(iov.copy_out_from(&mut cursor, 7, &mut got));
        assert!(!iov.copy_out_from(&mut cursor, 11, &mut got), "the cursor does not loosen it");
        assert!(!iov.copy_out(u64::MAX, &mut got), "an offset that would overflow is refused");
    }

    /// A cursor advanced deep into one list names an entry a shorter list does not have, at a
    /// base the shorter list's own offsets can still exceed.
    #[test]
    fn a_cursor_from_another_list_restarts_on_this_one() {
        let mut long = numbered(&[1; 40]);
        let long_entries = scattered(&mut long);
        let long_iov = Iov::new(&long_entries);
        let mut cursor = Cursor::default();
        let mut got = [0u8; 4];
        assert!(long_iov.copy_out_from(&mut cursor, 30, &mut got));
        assert_eq!(got, [31, 32, 33, 34]);
        let mut short = numbered(&[20, 20]);
        let short_entries = scattered(&mut short);
        let short_iov = Iov::new(&short_entries);
        assert!(short_iov.copy_out_from(&mut cursor, 30, &mut got));
        assert_eq!(got, [31, 32, 33, 34], "the short list's bytes, walked from its head");
    }
}

/// Proofs over every scatter list and every range a transfer can ask for, run by `cargo kani`.
///
/// `walk_from` hands its callback raw pointers into guest pages, and the copy that follows is
/// sound only if every piece lies inside the entry it came from -- the SAFETY comments on
/// [`Iov::copy_out_from`] and [`Iov::copy_in_from`] rest on it. These prove it for lists of up to
/// three entries of any 32-bit length, and any start and length. Entry bases are distinct
/// addresses nothing dereferences: the walk only does arithmetic on them.
#[cfg(kani)]
mod proofs {
    use super::*;
    use crate::abi::{GuestIov, VmmPtr};

    const ENTRIES: usize = 3;
    /// Where entry `i` of a list starts. Far enough apart that no entry reaches the next.
    const fn base(i: usize) -> usize {
        (i + 1) << 40
    }

    fn any_list() -> [GuestIov; ENTRIES] {
        core::array::from_fn(|i| GuestIov {
            base: VmmPtr(base(i) as *mut core::ffi::c_void),
            len: kani::any::<u32>() as usize,
        })
    }

    /// The pieces one walk handed its callback, as `(address, into, len)`.
    #[derive(Clone, Copy, PartialEq)]
    struct Pieces {
        got: [(usize, usize, usize); ENTRIES + 1],
        n: usize,
    }

    fn walk(iov: &Iov<'_>, cursor: &mut Cursor, at: u64, len: usize) -> (bool, Pieces) {
        let mut p = Pieces { got: [(0, 0, 0); ENTRIES + 1], n: 0 };
        let ok = iov.walk_from(cursor, at, len, |ptr, into, n| {
            assert!(p.n <= ENTRIES, "a walk handed back more pieces than there are entries");
            p.got[p.n] = (ptr as usize, into, n);
            p.n += 1;
        });
        (ok, p)
    }

    /// Where logical offset `pos` of the list lives: its entry's base plus the offset into it.
    fn address_of(list: &[GuestIov], pos: u64) -> Option<(usize, usize)> {
        let mut start = 0u64;
        for (i, e) in list.iter().enumerate() {
            if pos < start + e.len as u64 {
                return Some((i, base(i) + (pos - start) as usize));
            }
            start += e.len as u64;
        }
        None
    }

    /// A walk succeeds exactly when the range fits, hands back contiguous pieces that add up to
    /// it, and puts each piece where its logical offset lives, inside its own entry. A refused
    /// walk visits nothing.
    #[kani::proof]
    #[kani::unwind(5)]
    fn every_piece_lies_inside_its_entry() {
        let all = any_list();
        let n: usize = kani::any();
        kani::assume(n <= ENTRIES);
        let list = &all[..n];
        let iov = Iov::new(list);
        let (at, len): (u64, usize) = (kani::any(), kani::any());

        let (ok, p) = walk(&iov, &mut Cursor::default(), at, len);
        let fits = at.checked_add(len as u64).is_some_and(|end| end <= iov.len());
        assert!(ok == fits, "a walk's verdict disagrees with the range");
        if !ok {
            assert!(p.n == 0, "a refused walk visited a piece");
            return;
        }
        let mut done = 0usize;
        for &(addr, into, take) in &p.got[..p.n] {
            assert!(into == done, "the pieces are not contiguous");
            if take > 0 {
                let (i, want) = address_of(list, at + into as u64).expect("inside the list");
                assert!(addr == want, "a piece starts somewhere its offset does not live");
                let skip = addr - base(i);
                assert!(skip + take <= list[i].len, "a piece runs past its entry");
            }
            done += take;
        }
        assert!(done == len, "the pieces do not add up to the range");
        kani::cover!(
            p.got[..p.n].iter().filter(|g| g.2 > 0).count() >= 2,
            "a range across entries"
        );
    }

    /// A walk resumed from where an earlier one left the cursor does exactly what a fresh walk
    /// does, whichever order the two ranges come in.
    #[kani::proof]
    #[kani::unwind(5)]
    fn a_resumed_walk_matches_a_fresh_one() {
        let all = any_list();
        let iov = Iov::new(&all);
        let mut cursor = Cursor::default();
        let _ = walk(&iov, &mut cursor, kani::any(), kani::any());

        let (at, len): (u64, usize) = (kani::any(), kani::any());
        let resumed = walk(&iov, &mut cursor, at, len);
        let fresh = walk(&iov, &mut Cursor::default(), at, len);
        assert!(resumed == fresh, "a resumed walk went somewhere a fresh one does not");
        kani::cover!(cursor.entry > 0, "a cursor carried past the first entry");
    }
}
