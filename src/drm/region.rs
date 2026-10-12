// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! The response region: the shared memory a native context's guest reads answers out of.
//!
//! The guest creates it first, as its blob 0, and the host never lets go of a share of it while
//! the context stands. It is one value -- the mapping, how much of it the guest asked for, and
//! where its header ends -- so an offset into it is checked against all three in one place.
//!
//! The guest writes this memory while the host does, so nothing here reads a guest-written value
//! twice: a response's offset is checked and used from the one copy the decoder made, and the
//! single guest value this reads, a response's pre-set length, is loaded once.

use super::Refused;
use super::proto;
use crate::guest_mem::GuestMap;
use std::sync::Arc;

pub struct ResponseRegion {
    map: Arc<GuestMap>,
    /// Where the backend's header ends and the responses begin.
    header: usize,
    /// The bytes the guest asked for. The mapping may be longer -- minted memory is whole pages --
    /// but the guest addresses only what it asked for, and so does the host.
    size: usize,
}

impl ResponseRegion {
    /// The region over `map`, for a guest that asked for `size` bytes of which the first `header`
    /// are the backend's. Writes the responses' start into the header, which is how the guest
    /// learns where they begin.
    pub fn new(map: Arc<GuestMap>, size: u64, header: usize) -> Result<ResponseRegion, Refused> {
        // The C's bounds: room for the header, and a size its 32-bit offsets can address.
        let size = usize::try_from(size).map_err(|_| Refused::RegionSize(size))?;
        if size < header || size > u32::MAX as usize || size > map.len() {
            return Err(Refused::RegionSize(size as u64));
        }
        let region = ResponseRegion { map, header, size };
        let header = u32::try_from(header).expect("a backend's header is a few dozen bytes");
        assert!(
            region.map.store_u32(proto::shmem::RSP_MEM_OFFSET, header),
            "the header holds the vdrm header it extends"
        );
        Ok(region)
    }

    /// Whether `len` bytes at response offset `off` lie in the responses.
    pub fn holds(&self, off: u32, len: usize) -> bool {
        let room = self.size - self.header;
        (off as usize) <= room && len <= room - off as usize
    }

    /// Copy a staged response in at `off`, no longer than the length the guest pre-set there.
    ///
    /// The guest says how long a response it has room for by writing the length before it sends
    /// the request; a response longer than that is cut, and the length written back is what was
    /// copied. `staged` was bounds-checked against `off` when it was staged.
    pub fn publish(&self, off: u32, staged: &[u8]) {
        assert!(self.holds(off, staged.len()), "a response is staged only where it fits");
        let at = self.header + off as usize + proto::RSP_LEN;
        // Every response is at least its length word, so the word is in bounds. Loaded once: the
        // guest may be rewriting it.
        let room = self.map.load_u32(at).expect("a staged response holds its length word");
        let len = staged.len().min(room as usize);
        assert!(self.map.copy_in(at, &staged[..len]), "checked by holds");
        // Written last, over whatever of it the copy wrote, as the C does.
        assert!(self.map.store_u32(at, len as u32), "read from just above");
    }

    /// Tell the guest every request up to `seqno` is done.
    pub fn finished(&self, seqno: u32) {
        assert!(self.map.store_u32(proto::shmem::SEQNO, seqno), "the header holds the seqno");
    }

    /// Write one of the backend's own header fields. The header is the host's: the guest reads it
    /// and the host never reads it back.
    pub fn write_header(&self, at: usize, bytes: &[u8]) {
        assert!(at + bytes.len() <= self.header, "a header field lies in the header");
        assert!(self.map.copy_in(at, bytes), "the header is inside the mapping");
    }

    /// Store a word of the backend's header atomically, for a field the guest polls.
    pub fn store_header_u32(&self, at: usize, value: u32) {
        assert!(at + 4 <= self.header, "a header field lies in the header");
        assert!(self.map.store_u32(at, value), "the header is inside the mapping");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn region(size: u64) -> (Arc<GuestMap>, ResponseRegion) {
        let map = Arc::new(GuestMap::anonymous(4096).unwrap());
        let r = ResponseRegion::new(Arc::clone(&map), size, 88).unwrap();
        (map, r)
    }

    #[test]
    fn the_region_says_where_responses_start() {
        let (map, _r) = region(4096);
        assert_eq!(map.load_u32(proto::shmem::RSP_MEM_OFFSET), Some(88));
    }

    #[test]
    fn a_region_too_small_for_its_header_is_refused() {
        let map = Arc::new(GuestMap::anonymous(4096).unwrap());
        assert!(ResponseRegion::new(Arc::clone(&map), 87, 88).is_err());
        let past = map.len() as u64 + 1;
        assert!(ResponseRegion::new(map, past, 88).is_err(), "larger than the mapping");
    }

    #[test]
    fn the_responses_are_bounded_by_what_the_guest_asked_for() {
        let (_map, r) = region(1000);
        assert!(r.holds(0, 912));
        assert!(!r.holds(0, 913));
        assert!(r.holds(904, 8));
        assert!(!r.holds(905, 8));
        assert!(!r.holds(u32::MAX, 0));
    }

    #[test]
    fn a_response_is_cut_to_the_room_the_guest_preset() {
        let (map, r) = region(4096);
        assert!(map.store_u32(88 + 16, 12));
        r.publish(16, &[0xaau8; 32]);
        assert_eq!(map.load_u32(88 + 16), Some(12));
        let mut back = [0u8; 16];
        assert!(map.copy_out(88 + 16, &mut back));
        assert_eq!(&back[4..12], &[0xaa; 8]);
        assert_eq!(&back[12..], &[0; 4], "nothing past the guest's room");
    }

    #[test]
    fn a_full_response_carries_its_own_length() {
        let (map, r) = region(4096);
        assert!(map.store_u32(88, 4096));
        r.publish(0, &[0x55u8; 24]);
        assert_eq!(map.load_u32(88), Some(24));
    }
}
