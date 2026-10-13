// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! One native context: its backend, its response region, and the framing that turns a submission
//! into requests. What every backend shares, so a backend is only its commands.

use super::proto::{self, ReqHeader};
use super::region::ResponseRegion;
use super::timeline::Timelines;
use super::{Buffer, NativeBlob, Refused, Source};
use crate::guest_mem::GuestMap;
use crate::ids::{BlobId, FenceId, ResourceHandle, RingIdx};
use std::os::fd::OwnedFd;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// `VIRTGPU_BLOB_FLAG_USE_MAPPABLE`, which the response region must be created with and alone.
const BLOB_FLAG_USE_MAPPABLE: u32 = 1;

/// What a backend is: the commands one kernel driver's guest sends, and the header its response
/// region starts with. Everything else -- framing, the region, the seqno -- is the context's.
pub trait Backend: Send {
    /// The alignment every request length and response offset must have.
    fn alignment(&self) -> usize;

    /// How many bytes of the response region are the backend's header, `vdrm_shmem` included.
    fn header_len(&self) -> usize;

    /// The response region was just attached: fill in whatever of the header the guest reads
    /// before it sends anything.
    fn region_attached(&mut self, region: &ResponseRegion);

    /// Run one request. `req` is the whole request, header included, exactly as long as the guest
    /// said. An answer is staged through `reply`, and only a request that returns `Ok` has it
    /// published. Any wait the request asks for goes through `waits`.
    fn dispatch(
        &mut self,
        cmd: u32,
        req: &[u8],
        reply: &mut Reply<'_>,
        region: Option<&ResponseRegion>,
        waits: &mut Waits,
    ) -> Result<(), Refused>;

    /// Bind the buffer made under `blob` to `handle`, and give the resource its share. `size` is
    /// the blob's, which the buffer must cover.
    fn get_blob(
        &mut self,
        handle: ResourceHandle,
        blob: NativeBlob,
        size: u64,
        flags: u32,
        region: Option<&ResponseRegion>,
    ) -> Result<Buffer, Refused>;

    /// Name `source` as `handle` in this context, if it does not already: the copy on this
    /// context's device file, imported now if there is none.
    fn attach(&mut self, handle: ResourceHandle, source: Source<'_>) -> Result<(), Refused>;

    /// Let go of whatever buffer `handle` names in this context.
    fn forget(&mut self, handle: ResourceHandle);

    /// How many hardware rings the guest may submit to; it numbers them from 1.
    fn rings(&self) -> u32;

    /// What the newest submission on `ring` signals, if the ring has submitted since this was
    /// last taken.
    fn take_last_submit(&mut self, ring: RingIdx) -> Option<OwnedFd>;
}

/// Where a request's answer is staged before the request is known to have succeeded.
///
/// A handler stages before it does its work, so a response the region has no room for refuses the
/// request before anything happens in the kernel -- the C's order, and the one that leaves nothing
/// half-done behind a refusal.
pub struct Reply<'a> {
    region: Option<&'a ResponseRegion>,
    rsp_off: u32,
    staged: Option<Vec<u8>>,
}

impl Reply<'_> {
    /// Room for a `len`-byte answer, zeroed. Refused when there is no region yet, or it has no room
    /// at the offset the request named.
    pub fn stage(&mut self, len: usize) -> Result<&mut [u8], Refused> {
        let region = self.region.ok_or(Refused::NoRegion)?;
        if !region.holds(self.rsp_off, len) {
            return Err(Refused::ResponseOutOfBounds { off: self.rsp_off, len });
        }
        assert!(self.staged.is_none(), "a request stages one answer");
        Ok(self.staged.insert(vec![0; len]))
    }
}

/// What one submission may still spend blocked, across all its requests.
///
/// A submission runs on the renderer's thread, so a request that waits for the GPU stalls every
/// context, not only its own. The guest names how long; the host decides how long it gets, and a
/// submission of many waits gets no more than one.
pub struct Waits {
    left: Duration,
}

impl Waits {
    /// The most one submission blocks: as long as any guest Mesa asks for, RADV's idle wait
    /// included.
    pub const BOUND: Duration = Duration::from_secs(1);

    fn new() -> Waits {
        Waits { left: Waits::BOUND }
    }

    /// Run `wait` with a timeout of `want`, or of what the submission has left if that is less,
    /// and charge it for the time `wait` took.
    pub fn bounded<T>(&mut self, want: Duration, wait: impl FnOnce(Duration) -> T) -> T {
        let started = Instant::now();
        let answer = wait(want.min(self.left));
        self.left = self.left.saturating_sub(started.elapsed());
        answer
    }
}

pub struct Context {
    pub name: String,
    backend: Box<dyn Backend>,
    region: Option<ResponseRegion>,
    /// `None` only for a backend with no rings.
    timelines: Option<Timelines>,
}

impl Context {
    pub fn new(name: String, backend: Box<dyn Backend>, timelines: Option<Timelines>) -> Context {
        assert!(
            timelines.is_some() || backend.rings() == 0,
            "a backend with rings has a waiter for them"
        );
        Context { name, backend, region: None, timelines }
    }

    /// Order `fence` behind the newest submission on `ring`, or behind the ring's earlier fences
    /// when it has submitted nothing since. `Ok(false)` is ring 0, which the caller retires.
    pub fn submit_fence(&mut self, ring: RingIdx, fence: FenceId) -> Result<bool, Refused> {
        if ring.0 == 0 {
            return Ok(false);
        }
        if ring.0 > self.backend.rings() {
            return Err(Refused::Ring(ring.0));
        }
        let sync = self.backend.take_last_submit(ring);
        self.timelines.as_ref().expect("checked at new").push(ring, sync, fence);
        Ok(true)
    }

    /// Take `map` as this context's response region: the guest's blob 0, which it asked for
    /// `size` bytes of with `flags`. There is one per context, and it must be mappable and
    /// nothing else, as the C requires.
    pub fn attach_region(
        &mut self,
        map: Arc<GuestMap>,
        size: u64,
        flags: u32,
    ) -> Result<(), Refused> {
        if flags != BLOB_FLAG_USE_MAPPABLE {
            return Err(Refused::RegionFlags(flags));
        }
        if self.region.is_some() {
            return Err(Refused::SecondRegion);
        }
        let region = ResponseRegion::new(map, size, self.backend.header_len())?;
        self.backend.region_attached(&region);
        self.region = Some(region);
        Ok(())
    }

    /// Bind a buffer this context made to a resource. A zero blob id is the response region,
    /// which takes the other path; an id wider than 32 bits is one no guest could have made.
    pub fn get_blob(
        &mut self,
        handle: ResourceHandle,
        blob: BlobId,
        size: u64,
        flags: u32,
    ) -> Result<Buffer, Refused> {
        let blob = NativeBlob::new(blob.0)?;
        self.backend.get_blob(handle, blob, size, flags, self.region.as_ref())
    }

    pub fn attach(&mut self, handle: ResourceHandle, source: Source<'_>) -> Result<(), Refused> {
        self.backend.attach(handle, source)
    }

    pub fn forget(&mut self, handle: ResourceHandle) {
        self.backend.forget(handle);
    }

    /// Run a submission: requests back to back, each led by its header.
    ///
    /// The first request that fails ends the submission, and its answer and seqno are not
    /// published -- the C's behaviour, which the guest's driver is written against. Each header is
    /// read once, from the caller's buffer, into a value every check and every use then reads.
    pub fn submit(&mut self, buf: &[u8]) -> Result<(), Refused> {
        let align = self.backend.alignment();
        let mut rest = buf;
        let mut waits = Waits::new();
        while rest.len() >= proto::HEADER_LEN {
            let hdr: ReqHeader = proto::decode(rest);
            let len = hdr.len as usize;
            if len < proto::HEADER_LEN || len > rest.len() || !len.is_multiple_of(align) {
                return Err(Refused::RequestLength(hdr.len));
            }
            if !(hdr.rsp_off as usize).is_multiple_of(align) {
                return Err(Refused::ResponseOffset(hdr.rsp_off));
            }
            let (req, tail) = rest.split_at(len);
            self.run(hdr, req, &mut waits)?;
            rest = tail;
        }
        if !rest.is_empty() {
            return Err(Refused::Trailing(rest.len()));
        }
        Ok(())
    }

    fn run(&mut self, hdr: ReqHeader, req: &[u8], waits: &mut Waits) -> Result<(), Refused> {
        let region = self.region.as_ref();
        let mut reply = Reply { region, rsp_off: hdr.rsp_off, staged: None };
        self.backend.dispatch(hdr.cmd, req, &mut reply, region, waits)?;
        // A command with no answer may come before the region exists; it simply has nowhere to
        // say it finished.
        let Some(region) = region else { return Ok(()) };
        if let Some(staged) = reply.staged {
            region.publish(hdr.rsp_off, &staged);
        }
        region.finished(hdr.seqno);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A backend with two commands: 1 answers with eight bytes of 0x11, 2 fails.
    struct Echo;

    impl Backend for Echo {
        fn alignment(&self) -> usize {
            8
        }
        fn header_len(&self) -> usize {
            16
        }
        fn region_attached(&mut self, region: &ResponseRegion) {
            region.store_header_u32(8, 0xbeef);
        }
        fn dispatch(
            &mut self,
            cmd: u32,
            _req: &[u8],
            reply: &mut Reply<'_>,
            _region: Option<&ResponseRegion>,
            _waits: &mut Waits,
        ) -> Result<(), Refused> {
            match cmd {
                1 => {
                    reply.stage(8)?[4..].fill(0x11);
                    Ok(())
                }
                _ => Err(Refused::UnknownCommand(cmd)),
            }
        }
        fn get_blob(
            &mut self,
            _handle: ResourceHandle,
            blob: NativeBlob,
            _size: u64,
            _flags: u32,
            _region: Option<&ResponseRegion>,
        ) -> Result<Buffer, Refused> {
            Err(Refused::NoSuchBlob(blob.get()))
        }
        fn attach(&mut self, _handle: ResourceHandle, _source: Source<'_>) -> Result<(), Refused> {
            Ok(())
        }
        fn forget(&mut self, _handle: ResourceHandle) {}
        fn rings(&self) -> u32 {
            0
        }
        fn take_last_submit(&mut self, _ring: RingIdx) -> Option<OwnedFd> {
            None
        }
    }

    fn req(cmd: u32, seqno: u32, rsp_off: u32) -> Vec<u8> {
        bytemuck::bytes_of(&ReqHeader { cmd, len: 16, seqno, rsp_off }).to_vec()
    }

    fn context() -> (Arc<GuestMap>, Context) {
        let map = Arc::new(GuestMap::anonymous(4096).unwrap());
        let mut c = Context::new("t".into(), Box::new(Echo), None);
        c.attach_region(Arc::clone(&map), 4096, BLOB_FLAG_USE_MAPPABLE).unwrap();
        (map, c)
    }

    #[test]
    fn a_request_publishes_its_answer_then_its_seqno() {
        let (map, mut c) = context();
        assert!(map.store_u32(16 + 8, 64));
        c.submit(&req(1, 7, 8)).unwrap();
        assert_eq!(map.load_u32(proto::shmem::SEQNO), Some(7));
        assert_eq!(map.load_u32(16 + 8), Some(8));
        assert_eq!(map.load_u32(16 + 12), Some(0x1111_1111));
        assert_eq!(map.load_u32(8), Some(0xbeef), "the backend filled its header at attach");
    }

    #[test]
    fn a_failed_request_ends_the_submission_and_publishes_nothing() {
        let (map, mut c) = context();
        let mut buf = req(1, 1, 0);
        buf.extend(req(2, 2, 0));
        buf.extend(req(1, 3, 0));
        assert_eq!(c.submit(&buf), Err(Refused::UnknownCommand(2)));
        assert_eq!(map.load_u32(proto::shmem::SEQNO), Some(1));
    }

    #[test]
    fn framing_is_checked_before_anything_runs() {
        let (map, mut c) = context();
        let mut bad = req(1, 9, 0);
        bad[4..8].copy_from_slice(&12u32.to_le_bytes());
        assert_eq!(c.submit(&bad), Err(Refused::RequestLength(12)));
        assert_eq!(c.submit(&req(1, 9, 4)), Err(Refused::ResponseOffset(4)));
        let mut long = req(1, 9, 0);
        long[4..8].copy_from_slice(&32u32.to_le_bytes());
        assert_eq!(c.submit(&long), Err(Refused::RequestLength(32)));
        let mut trailing = req(1, 9, 0);
        trailing.extend([0u8; 8]);
        assert_eq!(c.submit(&trailing), Err(Refused::Trailing(8)));
        assert_eq!(map.load_u32(proto::shmem::SEQNO), Some(9), "the first request still ran");
    }

    #[test]
    fn an_answer_outside_the_region_refuses_the_request() {
        let (map, mut c) = context();
        assert_eq!(
            c.submit(&req(1, 5, 4096)),
            Err(Refused::ResponseOutOfBounds { off: 4096, len: 8 })
        );
        assert_eq!(map.load_u32(proto::shmem::SEQNO), Some(0));
    }

    #[test]
    fn the_region_is_attached_once_and_mappable_only() {
        let map = Arc::new(GuestMap::anonymous(4096).unwrap());
        let mut c = Context::new("t".into(), Box::new(Echo), None);
        assert_eq!(c.attach_region(Arc::clone(&map), 4096, 3), Err(Refused::RegionFlags(3)));
        c.attach_region(Arc::clone(&map), 4096, BLOB_FLAG_USE_MAPPABLE).unwrap();
        assert_eq!(c.attach_region(map, 4096, BLOB_FLAG_USE_MAPPABLE), Err(Refused::SecondRegion));
    }

    /// Waits that would each take the whole bound share it: later ones are cut short, and once it
    /// is spent a wait is a poll.
    #[test]
    fn a_submission_blocks_for_the_bound_however_many_waits_it_holds() {
        let mut waits = Waits::new();
        let step = Waits::BOUND * 2 / 5;
        let granted: Vec<Duration> = (0..4)
            .map(|_| {
                waits.bounded(Duration::MAX, |timeout| {
                    std::thread::sleep(timeout.min(step));
                    timeout
                })
            })
            .collect();
        assert_eq!(granted[0], Waits::BOUND);
        assert!(granted[1] <= Waits::BOUND - step, "{granted:?}");
        assert!(granted[2] <= Waits::BOUND - 2 * step, "{granted:?}");
        assert_eq!(granted[3], Duration::ZERO, "{granted:?}");
    }

    #[test]
    fn a_blob_id_no_guest_makes_is_refused_before_the_backend() {
        let (_map, mut c) = context();
        let h = ResourceHandle::new(1).unwrap();
        assert_eq!(c.get_blob(h, BlobId(0), 4096, 0).err(), Some(Refused::BlobId(0)));
        assert_eq!(c.get_blob(h, BlobId(1 << 32), 4096, 0).err(), Some(Refused::BlobId(1 << 32)));
        assert_eq!(c.get_blob(h, BlobId(5), 4096, 0).err(), Some(Refused::NoSuchBlob(5)));
    }

    #[test]
    fn ring_zero_is_the_callers_and_a_ring_past_the_last_is_refused() {
        let (_map, mut c) = context();
        assert_eq!(c.submit_fence(RingIdx(0), FenceId(1)), Ok(false));
        assert_eq!(c.submit_fence(RingIdx(1), FenceId(2)), Err(Refused::Ring(1)));
    }

    #[test]
    fn an_answer_with_no_region_refuses_the_request() {
        let mut c = Context::new("t".into(), Box::new(Echo), None);
        assert_eq!(c.submit(&req(1, 1, 0)), Err(Refused::NoRegion));
    }
}
