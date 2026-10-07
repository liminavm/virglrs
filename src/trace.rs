// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! `LIMINA_VREND_TRACE`: an in-memory recording of everything the classic renderer is asked to do,
//! in the format `harness/replay/vrend-replay` replays and `vrend-trace-decode.py` reads.
//!
//! It is the C tree's `vrend_trace` in the same bytes, so a capture from either leg feeds the same
//! tools. What it is for: a failure seen only in a guest becomes a stream the host replays in
//! seconds, against either renderer, without a VM.
//!
//! **It buffers and writes only when asked.** Every hook is an append into a ring allocated and
//! touched when the recorder is armed; nothing allocates per record beyond a transfer's or a blob's
//! own bytes, and nothing is written to a file until a dump is requested -- by a byte written to
//! the FIFO, or by the renderer's teardown. A recorder that wrote as it went would move the batch
//! boundaries it exists to observe.
//!
//! **Built only with the `trace` feature.** Without it the recorder has no state, every hook is an
//! empty inline call, and an armed environment is told so at startup rather than producing no
//! file. The ring itself is compiled either way, so its tests run in both configurations.
//!
//! What a replay needs, and so what this records: every command the guest sent that the renderer
//! accepted, the bytes each guest-to-host transfer carried, the pixels behind a blob a texture
//! copies, and every resource create and unref. Batch boundaries, transfer metadata, fences and
//! the bound target's size at each draw are recorded for the decoder.

use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::ids::{ContextId, ResourceHandle};

/// The dump's magic, "LMVT".
const MAGIC: u32 = 0x4c4d_5654;
/// The format version `vrend-replay` requires.
const VERSION: u32 = 2;
/// A record header's bytes: total length, type, command, context, sequence, time, payload length
/// and auxiliary word count.
const HEADER: usize = 32;
/// A resource event's bytes: a sequence and twelve words.
const RES_EVENT: usize = 56;
/// The resource log starts this big and doubles. Creates are never evicted, so it grows; the
/// ceiling only stops a runaway guest eating the host, and reaching it is said.
const RES_INITIAL: usize = 32_768;
const RES_MAX: usize = 4_194_304;
/// Blobs whose content is deduplicated by hash. A full table stops deduplicating rather than
/// recording one blob under another's hash.
const BLOB_SLOTS: usize = 64;

/// Record types, by the numbers the readers know them by.
#[derive(Clone, Copy)]
#[repr(u8)]
enum Kind {
    Submit = 1,
    Command = 2,
    DrawFb = 3,
    Transfer = 4,
    Fence = 5,
    Retire = 6,
    Pad = 7,
    TransferData = 9,
    BlobData = 10,
}

/// Which way a recorded transfer went, as the readers number it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Direction {
    ToHost = 1,
    FromHost = 2,
}

/// A transfer's box and where in the pages it starts, for the decoder.
#[derive(Clone, Copy, Debug)]
pub struct TransferBox {
    pub level: u32,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub offset: u64,
}

/// Where a blob's memory is, as the readers number it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlobMem {
    Guest = 1,
    Host3d = 2,
}

/// A resource arriving or leaving, in wire values: these happen on the control path, outside the
/// command stream, so a replay builds every resource from them.
#[derive(Clone, Copy, Debug)]
pub enum ResEvent {
    Create {
        handle: ResourceHandle,
        target: u32,
        format: u32,
        bind: u32,
        width: u32,
        height: u32,
        depth: u32,
        array_size: u32,
        last_level: u32,
        nr_samples: u32,
        flags: u32,
    },
    Blob {
        handle: ResourceHandle,
        size: u64,
        blob_mem: BlobMem,
        blob_flags: u32,
        blob_id: u64,
        ctx: Option<ContextId>,
    },
    Unref(ResourceHandle),
}

impl ResEvent {
    /// The C's `struct vrend_trace_res`, stamped with `seq`.
    fn encode(&self, seq: u64) -> [u8; RES_EVENT] {
        let lo = |v: u64| v as u32;
        let hi = |v: u64| (v >> 32) as u32;
        // kind, handle, target, format, bind, width, height, depth, array_size, last_level,
        // nr_samples, flags. A blob reuses the slots as the C does: its size across width and
        // height, its id across depth and array_size, and its context in flags.
        let words: [u32; 12] = match *self {
            ResEvent::Create {
                handle,
                target,
                format,
                bind,
                width,
                height,
                depth,
                array_size,
                last_level,
                nr_samples,
                flags,
            } => [
                0,
                handle.get(),
                target,
                format,
                bind,
                width,
                height,
                depth,
                array_size,
                last_level,
                nr_samples,
                flags,
            ],
            ResEvent::Blob { handle, size, blob_mem, blob_flags, blob_id, ctx } => [
                1,
                handle.get(),
                0,
                blob_mem as u32,
                blob_flags,
                lo(size),
                hi(size),
                lo(blob_id),
                hi(blob_id),
                0,
                0,
                ctx.map_or(0, ContextId::get),
            ],
            ResEvent::Unref(handle) => [2, handle.get(), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        };
        let mut out = [0u8; RES_EVENT];
        out[..8].copy_from_slice(&seq.to_le_bytes());
        for (i, w) in words.iter().enumerate() {
            out[8 + i * 4..12 + i * 4].copy_from_slice(&w.to_le_bytes());
        }
        out
    }
}

/// The recorder every hook calls. Cheap to clone: armed, the clones share one ring; built without
/// the `trace` feature, it holds nothing and every call is empty.
#[derive(Clone, Default)]
pub struct Recorder {
    #[cfg(feature = "trace")]
    ring: Option<std::sync::Arc<Ring>>,
}

impl std::fmt::Debug for Recorder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Recorder({})", if self.ring().is_some() { "armed" } else { "off" })
    }
}

impl Recorder {
    /// Arm from limina's variables: `LIMINA_VREND_TRACE=<MB>` sizes the ring, and
    /// `LIMINA_VREND_TRACE_OUT`, `_FIFO` and `_BLOB_MAX` say where it is written, what asks for a
    /// dump, and how many distinct frames of one blob it keeps. Unset or zero is off.
    pub fn from_env() -> Recorder {
        let mb = std::env::var("LIMINA_VREND_TRACE")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(0);
        Recorder::arm(mb)
    }

    #[cfg(not(feature = "trace"))]
    fn arm(mb: usize) -> Recorder {
        // Said rather than left silent: no file after an armed run reads as a broken recorder.
        if mb != 0 {
            eprintln!(
                "[virglrs] LIMINA_VREND_TRACE is set but this build has no `trace` feature; \
                 nothing is recorded"
            );
        }
        Recorder::default()
    }

    #[cfg(feature = "trace")]
    fn arm(mb: usize) -> Recorder {
        if mb == 0 {
            return Recorder::default();
        }
        let var = |name: &str, default: &str| {
            PathBuf::from(std::env::var(name).unwrap_or_else(|_| default.to_string()))
        };
        let out = var("LIMINA_VREND_TRACE_OUT", "/tmp/limina-vrend-trace.bin");
        let fifo = var("LIMINA_VREND_TRACE_FIFO", "/tmp/limina-vrend-trace.fifo");
        let blob_max = std::env::var("LIMINA_VREND_TRACE_BLOB_MAX")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(4);
        let ring = std::sync::Arc::new(Ring::new(mb * 1024 * 1024, Some(out.clone()), blob_max));
        let asks = fifo::listen(&fifo, std::sync::Arc::downgrade(&ring));
        eprintln!(
            "[virglrs] vrend command tracer ARMED: {mb} MB ring, blob content up to {blob_max} \
             frame(s) per blob, dump {} -> {}",
            match asks {
                Ok(()) => format!("on `echo x > {}`", fifo.display()),
                Err(e) => format!("at teardown only (no FIFO at {}: {e})", fifo.display()),
            },
            out.display()
        );
        Recorder { ring: Some(ring) }
    }

    fn ring(&self) -> Option<&Ring> {
        #[cfg(feature = "trace")]
        return self.ring.as_deref();
        #[cfg(not(feature = "trace"))]
        None
    }

    /// A batch of `bytes` begins on `ctx`.
    pub fn submit(&self, ctx: ContextId, bytes: usize) {
        if let Some(r) = self.ring() {
            r.put(Kind::Submit, 0, Some(ctx), &[bytes as u32], &[]);
        }
    }

    /// One command `ctx` sent and the renderer accepted, header word first.
    pub fn command(&self, ctx: ContextId, wire: &[u32]) {
        if let Some(r) = self.ring() {
            let cmd = wire.first().map_or(0, |h| (h & 0xff) as u8);
            r.put(Kind::Command, cmd, Some(ctx), &[], bytemuck::cast_slice(wire));
        }
    }

    /// The target a draw on `ctx` renders into: its first colour buffer's size, how many colour
    /// buffers are bound, and that buffer's GL name.
    pub fn draw_fb(&self, ctx: ContextId, width: u32, height: u32, cbufs: u32, gl_name: u32) {
        if let Some(r) = self.ring() {
            r.put(Kind::DrawFb, 0, Some(ctx), &[width, height, cbufs, gl_name], &[]);
        }
    }

    /// A transfer, which never appears in the command stream when it arrives on the control
    /// path. `ctx` is `None` for one the VMM made with no context.
    pub fn transfer(
        &self,
        ctx: Option<ContextId>,
        handle: ResourceHandle,
        direction: Direction,
        b: TransferBox,
    ) {
        if let Some(r) = self.ring() {
            let aux =
                [handle.get(), direction as u32, b.level, b.x, b.y, b.width, b.height, b.stride];
            r.put(Kind::Transfer, 0, ctx, &aux, &b.offset.to_le_bytes());
        }
    }

    /// The bytes a guest-to-host transfer read, from `offset` in its source pages. `bytes` copies
    /// them and is called only when armed, so a call costs nothing otherwise. Recorded under the
    /// destination's handle, which is how the replayer finds where to put them.
    pub fn transfer_bytes(
        &self,
        ctx: Option<ContextId>,
        handle: ResourceHandle,
        offset: u64,
        bytes: impl FnOnce() -> Option<Vec<u8>>,
    ) {
        let Some(r) = self.ring() else { return };
        let Some(bytes) = bytes().filter(|b| !b.is_empty()) else { return };
        let aux = [handle.get(), offset as u32, (offset >> 32) as u32];
        r.put(Kind::TransferData, 0, ctx, &aux, &bytes);
    }

    /// The bytes behind blob `handle` that a texture is about to copy, from its offset zero;
    /// `bytes` copies them, and is called only when armed. A blob's pixels are written GPU-side
    /// and never travel as a transfer, so without these a replay has its shape and none of its
    /// content. A frame equal to the last one recorded for the blob is skipped, and past the
    /// per-blob cap a changed frame is dropped and counted.
    pub fn blob_bytes(&self, handle: ResourceHandle, bytes: impl FnOnce() -> Option<Vec<u8>>) {
        let Some(r) = self.ring() else { return };
        if r.blob_max == 0 {
            return;
        }
        let Some(bytes) = bytes().filter(|b| !b.is_empty()) else { return };
        r.put_blob(handle, &bytes);
    }

    /// Whether anything is recorded: for a hook whose arguments cost a lookup to work out.
    pub fn armed(&self) -> bool {
        self.ring().is_some()
    }

    /// A fence taken on `ctx`, or on none for a global one.
    pub fn fence(&self, ctx: Option<ContextId>, id: u64) {
        if let Some(r) = self.ring() {
            r.put(Kind::Fence, 0, ctx, &[0], &id.to_le_bytes());
        }
    }

    /// A fence handed back to the VMM.
    pub fn retire(&self, ctx: Option<ContextId>, id: u64) {
        if let Some(r) = self.ring() {
            r.put(Kind::Retire, 0, ctx, &[], &id.to_le_bytes());
        }
    }

    /// A resource arriving or leaving.
    pub fn resource(&self, ev: ResEvent) {
        if let Some(r) = self.ring() {
            r.put_resource(ev);
        }
    }
}

/// The ring and the resource log, under one lock: records arrive from the renderer's thread and
/// retirements from the fence thread, and a sequence number is only meaningful if one writer
/// assigns it at a time.
#[cfg_attr(not(feature = "trace"), allow(dead_code))]
struct Ring {
    state: Mutex<State>,
    out: Option<PathBuf>,
    blob_max: u32,
}

#[cfg_attr(not(feature = "trace"), allow(dead_code))]
struct State {
    buf: Vec<u8>,
    /// Where the next record goes, the oldest live record, and the bytes between them. A record
    /// is never split across the end: a pad record fills the tail end instead, so a reader walks
    /// forward from `tail` by each record's length.
    head: usize,
    tail: usize,
    used: usize,
    seq: u64,
    /// Records dropped to make room: nonzero means the dump no longer reaches back to the start.
    evicted: u64,
    /// Records larger than half the ring, refused whole.
    refused: u64,
    base: Instant,
    base_realtime_ns: u64,
    /// Resource events, encoded, in arrival order. A log and not a table: handles are reused.
    res: Vec<[u8; RES_EVENT]>,
    res_full: bool,
    blobs: [BlobSeen; BLOB_SLOTS],
}

#[derive(Clone, Copy, Default)]
struct BlobSeen {
    handle: Option<ResourceHandle>,
    hash: u64,
    kept: u32,
    dropped: u32,
}

#[cfg_attr(not(feature = "trace"), allow(dead_code))]
impl Ring {
    /// A ring of `bytes`, every page touched now: a first-touch fault inside a hook would be the
    /// timing perturbation the recorder exists to avoid.
    fn new(bytes: usize, out: Option<PathBuf>, blob_max: u32) -> Ring {
        let mut buf = vec![0u8; bytes];
        for page in buf.chunks_mut(4096) {
            page[0] = std::hint::black_box(0);
        }
        let base_realtime_ns =
            SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64);
        Ring {
            state: Mutex::new(State {
                buf,
                head: 0,
                tail: 0,
                used: 0,
                seq: 0,
                evicted: 0,
                refused: 0,
                base: Instant::now(),
                base_realtime_ns,
                res: Vec::with_capacity(RES_INITIAL),
                res_full: false,
                blobs: [BlobSeen::default(); BLOB_SLOTS],
            }),
            out,
            blob_max,
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().expect("the trace lock is never held across a panic")
    }

    fn put(&self, kind: Kind, cmd: u8, ctx: Option<ContextId>, aux: &[u32], payload: &[u8]) {
        self.lock().put(kind, cmd, ctx, aux, payload);
    }

    fn put_resource(&self, ev: ResEvent) {
        let mut s = self.lock();
        if s.res.len() >= RES_MAX {
            if !s.res_full {
                s.res_full = true;
                eprintln!(
                    "[virglrs] trace: resource log full at {RES_MAX} entries; the trace is NOT \
                     replayable"
                );
            }
            return;
        }
        // Stamped with the sequence the NEXT record takes: the replayer applies an event after
        // the record whose sequence reaches it.
        let seq = s.seq;
        s.res.push(ev.encode(seq));
    }

    fn put_blob(&self, handle: ResourceHandle, bytes: &[u8]) {
        let hash = bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
            (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3)
        });
        let mut s = self.lock();
        let start = handle.get() as usize % BLOB_SLOTS;
        for i in 0..BLOB_SLOTS {
            let slot = &mut s.blobs[(start + i) % BLOB_SLOTS];
            match slot.handle {
                Some(h) if h == handle => {
                    if slot.hash == hash {
                        return;
                    }
                    slot.hash = hash;
                    if slot.kept >= self.blob_max {
                        slot.dropped += 1;
                        return;
                    }
                    slot.kept += 1;
                    break;
                }
                None => {
                    *slot = BlobSeen { handle: Some(handle), hash, kept: 1, dropped: 0 };
                    break;
                }
                Some(_) => {}
            }
        }
        s.put(Kind::BlobData, 0, None, &[handle.get()], bytes);
    }

    /// Write the dump: a header, the resource log, then the ring from its oldest record.
    fn dump(&self) -> std::io::Result<()> {
        let Some(out) = &self.out else { return Ok(()) };
        let s = self.lock();
        let bytes = s.dump_bytes();
        std::fs::write(out, &bytes)?;
        eprintln!(
            "[virglrs] trace: dumped {} bytes, {} records, {} evicted -> {}",
            s.used,
            s.seq,
            s.evicted,
            out.display()
        );
        if s.refused != 0 {
            eprintln!(
                "[virglrs] trace: {} record(s) larger than half the ring were refused; raise \
                 LIMINA_VREND_TRACE to keep them",
                s.refused
            );
        }
        let (dropped, capped) = s
            .blobs
            .iter()
            .filter(|b| b.dropped != 0)
            .fold((0, 0), |(d, c), b| (d + b.dropped, c + 1));
        if dropped != 0 {
            eprintln!(
                "[virglrs] trace: blob content: {dropped} changed frame(s) dropped past the cap \
                 of {} across {capped} blob(s); raise LIMINA_VREND_TRACE_BLOB_MAX to keep more",
                self.blob_max
            );
        }
        Ok(())
    }
}

#[cfg(feature = "trace")]
impl Drop for Ring {
    /// The last recorder going is the renderer's teardown: what was recorded is written then, as
    /// the C writes it at exit.
    fn drop(&mut self) {
        if let Err(e) = self.dump() {
            eprintln!("[virglrs] trace: dump failed: {e}");
        }
    }
}

#[cfg_attr(not(feature = "trace"), allow(dead_code))]
impl State {
    fn put(&mut self, kind: Kind, cmd: u8, ctx: Option<ContextId>, aux: &[u32], payload: &[u8]) {
        let cap = self.buf.len();
        let mut need = (HEADER + aux.len() * 4 + payload.len()).next_multiple_of(8);
        if need > cap / 2 {
            self.refused += 1;
            return;
        }
        // A sliver smaller than a header at the end would be unwalkable, so it is absorbed into
        // this record: at entry the room left at the end is always at least a header.
        let tail_room = cap - self.head;
        if tail_room >= need && tail_room - need < HEADER {
            need = tail_room;
        }
        if tail_room < need {
            while self.used + tail_room > cap {
                self.evict_one();
            }
            self.write_header(self.head, tail_room, Kind::Pad, 0, 0, 0, 0, 0, 0);
            self.used += tail_room;
            self.head = 0;
        }
        while self.used + need > cap {
            self.evict_one();
        }
        let at = self.head;
        let mono = self.base.elapsed().as_nanos() as u64;
        let ctx = ctx.map_or(0, |c| c.get() as u16);
        let seq = self.seq;
        self.write_header(at, need, kind, cmd, ctx, seq, mono, payload.len(), aux.len());
        let mut p = at + HEADER;
        for w in aux {
            self.buf[p..p + 4].copy_from_slice(&w.to_le_bytes());
            p += 4;
        }
        self.buf[p..p + payload.len()].copy_from_slice(payload);
        p += payload.len();
        self.buf[p..at + need].fill(0);
        self.seq += 1;
        self.head += need;
        if self.head >= cap {
            self.head = 0;
        }
        self.used += need;
    }

    #[allow(clippy::too_many_arguments)]
    fn write_header(
        &mut self,
        at: usize,
        total: usize,
        kind: Kind,
        cmd: u8,
        ctx: u16,
        seq: u64,
        mono: u64,
        payload: usize,
        aux: usize,
    ) {
        let h = &mut self.buf[at..at + HEADER];
        h[0..4].copy_from_slice(&(total as u32).to_le_bytes());
        h[4] = kind as u8;
        h[5] = cmd;
        h[6..8].copy_from_slice(&ctx.to_le_bytes());
        h[8..16].copy_from_slice(&seq.to_le_bytes());
        h[16..24].copy_from_slice(&mono.to_le_bytes());
        h[24..28].copy_from_slice(&(payload as u32).to_le_bytes());
        h[28..32].copy_from_slice(&(aux as u32).to_le_bytes());
    }

    fn evict_one(&mut self) {
        let t = self.tail;
        let total = u32::from_le_bytes(self.buf[t..t + 4].try_into().expect("four bytes")) as usize;
        assert!(total >= HEADER, "a live record at the tail is at least a header");
        let pad = self.buf[t + 4] == Kind::Pad as u8;
        self.tail += total;
        if self.tail >= self.buf.len() {
            self.tail = 0;
        }
        self.used -= total;
        if !pad {
            self.evicted += 1;
        }
    }

    fn dump_bytes(&self) -> Vec<u8> {
        let cap = self.buf.len();
        let split = |v: u64| [v as u32, (v >> 32) as u32];
        let mut hdr = [0u32; 16];
        hdr[0] = MAGIC;
        hdr[1] = VERSION;
        hdr[2] = (cap / (1024 * 1024)) as u32;
        hdr[3] = self.used as u32;
        hdr[4..6].copy_from_slice(&split(self.seq));
        hdr[6..8].copy_from_slice(&split(self.evicted));
        // Record times count from the arming, so the monotonic base is zero and the wall-clock
        // one lines the trace up against a log.
        hdr[10..12].copy_from_slice(&split(self.base_realtime_ns));
        hdr[12] = self.res.len() as u32;
        hdr[13] = self.res_full as u32;
        let mut out = Vec::with_capacity(64 + self.res.len() * RES_EVENT + self.used);
        hdr.iter().for_each(|w| out.extend_from_slice(&w.to_le_bytes()));
        self.res.iter().for_each(|e| out.extend_from_slice(e));
        let (mut pos, mut left) = (self.tail, self.used);
        while left > 0 {
            let chunk = (cap - pos).min(left);
            out.extend_from_slice(&self.buf[pos..pos + chunk]);
            left -= chunk;
            pos += chunk;
            if pos >= cap {
                pos = 0;
            }
        }
        out
    }
}

/// The FIFO a dump is asked for through: a thread blocks on it, so asking costs the renderer
/// nothing until it happens, and a dump requested while the guest is idle still happens.
#[cfg(feature = "trace")]
mod fifo {
    use std::io::Read;
    use std::path::Path;
    use std::sync::Weak;

    pub(super) fn listen(path: &Path, ring: Weak<super::Ring>) -> std::io::Result<()> {
        match std::fs::remove_file(path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }
        nix::unistd::mkfifo(path, nix::sys::stat::Mode::from_bits_truncate(0o600))
            .map_err(std::io::Error::from)?;
        let path = path.to_path_buf();
        std::thread::Builder::new().name("virglrs-trace".into()).spawn(move || {
            loop {
                // Opening blocks until a writer comes; each byte written is one dump. Held weakly:
                // the renderer's teardown is the last owner, and its dump is the final one.
                let Ok(mut f) = std::fs::File::open(&path) else {
                    std::thread::sleep(std::time::Duration::from_secs(1));
                    continue;
                };
                let mut byte = [0u8; 1];
                while matches!(f.read(&mut byte), Ok(1)) {
                    let Some(ring) = ring.upgrade() else { return };
                    if let Err(e) = ring.dump() {
                        eprintln!("[virglrs] trace: dump failed: {e}");
                    }
                }
            }
        })?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn handle(n: u32) -> ResourceHandle {
        ResourceHandle::new(n).expect("non-zero")
    }

    fn ctx(n: u32) -> ContextId {
        ContextId::new(n).expect("non-zero")
    }

    /// The records of a dump, as (type, cmd, ctx, seq, aux, payload).
    #[allow(clippy::type_complexity)]
    fn records(dump: &[u8]) -> (Vec<(u64, u32)>, Vec<(u8, u8, u16, u64, Vec<u32>, Vec<u8>)>) {
        let w = |at: usize| u32::from_le_bytes(dump[at..at + 4].try_into().unwrap());
        let q = |at: usize| u64::from_le_bytes(dump[at..at + 8].try_into().unwrap());
        assert_eq!(w(0), MAGIC);
        assert_eq!(w(4), VERSION);
        let used = w(12) as usize;
        let res_n = w(48) as usize;
        let res = (0..res_n).map(|i| (q(64 + i * RES_EVENT), w(64 + i * RES_EVENT + 8))).collect();
        let mut p = 64 + res_n * RES_EVENT;
        let end = p + used;
        assert_eq!(end, dump.len(), "the ring is the rest of the dump");
        let mut recs = Vec::new();
        while p < end {
            let total = w(p) as usize;
            let (kind, cmd) = (dump[p + 4], dump[p + 5]);
            let c = u16::from_le_bytes([dump[p + 6], dump[p + 7]]);
            let (seq, plen, aux_n) = (q(p + 8), w(p + 24) as usize, w(p + 28) as usize);
            let aux = (0..aux_n).map(|i| w(p + HEADER + i * 4)).collect();
            let at = p + HEADER + aux_n * 4;
            if kind != Kind::Pad as u8 {
                recs.push((kind, cmd, c, seq, aux, dump[at..at + plen].to_vec()));
            }
            p += total;
        }
        (res, recs)
    }

    fn armed(bytes: usize) -> Ring {
        Ring::new(bytes, None, 2)
    }

    /// A dump reads back as the records put into it, in order, in the C's layout: a command keeps
    /// its header word and its opcode, a transfer its box and offset, and a resource event is
    /// stamped with the sequence of the record that follows it, which is where the replayer
    /// applies it.
    #[test]
    fn a_dump_reads_back_its_records_in_the_c_layout() {
        let ring = armed(1 << 16);
        ring.put(Kind::Submit, 0, Some(ctx(3)), &[12], &[]);
        ring.put_resource(ResEvent::Unref(handle(9)));
        let wire = [(2u32 << 16) | 4, 7, 8];
        ring.put(Kind::Command, 4, Some(ctx(3)), &[], bytemuck::cast_slice(&wire));
        let (res, recs) = records(&ring.lock().dump_bytes());
        assert_eq!(res, [(1, 2)], "the unref is applied after the record with sequence 1");
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0], (1, 0, 3, 0, vec![12], vec![]));
        assert_eq!(recs[1].0, 2);
        assert_eq!((recs[1].1, recs[1].2, recs[1].3), (4, 3, 1));
        assert_eq!(recs[1].5, bytemuck::cast_slice::<u32, u8>(&wire));
    }

    /// Past its capacity the ring drops its oldest records, counts them, and still dumps a stream
    /// a reader can walk from start to end -- across the wrap and the pad that fills it.
    #[test]
    fn a_full_ring_evicts_its_oldest_records_and_stays_walkable() {
        let ring = armed(1024);
        for i in 0..100u32 {
            ring.put(Kind::Command, 1, Some(ctx(1)), &[], &vec![i as u8; (i % 7) as usize * 8]);
        }
        let s = ring.lock();
        assert!(s.evicted > 0, "a 1 KiB ring cannot hold 100 records");
        let (_, recs) = records(&s.dump_bytes());
        let seqs: Vec<u64> = recs.iter().map(|r| r.3).collect();
        assert_eq!(*seqs.last().unwrap(), 99, "the newest record survives");
        assert!(seqs.windows(2).all(|w| w[1] == w[0] + 1), "survivors are contiguous: {seqs:?}");
        assert_eq!(s.evicted + recs.len() as u64, 100, "every record is kept or counted");
    }

    /// A record larger than half the ring is refused whole rather than evicting everything.
    #[test]
    fn a_record_past_half_the_ring_is_refused() {
        let ring = armed(1024);
        ring.put(Kind::Submit, 0, None, &[1], &[]);
        ring.put(Kind::TransferData, 0, None, &[1, 0, 0], &[0; 600]);
        let s = ring.lock();
        assert_eq!((s.refused, s.evicted, s.seq), (1, 0, 1));
    }

    /// A blob's frame is recorded when it changes, not when it is read again unchanged, and past
    /// the cap a changed frame is counted rather than kept.
    #[test]
    fn blob_content_is_kept_when_it_changes_up_to_the_cap() {
        let ring = armed(1 << 16);
        let h = handle(5);
        ring.put_blob(h, &[1; 16]);
        ring.put_blob(h, &[1; 16]);
        ring.put_blob(h, &[2; 16]);
        ring.put_blob(h, &[3; 16]);
        let s = ring.lock();
        let (_, recs) = records(&s.dump_bytes());
        let frames: Vec<u8> = recs.iter().map(|r| r.5[0]).collect();
        assert_eq!(frames, [1, 2], "the repeat is skipped and the third change is past the cap");
        assert_eq!(s.blobs.iter().map(|b| b.dropped).sum::<u32>(), 1);
    }

    /// Built without the feature, or not armed, the recorder records nothing and costs nothing.
    #[test]
    fn an_unarmed_recorder_records_nothing() {
        let r = Recorder::default();
        let mut called = false;
        r.transfer_bytes(None, handle(1), 0, || {
            called = true;
            Some(vec![0; 16])
        });
        assert!(!called, "the bytes are not even copied");
        assert!(r.ring().is_none());
    }
}
