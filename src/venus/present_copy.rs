//! Presenting a copy of a venus scanout, taken in order with the guest's own work.
//!
//! A guest whose scanout flushes carry no fence is not held off the buffers it flips: its flip
//! completes on its own vblank timer, and a double-buffering compositor starts drawing its next
//! frame into the buffer the flip released. Showing that buffer, or a copy of it made whenever
//! the display gets round to it, shows whatever the guest has drawn into it since. So the copy is
//! made here, on the context's queue, behind the frame's own work and ahead of anything the guest
//! submits after it, and the display is handed the copy.
//!
//! **What orders it.** Queue submission order alone does not order execution. On KosmicKrisp
//! every encoder it closes ends with an all-stages queue barrier (`kk_stop_encoder`), which is
//! what makes a later submit on the same queue wait for this copy's reads; the barrier recorded
//! after the copy says the same thing in Vulkan's terms. Neither reaches another queue, so a
//! context with more than one is not copied here.
//!
//! **What it cannot order.** The flush reaches the renderer on the VMM's virtio-gpu thread and
//! the guest's commands through its ring, on another. The copy is ordered ahead of what the ring
//! has not yet submitted when it goes on the queue, not ahead of what it already has: a host
//! that handles a flush more than a frame late can still copy a buffer the guest has drawn into
//! again. Only a guest held on a flush fence closes that.
//!
//! **Where it lands.** A small ring of surfaces per scanout, minted like the scanout and
//! published like any other. A slot is not reused while its copy is pending, while it is one of the two most
//! recently handed out, or while any process holds it in use -- the window server does, for a
//! surface it is compositing.
//!
//! Two seams: `LIMINA_TEST_PRESENT_COPY_DELAY_MS` holds each copy back before it is submitted,
//! which lets the guest's later submits land first -- the case above this cannot order -- and
//! `LIMINA_PRESENT_COPY_TRACE` logs the first pixel of the guest's scanout as the flush arrives
//! and of the copy once it is done.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::ids::{ScanoutId, SurfaceId};
use crate::metal::Surface;
use crate::venus::driver::Storage;
use crate::venus::driver::copier::Import;

/// Slots one scanout's ring may grow to before a present is refused a copy.
const MAX_SLOTS: usize = 6;

/// The two test knobs, read when a context is made.
#[derive(Clone, Copy, Default)]
pub struct Knobs {
    /// `LIMINA_TEST_PRESENT_COPY_DELAY_MS`: how long to hold each copy back before submitting it.
    pub delay: Option<Duration>,
    /// `LIMINA_PRESENT_COPY_TRACE`.
    pub trace: bool,
}

impl Knobs {
    pub fn from_env() -> Knobs {
        let delay = std::env::var("LIMINA_TEST_PRESENT_COPY_DELAY_MS")
            .ok()
            .and_then(|ms| ms.parse::<u64>().ok())
            .map(Duration::from_millis);
        if let Some(d) = delay {
            eprintln!("[virglrs] TEST SEAM: every present copy is held back {d:?} before submit");
        }
        Knobs { delay, trace: std::env::var_os("LIMINA_PRESENT_COPY_TRACE").is_some() }
    }
}

/// Why a present got no copy.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CopyRefused {
    /// Every slot is still pending or on glass. Passing: the host is behind, and the frame is
    /// better dropped than shown some other way.
    Busy,
    /// No copy can be ordered here at all: not a surface, not one queue, or nowhere to mint.
    NotOrderable,
}

/// A surface's first pixel, as its four bytes in memory order.
fn first_pixel(surface: &Surface) -> u32 {
    let mut px = [0u8; 4];
    surface.read_into(&mut px);
    u32::from_be_bytes(px)
}

/// One surface of a ring, and everything that exists only because of it.
pub(super) struct Target {
    /// The Vulkan buffer over the surface's pages, made on the first copy into it. Declared
    /// before the surface so that it goes first: it must never outlive the pages it names.
    pub(super) import: Mutex<Option<Import>>,
    pub(super) surface: Surface,
    /// Handed out and not yet copied into.
    pending: AtomicBool,
}

impl Target {
    fn shape(&self) -> (u32, u32, u32) {
        (self.surface.width(), self.surface.height(), self.surface.bytes_per_row())
    }
}

/// One present's copy: from the guest's scanout, into a slot.
pub struct Job {
    /// The guest's scanout, held for as long as the copy may read it.
    pub(super) src: Storage,
    pub(super) dst: Arc<Target>,
    /// The scanout's first pixel as the flush arrived, when tracing.
    arrived: Option<u32>,
    /// How many of the ring's slots were in use by some process when this one was picked, for
    /// the trace: the window server holds one it is compositing, and this is how to see it does.
    in_use: usize,
}

impl Job {
    /// The surface the display is to show for this present.
    pub fn target(&self) -> SurfaceId {
        self.dst.surface.id()
    }
}

impl Drop for Job {
    fn drop(&mut self) {
        // Done or abandoned, the slot is free to be chosen again.
        self.dst.pending.store(false, Ordering::Release);
    }
}

/// One scanout's slots.
#[derive(Default)]
struct Lane {
    slots: Vec<Arc<Target>>,
    /// Ids of the two slots handed out most recently, newest last.
    recent: Vec<SurfaceId>,
}

/// A context's copy rings, one per scanout it presents to.
///
/// Per scanout, not per context: a guest driving two outputs from one context presents two
/// shapes in turn, and one ring holding both would evict and mint a surface on nearly every
/// present. A slot owns its import, so a slot let go of takes its Vulkan buffer with it.
#[derive(Default)]
pub struct Ring {
    knobs: Knobs,
    lanes: HashMap<ScanoutId, Lane>,
    /// Surfaces minted over the ring's life, across every scanout.
    minted: u64,
}

impl Ring {
    pub fn new(knobs: Knobs) -> Ring {
        Ring { knobs, ..Ring::default() }
    }

    /// A job copying `src`, presented on `scanout`, into a free slot, or why there is none.
    pub fn job(&mut self, scanout: ScanoutId, src: Storage) -> Result<Job, CopyRefused> {
        let from = src.surface().map_err(|_| CopyRefused::NotOrderable)?;
        let shape = (from.width(), from.height(), from.bytes_per_row());
        let lane = self.lanes.entry(scanout).or_default();
        // A mode change leaves slots of the old shape; drop the ones nothing is using.
        lane.slots.retain(|s| s.shape() == shape || s.pending.load(Ordering::Acquire));
        let free = lane.slots.iter().position(|s| {
            s.shape() == shape
                && !s.pending.load(Ordering::Acquire)
                && !lane.recent.contains(&s.surface.id())
                && !s.surface.in_use()
        });
        let at = match free {
            Some(at) => at,
            None if lane.slots.len() < MAX_SLOTS => {
                let surface = match Surface::scanout_like(from) {
                    Ok(s) if s.bytes_per_row() == shape.2 => s,
                    Ok(_) | Err(_) => return Err(CopyRefused::NotOrderable),
                };
                // Said every time: a slot is minted a few times per scanout and mode, so a log
                // that keeps printing this is a ring that is not reusing its surfaces.
                self.minted += 1;
                eprintln!(
                    "[virglrs] present copy: scanout {scanout}: minted surface {} ({}x{}), \
                     {} in its ring, {} minted by this context",
                    surface.id(),
                    shape.0,
                    shape.1,
                    lane.slots.len() + 1,
                    self.minted
                );
                lane.slots.push(Arc::new(Target {
                    import: Mutex::new(None),
                    surface,
                    pending: AtomicBool::new(false),
                }));
                lane.slots.len() - 1
            }
            None => return Err(CopyRefused::Busy),
        };
        let slot = &lane.slots[at];
        slot.pending.store(true, Ordering::Release);
        lane.recent.push(slot.surface.id());
        if lane.recent.len() > 2 {
            lane.recent.remove(0);
        }
        let arrived = self.knobs.trace.then(|| first_pixel(from));
        let in_use = if self.knobs.trace {
            lane.slots.iter().filter(|s| s.surface.in_use()).count()
        } else {
            0
        };
        Ok(Job { dst: Arc::clone(slot), src, arrived, in_use })
    }

    /// Let go of every import `gone` names, ahead of the device they were made on.
    pub(super) fn drop_imports(&mut self, gone: impl Fn(&Import) -> bool) {
        for slot in self.lanes.values().flat_map(|l| &l.slots) {
            let mut import = slot.import.lock().expect("an import lock is never poisoned");
            if import.as_ref().is_some_and(&gone) {
                *import = None;
            }
        }
    }
}

/// Log what the copy holds against what the scanout held as its flush arrived.
pub fn trace(job: &Job) {
    if let Some(arrived) = job.arrived {
        eprintln!(
            "[virglrs] copy trace: frame {} arrived bgra={arrived:08x} shown bgra={:08x} \
             in_use={}",
            job.dst.surface.id().0,
            first_pixel(&job.dst.surface),
            job.in_use
        );
    }
}

/// Copy `job` on the CPU, when the GPU copy could not be made. Not ordered against anything, and
/// better than presenting a slot whose contents are a frame from long ago.
pub fn copy_on_cpu(job: &Job) {
    if let Ok(src) = job.src.surface() {
        let dst = &job.dst.surface;
        let mut bytes = vec![0u8; src.alloc_size().min(dst.alloc_size()) as usize];
        let n = src.read_into(&mut bytes);
        dst.write_from(&bytes[..n]);
    }
    trace(job);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::Account;
    use crate::surface::PixelFormat;

    fn scanout(account: &Account, width: u32, height: u32) -> Storage {
        let surface = Surface::scanout(width, height, PixelFormat::Bgra, width * 4)
            .expect("the system minted a surface");
        Storage::minted_for_test(surface, account)
    }

    /// Two outputs presented in turn from one context, at different sizes, as a desktop with
    /// two monitors does. Each keeps its own few surfaces: a ring shared between them evicted the
    /// other output's slots and minted a surface on nearly every present.
    #[test]
    fn two_scanouts_presented_in_turn_keep_their_own_surfaces() {
        let account = Account::for_test(None);
        let (a, b) = (scanout(&account, 64, 8), scanout(&account, 32, 4));
        let mut ring = Ring::new(Knobs::default());
        for _ in 0..100 {
            for (id, src) in [(ScanoutId(0), &a), (ScanoutId(1), &b)] {
                let job = ring.job(id, src.clone()).expect("a free slot");
                assert_eq!(job.dst.surface.width(), src.surface().unwrap().width());
                // Copied and retired, as the present thread does before the next present.
                drop(job);
            }
        }
        // Three each: the two most recent are never reused, so the third is the first free.
        assert_eq!(ring.minted, 6, "each scanout should reuse its own three surfaces");
    }

    /// A scanout that changes mode lets go of the old shape's surfaces.
    #[test]
    fn a_mode_change_drops_the_old_shape() {
        let account = Account::for_test(None);
        let mut ring = Ring::new(Knobs::default());
        for (w, h) in [(64, 8), (32, 4)] {
            let src = scanout(&account, w, h);
            for _ in 0..10 {
                drop(ring.job(ScanoutId(0), src.clone()).expect("a free slot"));
            }
        }
        let lane = &ring.lanes[&ScanoutId(0)];
        assert!(lane.slots.iter().all(|s| s.shape() == (32, 4, 128)));
        assert_eq!(ring.minted, 6);
    }
}
