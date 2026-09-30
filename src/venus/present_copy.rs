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
//! **Where it lands.** A small ring of surfaces minted like the scanout and published like any
//! other. A slot is not reused while its copy is pending, while it is one of the two most
//! recently handed out, or while any process holds it in use -- the window server does, for a
//! surface it is compositing.
//!
//! Two seams: `LIMINA_TEST_PRESENT_COPY_DELAY_MS` holds each copy back before it is submitted,
//! which lets the guest's later submits land first -- the case above this cannot order -- and
//! `LIMINA_PRESENT_COPY_TRACE` logs the first pixel of the guest's scanout as the flush arrives
//! and of the copy once it is done.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::ids::SurfaceId;
use crate::metal::Surface;
use crate::venus::driver::Storage;

/// Slots a ring may grow to before a present is refused a copy.
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

/// A surface's first pixel, as its four bytes in memory order.
fn first_pixel(surface: &Surface) -> u32 {
    let mut px = [0u8; 4];
    surface.read_into(&mut px);
    u32::from_be_bytes(px)
}

/// One surface of the ring.
struct Slot {
    surface: Arc<Surface>,
    /// Handed out and not yet copied into.
    pending: Arc<AtomicBool>,
}

/// One present's copy: from the guest's scanout, into a slot.
pub struct Job {
    /// The guest's scanout, held for as long as the copy may read it.
    pub(super) src: Storage,
    pub(super) dst: Arc<Surface>,
    pending: Arc<AtomicBool>,
    /// The scanout's first pixel as the flush arrived, when tracing.
    arrived: Option<u32>,
}

impl Job {
    /// The surface the display is to show for this present.
    pub fn target(&self) -> SurfaceId {
        self.dst.id()
    }
}

impl Drop for Job {
    fn drop(&mut self) {
        // Done or abandoned, the slot is free to be chosen again.
        self.pending.store(false, Ordering::Release);
    }
}

/// A context's copy ring: which slot to hand out next.
#[derive(Default)]
pub struct Ring {
    knobs: Knobs,
    slots: Vec<Slot>,
    /// Ids of the two slots handed out most recently, newest last.
    recent: Vec<u32>,
}

impl Ring {
    pub fn new(knobs: Knobs) -> Ring {
        Ring { knobs, ..Ring::default() }
    }

    /// A job copying `src` into a free slot, or `None` when there is none to give -- every slot
    /// still busy -- or the scanout has no surface to copy.
    pub fn job(&mut self, src: Storage) -> Option<Job> {
        let from = src.surface().ok()?;
        let shape = (from.width(), from.height(), from.bytes_per_row());
        // A mode change leaves slots of the old shape; drop the ones nothing is using.
        self.slots.retain(|s| {
            (s.surface.width(), s.surface.height(), s.surface.bytes_per_row()) == shape
                || s.pending.load(Ordering::Acquire)
        });
        let free = self.slots.iter().position(|s| {
            (s.surface.width(), s.surface.height(), s.surface.bytes_per_row()) == shape
                && !s.pending.load(Ordering::Acquire)
                && !self.recent.contains(&s.surface.id().0)
                && !s.surface.in_use()
        });
        let at = match free {
            Some(at) => at,
            None if self.slots.len() < MAX_SLOTS => {
                let surface = match Surface::scanout_like(from) {
                    Ok(s) if s.bytes_per_row() == shape.2 => s,
                    Ok(_) | Err(_) => return None,
                };
                self.slots.push(Slot {
                    surface: Arc::new(surface),
                    pending: Arc::new(AtomicBool::new(false)),
                });
                self.slots.len() - 1
            }
            None => return None,
        };
        let slot = &self.slots[at];
        slot.pending.store(true, Ordering::Release);
        let id = slot.surface.id().0;
        self.recent.push(id);
        if self.recent.len() > 2 {
            self.recent.remove(0);
        }
        let arrived = self.knobs.trace.then(|| first_pixel(from));
        Some(Job {
            dst: Arc::clone(&slot.surface),
            pending: Arc::clone(&slot.pending),
            src,
            arrived,
        })
    }
}

/// Log what the copy holds against what the scanout held as its flush arrived.
pub fn trace(job: &Job) {
    if let Some(arrived) = job.arrived {
        eprintln!(
            "[virglrs] copy trace: frame {} arrived bgra={arrived:08x} shown bgra={:08x}",
            job.dst.id().0,
            first_pixel(&job.dst)
        );
    }
}

/// Copy `job` on the CPU, when the GPU copy could not be made. Not ordered against anything, and
/// better than presenting a slot whose contents are a frame from long ago.
pub fn copy_on_cpu(job: &Job) {
    if let Ok(src) = job.src.surface() {
        let mut bytes = vec![0u8; src.alloc_size().min(job.dst.alloc_size()) as usize];
        let n = src.read_into(&mut bytes);
        job.dst.write_from(&bytes[..n]);
    }
    trace(job);
}
