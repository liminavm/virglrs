// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! Asynchronous fence retirement.
//!
//! `RENDER_SERVER | THREAD_SYNC | ASYNC_FENCE_CB` is a behavioural contract, not a hint: venus
//! fences must retire on a thread that is NOT the one submitting, or the guest deadlocks inside
//! `vkQueueWaitIdle` waiting for a fence whose retirement is queued behind its own call. So even
//! though a skeleton has no GPU work to wait on, it retires through the thread -- otherwise P2
//! would inherit a synchronous path that looks fine until a guest waits on it.
//!
//! Ordering is per (context, ring), which is the only order the guest can observe: it reads a
//! ring's seqno, so fences on one ring must retire in the order they were created. Across rings
//! there is no order to preserve and none is imposed.

use std::collections::VecDeque;

// loom's primitives in a test build under `--cfg loom`, so a model can drive every interleaving
// of the threads here; std's otherwise. Nothing else in this module names a synchronisation type.
#[cfg(all(test, loom))]
use loom::{
    sync::{Arc, Condvar, Mutex},
    thread,
};
#[cfg(not(all(test, loom)))]
use std::{
    sync::{Arc, Condvar, Mutex},
    thread,
};

use crate::ids::{ClientFenceId, ContextId, FenceId, RingIdx};

/// Where a retired fence goes.
///
/// Implemented by whoever drives the renderer -- the C shim wraps the VMM's callback table in one
/// of these, and a Rust caller writes its own. It is the renderer's only way to tell anyone that
/// work has completed.
///
/// **Called from the renderer's retirement thread**, never from the thread that submitted: that
/// is what `THREAD_SYNC | ASYNC_FENCE_CB` buys, and a guest waiting in `vkQueueWaitIdle`
/// deadlocks without it. Two things follow for an implementor. Blocking here stalls every later
/// fence, on every context, because one thread delivers them all. And a panic here aborts the
/// process, since the crate builds `panic = "abort"` -- so a sink handles its own errors rather
/// than unwrapping.
pub trait FenceSink: Send {
    /// A fence on one context's ring has retired. Delivered in creation order within a ring; no
    /// order is imposed across rings, because the guest can observe none.
    fn context_fence(&mut self, ctx: ContextId, ring: RingIdx, fence: FenceId);

    /// A fence on the legacy global path has retired.
    fn global_fence(&mut self, fence: ClientFenceId);

    /// A present fence has retired: the work behind a flushed resource's contents has finished on
    /// the host, and the frame parked on it may be shown.
    ///
    /// It carries no context and no ring because it has neither. This is a fence the *VMM* asked
    /// for about a resource, not one the guest created on a stream, and the ring it used to be
    /// smuggled in on was a number reserved by convention in the guest's own index space.
    fn present_fence(&mut self, fence: FenceId);
}

enum Job {
    /// A context fence: retires through `write_context_fence` with its ring and id.
    Context(ContextId, RingIdx, FenceId),
    /// A legacy global fence: retires through `global_fence` with the client's own id.
    Global(ClientFenceId),
    /// A present fence: retires through `present_fence` with the cookie the VMM parked on.
    Present(FenceId),
    Stop,
}

struct Queue {
    jobs: VecDeque<Job>,
    stopped: bool,
}

/// The queue and the thread draining it, kept alive by everyone who can still retire through it.
///
/// The thread is stopped and joined when the last of them goes, which is what makes it impossible
/// to stop retirement while something can still queue a fence: a `Handle` held by the classic
/// fence waiter keeps this alive until the waiter itself is gone, whatever order anything is
/// declared or dropped in.
struct Inner {
    q: Arc<(Mutex<Queue>, Condvar)>,
    thread: Mutex<Option<thread::JoinHandle<()>>>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        push(&self.q, Job::Stop);
        if let Some(t) = self.thread.lock().expect("the thread slot is never poisoned").take() {
            // A fence the VMM is still waiting on must not be dropped on the floor: the thread
            // drains what is queued before it stops, and cleanup waits for that to finish.
            let _ = t.join();
        }
    }
}

pub struct Retirement {
    inner: Arc<Inner>,
}

impl Retirement {
    pub fn start(sink: Box<dyn FenceSink>, debug: crate::vrend::debug::Switches) -> Retirement {
        let q =
            Arc::new((Mutex::new(Queue { jobs: VecDeque::new(), stopped: false }), Condvar::new()));
        let qt = Arc::clone(&q);
        let thread = thread::Builder::new()
            .name("virglrs-fence".into())
            .spawn(move || run(sink, qt, debug))
            .expect("spawning the fence retirement thread");
        Retirement { inner: Arc::new(Inner { q, thread: Mutex::new(Some(thread)) }) }
    }

    pub fn retire_context(&self, ctx: ContextId, ring: RingIdx, fence: FenceId) {
        push(&self.inner.q, Job::Context(ctx, ring, fence));
    }

    pub fn retire_global(&self, fence: ClientFenceId) {
        push(&self.inner.q, Job::Global(fence));
    }

    pub fn retire_present(&self, fence: FenceId) {
        push(&self.inner.q, Job::Present(fence));
    }

    /// A handle for another thread to retire through.
    ///
    /// The classic fence waiter holds one: it decides *when* a fence has been answered, and this
    /// is how it says so, without owning the thread that delivers or the sink it delivers to.
    ///
    /// A handle keeps the retirement thread alive, so holding one is enough -- there is no order
    /// to get right and no destroy site to remember.
    pub fn handle(&self) -> Handle {
        Handle { inner: Arc::clone(&self.inner) }
    }
}

/// A way to retire a fence, for a thread that is not the one that owns [`Retirement`].
#[derive(Clone)]
pub struct Handle {
    inner: Arc<Inner>,
}

impl Handle {
    pub fn retire_context(&self, ctx: ContextId, ring: RingIdx, fence: FenceId) {
        push(&self.inner.q, Job::Context(ctx, ring, fence));
    }

    pub fn retire_global(&self, fence: ClientFenceId) {
        push(&self.inner.q, Job::Global(fence));
    }

    pub fn retire_present(&self, fence: FenceId) {
        push(&self.inner.q, Job::Present(fence));
    }
}

fn push(q: &Arc<(Mutex<Queue>, Condvar)>, job: Job) {
    let (m, cv) = &**q;
    let mut g = m.lock().expect("the fence queue lock is never held across a panic");
    if g.stopped {
        // Stopping twice is nothing; a *fence* arriving after the thread has gone is one the guest
        // may still be waiting on, and returning quietly here is how it would be lost. Nothing can
        // push after the stop, because the stop happens when the last thing that could push is
        // dropped -- so this asserts that, rather than guarding against it.
        assert!(
            matches!(job, Job::Stop),
            "a fence was queued for retirement after the retirement thread stopped"
        );
        return;
    }
    g.jobs.push_back(job);
    cv.notify_one();
}

fn run(
    mut sink: Box<dyn FenceSink>,
    q: Arc<(Mutex<Queue>, Condvar)>,
    debug: crate::vrend::debug::Switches,
) {
    let (m, cv) = &*q;
    loop {
        let job = {
            let mut g = m.lock().expect("the fence queue lock is never held across a panic");
            loop {
                match g.jobs.pop_front() {
                    Some(j) => break j,
                    None => {
                        g = cv.wait(g).expect("the fence queue lock is never held across a panic")
                    }
                }
            }
        };
        if debug.enabled(crate::vrend::debug::Switch::Fence) {
            let what = match &job {
                Job::Context(ctx, ring, fence) => {
                    format!("context ctx={ctx:?} ring={ring:?} id={}", fence.0)
                }
                Job::Global(id) => format!("global id={}", id.0),
                Job::Present(id) => format!("present id={}", id.0),
                Job::Stop => "stop".to_string(),
            };
            eprintln!("[virglrs] fence: delivering {what} to the VMM");
        }
        match job {
            Job::Context(ctx, ring, fence) => sink.context_fence(ctx, ring, fence),
            Job::Global(id) => sink.global_fence(id),
            Job::Present(id) => sink.present_fence(id),
            Job::Stop => {
                m.lock().expect("the fence queue lock is never held across a panic").stopped = true;
                return;
            }
        }
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use std::sync::mpsc::{Sender, channel};

    /// A sink that reports everything it is handed, in the order it arrives.
    struct Recorder(Sender<(u32, u32, u64)>);

    impl FenceSink for Recorder {
        fn context_fence(&mut self, ctx: ContextId, ring: RingIdx, fence: FenceId) {
            let _ = self.0.send((ctx.get(), ring.0, fence.0));
        }

        fn present_fence(&mut self, _: FenceId) {}

        fn global_fence(&mut self, fence: ClientFenceId) {
            let _ = self.0.send((0, 0, fence.0 as u64));
        }
    }

    /// A holder of a `Handle` can still retire after the `Retirement` it came from is dropped.
    ///
    /// This is what makes the classic fence waiter safe to own from anywhere: it drains and
    /// retires while it is being dropped, and nothing has to arrange for that to happen before
    /// retirement stops. Written as the negative of the bug it prevents -- with the thread stopped
    /// early, `push` asserts and this fence is never delivered.
    #[test]
    fn a_handle_outliving_its_retirement_still_delivers() {
        let (tx, rx) = channel();
        let r = Retirement::start(Box::new(Recorder(tx)), crate::vrend::debug::Switches::default());
        let h = r.handle();
        drop(r);
        h.retire_global(ClientFenceId(7));
        drop(h);

        let got: Vec<_> = rx.try_iter().collect();
        assert_eq!(got, vec![(0, 0, 7)], "a fence queued after its Retirement dropped was lost");
    }

    /// Dropping the renderer must not drop fences the VMM is still waiting on.
    ///
    /// The queue is asynchronous, so at drop there is almost always work outstanding -- a guest
    /// blocked in `vkQueueWaitIdle` on one of those fences waits forever if it is discarded. This
    /// was a comment on `Drop` until the sink became a trait; a C function pointer could not be
    /// written in a test, so nothing checked it.
    ///
    /// Ordering rides along: a ring's fences are the one order a guest can observe, because it
    /// reads that ring's seqno.
    #[test]
    fn every_queued_fence_is_delivered_in_ring_order_before_the_queue_stops() {
        const N: u64 = 500;
        let (tx, rx) = channel();
        let r = Retirement::start(Box::new(Recorder(tx)), crate::vrend::debug::Switches::default());

        let ctx = ContextId::new(7).unwrap();
        for i in 1..=N {
            r.retire_context(ctx, RingIdx(1), FenceId(i));
        }
        r.retire_global(ClientFenceId(99));
        drop(r);

        // Drained without blocking, on purpose: a blocking read would wait for the sender to be
        // dropped and so would pass whether or not `drop` waited for anything. What is under test
        // is that delivery has ALREADY happened by the time `drop` returns.
        let got: Vec<_> = rx.try_iter().collect();
        assert_eq!(got.len() as u64, N + 1, "a fence was dropped on the floor");
        for (i, (c, ring, fence)) in got.iter().take(N as usize).enumerate() {
            assert_eq!((*c, *ring, *fence), (7, 1, i as u64 + 1), "a ring retired out of order");
        }
        assert_eq!(got[N as usize], (0, 0, 99));
    }
}

/// Every interleaving of a fence retired from another thread against the owner letting go, run
/// under `RUSTFLAGS="--cfg loom" cargo test --lib fence::loom_models`.
///
/// The tests above run the one schedule the machine happens to pick. Here loom runs them all:
/// either thread may drop the last share of the queue, the retirement thread may be waiting or
/// running at each push, and the stop may land anywhere after the last push can.
#[cfg(all(test, loom))]
mod loom_models {
    use super::*;

    /// A sink that records `(ring, fence)` in arrival order, behind loom's own lock so the
    /// model sees the handoff.
    struct Recorder(Arc<Mutex<Vec<(u32, u64)>>>);

    impl FenceSink for Recorder {
        fn context_fence(&mut self, _ctx: ContextId, ring: RingIdx, fence: FenceId) {
            self.0.lock().unwrap().push((ring.0, fence.0));
        }

        fn present_fence(&mut self, _: FenceId) {}

        fn global_fence(&mut self, fence: ClientFenceId) {
            self.0.lock().unwrap().push((u32::MAX, fence.0 as u64));
        }
    }

    /// Whatever order the two threads let go in, every fence queued is delivered before the
    /// retirement thread stops, each ring in the order its fences were queued, and the
    /// queued-after-stop assert never fires.
    #[test]
    fn every_fence_is_delivered_in_ring_order_whoever_lets_go_last() {
        // Counted outside the model, with std's atomic: loom reruns the closure once per
        // schedule, and a model that ran once would pass having tried nothing.
        static SCHEDULES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        loom::model(|| {
            SCHEDULES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let got = Arc::new(Mutex::new(Vec::new()));
            let r = Retirement::start(
                Box::new(Recorder(Arc::clone(&got))),
                crate::vrend::debug::Switches::default(),
            );
            let h = r.handle();
            let ctx = ContextId::new(7).unwrap();
            let other = thread::spawn(move || {
                h.retire_context(ctx, RingIdx(1), FenceId(1));
                h.retire_context(ctx, RingIdx(1), FenceId(2));
            });
            r.retire_context(ctx, RingIdx(2), FenceId(9));
            drop(r);
            other.join().unwrap();

            // Whichever thread dropped the last share joined the retirement thread before it
            // returned, and both are done now, so delivery is complete -- not merely under way.
            let got = got.lock().unwrap().clone();
            let ring =
                |n: u32| got.iter().filter(|(r, _)| *r == n).map(|(_, f)| *f).collect::<Vec<_>>();
            assert_eq!(got.len(), 3, "a fence was lost: {got:?}");
            assert_eq!(ring(1), [1, 2], "a ring retired out of order: {got:?}");
            assert_eq!(ring(2), [9], "a fence was lost: {got:?}");
        });
        let ran = SCHEDULES.load(std::sync::atomic::Ordering::Relaxed);
        assert!(ran > 100, "loom ran {ran} schedules; the model is not exploring");
        eprintln!("[fence] loom ran {ran} schedules");
    }
}
