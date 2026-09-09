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
use std::sync::{Arc, Condvar, Mutex};

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
}

enum Job {
    /// A context fence: retires through `write_context_fence` with its ring and id.
    Context(ContextId, RingIdx, FenceId),
    /// A legacy global fence: retires through `global_fence` with the client's own id.
    Global(ClientFenceId),
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
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
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
    pub fn start(sink: Box<dyn FenceSink>) -> Retirement {
        let q =
            Arc::new((Mutex::new(Queue { jobs: VecDeque::new(), stopped: false }), Condvar::new()));
        let qt = Arc::clone(&q);
        let thread = std::thread::Builder::new()
            .name("virglrs-fence".into())
            .spawn(move || run(sink, qt))
            .expect("spawning the fence retirement thread");
        Retirement { inner: Arc::new(Inner { q, thread: Mutex::new(Some(thread)) }) }
    }

    pub fn retire_context(&self, ctx: ContextId, ring: RingIdx, fence: FenceId) {
        push(&self.inner.q, Job::Context(ctx, ring, fence));
    }

    pub fn retire_global(&self, fence: ClientFenceId) {
        push(&self.inner.q, Job::Global(fence));
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

fn run(mut sink: Box<dyn FenceSink>, q: Arc<(Mutex<Queue>, Condvar)>) {
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
        if crate::vrend::debug::enabled(crate::vrend::debug::Switch::Fence) {
            let what = match &job {
                Job::Context(ctx, ring, fence) => {
                    format!("context ctx={ctx:?} ring={ring:?} id={}", fence.0)
                }
                Job::Global(id) => format!("global id={}", id.0),
                Job::Stop => "stop".to_string(),
            };
            eprintln!("[virglrs] fence: delivering {what} to the VMM");
        }
        match job {
            Job::Context(ctx, ring, fence) => sink.context_fence(ctx, ring, fence),
            Job::Global(id) => sink.global_fence(id),
            Job::Stop => {
                m.lock().expect("the fence queue lock is never held across a panic").stopped = true;
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::{Sender, channel};

    /// A sink that reports everything it is handed, in the order it arrives.
    struct Recorder(Sender<(u32, u32, u64)>);

    impl FenceSink for Recorder {
        fn context_fence(&mut self, ctx: ContextId, ring: RingIdx, fence: FenceId) {
            let _ = self.0.send((ctx.get(), ring.0, fence.0));
        }

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
        let r = Retirement::start(Box::new(Recorder(tx)));
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
        let r = Retirement::start(Box::new(Recorder(tx)));

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
