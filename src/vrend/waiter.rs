// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! Waiting for a classic fence's GL work, off the thread that took it.
//!
//! A classic fence has to mean the GL work it fences has run: Metal orders no queue against
//! another, so a venus compositor importing the surface samples on a Vulkan queue with nothing
//! ordering it against ours, and an early fence hands it the frame before. The renderer used to
//! answer that with `glFinish` on every context, taken on the thread that services virtio-gpu for
//! *every* context, while holding the renderer. One heavy GL client then paid for everyone: a
//! cursor update queued behind a drain of the aquarium's frame, and the desktop went sticky under
//! load in a way no throughput number showed.
//!
//! So the wait moves here. The submitting thread takes a sync object and flushes -- which costs
//! it a flush, not a drain -- and hands the fence to this thread, which waits on a context of its
//! own and retires through [`crate::fence::Handle`] when the work has run. It holds no renderer
//! and no renderer lock, so nothing else queues behind it.
//!
//! **Ordering.** One queue, first in first out, so fences retire in the order they were created.
//! That is more order than the contract needs -- [`crate::fence`] promises it only within a
//! (context, ring) -- and on this stack it costs nothing, because zink puts a display's contexts
//! on one timeline with monotonic batch ids, so a later sync cannot signal before an earlier one
//! anyway. What it buys is the property limina's gpu device depends on: the guest kernel signals
//! every fence with an id at or below the one delivered, so a younger fence retiring past an older
//! in-flight one would signal work that has not run.
//!
//! **This thread may block for as long as the GPU takes**, which is the point, and is why nothing
//! else is delivered through it. Venus fences do not come here: they carry their waits inside the
//! command stream and retire straight through [`crate::fence::Retirement`], so a slow GL client
//! cannot delay a compositor's Vulkan fence.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};

use super::egl::{self, ThreadDisplay};
use super::gl::{Fence, FenceWait, Gl};
use crate::fence;
use crate::ids::{ClientFenceId, ContextId, FenceId, RingIdx};

/// How long one `glClientWaitSync` is given before we ask again.
///
/// A wait is retried rather than trusted once: the spec lets a driver report a timeout at its own
/// cap rather than at the one asked for, so a single expired wait says nothing about whether the
/// work will ever complete. The C waits in one-second slices for the same reason.
const SLICE_NS: u64 = 1_000_000_000;

/// What to retire once the work has run.
enum Retire {
    Context(ContextId, RingIdx, FenceId),
    Global(ClientFenceId),
}

struct Job {
    /// The work to wait for, or `None` for a fence with nothing to wait on -- a context that had
    /// taken no sync, or a driver that refused one. It still travels the queue, because leaving it
    /// out would let it overtake a fence ahead of it that is still in flight.
    fence: Option<Fence>,
    retire: Retire,
}

struct Queue {
    jobs: VecDeque<Job>,
    stopped: bool,
}

/// The classic fence waiter: a thread, and the GL context it waits on.
pub struct Waiter {
    q: Arc<(Mutex<Queue>, Condvar)>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Waiter {
    /// Start the waiter on a context of `ctx0`'s share group.
    ///
    /// The context is created here, on the caller's thread, and made current on the waiter's --
    /// currency is per thread, so the two never contend for it.
    pub fn start(
        display: ThreadDisplay,
        ctx: egl::Context,
        gl: Gl,
        sink: fence::Handle,
    ) -> Waiter {
        let q =
            Arc::new((Mutex::new(Queue { jobs: VecDeque::new(), stopped: false }), Condvar::new()));
        let qt = Arc::clone(&q);
        let thread = std::thread::Builder::new()
            .name("virglrs-glwait".into())
            .spawn(move || run(display, ctx, gl, sink, qt))
            .expect("spawning the classic fence waiter");
        Waiter { q, thread: Some(thread) }
    }

    /// Queue a fence to retire once `fence`'s work has run.
    pub fn retire_context(
        &self,
        fence: Option<Fence>,
        ctx: ContextId,
        ring: RingIdx,
        id: FenceId,
    ) {
        self.push(Job { fence, retire: Retire::Context(ctx, ring, id) });
    }

    /// Queue a global-ring fence to retire once `fence`'s work has run.
    pub fn retire_global(&self, fence: Option<Fence>, id: ClientFenceId) {
        self.push(Job { fence, retire: Retire::Global(id) });
    }

    fn push(&self, job: Job) {
        let (m, cv) = &*self.q;
        let mut g = m.lock().expect("the waiter queue lock is never held across a panic");
        assert!(!g.stopped, "a classic fence was queued after the waiter stopped");
        g.jobs.push_back(job);
        cv.notify_one();
    }
}

impl Drop for Waiter {
    fn drop(&mut self) {
        {
            let (m, cv) = &*self.q;
            let mut g = m.lock().expect("the waiter queue lock is never held across a panic");
            g.stopped = true;
            cv.notify_one();
        }
        if let Some(t) = self.thread.take() {
            // Every queued fence is waited for and retired before this returns. A guest blocked on
            // one of them is not woken by anything else, and the retirement it goes to must still
            // be alive to take it -- which is why `Renderer` drops `vrend` before `fences`.
            let _ = t.join();
        }
    }
}

fn run(
    display: ThreadDisplay,
    ctx: egl::Context,
    gl: Gl,
    sink: fence::Handle,
    q: Arc<(Mutex<Queue>, Condvar)>,
) {
    display.make_current(&ctx).expect("the fence waiter's own context is made current");
    let (m, cv) = &*q;
    loop {
        let job = {
            let mut g = m.lock().expect("the waiter queue lock is never held across a panic");
            loop {
                if let Some(j) = g.jobs.pop_front() {
                    break j;
                }
                if g.stopped {
                    // Drained: nothing is owed, so the context can go.
                    drop(g);
                    let _ = display.release_current();
                    return;
                }
                g = cv.wait(g).expect("the waiter queue lock is never held across a panic");
            }
        };
        if let Some(fence) = job.fence {
            wait_out(&gl, &fence);
            gl.fence_delete(fence);
        }
        match job.retire {
            Retire::Context(ctx, ring, id) => sink.retire_context(ctx, ring, id),
            Retire::Global(id) => sink.retire_global(id),
        }
    }
}

/// Wait until the fence's work has run, or until the driver says it never will.
///
/// A timeout is not an answer, so it is retried. A refusal is: the fence is retired anyway, on the
/// same reasoning the whole path rests on -- a fence that never retires hangs the guest, and a
/// guest that is told too early draws one stale frame.
fn wait_out(gl: &Gl, fence: &Fence) {
    loop {
        match gl.fence_wait(fence, SLICE_NS) {
            FenceWait::Signalled => return,
            FenceWait::Timeout => continue,
            FenceWait::Failed => {
                eprintln!("[virglrs] vrend: the driver refused a fence wait; retiring it anyway");
                return;
            }
        }
    }
}
