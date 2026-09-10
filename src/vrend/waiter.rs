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
//! So the wait moves here. The submitting thread takes a sync object and flushes -- which costs it
//! a flush rather than a wait for the GPU, though not nothing: measured, `glFenceSync` is 21% of
//! the worker under a draw-call-heavy workload and 79% of that is blocked in mesa's `tc_flush`,
//! waiting for the threaded context's own thread to drain its batch queue -- and hands the fence to
//! this thread, which waits on a context of its
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

/// What answers a classic fence.
///
/// The two are not "a sync" and "no sync": they are two different ways of being answered, and
/// collapsing them into an `Option` is what let a fence with nothing of its own to wait for be
/// mistaken for one whose wait had been skipped.
pub enum Answer {
    /// One sync per GL queue the fenced work could be on. It retires once every one has signalled.
    ///
    /// **Not one sync.** Each sub-context has a command queue of its own and a sync covers only the
    /// context it was taken on (see `Context::gl_contexts`), so a sync on whichever sub-context
    /// happened to be current left a sibling's renders unwaited-for -- and ctx0's uploads with
    /// them, which is where a `Vrend::transfer` with no context named puts its work. The set is the
    /// one `Vrend::finish_contexts` finishes, and the two have to cover the same queues: the whole
    /// point of the fence path is that it replaces that finish without changing what a fence means.
    ///
    /// Never empty. A fence answered by no sync at all would retire as soon as the waiter reached
    /// it, which is early, so `Vrend::decide_fence` asserts rather than building one.
    Syncs(Vec<Fence>),
    /// Nothing of its own to wait for, so it retires behind whatever is already queued.
    ///
    /// This is the whole answer for a fence on a command that queued no GL work -- a
    /// `RESOURCE_CREATE_3D` allocates storage and renders nothing, and two thirds of the fences
    /// under a texture-churning workload are exactly that. The guest puts every Global-ring fence
    /// on one `dma_fence` context and signals every id at or below the one delivered, and this
    /// queue is FIFO, so retiring behind the queue *is* retiring behind every render fenced before
    /// it. Nothing is waited for twice and nothing is answered early.
    Ordered,
}

impl Answer {
    /// What this answer is called in a trace.
    pub fn name(&self) -> &'static str {
        match self {
            Answer::Syncs(_) => "sync",
            Answer::Ordered => "ordered",
        }
    }
}

struct Job {
    /// How this fence is answered; see [`Answer`]. An `Ordered` job still travels the queue,
    /// because leaving it out would let it overtake a fence ahead of it that is still in flight.
    fence: Answer,
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
    pub fn start(display: ThreadDisplay, ctx: egl::Context, gl: Gl, sink: fence::Handle) -> Waiter {
        let q =
            Arc::new((Mutex::new(Queue { jobs: VecDeque::new(), stopped: false }), Condvar::new()));
        let qt = Arc::clone(&q);
        let thread = std::thread::Builder::new()
            .name("virglrs-glwait".into())
            .spawn(move || run(display, ctx, gl, sink, qt))
            .expect("spawning the classic fence waiter");
        Waiter { q, thread: Some(thread) }
    }

    /// Queue a fence to retire once its work has run.
    pub fn retire_context(&self, fence: Answer, ctx: ContextId, ring: RingIdx, id: FenceId) {
        self.push(Job { fence, retire: Retire::Context(ctx, ring, id) });
    }

    /// Queue a global-ring fence to retire once its work has run.
    pub fn retire_global(&self, fence: Answer, id: ClientFenceId) {
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
        if crate::vrend::debug::enabled(crate::vrend::debug::Switch::Fence) {
            eprintln!("[virglrs] fence: waiter woke, answer={}", job.fence.name());
        }
        if let Answer::Syncs(fences) = job.fence {
            // Every one, and each spent as it is waited out: they are independent queues, so the
            // last to signal is what the fence is waiting for and the order they are waited in
            // does not matter.
            for fence in fences {
                wait_out(&gl, &fence);
                gl.fence_delete(fence);
            }
        }
        if crate::vrend::debug::enabled(crate::vrend::debug::Switch::Fence) {
            eprintln!("[virglrs] fence: waiter done waiting, retiring");
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

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::surface::{self, PixelFormat};
    use crate::vrend::egl::{Flavour, Version, Winsys};
    use crate::vrend::gl::gles::*;
    use crate::vrend::gl::types::GLsizei;

    const W: u32 = 1024;
    const H: u32 = 1024;
    /// Big enough that the driver cannot have finished by the time the CPU looks, and cheap enough
    /// that the test is not a benchmark. Raised until the control arms.
    const FIRST_PASSES: u32 = 400;

    /// Queue `passes` full-surface clears, the last one `last`, and flush without waiting.
    fn queue(gl: &Gl, passes: u32, last: [f32; 4]) {
        for i in 0..passes {
            // Every pass writes the whole surface, so only the last one can be what a reader sees
            // if the work ran to completion -- and an earlier colour is what it sees if it did not.
            let c = if i + 1 == passes { last } else { [1.0, 0.0, 1.0, 1.0] };
            gl.clear_color(c);
            gl.clear(GL_COLOR_BUFFER_BIT);
        }
    }

    /// The blue channel of the surface's first pixel, read on the CPU through IOSurface.
    ///
    /// This is the foreign consumer the fence exists for: `IOSurfaceLock` and a load, ordered
    /// against our GL queue by nothing at all.
    fn first_pixel_blue(s: &surface::Surface) -> u8 {
        // `read_rows`, not `read_plane_row`: a plain surface has no *planes*, so its plane count
        // is zero and the per-plane accessor answers `None` for every index.
        let stride = s.bytes_per_row() as usize;
        let mut row = vec![0u8; stride];
        assert_eq!(s.read_rows(&mut row, stride, 1), 1, "the surface's first row reads back");
        row[0]
    }

    /// A surface, and a framebuffer over it that `queue` renders into.
    ///
    /// Returned rather than inlined because two tests need the same target, and the second one is
    /// only meaningful if it is rendering into exactly what the first one does.
    fn render_target(winsys: &Winsys, gl: &Gl) -> Arc<surface::Surface> {
        let surface =
            Arc::new(surface::Surface::plain(W, H, PixelFormat::Bgra).expect("an IOSurface"));
        let image = winsys
            .image_from_iosurface(Arc::clone(&surface) as Arc<dyn surface::Held>)
            .expect("an EGL image over the surface");
        let tex = gl.gen_texture();
        gl.bind_texture(GL_TEXTURE_2D, Some(tex));
        gl.egl_image_target_texture_2d(GL_TEXTURE_2D, &image);
        let fb = gl.gen_framebuffer();
        gl.bind_framebuffer(GL_FRAMEBUFFER, Some(fb));
        gl.framebuffer_texture_2d(GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, Some(tex), 0);
        assert_eq!(
            gl.check_framebuffer_status(),
            GL_FRAMEBUFFER_COMPLETE,
            "the IOSurface-backed texture is a complete framebuffer"
        );
        gl.viewport(0, 0, W as GLsizei, H as GLsizei);
        surface
    }

    /// A sink that says which fences were retired, in order.
    struct Recorder(std::sync::mpsc::Sender<ClientFenceId>);

    impl crate::fence::FenceSink for Recorder {
        fn context_fence(&mut self, _ctx: ContextId, _ring: RingIdx, _fence: FenceId) {}
        fn global_fence(&mut self, fence: ClientFenceId) {
            let _ = self.0.send(fence);
        }
    }

    /// An `Ordered` fence retires behind the work queued before it, which is the whole claim the
    /// design rests on.
    ///
    /// A fence with nothing of its own to wait for takes no sync and no finish -- under a
    /// texture-churning workload two thirds of all fences are exactly that, every one a
    /// `RESOURCE_CREATE_3D` that rendered nothing. What makes it safe is not a wait but the
    /// queue's order: every render fenced before it is already ahead of it, so a consumer woken by
    /// the `Ordered` fence sees that render complete.
    ///
    /// **The control is the point**, as in the test above. The same `Ordered` fence with nothing
    /// ahead of it must come back stale; if it does not, the render is completing on its own and
    /// this test would pass with the ordering deleted.
    #[test]
    fn an_ordered_fence_retires_behind_the_work_queued_before_it() {
        let _display = crate::vrend::one_display_at_a_time();
        let winsys = Winsys::open(Flavour::Gles).expect("the surfaceless display opens");
        let ctx = winsys
            .create_context(Version { major: 3, minor: 1 }, None)
            .expect("a GLES 3.1 context");
        winsys.make_current(&ctx).expect("ctx is current on this thread");
        let gl = Gl::new(winsys.gles());
        let surface = render_target(&winsys, &gl);

        let (tx, retired) = std::sync::mpsc::channel();
        let retirement = crate::fence::Retirement::start(Box::new(Recorder(tx)));
        let display = winsys.thread_display().expect("a winsys of our own lends its display");
        let wait_ctx = winsys
            .create_context(Version { major: 3, minor: 1 }, Some(&ctx))
            .expect("a shared context");
        let waiter = Waiter::start(display, wait_ctx, Gl::new(winsys.gles()), retirement.handle());

        let black = |gl: &Gl| {
            gl.clear_color([0.0, 0.0, 0.0, 1.0]);
            gl.clear(GL_COLOR_BUFFER_BIT);
            gl.finish();
        };

        // The control: an Ordered fence with an EMPTY queue ahead of it. It must retire while the
        // render is still in flight, or nothing below distinguishes ordering from luck.
        let mut passes = FIRST_PASSES;
        let mut alone = 0xff;
        for _ in 0..6 {
            black(&gl);
            queue(&gl, passes, [0.0, 0.0, 1.0, 1.0]);
            gl.flush();
            waiter.retire_global(Answer::Ordered, ClientFenceId(1));
            assert_eq!(retired.recv().expect("the sink answers"), ClientFenceId(1));
            alone = first_pixel_blue(&surface);
            if alone != 0xff {
                break;
            }
            gl.finish();
            passes *= 4;
        }
        assert_ne!(
            alone, 0xff,
            "an Ordered fence with nothing ahead of it still came back complete at every size \
             tried, so this test cannot tell ordering from a render that finished on its own"
        );

        // The same Ordered fence, this time behind a Sync for that render. FIFO is what makes it
        // wait, and the reader must now see the finished colour.
        black(&gl);
        queue(&gl, passes, [0.0, 0.0, 1.0, 1.0]);
        let sync = gl.fence().expect("the driver gives a sync object");
        waiter.retire_context(
            Answer::Syncs(vec![sync]),
            ContextId::new(1).expect("a context id"),
            RingIdx(0),
            FenceId(2),
        );
        waiter.retire_global(Answer::Ordered, ClientFenceId(3));
        assert_eq!(retired.recv().expect("the sink answers"), ClientFenceId(3));
        let behind = first_pixel_blue(&surface);

        eprintln!("[ordered] {passes} passes: alone={alone:#04x} behind={behind:#04x}");
        assert_eq!(
            behind, 0xff,
            "an Ordered fence retired before the render a Sync queued ahead of it had run"
        );
    }

    /// A CPU reader must see the render a classic fence waited for.
    ///
    /// This is the hazard the fence path exists for, in the smallest form that still has it: a
    /// consumer with no ordering against our GL queue, reading a surface a GL context rendered
    /// into. A venus compositor importing the surface is the real case, and this stands in for it
    /// because `IOSurfaceLock` is just as unordered as a Vulkan queue and needs no compositor.
    ///
    /// **The control is the point.** Asserting only that the waited read is right would pass on a
    /// machine where the work happened to be finished anyway, and would go on passing if the wait
    /// were deleted. So the same sequence is read *without* the wait first, and the test refuses
    /// to conclude anything unless that read is stale -- the needle has to be shown live before
    /// the assertion means anything.
    #[test]
    fn a_cpu_reader_sees_the_render_the_fence_waited_for() {
        let _display = crate::vrend::one_display_at_a_time();
        let winsys = Winsys::open(Flavour::Gles).expect("the surfaceless display opens");
        let ctx = winsys
            .create_context(Version { major: 3, minor: 1 }, None)
            .expect("a GLES 3.1 context");
        winsys.make_current(&ctx).expect("ctx is current on this thread");
        let gl = Gl::new(winsys.gles());

        let surface =
            Arc::new(surface::Surface::plain(W, H, PixelFormat::Bgra).expect("an IOSurface"));
        let image = winsys
            .image_from_iosurface(Arc::clone(&surface) as Arc<dyn surface::Held>)
            .expect("an EGL image over the surface");

        let tex = gl.gen_texture();
        gl.bind_texture(GL_TEXTURE_2D, Some(tex));
        gl.egl_image_target_texture_2d(GL_TEXTURE_2D, &image);
        let fb = gl.gen_framebuffer();
        gl.bind_framebuffer(GL_FRAMEBUFFER, Some(fb));
        gl.framebuffer_texture_2d(GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, Some(tex), 0);
        assert_eq!(
            gl.check_framebuffer_status(),
            GL_FRAMEBUFFER_COMPLETE,
            "the IOSurface-backed texture is a complete framebuffer"
        );
        gl.viewport(0, 0, W as GLsizei, H as GLsizei);

        // The reader thread is started and made current BEFORE any of this is timed, and it
        // answers over a channel. That is the whole experiment: spawning a thread and binding a
        // context costs milliseconds, which is longer than the render, so a reader that does that
        // work after the flush finds the GPU idle whether or not it waited -- and the test would
        // pass with the wait deleted. The only difference between the two reads below is the wait.
        let display = winsys.thread_display().expect("a winsys of our own lends its display");
        let wait_ctx = winsys
            .create_context(Version { major: 3, minor: 1 }, Some(&ctx))
            .expect("a shared context");
        let wait_gl = Gl::new(winsys.gles());
        let seen = Arc::clone(&surface);
        let (ask, asked) = std::sync::mpsc::channel::<(Fence, bool)>();
        let (told, answer) = std::sync::mpsc::channel::<u8>();
        let reader = std::thread::spawn(move || {
            display.make_current(&wait_ctx).expect("the reader's context is current");
            while let Ok((fence, wait)) = asked.recv() {
                if wait {
                    wait_out(&wait_gl, &fence);
                }
                wait_gl.fence_delete(fence);
                told.send(first_pixel_blue(&seen)).expect("the asker is still listening");
            }
        });

        // Settle on black, so "stale" has a value and is not whatever the allocation held.
        let black = |gl: &Gl| {
            gl.clear_color([0.0, 0.0, 0.0, 1.0]);
            gl.clear(GL_COLOR_BUFFER_BIT);
            gl.finish();
        };
        black(&gl);
        assert_eq!(first_pixel_blue(&surface), 0, "the surface starts black");

        // Arm the control at the same point the assertion is made: the reader is told not to
        // wait, and must come back stale. If it does not, the render is finishing before the
        // channel round trip and this test cannot tell a wait from no wait -- so grow it.
        let mut passes = FIRST_PASSES;
        let mut unwaited = 0xff;
        for _ in 0..6 {
            black(&gl);
            queue(&gl, passes, [0.0, 0.0, 1.0, 1.0]);
            let fence = gl.fence().expect("the driver gives a sync object");
            ask.send((fence, false)).expect("the reader is listening");
            unwaited = answer.recv().expect("the reader answers");
            if unwaited != 0xff {
                break;
            }
            gl.finish();
            passes *= 4;
        }
        assert_ne!(
            unwaited, 0xff,
            "an unwaited read came back complete at every size tried, so this test cannot tell a \
             working wait from a deleted one -- it proves nothing as written"
        );

        // Now the same thing, waited. Same thread, same context, same round trip.
        black(&gl);
        queue(&gl, passes, [0.0, 0.0, 1.0, 1.0]);
        let fence = gl.fence().expect("the driver gives a sync object");
        ask.send((fence, true)).expect("the reader is listening");
        let waited = answer.recv().expect("the reader answers");

        drop(ask);
        reader.join().expect("the reading thread does not panic");

        // Printed, not judged: what the control cost to arm, so a later reader can see whether it
        // armed easily or barely, and on what size of render.
        eprintln!(
            "[leg] passes={passes} unwaited=0x{unwaited:02x} waited=0x{waited:02x} \
             (0xff is the render this fence stands for)"
        );
        assert_eq!(
            waited, 0xff,
            "a CPU reader saw {waited:#04x} where the fence said the render had run -- the same \
             read was stale ({unwaited:#04x}) without the wait, so the wait is what makes it true"
        );

        gl.delete_framebuffer(fb);
    }
}
