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

    /// Queue a fence to retire once `fence`'s work has run.
    pub fn retire_context(&self, fence: Option<Fence>, ctx: ContextId, ring: RingIdx, id: FenceId) {
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::metal::{self, PixelFormat};
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
    fn first_pixel_blue(s: &metal::Surface) -> u8 {
        s.read_plane_row(0, 0).expect("the surface has a row 0")[0]
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
    #[ignore = "needs the zink-on-KosmicKrisp environment"]
    fn a_cpu_reader_sees_the_render_the_fence_waited_for() {
        let winsys = Winsys::open(Flavour::Gles).expect("the surfaceless display opens");
        let ctx = winsys
            .create_context(Version { major: 3, minor: 1 }, None)
            .expect("a GLES 3.1 context");
        winsys.make_current(&ctx).expect("ctx is current on this thread");
        let gl = Gl::new(winsys.gles());

        let surface =
            Arc::new(metal::Surface::plain(W, H, PixelFormat::Bgra).expect("an IOSurface"));
        let image = winsys
            .image_from_iosurface(Arc::clone(&surface) as Arc<dyn metal::Held>)
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

        // A second context of the same share group, on another thread, is where the wait happens --
        // exactly as the waiter does it.
        let display = winsys.thread_display();
        let wait_ctx = winsys
            .create_context(Version { major: 3, minor: 1 }, Some(&ctx))
            .expect("a shared ctx");
        let wait_gl = Gl::new(winsys.gles());

        // Settle on black, so "stale" has a value and is not whatever the allocation held.
        gl.clear_color([0.0, 0.0, 0.0, 1.0]);
        gl.clear(GL_COLOR_BUFFER_BIT);
        gl.finish();
        assert_eq!(first_pixel_blue(&surface), 0, "the surface starts black");

        // Arm the control: queue work ending in full blue and look before waiting. If the reader
        // already sees blue the workload is too small to expose anything, so make it bigger.
        let mut passes = FIRST_PASSES;
        let mut unwaited = 0xff;
        for _ in 0..5 {
            gl.clear_color([0.0, 0.0, 0.0, 1.0]);
            gl.clear(GL_COLOR_BUFFER_BIT);
            gl.finish();
            queue(&gl, passes, [0.0, 0.0, 1.0, 1.0]);
            gl.flush();
            unwaited = first_pixel_blue(&surface);
            if unwaited != 0xff {
                break;
            }
            gl.finish();
            passes *= 4;
        }
        assert_ne!(
            unwaited, 0xff,
            "could not build a render slow enough for an unwaited read to be stale, so this test \
             cannot tell a working wait from a deleted one -- it proves nothing as written"
        );

        // Now the fence, waited the way the waiter waits it: a sync taken on the rendering
        // context, and a wait on a different context on a different thread.
        queue(&gl, passes, [0.0, 0.0, 1.0, 1.0]);
        let fence = gl.fence().expect("the driver gives a sync object");
        // Moved, not borrowed: a GL context is `Send` and not `Sync`, because being current is a
        // property of one thread. That is the same reason the waiter owns its context outright.
        let seen = Arc::clone(&surface);
        let waited = std::thread::spawn(move || {
            display.make_current(&wait_ctx).expect("the waiter's context is current");
            wait_out(&wait_gl, &fence);
            wait_gl.fence_delete(fence);
            first_pixel_blue(&seen)
        })
        .join()
        .expect("the waiting thread does not panic");

        assert_eq!(
            waited, 0xff,
            "a CPU reader saw {waited:#04x} where the fence said the render had run -- the same \
             read was stale ({unwaited:#04x}) without the wait, so the wait is what makes it true"
        );

        gl.delete_framebuffer(fb);
    }
}
