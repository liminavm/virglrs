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
use std::io::{ErrorKind, Read, Write};
use std::os::fd::{AsFd, OwnedFd};
use std::os::unix::net::UnixStream;
use std::sync::{Arc, Condvar, Mutex};

use super::debug;
use super::egl::{self, ThreadDisplay};
use super::gl::{Fence, FenceWait, Gl};
use super::in_flight::Ticket;
use super::video::pending::Landing;
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
    Present(FenceId),
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

/// Everything a fence waits on before it retires, in the order the waiter waits for it.
pub struct Owed {
    /// Decoded pictures the fence covers that may not have landed yet: a hardware decode runs on
    /// its codec's own thread, so a fence created after an END_FRAME has to wait for the picture
    /// as well as for the GL work. Waited out before the syncs, and empty for every fence on a
    /// context with no decode in flight. See [`super::video::pending`].
    pub pictures: Vec<Arc<Landing>>,
    /// How this fence is answered; see [`Answer`]. An `Ordered` job still travels the queue,
    /// because leaving it out would let it overtake a fence ahead of it that is still in flight.
    pub fence: Answer,
    /// This fence's place in its context's in-flight count, given back as soon as the GPU has
    /// passed it -- see [`super::in_flight`]. `None` for a fence no context is bounded by.
    pub ticket: Option<Ticket>,
    /// Whether a query result this fence covers was still unanswered when the fence was taken.
    /// If so, the render thread must look at it again once the work has run and before the fence
    /// retires; see [`Pump`].
    pub queries: bool,
}

struct Job {
    owed: Owed,
    retire: Retire,
}

/// How the waiter gets parked queries answered before a fence that covers them retires.
///
/// The guest asks for a query result once, at end-query, and then waits for the fence behind
/// that request. When that fence retires it reads the buffer directly and expects the result to
/// be there. A query object belongs to the GL context that made it, and only the render thread
/// has that context current, so this thread cannot read the result itself. Instead it rings the
/// VMM through [`Pump::descriptor`] and holds the fence until the render thread has run
/// [`Pump::serve`]. This is the C's poll eventfd and `vrend_renderer_check_queries`.
pub struct Pump {
    state: Mutex<PumpState>,
    served: Condvar,
    /// This thread's end: one byte means "poll me", and a full buffer already means that.
    ring: UnixStream,
    /// The VMM's end, readable while a ring is unanswered. A socket rather than an eventfd,
    /// so it is the same on both hosts.
    bell: UnixStream,
}

#[derive(Default)]
struct PumpState {
    /// Rings made, counted so that a serve answers exactly the rings made before it looked.
    asked: u64,
    served: u64,
    /// Whether the VMM took [`Pump::descriptor`], which is its promise to call
    /// [`Pump::serve`] when it is readable. Until then nothing would answer a ring.
    subscribed: bool,
    /// Set when the waiter stops. A ring nobody will answer must not hold the join.
    stopped: bool,
}

impl Pump {
    fn new() -> std::io::Result<Pump> {
        let (bell, ring) = UnixStream::pair()?;
        bell.set_nonblocking(true)?;
        ring.set_nonblocking(true)?;
        Ok(Pump { state: Mutex::default(), served: Condvar::new(), ring, bell })
    }

    /// A descriptor that is readable while the waiter needs [`Pump::serve`] to run. Taking it is
    /// the caller's promise to serve whenever it is: from then on fences wait for the serve.
    pub fn descriptor(&self) -> std::io::Result<OwnedFd> {
        let fd = self.bell.as_fd().try_clone_to_owned()?;
        self.state.lock().expect("the pump lock is never held across a panic").subscribed = true;
        Ok(fd)
    }

    /// Whether anyone has promised to serve; see [`Pump::descriptor`].
    pub fn subscribed(&self) -> bool {
        self.state.lock().expect("the pump lock is never held across a panic").subscribed
    }

    /// The waiter's half: ring, then wait until a serve that started after the ring has finished.
    fn ask(&self) {
        let mut g = self.state.lock().expect("the pump lock is never held across a panic");
        g.asked += 1;
        let mine = g.asked;
        loop {
            match (&self.ring).write(&[1]) {
                // A full buffer is a bell that is already ringing.
                Ok(_) => break,
                Err(e) if e.kind() == ErrorKind::WouldBlock => break,
                Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                Err(e) => panic!("ringing the renderer's poll descriptor failed: {e}"),
            }
        }
        while g.served < mine && !g.stopped {
            g = self.served.wait(g).expect("the pump lock is never held across a panic");
        }
    }

    /// The render thread's half: silence the bell, run `check`, and release every ring made
    /// before it.
    ///
    /// The bell is drained before `asked` is read, so a ring that lands after the drain either
    /// counts towards this serve or leaves a byte for the next one. A serve with nothing asked
    /// just runs `check`.
    pub fn serve(&self, check: impl FnOnce()) {
        let mut sink = [0u8; 64];
        loop {
            match (&self.bell).read(&mut sink) {
                Ok(0) => break,
                Ok(_) => continue,
                Err(e) if e.kind() == ErrorKind::WouldBlock => break,
                Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                Err(e) => panic!("draining the renderer's poll descriptor failed: {e}"),
            }
        }
        let target = self.state.lock().expect("the pump lock is never held across a panic").asked;
        check();
        let mut g = self.state.lock().expect("the pump lock is never held across a panic");
        g.served = g.served.max(target);
        self.served.notify_all();
    }

    fn stop(&self) {
        self.state.lock().expect("the pump lock is never held across a panic").stopped = true;
        self.served.notify_all();
    }
}

struct Queue {
    jobs: VecDeque<Job>,
    stopped: bool,
}

/// The classic fence waiter: a thread, and the GL context it waits on.
pub struct Waiter {
    q: Arc<(Mutex<Queue>, Condvar)>,
    pump: Arc<Pump>,
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
        debug: debug::Switches,
    ) -> Waiter {
        let q =
            Arc::new((Mutex::new(Queue { jobs: VecDeque::new(), stopped: false }), Condvar::new()));
        let qt = Arc::clone(&q);
        let pump = Arc::new(Pump::new().expect("a socket pair for the renderer's poll descriptor"));
        let pt = Arc::clone(&pump);
        let thread = std::thread::Builder::new()
            .name("virglrs-glwait".into())
            .spawn(move || run(display, ctx, gl, sink, qt, &pt, debug))
            .expect("spawning the classic fence waiter");
        Waiter { q, pump, thread: Some(thread) }
    }

    /// What the render thread serves when the VMM polls. See [`Pump`].
    pub fn pump(&self) -> &Arc<Pump> {
        &self.pump
    }

    /// Queue a fence to retire once everything it is [`Owed`] has happened.
    pub fn retire_context(&self, owed: Owed, ctx: ContextId, ring: RingIdx, id: FenceId) {
        self.push(Job { owed, retire: Retire::Context(ctx, ring, id) });
    }

    /// Queue a global-ring fence to retire once everything it is [`Owed`] has happened.
    pub fn retire_global(&self, owed: Owed, id: ClientFenceId) {
        self.push(Job { owed, retire: Retire::Global(id) });
    }

    /// Queue a present fence to retire once the work behind a flushed surface has run, and the
    /// rest of what it is [`Owed`] has happened.
    pub fn retire_present(&self, owed: Owed, id: FenceId) {
        self.push(Job { owed, retire: Retire::Present(id) });
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
        // A fence held for a poll that will not come is released unanswered: the guest reads a
        // query that is not done, which its own fallback re-reads, rather than never waking.
        self.pump.stop();
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
    pump: &Pump,
    debug: debug::Switches,
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
        if debug.enabled(debug::Switch::Fence) {
            eprintln!("[virglrs] fence: waiter woke, answer={}", job.owed.fence.name());
        }
        // The pictures first. A decode thread needs nothing from this one, so waiting here cannot
        // stall anything but the fences behind this one -- which is the order they owe anyway.
        for picture in &job.owed.pictures {
            picture.wait();
        }
        if let Answer::Syncs(fences) = job.owed.fence {
            // Every one, and each spent as it is waited out: they are independent queues, so the
            // last to signal is what the fence is waiting for and the order they are waited in
            // does not matter.
            for fence in fences {
                wait_out(&gl, &fence);
                gl.fence_delete(fence);
            }
        }
        // The GPU is past this fence's work, so it no longer counts against its context. Given
        // back before the retirement, which goes through the VMM's locks: a batch waiting on the
        // count must not also wait on those.
        drop(job.owed.ticket);
        // The work has run, so a parked query it covers is ready; the render thread writes it
        // before the guest is told it may read. After the ticket, because a batch held at the
        // fence depth would otherwise hold the render thread away from the poll.
        if job.owed.queries {
            pump.ask();
        }
        if debug.enabled(debug::Switch::Fence) {
            eprintln!("[virglrs] fence: waiter done waiting, retiring");
        }
        match job.retire {
            Retire::Context(ctx, ring, id) => sink.retire_context(ctx, ring, id),
            Retire::Global(id) => sink.retire_global(id),
            Retire::Present(id) => sink.retire_present(id),
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
    /// The render's cost, in iterations of the fragment loop: measured 2026-10-02 on an M1 Max at
    /// about 15 ms of GPU for well under a millisecond of CPU. Raised until the control arms.
    const FIRST_ITERATIONS: i32 = 1024;
    /// Where raising stops. Four times 16384 measured *faster* than 16384 on KosmicKrisp
    /// (2026-10-02) -- the GPU gives up on a fragment that long -- so a ladder past it would
    /// measure nothing.
    const LAST_ITERATIONS: i32 = 16384;

    /// A render whose cost is all on the GPU: one full-surface draw whose every fragment loops,
    /// then one that adds blue.
    ///
    /// The cost has to be GPU time the CPU does not share. Full-surface clears were tried and are
    /// not that: the driver collapses or overlaps them, so growing their count grew the CPU's
    /// recording as fast as the GPU's work, and a reader slowed by a loaded CPU found the work
    /// done at every size -- the control failed under load and passed idle. Here the CPU records
    /// two draws whatever the size, and only the fragment loop grows.
    ///
    /// Both draws blend additively, onto a surface the caller has cleared to black. The loop's
    /// draw adds zero -- through a test the compiler cannot prove, so the loop is not folded away
    /// -- and blending keeps the tiler from discarding it under the draw that follows. So the
    /// surface stays black until the second draw lands, and is exactly blue once it has.
    struct Load {
        iterations: crate::vrend::gl::UniformLocation,
        colour: crate::vrend::gl::UniformLocation,
    }

    impl Load {
        fn new(gl: &Gl) -> Load {
            const VS: &str = "#version 310 es
                void main() {
                    // One triangle that covers the surface, from the vertex index alone.
                    vec2 p = vec2(float((gl_VertexID & 1) << 2) - 1.0,
                                  float((gl_VertexID & 2) << 1) - 1.0);
                    gl_Position = vec4(p, 0.0, 1.0);
                }";
            const FS: &str = "#version 310 es
                precision highp float;
                uniform int iterations;
                uniform vec4 colour;
                out vec4 c;
                void main() {
                    float x = gl_FragCoord.x * 0.001;
                    for (int i = 0; i < iterations; i++)
                        x = fract(sin(x) * 43758.5453 + 0.1);
                    // Always true, since fract() is below 1 -- but not to the compiler.
                    c = colour * (x < 2.0 ? 1.0 : 0.5);
                }";
            let program = gl.create_program().expect("a program");
            for (kind, source) in [(GL_VERTEX_SHADER, VS), (GL_FRAGMENT_SHADER, FS)] {
                let shader = gl.create_shader(kind).expect("a shader");
                gl.compile_shader(shader, source).expect("the load's shader compiles");
                gl.attach_shader(program, shader);
                gl.delete_shader(shader);
            }
            gl.link_program(program).expect("the load's program links");
            let mut bound = crate::vrend::gl::BoundProgram::default();
            gl.use_program(&mut bound, Some(program));
            gl.bind_vertex_array(Some(gl.gen_vertex_array()));
            gl.blend_func_separate(GL_ONE, GL_ONE, GL_ONE, GL_ONE);
            gl.enable(GL_BLEND);
            let uniform = |name| gl.get_uniform_location(program, name).expect("a live uniform");
            Load { iterations: uniform("iterations"), colour: uniform("colour") }
        }

        /// Queue the render, `iterations` long, and flush without waiting.
        fn queue(&self, gl: &Gl, iterations: i32) {
            gl.uniform_1i(self.iterations, iterations);
            gl.uniform_4f(self.colour, [0.0, 0.0, 0.0, 0.0]);
            gl.draw_arrays(GL_TRIANGLES, 0, 3);
            gl.uniform_1i(self.iterations, 0);
            gl.uniform_4f(self.colour, [0.0, 0.0, 1.0, 0.0]);
            gl.draw_arrays(GL_TRIANGLES, 0, 3);
            gl.flush();
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
            .image_from_surface(Arc::clone(&surface) as Arc<dyn surface::Held>)
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

    /// What an `Ordered` fence with nothing else to wait for is owed: the `pictures`, if any.
    fn ordered(pictures: Vec<Arc<Landing>>) -> Owed {
        Owed { pictures, fence: Answer::Ordered, ticket: None, queries: false }
    }

    /// A sink that says which fences were retired, in order.
    struct Recorder(std::sync::mpsc::Sender<ClientFenceId>);

    impl crate::fence::FenceSink for Recorder {
        fn context_fence(&mut self, _ctx: ContextId, _ring: RingIdx, _fence: FenceId) {}
        fn present_fence(&mut self, _: FenceId) {}

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
        let gl = Gl::new(winsys.procs(), crate::vrend::features::Api::Gles(30));
        let surface = render_target(&winsys, &gl);

        let (tx, retired) = std::sync::mpsc::channel();
        let retirement = crate::fence::Retirement::start(
            Box::new(Recorder(tx)),
            crate::vrend::debug::Switches::default(),
        );
        let display = winsys.thread_display().expect("a winsys of our own lends its display");
        let wait_ctx = winsys
            .create_context(Version { major: 3, minor: 1 }, Some(&ctx))
            .expect("a shared context");
        let waiter = Waiter::start(
            display,
            wait_ctx,
            Gl::new(winsys.procs(), crate::vrend::features::Api::Gles(30)),
            retirement.handle(),
            debug::Switches::default(),
        );

        let black = |gl: &Gl| {
            gl.clear_color([0.0, 0.0, 0.0, 1.0]);
            gl.clear(GL_COLOR_BUFFER_BIT);
            gl.finish();
        };

        // The control: an Ordered fence with an EMPTY queue ahead of it. It must retire while the
        // render is still in flight, or nothing below distinguishes ordering from luck.
        let load = Load::new(&gl);
        let mut iterations = FIRST_ITERATIONS;
        let mut alone;
        loop {
            black(&gl);
            load.queue(&gl, iterations);
            waiter.retire_global(ordered(Vec::new()), ClientFenceId(1));
            assert_eq!(retired.recv().expect("the sink answers"), ClientFenceId(1));
            alone = first_pixel_blue(&surface);
            if alone != 0xff || iterations == LAST_ITERATIONS {
                break;
            }
            gl.finish();
            iterations *= 4;
        }
        assert_ne!(
            alone, 0xff,
            "an Ordered fence with nothing ahead of it still came back complete at every size \
             tried, so this test cannot tell ordering from a render that finished on its own"
        );

        // The same Ordered fence, this time behind a Sync for that render. FIFO is what makes it
        // wait, and the reader must now see the finished colour.
        black(&gl);
        load.queue(&gl, iterations);
        let sync = gl.fence().expect("the driver gives a sync object");
        // With a ticket, so the same run checks that the context's in-flight count is given back
        // once the GPU is past the sync -- and not before: nothing retires ahead of this job.
        let gate = crate::vrend::in_flight::Gate::default();
        waiter.retire_context(
            Owed {
                pictures: Vec::new(),
                fence: Answer::Syncs(vec![sync]),
                ticket: Some(gate.ticket()),
                queries: false,
            },
            ContextId::new(1).expect("a context id"),
            RingIdx(0),
            FenceId(2),
        );
        waiter.retire_global(ordered(Vec::new()), ClientFenceId(3));
        assert_eq!(retired.recv().expect("the sink answers"), ClientFenceId(3));
        let behind = first_pixel_blue(&surface);
        assert_eq!(gate.queued(), 0, "the fence retired but still counts as in flight");

        eprintln!("[ordered] {iterations} iterations: alone={alone:#04x} behind={behind:#04x}");
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
        let gl = Gl::new(winsys.procs(), crate::vrend::features::Api::Gles(30));

        let surface =
            Arc::new(surface::Surface::plain(W, H, PixelFormat::Bgra).expect("an IOSurface"));
        let image = winsys
            .image_from_surface(Arc::clone(&surface) as Arc<dyn surface::Held>)
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
        let wait_gl = Gl::new(winsys.procs(), crate::vrend::features::Api::Gles(30));
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
        let load = Load::new(&gl);
        let mut iterations = FIRST_ITERATIONS;
        let mut unwaited;
        loop {
            black(&gl);
            load.queue(&gl, iterations);
            let fence = gl.fence().expect("the driver gives a sync object");
            ask.send((fence, false)).expect("the reader is listening");
            unwaited = answer.recv().expect("the reader answers");
            if unwaited != 0xff || iterations == LAST_ITERATIONS {
                break;
            }
            gl.finish();
            iterations *= 4;
        }
        assert_ne!(
            unwaited, 0xff,
            "an unwaited read came back complete at every size tried, so this test cannot tell a \
             working wait from a deleted one -- it proves nothing as written"
        );

        // Now the same thing, waited. Same thread, same context, same round trip.
        black(&gl);
        load.queue(&gl, iterations);
        let fence = gl.fence().expect("the driver gives a sync object");
        ask.send((fence, true)).expect("the reader is listening");
        let waited = answer.recv().expect("the reader answers");

        drop(ask);
        reader.join().expect("the reading thread does not panic");

        // Printed, not judged: what the control cost to arm, so a later reader can see whether it
        // armed easily or barely, and on what size of render.
        eprintln!(
            "[leg] iterations={iterations} unwaited=0x{unwaited:02x} waited=0x{waited:02x} \
             (0xff is the render this fence stands for)"
        );
        assert_eq!(
            waited, 0xff,
            "a CPU reader saw {waited:#04x} where the fence said the render had run -- the same \
             read was stale ({unwaited:#04x}) without the wait, so the wait is what makes it true"
        );

        gl.delete_framebuffer(fb);
    }

    /// A fence taken while a picture is decoding retires only once that picture has landed.
    ///
    /// Before decodes ran on their own threads, an END_FRAME was finished before any later fence
    /// could be taken, and the guest trusts every fence at or below one it is told of. The fence
    /// here has no GL work of its own, so nothing but the picture holds it back.
    #[test]
    fn a_fence_waits_for_the_pictures_decoding_ahead_of_it() {
        use std::time::Duration;

        use crate::vrend::video::pending::{Landing, Outcome};

        let _display = crate::vrend::one_display_at_a_time();
        let winsys = Winsys::open(Flavour::Gles).expect("the surfaceless display opens");
        let ctx = winsys
            .create_context(Version { major: 3, minor: 1 }, None)
            .expect("a GLES 3.1 context");
        winsys.make_current(&ctx).expect("ctx is current on this thread");

        let (tx, retired) = std::sync::mpsc::channel();
        let retirement = crate::fence::Retirement::start(
            Box::new(Recorder(tx)),
            crate::vrend::debug::Switches::default(),
        );
        let display = winsys.thread_display().expect("a winsys of our own lends its display");
        let wait_ctx = winsys
            .create_context(Version { major: 3, minor: 1 }, Some(&ctx))
            .expect("a shared context");
        let waiter = Waiter::start(
            display,
            wait_ctx,
            Gl::new(winsys.procs(), crate::vrend::features::Api::Gles(30)),
            retirement.handle(),
            debug::Switches::default(),
        );

        let landing = Landing::new();
        waiter.retire_global(ordered(vec![Arc::clone(&landing)]), ClientFenceId(1));
        assert!(
            retired.recv_timeout(Duration::from_millis(100)).is_err(),
            "the fence retired while the picture it was taken over was still decoding"
        );
        landing.land(Outcome::Nothing);
        assert_eq!(
            retired.recv_timeout(Duration::from_secs(5)).expect("the fence retires once it lands"),
            ClientFenceId(1)
        );
    }
}
