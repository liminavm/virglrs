// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! A decoded picture on its way into a decode target, between the END_FRAME that asked for it and
//! the first read that needs it.
//!
//! The hardware decode runs on a codec's own thread (see [`super::Codec`]), so the thread that
//! serves the control queue no longer waits for VideoToolbox. What that thread gives up is the
//! guarantee it used to have for free: a decode finished before the next command ran, so every
//! read of a target saw its picture. Here that guarantee is kept by the reader instead. Each
//! target texture carries a [`Pending`] while its picture is in flight, and whatever reads or
//! writes the texture settles it first -- waiting for the picture if it has not landed, and doing
//! the part of delivery that has to happen on the render thread.
//!
//! **The ticket lives on the texture, not on the codec or the context.** A target is read by
//! contexts that never decoded into it -- a compositor sampling a browser's frame -- and the
//! texture is the one thing every reader already reaches. A context-level list would be a second
//! container for a fact the texture holds, and a reader in another context could not find it.
//!
//! **Nothing here points back at the target.** A per-plane entry carries the plane's upload
//! recipe and the picture's [`Landing`]; it never holds the buffer or a texture, because the
//! texture holds the entry and the cycle would keep a picture -- and the decoder pool slot behind
//! it -- alive for as long as nobody read the target.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{SyncSender, TrySendError};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use super::TargetFormat;
use crate::decode::Picture;
use crate::vrend::formats::GlFormat;
use crate::vrend::gl::gles::{GL_TEXTURE_2D, GL_TEXTURE_BINDING_2D};
use crate::vrend::gl::{Gl, TextureName};
use crate::vrend::resource::Planes;

/// What a decode thread left for the render thread.
pub enum Outcome {
    /// A picture for per-plane targets, uploaded plane by plane as each plane's texture settles.
    Picture(Picture),
    /// The picture was written into a composite target's surface planes on the decode thread.
    /// What remains is to record that the planes moved, which is the render thread's to do.
    Written,
    /// Nothing reached the target: the host refused the frame, returned it wrong, or the frame
    /// was decoded only for its reference value. The target keeps what it held, which is what a
    /// synchronous decode that failed left too.
    Nothing,
}

enum Stage {
    InFlight,
    Landed(Outcome),
}

/// One decode's result, shared by the thread that produces it and everything waiting for it.
///
/// Waited on by the render thread when a read needs the picture, and by the fence waiter before
/// it retires a fence created after the END_FRAME. Only the render thread consumes what landed.
pub struct Landing {
    stage: Mutex<Stage>,
    landed: Condvar,
}

impl Landing {
    pub fn new() -> Arc<Landing> {
        Arc::new(Landing { stage: Mutex::new(Stage::InFlight), landed: Condvar::new() })
    }

    /// The decode thread's half: the job is done, whatever became of it.
    ///
    /// Every waiter is woken. There can be two -- the render thread settling a read and the fence
    /// waiter -- and waking one would leave the other asleep on a picture that has landed.
    pub fn land(&self, outcome: Outcome) {
        *self.lock() = Stage::Landed(outcome);
        self.landed.notify_all();
    }

    /// Whether the job is done. Asked without waiting, by the paths that must not block.
    pub fn is_landed(&self) -> bool {
        matches!(*self.lock(), Stage::Landed(_))
    }

    /// Block until the job is done. The fence waiter's wait: it needs to know the picture is in
    /// place, and consumes nothing.
    pub fn wait(&self) {
        drop(self.landed_guard());
    }

    fn landed_guard(&self) -> MutexGuard<'_, Stage> {
        let mut stage = self.lock();
        while matches!(*stage, Stage::InFlight) {
            stage = self.landed.wait(stage).expect("the landing lock is never held across a panic");
        }
        stage
    }

    fn lock(&self) -> MutexGuard<'_, Stage> {
        self.stage.lock().expect("the landing lock is never held across a panic")
    }
}

/// How many target textures carry an unsettled [`Pending`], renderer-wide.
///
/// The barrier sits on hot paths -- every draw's sampler bind looks its resources up -- and the
/// answer there is almost always that no decode is in flight anywhere. This is the one load that
/// says so. It is kept by the entries themselves: a [`Pending`] counts itself when it is made and
/// uncounts itself when it is dropped, so no settle path, replacement or teardown has to remember.
///
/// It also counts the reads that had to wait. A read that outruns the decoder blocks the render
/// thread exactly as every decode used to, so a regression here would look like the old stall
/// with nothing to name it; `VIRGLRS_SUBMIT_STATS` prints these beside the decode timings.
#[derive(Clone, Default)]
pub struct Unsettled(Arc<Counters>);

#[derive(Default)]
struct Counters {
    pending: AtomicUsize,
    reads: Window,
    replaces: Window,
    queue: Window,
    decoded: DecodeWindows,
}

/// Where a decode's time goes on its codec's thread.
#[derive(Default)]
struct DecodeWindows {
    queued: Window,
    session: Window,
    write: Window,
}

/// One kind of wait the render thread made for the decoder, counted since the last report.
#[derive(Default)]
struct Window {
    waits: AtomicU64,
    waited_us: AtomicU64,
    longest_us: AtomicU64,
}

impl Window {
    fn record(&self, took: Duration) {
        let us = u64::try_from(took.as_micros()).unwrap_or(u64::MAX);
        self.waits.fetch_add(1, Ordering::Relaxed);
        self.waited_us.fetch_add(us, Ordering::Relaxed);
        self.longest_us.fetch_max(us, Ordering::Relaxed);
    }

    fn take(&self) -> Waited {
        Waited {
            count: self.waits.swap(0, Ordering::Relaxed),
            total: Duration::from_micros(self.waited_us.swap(0, Ordering::Relaxed)),
            longest: Duration::from_micros(self.longest_us.swap(0, Ordering::Relaxed)),
        }
    }
}

/// How often one kind of wait happened in a report's window, how long it took in all, and the
/// longest single one.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Waited {
    pub count: u64,
    pub total: Duration,
    pub longest: Duration,
}

/// Where the decode thread's time went, per phase, in a report's window.
///
/// `queued` is from the send to the decode thread taking the job, `session` is the VideoToolbox
/// decode, and `write` is copying the picture into a composite target's planes (none for a
/// per-plane target, whose planes upload on the render thread). A fence taken behind a decode
/// waits at most the sum, which is what makes these the bound on how long one context's decode
/// can hold back another's fences.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct DecodeTimes {
    pub queued: Waited,
    pub session: Waited,
    pub write: Waited,
}

/// One decode's phases, measured on the decode thread; see [`DecodeTimes`].
#[derive(Clone, Copy, Default)]
pub struct Phases {
    pub queued: Duration,
    pub session: Option<Duration>,
    pub write: Option<Duration>,
}

/// Every way the render thread waits for the decoder, each counted on its own.
///
/// Only `reads` is a read outrunning its picture. The other two are waits END_FRAME itself makes,
/// which a read count cannot see: `replaces` is a decode into a target whose previous picture has
/// not landed and was never read, and `queue` is a decode sent into a codec whose queue is full.
/// Under a clamped or overloaded host all three grow, and telling them apart from plain CPU
/// denial is what they are for.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Stalls {
    pub reads: Waited,
    pub replaces: Waited,
    pub queue: Waited,
}

impl Unsettled {
    /// Whether any target anywhere has a picture not yet settled.
    pub fn any(&self) -> bool {
        self.0.pending.load(Ordering::Acquire) != 0
    }

    /// Record one decode's phases. Called on the decode thread; a phase that did not run -- a
    /// session that could not be built, a picture with no planes to write -- is not counted.
    pub fn record_decode(&self, phases: Phases) {
        let d = &self.0.decoded;
        d.queued.record(phases.queued);
        if let Some(took) = phases.session {
            d.session.record(took);
        }
        if let Some(took) = phases.write {
            d.write.record(took);
        }
    }

    /// The decode thread's phase times since the last call. Taken, like [`Self::take_waits`].
    pub fn take_decode_times(&self) -> DecodeTimes {
        let d = &self.0.decoded;
        DecodeTimes { queued: d.queued.take(), session: d.session.take(), write: d.write.take() }
    }

    /// Every wait for the decoder since the last call. Taken, so each report covers its own
    /// window.
    pub fn take_waits(&self) -> Stalls {
        Stalls {
            reads: self.0.reads.take(),
            replaces: self.0.replaces.take(),
            queue: self.0.queue.take(),
        }
    }

    fn count(&self) -> Counted {
        self.0.pending.fetch_add(1, Ordering::AcqRel);
        Counted(Arc::clone(&self.0))
    }
}

/// Send `item` into a bounded queue, counting the send as a queue wait in `unsettled` if the queue
/// was full and the render thread had to block for room.
pub fn enqueue<T>(queue: &SyncSender<T>, item: T, unsettled: &Unsettled) {
    let item = match queue.try_send(item) {
        Ok(()) => return,
        Err(TrySendError::Full(item)) => item,
        Err(TrySendError::Disconnected(_)) => panic!("a codec's decode thread outlives its queue"),
    };
    let began = Instant::now();
    queue.send(item).expect("a codec's decode thread outlives its queue");
    unsettled.0.queue.record(began.elapsed());
}

/// One unit of [`Unsettled`], given back on drop.
struct Counted(Arc<Counters>);

impl Drop for Counted {
    fn drop(&mut self) {
        self.0.pending.fetch_sub(1, Ordering::AcqRel);
    }
}

/// What settling a texture has left to do once its picture has landed.
pub enum Recipe {
    /// A composite target: note that its planes moved, so the conversion into the base texture
    /// runs for whatever samples it whole.
    Composite,
    /// One plane of a per-plane target: upload that plane of the picture into this texture.
    Plane(PlaneUpload),
}

/// Everything a per-plane upload needs except the texture it lands in, which is the one holding
/// the entry.
pub struct PlaneUpload {
    /// Which of the target's planes this texture is.
    pub index: usize,
    /// The layout the target was allocated in, which maps a target plane to a picture plane.
    pub layout: TargetFormat,
    pub gl: GlFormat,
    pub block_bytes: u32,
    pub width: u32,
    pub height: u32,
}

/// A picture in flight into one target texture.
pub struct Pending {
    landing: Arc<Landing>,
    recipe: Recipe,
    counted: Counted,
}

impl Pending {
    pub fn new(landing: Arc<Landing>, recipe: Recipe, unsettled: &Unsettled) -> Pending {
        Pending { landing, recipe, counted: unsettled.count() }
    }
}

/// Whether to wait for a picture that has not landed.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Wait {
    /// A read that needs the picture: block until it lands.
    Block,
    /// A walk that only wants what is already there, and must not stall the render thread.
    IfLanded,
}

/// What settling a texture did.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Settled {
    /// Nothing was pending.
    Clean,
    /// A picture was pending and is now delivered. `waited` says whether the render thread had to
    /// block for it -- a read that outran the decoder, which is the old stall in miniature and is
    /// counted. `fill` says a composite view now owes the conversion.
    Delivered { waited: bool, fill: bool },
    /// Still in flight, and the caller asked not to wait.
    InFlight,
}

/// The pending picture a texture carries, if any. See the module docs.
///
/// The lock is uncontended in practice -- only the render thread attaches and settles, and the
/// decode thread never touches a texture -- and buys the `Sync` a texture shared between contexts
/// needs, exactly as [`crate::vrend::resource::Texture`]'s view table does.
#[derive(Default)]
pub struct Slot(Mutex<Option<Pending>>);

impl Slot {
    /// Put a new picture in flight into this texture.
    ///
    /// A picture already pending here is delivered first, waiting for it if it has to: a target
    /// decoded into twice before anything read it must still end up holding the first picture
    /// if the second decode fails, which is what a synchronous decode left it holding. It is
    /// rare -- a guest reusing a target before anything read it -- and it costs one wait.
    ///
    /// Such a wait is counted as a replacement, not as a read (see [`Stalls`]).
    pub fn attach(&self, gl: &Gl, name: TextureName, planes: Option<&Planes>, pending: Pending) {
        let replaced = self.lock().replace(pending);
        if let Some(replaced) = replaced {
            let began = (!replaced.landing.is_landed()).then(Instant::now);
            let counters = Arc::clone(&replaced.counted.0);
            deliver(replaced, gl, name, planes);
            if let Some(began) = began {
                counters.replaces.record(began.elapsed());
            }
        }
    }

    /// Whether a picture is in flight into this texture and has not landed. Asked by the walks
    /// that must not block, to leave such a target alone rather than work on half a picture.
    pub fn in_flight(&self) -> bool {
        self.lock().as_ref().is_some_and(|p| !p.landing.is_landed())
    }

    /// The landing this texture is waiting on, for a fence to wait on too.
    pub fn landing(&self) -> Option<Arc<Landing>> {
        self.lock().as_ref().map(|p| Arc::clone(&p.landing))
    }

    /// Deliver this texture's pending picture, if it has one.
    ///
    /// `name` and `planes` are the texture's own, passed in because the texture holds this slot.
    /// A per-plane upload runs in whatever GL context is current, so it pins the unpack state and
    /// puts back the `GL_TEXTURE_2D` binding it borrows: the caller may be in the middle of
    /// binding a draw's samplers.
    pub fn settle(
        &self,
        gl: &Gl,
        name: TextureName,
        planes: Option<&Planes>,
        wait: Wait,
    ) -> Settled {
        // The entry is taken out before anything waits, so the texture's lock is never held
        // across a decode. Only the render thread settles, so nothing can attach behind it.
        let Some(pending) = self.lock().take() else {
            return Settled::Clean;
        };
        let waited = !pending.landing.is_landed();
        if waited && wait == Wait::IfLanded {
            *self.lock() = Some(pending);
            return Settled::InFlight;
        }
        let began = waited.then(Instant::now);
        let counted = Arc::clone(&pending.counted.0);
        let fill = deliver(pending, gl, name, planes);
        if let Some(began) = began {
            counted.reads.record(began.elapsed());
        }
        Settled::Delivered { waited, fill }
    }

    fn lock(&self) -> MutexGuard<'_, Option<Pending>> {
        self.0.lock().expect("the pending slot lock is never held across a panic")
    }
}

/// Wait for a pending picture and do the render thread's half of delivering it. Returns whether a
/// composite view now owes the conversion.
fn deliver(pending: Pending, gl: &Gl, name: TextureName, planes: Option<&Planes>) -> bool {
    let stage = pending.landing.landed_guard();
    let Stage::Landed(outcome) = &*stage else {
        unreachable!("landed_guard returns only once the job has landed");
    };
    match (&pending.recipe, outcome) {
        (Recipe::Composite, Outcome::Written) => {
            planes.expect("a composite recipe is only attached to a composite target").delivered()
        }
        (Recipe::Plane(upload), Outcome::Picture(picture)) => {
            upload_plane(gl, name, upload, picture);
            false
        }
        // Nothing reached the target, or the recipe and the outcome are for different kinds of
        // target -- which the attach cannot produce, and which delivers nothing either way.
        _ => false,
    }
}

/// Upload one plane of a decoded picture into its texture.
///
/// The arithmetic is the synchronous path's, unchanged: clamp to what the source plane holds, and
/// stride by the decoder's pitch. See the notes on [`super::Buffer`]'s per-plane delivery.
fn upload_plane(gl: &Gl, name: TextureName, upload: &PlaneUpload, picture: &Picture) {
    let Some(locked) = picture.lock() else {
        eprintln!(
            "[virglrs] video: a decoded picture could not be mapped; the plane keeps what it held"
        );
        return;
    };
    let count = locked.plane_count();
    // A target with more planes than the picture is the guest's choice of layout, not an error:
    // the planes past the picture's are left alone, as they always were.
    if upload.index >= count {
        return;
    }
    let Some(source) = locked.plane(upload.layout.source_plane(upload.index, count)) else {
        return;
    };
    let row_pixels = source.pitch as u32 / upload.block_bytes;
    let w = upload.width.min(row_pixels);
    let h = upload.height.min(source.height);
    if w == 0 || h == 0 {
        return;
    }
    let previous = gl.get_integer(GL_TEXTURE_BINDING_2D) as u32;
    gl.unpack_tight();
    gl.bind_texture(GL_TEXTURE_2D, Some(name));
    let ok = gl.tex_sub_image_2d_padded(
        GL_TEXTURE_2D,
        0,
        0,
        0,
        w as i32,
        h as i32,
        upload.gl.glformat,
        upload.gl.gltype,
        source.bytes,
        row_pixels as i32,
    );
    gl.bind_texture_name(GL_TEXTURE_2D, previous);
    // The source is a slice sized by CoreVideo, the rectangle is clamped to it just above, and
    // the plane's bytes per pixel is the same number the upload reads by -- `Plane::new` refused
    // the formats where the two disagree. So a refusal here is this arithmetic being wrong: a
    // host bug, and one that would otherwise show as a target holding the previous frame.
    assert!(ok, "a decoded plane clamped to its own extent does not fit it");
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::vrend::egl::{Flavour, Version, Winsys};

    /// A GL handle for the settle's signature. None of these tests reaches a GL call -- nothing
    /// they land is a picture -- but the settle is written against the one it is handed.
    fn gl() -> (std::sync::MutexGuard<'static, ()>, Winsys, crate::vrend::egl::Context, Gl) {
        let display = crate::vrend::one_display_at_a_time();
        let winsys = Winsys::open(Flavour::Gles).expect("the surfaceless display opens");
        let ctx = winsys
            .create_context(Version { major: 3, minor: 1 }, None)
            .expect("a GLES 3.1 context");
        winsys.make_current(&ctx).expect("ctx is current on this thread");
        let gl = Gl::new(winsys.gles());
        (display, winsys, ctx, gl)
    }

    fn name() -> TextureName {
        TextureName::unbacked(1)
    }

    /// Land `landing` with nothing, `after` from now, on another thread.
    fn land_later(landing: &Arc<Landing>, after: Duration) -> std::thread::JoinHandle<()> {
        let landing = Arc::clone(landing);
        std::thread::spawn(move || {
            std::thread::sleep(after);
            landing.land(Outcome::Nothing);
        })
    }

    /// The fence waiter and a reader can wait on one picture at once, and both wake when it lands.
    #[test]
    fn a_landing_wakes_every_waiter() {
        let landing = Landing::new();
        let waiters: Vec<_> = (0..2)
            .map(|_| {
                let landing = Arc::clone(&landing);
                std::thread::spawn(move || landing.wait())
            })
            .collect();
        std::thread::sleep(Duration::from_millis(20));
        landing.land(Outcome::Nothing);
        for waiter in waiters {
            waiter.join().expect("each waiter wakes");
        }
        assert!(landing.is_landed());
    }

    /// The count the hot paths ask is kept by the entries themselves, so dropping one -- settled,
    /// replaced or torn down with its texture -- is all it takes to uncount it.
    #[test]
    fn a_pending_picture_counts_itself_for_as_long_as_it_lives() {
        let unsettled = Unsettled::default();
        assert!(!unsettled.any());
        let first = Pending::new(Landing::new(), Recipe::Composite, &unsettled);
        let second = Pending::new(Landing::new(), Recipe::Composite, &unsettled);
        assert!(unsettled.any());
        drop(first);
        assert!(unsettled.any(), "one is still pending");
        drop(second);
        assert!(!unsettled.any());
    }

    /// A walk that must not block leaves an unlanded picture where it was, still counted, and a
    /// later read settles it.
    #[test]
    fn a_walk_that_must_not_wait_leaves_an_unlanded_picture_pending() {
        let (_display, _winsys, _ctx, gl) = gl();
        let unsettled = Unsettled::default();
        let slot = Slot::default();
        let landing = Landing::new();
        slot.attach(
            &gl,
            name(),
            None,
            Pending::new(Arc::clone(&landing), Recipe::Composite, &unsettled),
        );

        assert_eq!(slot.settle(&gl, name(), None, Wait::IfLanded), Settled::InFlight);
        assert!(slot.in_flight() && unsettled.any(), "the picture is still on its way");

        landing.land(Outcome::Nothing);
        assert_eq!(
            slot.settle(&gl, name(), None, Wait::IfLanded),
            Settled::Delivered { waited: false, fill: false }
        );
        assert!(!unsettled.any());
        assert_eq!(slot.settle(&gl, name(), None, Wait::Block), Settled::Clean);
    }

    /// A read that outruns the decoder waits for the picture, and the wait is counted where the
    /// submit stats will find it.
    #[test]
    fn a_read_that_outruns_the_decoder_waits_and_is_counted() {
        let (_display, _winsys, _ctx, gl) = gl();
        let unsettled = Unsettled::default();
        let slot = Slot::default();
        let landing = Landing::new();
        slot.attach(
            &gl,
            name(),
            None,
            Pending::new(Arc::clone(&landing), Recipe::Composite, &unsettled),
        );

        let lander = land_later(&landing, Duration::from_millis(40));
        assert_eq!(
            slot.settle(&gl, name(), None, Wait::Block),
            Settled::Delivered { waited: true, fill: false }
        );
        lander.join().expect("the lander finishes");
        let stalls = unsettled.take_waits();
        assert_eq!(stalls.reads.count, 1);
        assert!(stalls.reads.total >= Duration::from_millis(30), "the wait was {stalls:?}");
        assert_eq!(stalls.reads.total, stalls.reads.longest);
        assert_eq!((stalls.replaces.count, stalls.queue.count), (0, 0), "a read is only a read");
        assert_eq!(unsettled.take_waits(), Stalls::default(), "each report takes its own window");
    }

    /// A second decode into a target that nothing read yet delivers the first picture before it
    /// replaces it, waiting if it has to: the first picture is what the target holds if the
    /// second decode fails.
    #[test]
    fn a_target_decoded_into_twice_takes_the_first_picture_before_the_second() {
        let (_display, _winsys, _ctx, gl) = gl();
        let unsettled = Unsettled::default();
        let slot = Slot::default();
        let first = Landing::new();
        slot.attach(
            &gl,
            name(),
            None,
            Pending::new(Arc::clone(&first), Recipe::Composite, &unsettled),
        );
        let lander = land_later(&first, Duration::from_millis(20));

        let second = Landing::new();
        slot.attach(
            &gl,
            name(),
            None,
            Pending::new(Arc::clone(&second), Recipe::Composite, &unsettled),
        );
        // Read before the join, which would land it whether or not the attach waited.
        let landed_by_the_attach = first.is_landed();
        lander.join().expect("the lander finishes");
        assert!(landed_by_the_attach, "the attach did not wait for the first picture");
        assert!(slot.in_flight(), "the second is what is pending now");
        let stalls = unsettled.take_waits();
        assert_eq!(stalls.reads.count, 0, "replacing is not a read");
        assert_eq!(stalls.replaces.count, 1, "the wait for the first picture was not counted");
        assert!(stalls.replaces.total > Duration::ZERO);
    }

    /// A send into a queue with room is not a wait, and costs nothing to count.
    #[test]
    fn a_send_into_a_queue_with_room_is_not_counted() {
        let unsettled = Unsettled::default();
        let (queue, _jobs) = std::sync::mpsc::sync_channel(1);
        enqueue(&queue, 1u32, &unsettled);
        assert_eq!(unsettled.take_waits(), Stalls::default());
    }

    /// A send into a full queue blocks the render thread until the decode thread makes room, and
    /// that wait is counted as the queue's, not as a read's.
    #[test]
    fn a_send_into_a_full_queue_waits_and_is_counted() {
        let unsettled = Unsettled::default();
        let (queue, jobs) = std::sync::mpsc::sync_channel(1);
        enqueue(&queue, 1u32, &unsettled);
        let drainer = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(40));
            let taken: Vec<u32> = jobs.iter().take(2).collect();
            taken
        });
        enqueue(&queue, 2u32, &unsettled);
        assert_eq!(drainer.join().expect("the drainer finishes"), vec![1, 2], "order is kept");
        let stalls = unsettled.take_waits();
        assert_eq!(stalls.queue.count, 1);
        assert!(stalls.queue.total >= Duration::from_millis(30), "the wait was {stalls:?}");
        assert_eq!(stalls.reads.count, 0, "a full queue is not a read");
    }

    /// A decode's phases land in their own windows, and a phase that did not run is not counted
    /// as one that took no time.
    #[test]
    fn a_decode_records_only_the_phases_it_ran() {
        let unsettled = Unsettled::default();
        unsettled.record_decode(Phases {
            queued: Duration::from_millis(1),
            session: Some(Duration::from_millis(3)),
            write: Some(Duration::from_millis(2)),
        });
        unsettled.record_decode(Phases {
            queued: Duration::from_millis(5),
            session: None,
            write: None,
        });
        let times = unsettled.take_decode_times();
        assert_eq!((times.queued.count, times.session.count, times.write.count), (2, 1, 1));
        assert_eq!(times.queued.longest, Duration::from_millis(5));
        assert_eq!(times.session.total, Duration::from_millis(3));
        assert_eq!(unsettled.take_decode_times(), DecodeTimes::default(), "taken per window");
        assert_eq!(unsettled.take_waits(), Stalls::default(), "a decode is not a wait");
    }
}
