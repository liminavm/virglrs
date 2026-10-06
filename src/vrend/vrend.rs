// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! The classic renderer's root: the winsys, the driver, what it can do, and every GL object the
//! guests own.
//!
//! One `Vrend` per renderer, owned by it. It holds the winsys and the context every API-path
//! operation runs on (the C's `ctx0`), the feature and format tables probed once at init, the
//! host side of every classic resource keyed by the handle the renderer files it under, and one
//! GL context per guest context, all sharing ctx0's objects.
//!
//! Which context is current is tracked here and switched only when it changes, the way
//! `vrend_hw_switch_context` does: every entry point names the context it needs, and the switch
//! is one place rather than a habit.

use super::blitter;
use super::caps;
use super::context::{Context, Fault, Guest, Host, Todo, Unfed};
use super::current::{Current, GlContext};
use super::egl::{self, EglError, Flavour, GlContexts, Version, Winsys};
use super::features::{Api, Feature, Features};
use super::formats::Table;
use super::gl::gles::GL_CONTEXT_CORE_PROFILE_BIT;
use super::gl::{self, GLenum, Gl, RobustReads};
use super::journal::{Census, Seq};
use super::pipe::TextureTarget;
use super::resource::{self, Args, Limits, Refusal, Resource};
use super::shader;
use super::tally;
use super::transfer::{self, Info};
use super::waiter::{self, Answer, Owed};
use crate::config::{Config, HostGl};
use crate::decode;
use crate::guest_mem::{Iov, PixelSource};
use crate::ids::{BlobId, ClientFenceId, ContextId, FenceId, ResourceHandle, RingIdx};
use crate::renderer::ClassicCtx;
use crate::surface;
use std::fmt;
use std::sync::Arc;

/// Why the classic renderer could not come up.
#[derive(Debug)]
pub enum InitError {
    Egl(EglError),
    /// No context of the GL asked for could be made.
    NoContext,
    /// The context that was made speaks a GL vrend does not run on, carrying its `GL_VERSION`.
    /// Reachable when something else minted it: a display of ours is asked for the GL this
    /// renderer runs on, and gives it or nothing.
    UnservedGl {
        version: String,
        why: UnservedGl,
    },
}

/// Why a context's GL is not one vrend runs on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UnservedGl {
    /// Desktop GL, which the caller did not ask for: see `Config::host_gl`.
    NotAskedFor,
    /// A compatibility-profile context. The C has a leg for one; this renderer does not.
    Compatibility,
    /// Desktop GL older than 3.3, the oldest core profile with what vrend's translation needs.
    TooOld,
}

impl fmt::Display for InitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InitError::Egl(e) => write!(f, "{e}"),
            InitError::NoContext => f.write_str("no context of the GL asked for could be created"),
            InitError::UnservedGl { version, why } => {
                let why = match why {
                    UnservedGl::NotAskedFor => {
                        "desktop GL, which `Config::host_gl` did not ask for"
                    }
                    UnservedGl::Compatibility => "a compatibility profile, which is not served",
                    UnservedGl::TooOld => "older than the 3.3 core profile vrend needs",
                };
                write!(f, "the GL context is {version}: {why}")
            }
        }
    }
}

impl From<EglError> for InitError {
    fn from(e: EglError) -> InitError {
        InitError::Egl(e)
    }
}

pub struct Vrend {
    winsys: Winsys,
    gl: Gl,
    pub features: Features,
    pub formats: Table,
    pub limits: Limits,
    /// What the translator may assume of the host, read once alongside the limits.
    shader_cfg: shader::Config,
    /// What the guest's driver is told of the host, probed once from the same answers.
    caps: caps::CapsV2,
    /// What VideoToolbox decodes here, or `None` when the caller did not ask for video.
    ///
    /// Probing is what registers the supplemental decoders, so this is also the record that
    /// registration happened: nothing can ask this host about a codec without holding one.
    video: Option<decode::Support>,
    ctx0: egl::Context,
    /// The version guest contexts are made with: the newest the driver gave ctx0.
    version: Version,
    current: Current,
    /// Vrend's half of the resource table. Its owner is the renderer's table: an entry here
    /// lives exactly as long as the [`resource::Claim`] the renderer holds under the same
    /// handle, and [`resource::Slots`] is what makes that true rather than a rule to follow.
    resources: resource::Slots,
    contexts: crate::Map<ContextId, Context>,
    pub todo: Todo,
    /// What the command path costs per guest command. Inert unless armed -- see
    /// [`tally::Tally`], which says why a profiler cannot answer this.
    tally: tally::Tally,
    /// The buffer every texture transfer stages through; see [`transfer::Staging`].
    staging: transfer::Staging,
    /// The shader blitter and its GL context, built on the first blit that needs one. A renderer
    /// that never takes the blitter's path never pays for it.
    blitter: Option<blitter::Blitter>,
    /// The thread classic fences are waited on, and `None` where the driver would not give it a
    /// context of its own -- in which case the fence path falls back to finishing inline, which is
    /// correct and merely slow. The C does the same when its sync context fails to come up.
    waiter: Option<waiter::Waiter>,
    /// Where a fence goes once it is answered. Held here as well as by the waiter, because a
    /// renderer with no waiter answers its fences inline and still has to retire them.
    fences: crate::fence::Handle,
    /// Texture storage the guest has freed that something else still holds a share of.
    ///
    /// The share cannot delete itself when the last holder lets go, because deleting needs a
    /// current GL context and the driver, and a drop has neither. So it is parked here and swept
    /// from the next place that has both.
    doomed: Vec<Arc<resource::Texture>>,
    /// Which resources copy guest pages, as of the batch it was last asked in.
    pixels: resource::Refresh,
    /// How many decode targets, across every context, have a picture in flight that nothing has
    /// settled yet. See [`crate::vrend::video::pending::Unsettled`].
    unsettled: super::video::pending::Unsettled,
    /// Batches run, ever. The unit a copy of a guest's pages is kept fresh in: within one batch
    /// the guest has had no opportunity to run, so one read serves every draw in it.
    batch: u64,
    /// Classic's handle to the renderer's host-memory ledger -- see [`crate::budget`]. What it
    /// charges is the IOSurfaces this arm mints and nothing else; ordinary GL storage is the
    /// driver's and this process cannot see it.
    budget: crate::budget::Classic,
    /// limina's trace knobs, as the renderer read them when it was built.
    traces: super::debug::Traces,
    /// `VIRGLRS_DEBUG`'s switches, as the renderer read them when it was built.
    debug: super::debug::Switches,
    /// `VIRGLRS_FENCE_FINISH=1`: every fence finishes every context inline, on this thread -- the
    /// behaviour the fence path replaced, kept so the two can be compared on one build the way the
    /// cost of the finish was measured in the first place. Retirement still goes through the
    /// waiter's queue, so the comparison changes what a fence costs and not what it means.
    fence_finish: bool,
    /// How many fences one context may have in the waiter's queue before its next batch waits for
    /// the oldest to retire; 0 for no bound. See [`super::in_flight`].
    fence_depth: usize,
    /// What the bound has cost since it last said so.
    throttled: Throttled,
}

/// How many fences one context may have in flight before its next batch waits, unless
/// `VIRGLRS_CLASSIC_FENCE_DEPTH` says otherwise. A guest asks for about one fence per execbuffer,
/// and a desktop's contexts sit at a handful; this is well above that and well below the thousands
/// of render passes an unbounded context was measured queueing.
const FENCE_DEPTH_DEFAULT: usize = 16;

/// How long a batch waits for its context to drop below the bound before running anyway. Long
/// enough for any real frame to retire, short enough that a sync which never signals costs the
/// other contexts a hiccup rather than the desktop.
const FENCE_DEPTH_PATIENCE: std::time::Duration = std::time::Duration::from_secs(2);

/// `VIRGLRS_CLASSIC_FENCE_DEPTH`: the bound, `0` for none. Anything that is not a number keeps
/// the default and says so, rather than silently switching the bound off.
fn fence_depth_from_env() -> usize {
    match std::env::var("VIRGLRS_CLASSIC_FENCE_DEPTH") {
        Err(_) => FENCE_DEPTH_DEFAULT,
        Ok(v) => v.trim().parse().unwrap_or_else(|_| {
            eprintln!(
                "[virglrs] VIRGLRS_CLASSIC_FENCE_DEPTH={v:?} is not a count; keeping {FENCE_DEPTH_DEFAULT}"
            );
            FENCE_DEPTH_DEFAULT
        }),
    }
}

/// How often the bound reports what it has cost, at most.
const THROTTLE_REPORT: std::time::Duration = std::time::Duration::from_secs(5);

/// Batches the in-flight bound held back since the last report. Printed at most every
/// [`THROTTLE_REPORT`], and only when it did something: a context that never reaches the bound
/// never prints.
#[derive(Default)]
struct Throttled {
    waits: u64,
    waited: std::time::Duration,
    gave_up: u64,
    last_report: Option<std::time::Instant>,
}

impl Throttled {
    fn note(&mut self, ctx: ContextId, waited: super::in_flight::Waited, depth: usize) {
        use super::in_flight::Waited;
        match waited {
            Waited::No => return,
            Waited::For(d) => {
                self.waits += 1;
                self.waited += d;
            }
            Waited::GaveUp(d) => {
                self.waits += 1;
                self.waited += d;
                self.gave_up += 1;
                eprintln!(
                    "[virglrs] vrend: ctx {ctx:?} still had {depth} fences in flight after {d:?}; \
                     running its batch anyway -- a sync that does not signal?"
                );
            }
        }
        let now = std::time::Instant::now();
        if self.last_report.is_some_and(|t| now - t < THROTTLE_REPORT) {
            return;
        }
        eprintln!(
            "[virglrs] vrend: fence depth bound ({depth}) held {} batches for {:?} in all \
             ({} gave up) since the last report; latest ctx {ctx:?}",
            self.waits, self.waited, self.gave_up
        );
        *self = Throttled { last_report: Some(now), ..Throttled::default() };
    }
}

/// The versions tried, newest first: the C's `gl_versions` ladder, whose desktop rows stop at the
/// oldest core profile vrend serves. On a display of ours the flavour picks its half; under an
/// embedder that may hand back either API, the whole ladder is walked when desktop GL was asked
/// for, as the C walks it.
const VERSIONS: [Version; 11] = [
    Version { major: 4, minor: 6 },
    Version { major: 4, minor: 5 },
    Version { major: 4, minor: 4 },
    Version { major: 4, minor: 3 },
    Version { major: 4, minor: 2 },
    Version { major: 4, minor: 1 },
    Version { major: 4, minor: 0 },
    Version { major: 3, minor: 3 },
    Version { major: 3, minor: 2 },
    Version { major: 3, minor: 1 },
    Version { major: 3, minor: 0 },
];
const DESKTOP_VERSIONS: &[Version] = VERSIONS.split_at(8).0;
const GLES_VERSIONS: &[Version] = VERSIONS.split_at(8).1;

/// Why a `RESOURCE_CREATE_BLOB` naming a classic context's blob id was refused.
///
/// Each is the guest's error and each wants a different investigation, which is why they are not
/// one "invalid": nothing described the id at all says the `PIPE_RESOURCE_CREATE` never arrived
/// or already went to another claim; a size mismatch says the guest's two halves disagree about
/// how big its allocation is; unmappable says the host built the buffer and the driver would not
/// hand its pages over.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ClaimRefused {
    /// No resource stands under that id in that context.
    NotDescribed,
    /// The blob is larger than the resource backing it.
    Oversize { asked: u64, allocated: u32 },
    /// The driver refused the persistent mapping, or the storage never admitted one.
    Unmappable,
}

/// Why a replay step did not run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ReplayRefused {
    /// No classic context stands under that id.
    NoContext,
    /// The context is not between `replay_begin` and `replay_end`, so it is live and a journal
    /// fed to it would be replayed over what the guest has built since.
    NotReplaying,
    /// A retained command poisoned the context, and the feed stopped there.
    Poisoned,
    /// The context would not take the journal it was handed, for this reason.
    JournalRefused(&'static str),
}

/// Whether a blob of `size` bytes may be published from a resource of `width` bytes.
///
/// The wire carries the two independently and the C reconciles neither -- `vrend_get_blob_pipe`
/// takes `blob_size` as `UNUSED`. They are one fact with two spellings, and the guest is the only
/// thing that can make them disagree: asking to publish more than was allocated is asking the VMM
/// to map whatever the driver put after the buffer. Refused here, where both halves are in hand,
/// and never clamped -- a clamp reports success for a mapping the guest did not ask for and will
/// index past.
///
/// Smaller is not a mismatch. A guest may publish part of what it allocated, and the mapping it
/// is given is still bounded by the buffer.
fn publishable(size: u64, width: u32) -> Result<(), ClaimRefused> {
    if size > width as u64 {
        return Err(ClaimRefused::Oversize { asked: size, allocated: width });
    }
    Ok(())
}

impl fmt::Display for ClaimRefused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClaimRefused::NotDescribed => write!(f, "no resource was described under that blob id"),
            ClaimRefused::Oversize { asked, allocated } => {
                write!(f, "asked to publish {asked} bytes of a {allocated}-byte resource")
            }
            ClaimRefused::Unmappable => write!(f, "the buffer's pages cannot be mapped"),
        }
    }
}

/// A cursor image, as read back from the resource the guest set it from.
///
/// The extent travels with the pixels because neither means anything alone: a caller told a size
/// separately from the buffer it measures is a caller that can be told the wrong shape for the
/// bytes it has.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Cursor {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

impl Vrend {
    /// Open the winsys, bring ctx0 up on this thread and probe the driver.
    ///
    /// `contexts` is an embedder's context factory, when there is one; without it the winsys is
    /// this renderer's own surfaceless display. See [`GlContexts`].
    pub fn new(
        config: Config,
        budget: &Arc<crate::budget::Budget>,
        fences: crate::fence::Handle,
        contexts: Option<Box<dyn GlContexts>>,
        condemned: resource::Condemned,
        traces: super::debug::Traces,
        debug: super::debug::Switches,
    ) -> Result<Vrend, InitError> {
        let fences_for_inline = fences.clone();
        // An embedder's winsys arrives with ctx0 already made and current, because its display is
        // discovered through that context; ours is opened first and asked for one.
        let (winsys, ctx0, version) = match contexts {
            Some(contexts) => Winsys::embedded(
                contexts,
                match config.host_gl {
                    HostGl::Gles => GLES_VERSIONS,
                    HostGl::Desktop => &VERSIONS,
                },
            )?,
            None => {
                let (flavour, versions) = match config.host_gl {
                    HostGl::Gles => (Flavour::Gles, GLES_VERSIONS),
                    HostGl::Desktop => (Flavour::Gl, DESKTOP_VERSIONS),
                };
                let winsys = Winsys::open(flavour)?;
                let mut made = None;
                for &v in versions {
                    if let Ok(c) = winsys.create_context(v, None) {
                        made = Some((c, v));
                        break;
                    }
                }
                let (ctx0, version) = made.ok_or(InitError::NoContext)?;
                winsys.make_current(&ctx0)?;
                (winsys, ctx0, version)
            }
        };
        let procs = winsys.procs();
        let version_string = gl::version_string(&procs);
        // Whose choice the client API was depends on who minted the context, so it is read back
        // rather than assumed.
        let api = host_api(&version_string, || gl::profile_mask(&procs), config.host_gl)
            .map_err(|why| InitError::UnservedGl { version: version_string.clone(), why })?;
        let gl = Gl::new(procs, api);
        let mut features = Features::probe(api, gl.extensions());
        if !winsys.has_gl_colorspace() {
            features.clear(Feature::srgb_write_control);
        }
        features.reconcile(&gl);
        let robust = RobustReads::choose(api, &features);
        let gl = gl.reading(robust);
        let limits = Limits::query(&gl, &features);
        let shader_cfg = shader::Config::probe(&gl, &features, &limits);
        let formats = Table::probe(&gl, &features);
        let video = config.video.then(decode::Support::probe);
        if let Some(support) = video {
            // Detached: the thread holds nothing of the renderer, and a real session built while
            // it is still running only pays what it would have paid anyway.
            drop(decode::warm_up(&support));
            let names: Vec<&str> = decode::Codec::ALL
                .iter()
                .filter(|c| support.decodes(**c))
                .map(|c| c.name())
                .collect();
            eprintln!(
                "[virglrs] vrend: hardware video decode {}",
                if names.is_empty() {
                    "UNAVAILABLE -- this host has silicon for none of the codecs we serve".into()
                } else {
                    names.join(" ")
                },
            );
        }
        let mut caps = caps::CapsV2::probe(&gl, &features, &limits, &formats, video.as_ref());
        // The C's gbm-layout feature: shared buffers allocated linear, and the guest told every
        // shared buffer's layout. On by default (`Config::linear_shared`, where the C has it off)
        // and only with venus, as the C ties it to `VIRGL_RENDERER_VENUS`: the layout matters to a
        // venus import, and without venus there is nothing to pay the tiling for.
        #[cfg(not(target_os = "macos"))]
        let mut winsys = winsys;
        #[cfg(not(target_os = "macos"))]
        if config.linear_shared && config.venus {
            match winsys.linear_shared() {
                Ok(node) => eprintln!(
                    "[virglrs] vrend: shared buffers are linear, allocated on {}",
                    node.display()
                ),
                Err(why) => eprintln!(
                    "[virglrs] vrend: linear shared buffers are not available \
                     ({why}); shared buffers keep the driver's tiling"
                ),
            }
        }
        if winsys.reports_layouts() {
            caps.capability_bits_v2 |= caps::cap2::RESOURCE_LAYOUT;
        }
        // Adopting is the one storage question both hosts answer, and the one that fails silently
        // per window when the answer is no: a compositor samples a client's buffer as a blank
        // texture. So it is said here, once, as the property of the host it is.
        let adoption = match winsys.shared_storage_refusal(&features) {
            None => String::new(),
            Some(why) => format!(
                "; NO SHARED STORAGE CAN BE ADOPTED ({why}) -- every client window a compositor \
                 samples will be blank"
            ),
        };
        eprintln!(
            "[virglrs] vrend: {version_string} ({api}), {} formats, {} features, \
             {}{adoption}",
            formats.entries().count(),
            features.present().count(),
            if !cfg!(target_os = "macos") {
                // Not a shortfall here: this host hands the VMM a texture name and it scans out
                // from that, so there is no surface to import and nothing is copied for want of
                // one.
                "storage: GL textures, scanned out by name"
            } else if features.adopts_iosurfaces() {
                "iosurface storage available"
            } else {
                "iosurface storage UNAVAILABLE -- no scanout or shared buffer can be imported \
                 without a copy, and every one will be blank"
            },
        );
        // The waiter gets a context of ctx0's share group, made current on its own thread. A
        // driver that will not give a second context is not a reason to refuse to start: the fence
        // path falls back to finishing inline, which is what this renderer did before there was a
        // waiter at all. The C makes the same choice when its sync context fails.
        // An embedder's contexts belong to the renderer's thread, so there is no display handle
        // for a waiter to hold and no waiter to start -- which is also what the C does here: a VMM
        // that asked for neither THREAD_SYNC nor ASYNC_FENCE_CB has no sync thread either, and
        // checks its fences when it polls.
        let waiter = match winsys.thread_display() {
            None => None,
            Some(display) => match winsys.create_context(version, Some(&ctx0)) {
                Ok(wait_ctx) => Some(waiter::Waiter::start(
                    display,
                    wait_ctx,
                    Gl::new(winsys.procs(), api).reading(robust),
                    fences,
                    debug,
                )),
                Err(e) => {
                    eprintln!(
                        "[virglrs] vrend: no context for the fence waiter ({e}); \
                         classic fences will finish inline"
                    );
                    None
                }
            },
        };
        let fences = fences_for_inline;
        let unsettled = super::video::pending::Unsettled::default();
        let tally = tally::Tally::from_env(&unsettled);
        Ok(Vrend {
            winsys,
            gl,
            features,
            formats,
            limits,
            shader_cfg,
            caps,
            video,
            ctx0,
            version,
            current: Current::ctx0(),
            resources: resource::Slots::new(condemned),
            contexts: crate::Map::default(),
            todo: Todo::default(),
            tally,
            staging: transfer::Staging::default(),
            blitter: None,
            waiter,
            fences,
            doomed: Vec::new(),
            unsettled,
            batch: 0,
            pixels: resource::Refresh::default(),
            budget: crate::budget::Classic::open(budget),
            traces,
            debug,
            fence_finish: std::env::var("VIRGLRS_FENCE_FINISH").as_deref() == Ok("1"),
            fence_depth: fence_depth_from_env(),
            throttled: Throttled::default(),
        })
    }

    /// The classic capsets, as probed at init.
    pub fn caps(&self) -> &caps::CapsV2 {
        &self.caps
    }

    /// What this host decodes in hardware, or `None` when the caller did not ask for video.
    ///
    /// `None` and a support that decodes nothing are different answers and stay different: the
    /// first is a configuration, the second is this machine's silicon.
    pub fn video(&self) -> Option<&decode::Support> {
        self.video.as_ref()
    }

    pub fn gl(&self) -> &Gl {
        &self.gl
    }

    /// Make ctx0 current.
    fn switch_ctx0(&mut self) {
        self.current
            .switch_to(&self.winsys, &self.ctx0, GlContext::Ctx0)
            .expect("ctx0 was current once and still exists");
    }

    /// The host a context's commands run against, and the contexts beside it: two disjoint
    /// borrows of this renderer, so a context can run against the rest of it.
    fn split<'a>(
        &'a mut self,
        ctx: ContextId,
        guest: &'a dyn Guest,
    ) -> (Host<'a>, &'a mut crate::Map<ContextId, Context>) {
        // A batch is the other place with a context to spare, and the only one a workload that
        // frees resources without creating any ever reaches. Before the batch, never inside it: a
        // switch under a running command would leave a context the command still expects.
        self.sweep_condemned();
        let Vrend {
            winsys,
            gl,
            features,
            formats,
            limits,
            shader_cfg,
            caps: _,
            video,
            ctx0,
            version,
            current,
            resources,
            contexts,
            todo,
            tally,
            blitter,
            doomed: _,
            batch,
            pixels,
            budget,
            unsettled,
            // Neither belongs to a context's commands: the waiter is a thread, and the handle is
            // where a fence goes once answered.
            staging,
            waiter: _,
            fences: _,
            traces,
            debug,
            fence_finish: _,
            fence_depth: _,
            throttled: _,
        } = self;
        let host = Host {
            batch: *batch,
            traces: *traces,
            debug: *debug,
            tally,
            staging,
            budget,
            gl,
            winsys,
            version: *version,
            share: ctx0,
            features,
            formats,
            limits,
            shader_cfg,
            resources: resources.sync(),
            pixels,
            guest,
            ctx,
            current,
            todo,
            blitter,
            video: video.as_ref(),
            unsettled,
        };
        (host, contexts)
    }

    /// What has been retained for a rebuild: every live context's objects and current state,
    /// plus the type each live blob was given.
    ///
    /// The resources are counted here and not in `Context` because that is where they live -- one
    /// table shared by every context, so no single context can answer for it.
    pub fn journal_census(&mut self) -> Census {
        let mut c = Census::default();
        for ctx in self.contexts.values() {
            c += ctx.journal_census();
        }
        for wire in self.resources.preamble() {
            c.add_wire(wire.len(), true);
        }
        c
    }

    /// One context's journal, as the bytes the VMM stores and hands back.
    ///
    /// `None` for a context that is not here. An empty journal still serializes: a context that
    /// built nothing is a fact worth restoring accurately, and the alternative -- answering
    /// "no journal" -- is what the VMM reads as "this context is not mine to rebuild".
    pub fn journal_export(&mut self, ctx: ClassicCtx) -> Option<Vec<u8>> {
        self.journal_of(ctx.id())
    }

    fn journal_of(&mut self, id: ContextId) -> Option<Vec<u8>> {
        // Named separately, because one is read while the other is reconciled: a `&mut self`
        // method could not hold both.
        let Vrend { contexts, resources, .. } = self;
        let ctx = contexts.get(&id)?;
        Some(crate::vrend::journal::serialize(&ctx.journal(resources.preamble())))
    }

    /// Each live context's journal: how many bytes it exports, and how many entries those bytes
    /// read back as.
    ///
    /// The round-trip is the point. A census counts what was retained, which a serializer bug
    /// would leave untouched; parsing our own output back is the cheapest thing that actually
    /// exercises the format on a real world rather than on a fixture we wrote.
    pub fn journal_report(&mut self) -> Vec<(ContextId, usize, Result<usize, &'static str>)> {
        // The ids are collected first: exporting one reconciles the resource table, which is a
        // borrow of this renderer that cannot be held while walking its contexts.
        let ids: Vec<ContextId> = self.contexts.keys().copied().collect();
        ids.into_iter()
            .filter_map(|id| {
                let bytes = self.journal_of(id)?;
                let read_back = crate::vrend::journal::parse(&bytes).map(|e| e.len());
                Some((id, bytes.len(), read_back))
            })
            .collect()
    }

    /// Begin rebuilding a classic context.
    pub fn replay_begin(&mut self, ctx: ClassicCtx) -> Result<(), ReplayRefused> {
        self.contexts.get_mut(&ctx.id()).ok_or(ReplayRefused::NoContext)?.replay_begin();
        Ok(())
    }

    /// Hand a classic context the journal it will be rebuilt from.
    pub fn journal_restore(
        &mut self,
        ctx: ClassicCtx,
        bytes: &[u8],
    ) -> Result<usize, ReplayRefused> {
        self.contexts
            .get_mut(&ctx.id())
            .ok_or(ReplayRefused::NoContext)?
            .replay_restore(bytes)
            .map_err(ReplayRefused::JournalRefused)
    }

    /// Feed a classic context's retained commands up to `upto`.
    pub fn replay_upto(
        &mut self,
        ctx: ClassicCtx,
        guest: &dyn Guest,
        upto: Seq,
    ) -> Result<(), ReplayRefused> {
        let id = ctx.id();
        if !self.contexts.contains_key(&id) {
            return Err(ReplayRefused::NoContext);
        }
        let (mut host, contexts) = self.split(id, guest);
        contexts.get_mut(&id).expect("checked above").replay_upto(&mut host, upto).map_err(|why| {
            match why {
                Unfed::NotReplaying => ReplayRefused::NotReplaying,
                Unfed::Poisoned => ReplayRefused::Poisoned,
            }
        })
    }

    /// Finish rebuilding a classic context, and report what it could not use.
    pub fn replay_end(&mut self, ctx: ClassicCtx) -> Result<(), ReplayRefused> {
        self.contexts.get_mut(&ctx.id()).ok_or(ReplayRefused::NoContext)?.replay_end();
        Ok(())
    }

    // ---- contexts ----

    pub fn context_create(&mut self, ctx: ClassicCtx, guest: &dyn Guest) -> Result<(), EglError> {
        let id = ctx.id();
        let (mut host, contexts) = self.split(id, guest);
        let c = Context::new(&mut host)?;
        contexts.insert(id, c);
        Ok(())
    }

    pub fn context_destroy(&mut self, ctx: ClassicCtx, guest: &dyn Guest) {
        let id = ctx.id();
        let (mut host, contexts) = self.split(id, guest);
        if let Some(c) = contexts.remove(&id) {
            c.destroy(&mut host);
        }
        self.switch_ctx0();
        // The framebuffers that went with the context were the last holders of any storage the
        // guest freed while they drew into it.
        self.sweep_doomed();
    }

    /// Run a batch on a context. `None` for a context this renderer does not have.
    pub fn submit(
        &mut self,
        ctx: ClassicCtx,
        words: &[u32],
        guest: &dyn Guest,
    ) -> Option<Result<(), Fault>> {
        let id = ctx.id();
        self.hold_to_fence_depth(id);
        // One tick per batch, before anything in it runs. It is what says a guest has had no
        // opportunity to rewrite its pages since a copy of them was taken -- see
        // [`resource::GuestPixels`] -- so it must move exactly when that stops being true.
        self.batch += 1;
        // Read before `split` borrows the tally into the `Host`, and closed after it is given
        // back: one clock pair for the whole batch, never one per command.
        let began = self.tally.batch_began();
        let (mut host, contexts) = self.split(id, guest);
        let ran = contexts.get_mut(&id).map(|c| c.submit(&mut host, words));
        self.tally.batch_ended(began, words.len());
        ran
    }

    /// Wait, before running a batch for `id`, until that context has fewer than `fence_depth`
    /// fences in the waiter's queue. See [`super::in_flight`].
    ///
    /// Only with a waiter: without one every fence is answered inline and nothing is ever queued.
    fn hold_to_fence_depth(&mut self, id: ContextId) {
        if self.fence_depth == 0 || self.waiter.is_none() {
            return;
        }
        let Some(gate) = self.contexts.get(&id).map(|c| c.in_flight().clone()) else {
            return;
        };
        let waited = gate.wait_below(self.fence_depth, FENCE_DEPTH_PATIENCE);
        self.throttled.note(id, waited, self.fence_depth);
    }

    /// The in-flight ticket a fence taken for `on` carries through the waiter: one when the fence
    /// is answered by syncs on a context we have, since only that is GPU work still to run.
    fn in_flight_ticket(
        &self,
        on: Option<ContextId>,
        answer: &Answer,
    ) -> Option<super::in_flight::Ticket> {
        match answer {
            Answer::Syncs(_) => {
                on.and_then(|id| self.contexts.get(&id)).map(|c| c.in_flight().ticket())
            }
            Answer::Ordered => None,
        }
    }

    // ---- resources ----

    /// Create the host side of a classic resource, on ctx0.
    pub fn resource_create(&mut self, handle: ResourceHandle, args: Args) -> Result<(), Refusal> {
        // Ahead of both the assert and the charge: a handle the guest has freed is free here too,
        // and the storage it gave back is refunded before this asks the budget for more.
        self.sweep_condemned();
        assert!(
            !self.resources.sync().contains_key(&handle),
            "the renderer checked the handle was free"
        );
        self.switch_ctx0();
        let res = Resource::create(
            &self.gl,
            &self.winsys,
            &self.features,
            &self.formats,
            &self.limits,
            &self.budget,
            args,
        )?;
        self.resources.sync().insert(handle, resource::Slot::Resource(Box::new(res)));
        Ok(())
    }

    /// Claim the resource a classic context described under `blob`, giving it `handle` and a
    /// persistent host mapping.
    ///
    /// `vrend_get_blob_pipe` plus `vrend_renderer_resource_map`, as one step. The C leaves them
    /// apart and the VMM maps later, which means a resource can exist in the table having refused
    /// the only thing it was created to do. Here the map is part of the claim: a buffer that
    /// cannot be mapped is not published, and `CREATE_BLOB` says so to the guest that asked.
    ///
    /// `size` is what the guest asked to publish, and `args.width` is what was allocated. The
    /// wire lets them disagree -- the C ignores `blob_size` entirely -- and a guest asking to
    /// publish more than it allocated is asking the VMM to map whatever follows the buffer into
    /// its address space. Refused, never clamped.
    pub fn claim_described(
        &mut self,
        ctx: ClassicCtx,
        blob: BlobId,
        handle: ResourceHandle,
        size: u64,
    ) -> Result<Args, ClaimRefused> {
        self.sweep_condemned();
        assert!(
            !self.resources.sync().contains_key(&handle),
            "the renderer checked the handle was free"
        );
        let mut res = self
            .contexts
            .get_mut(&ctx.id())
            .and_then(|c| c.claim_described(blob))
            .ok_or(ClaimRefused::NotDescribed)?;
        let args = res.args;
        // Any GL context of the share group can map the buffer -- ctx0 is the one that is always
        // there, and using it means the answer does not depend on which sub-context the guest
        // happened to leave current.
        self.switch_ctx0();
        let refused = publishable(size, args.width)
            .err()
            .or_else(|| (!res.map_persistent(&self.gl)).then_some(ClaimRefused::Unmappable));
        if let Some(why) = refused {
            // Nothing can be attached to it: it has had no handle, so no view, framebuffer or
            // transfer has ever been able to name it. `destroy` handing storage back here would
            // mean a described resource had been reachable, which is the thing this path exists
            // to prevent.
            assert!(
                res.destroy(&self.gl).is_none(),
                "a resource with no handle is attached to nothing"
            );
            return Err(why);
        }
        self.resources.sync().insert(handle, resource::Slot::Resource(Box::new(res)));
        Ok(args)
    }

    /// Where a claimed resource's buffer is mapped, and how far it runs.
    ///
    /// `None` for every resource that was never published to a guest, which is every ordinary
    /// classic one: the address exists only because [`Self::claim_described`] took it.
    pub fn resource_mapping(&mut self, handle: ResourceHandle) -> Option<(usize, u64)> {
        let res = self.resource(handle)?;
        Some((res.mapped?, res.args.width as u64))
    }

    /// A blob attached to a classic context: storage, and nothing yet that says what it is.
    ///
    /// A blob reaches vrend only here. It is created without a type -- no format, no extent --
    /// so there is nothing for `resource_create` to make, and the `SET_TYPE` that describes it
    /// arrives later in the command stream. Holding the storage from the attach is what gives
    /// that command something to adopt, and what lets a handle touched in between fault as
    /// untyped rather than as absent.
    ///
    /// Idempotent: the guest may attach one resource to several contexts, and a later attach
    /// must not un-type what an earlier one's `SET_TYPE` already settled.
    ///
    /// The witness is the proof that a classic context is the one attaching. Attachment itself
    /// lives in the renderer's table, so nothing here is keyed by it -- what it settles is that a
    /// venus context attaching its own blob cannot reach this and leave a share parked here.
    pub fn resource_attach_blob(
        &mut self,
        _by: ClassicCtx,
        handle: ResourceHandle,
        storage: Option<surface::Adoptable>,
    ) {
        self.resources
            .sync()
            .entry(handle)
            .or_insert_with(|| resource::Slot::Untyped(resource::Untyped::new(storage)));
    }

    /// Deliver the decoded picture in flight into a resource, before the VMM or a transfer
    /// reaches its pixels.
    ///
    /// The control-queue half of the barrier a context's lookups put in front of every command
    /// (see `Host::settle`): these paths read and write a resource with no context command in
    /// between -- a scanout flush publishes a surface, a transfer copies to or from the guest, a
    /// cursor is read back -- so each settles first. A per-plane upload runs in whatever context
    /// is current, and puts back what it borrows. It is flushed, because the reader is not that
    /// context: GL makes one context's commands visible to another only once they are submitted.
    fn settle(&mut self, handle: ResourceHandle) {
        if !self.unsettled.any() {
            return;
        }
        let texture = self
            .resources
            .sync()
            .get(&handle)
            .and_then(resource::Slot::resource)
            .and_then(Resource::texture);
        let Some(texture) = texture else { return };
        let settled = texture.settle(&self.gl, super::video::pending::Wait::Block);
        if matches!(settled, super::video::pending::Settled::Delivered { .. }) {
            self.gl.flush();
        }
    }

    /// The IOSurface a resource is presented from, if its storage is one. Asked of the resource
    /// every time: the surface goes with the resource, and there is no other place to hold one.
    ///
    /// Settled first: the scanout paths all come through here, and what they publish or read is
    /// the surface's pixels.
    pub fn resource_surface(&mut self, handle: ResourceHandle) -> Option<&surface::Surface> {
        self.settle(handle);
        self.resources.sync().get(&handle)?.resource()?.surface()
    }

    /// The GL texture a resource's storage is, when its storage is one.
    ///
    /// Asked of the resource every time rather than mirrored anywhere: the name is the texture's
    /// and dies with it, and a copy kept elsewhere would outlive the object it names.
    pub fn resource_texture(&mut self, handle: ResourceHandle) -> Option<gl::TextureName> {
        self.settle(handle);
        Some(self.resources.sync().get(&handle)?.resource()?.texture()?.name)
    }

    /// The pixels behind a cursor resource: `vrend_renderer_get_cursor_contents`.
    ///
    /// For a VMM that draws the pointer itself. QEMU does: the guest puts its cursor on a plane,
    /// virtio-gpu turns that into cursor-queue commands, and the VMM then needs the *image* --
    /// which lives in a host texture it cannot read. Motion and clicks travel elsewhere entirely,
    /// so a renderer that answers nothing here costs a visible pointer and nothing else, which is
    /// exactly how it goes unnoticed.
    ///
    /// `None` for anything that cannot be a cursor. The size ceiling and the 2D-only rule are the
    /// C's, and they are what keep this from being a general readback of any resource by a caller
    /// that only has a handle.
    pub fn cursor_contents(&mut self, handle: ResourceHandle) -> Option<Cursor> {
        // ctx0 first, as the C does: the readback binds a framebuffer, and doing that in
        // whichever context ran last would change a binding the guest still expects to be its
        // own. It also has to happen before the resource is borrowed, since it needs `&mut self`.
        self.switch_ctx0();
        self.settle(handle);
        let res = self.resources.sync().get(&handle)?.resource()?;
        // Multisampled is refused here rather than left to fail downstream. It would: attaching
        // one and reading it back is an error GL reports, so the answer is already `None`. But
        // this is the C's guard set and the refusals are supposed to be the readable half of it
        // -- `resource.rs` spells the same trio out wherever it asks this question -- and a
        // refusal named here costs no framebuffer to discover.
        if res.args.target != TextureTarget::Texture2d || res.args.nr_samples > 1 {
            return None;
        }
        if res.args.width > 128 || res.args.height > 128 {
            return None;
        }
        let desc = res.args.format.describe()?;
        let mut pixels =
            vec![
                0u8;
                usize::try_from(desc.size_2d(desc.stride(res.args.width), res.args.height)?)
                    .ok()?
            ];
        transfer::read_whole_2d(&self.gl, &self.features, &self.formats, res, &mut pixels).ok()?;
        Some(Cursor { width: res.args.width, height: res.args.height, pixels })
    }

    /// Read a scanout back *through the descriptor it exports*, importing it and letting the GPU
    /// detile.
    ///
    /// The exporting host's answer to the question the minting host answers by locking an
    /// IOSurface and copying its rows. Both read the storage a compositor would present from,
    /// which is the point: reading this resource's texture instead would be cheaper, would give
    /// the right pixels, and would answer a different question -- it would pass just as happily
    /// if the exported descriptor named the wrong memory, the wrong pitch or the wrong format,
    /// which is exactly what this call exists to catch.
    ///
    /// The round trip is what makes the descriptor load-bearing. It is imported as an EGL image,
    /// taken as a texture's storage and read back through a framebuffer, so the GPU applies the
    /// modifier -- a tiled buffer's bytes in row order are not the picture, and no CPU mapping
    /// could do this correctly.
    ///
    /// Costly, and deliberately on the slow path: this is what a headless sink calls when it has
    /// no other way to see a frame, not something a present goes through.
    #[cfg(not(target_os = "macos"))]
    pub fn read_scanout_through_export(
        &mut self,
        handle: ResourceHandle,
        dst: &mut [u8],
        stride: usize,
        height: u32,
    ) -> Option<u32> {
        use super::gl::gles::*;
        use super::gl::{GLint, GLsizei};

        // ctx0 first, as the readback below binds a framebuffer and a texture: doing that in
        // whichever context ran last would change bindings the guest still expects to be its own.
        self.switch_ctx0();
        self.settle(handle);
        let (held, width, full_height) = {
            let res = self.resources.sync().get(&handle)?.resource()?;
            (res.surface_share()?, res.args.width, res.args.height)
        };
        let image = match self.winsys.image_from_surface(held) {
            Ok(image) => image,
            Err(e) => {
                eprintln!("[virglrs] vrend: {handle:?}: cannot import its own export ({e})");
                return None;
            }
        };

        let rows = height.min(full_height);
        let name = self.gl.gen_texture();
        self.gl.bind_texture(GL_TEXTURE_2D, Some(name));
        self.gl.drain_errors();
        self.gl.egl_image_target_texture_2d(GL_TEXTURE_2D, &image);
        // A sampler state the driver will accept: an imported image has no mip levels, and a
        // framebuffer attachment of an incomplete texture is a status this would fail on later
        // with nothing saying why.
        for p in [GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER] {
            self.gl.tex_parameter_i(GL_TEXTURE_2D, p, GL_NEAREST as GLint);
        }
        self.gl.bind_texture(GL_TEXTURE_2D, None);

        let fb = self.gl.gen_framebuffer();
        self.gl.bind_framebuffer(GL_FRAMEBUFFER, Some(fb));
        self.gl.framebuffer_texture_2d(GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, Some(name), 0);
        let status = self.gl.check_framebuffer_status();
        let mut got = 0;
        if status == GL_FRAMEBUFFER_COMPLETE {
            // Tightly packed, then re-pitched below: `Gl::read_pixels` reads at a tight pack
            // state, and asking it to write at the caller's stride would make the pack state a
            // second place the row length is decided.
            let mut packed = vec![0u8; width as usize * rows as usize * 4];
            if self.gl.read_pixels(
                0,
                0,
                width as GLsizei,
                rows as GLsizei,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                &mut packed,
            ) {
                let row = (width as usize * 4).min(stride);
                for y in 0..rows as usize {
                    let (at, to) = (y * width as usize * 4, y * stride);
                    if to + row > dst.len() {
                        break;
                    }
                    dst[to..to + row].copy_from_slice(&packed[at..at + row]);
                    // To BGRA, which is what this call owes its caller: a scanout's pixels in the
                    // order an IOSurface holds them, so the minting host and this one answer the
                    // same question. `glReadPixels` gives whatever order it was ASKED for and
                    // converts as it goes, so what came back is RGBA regardless of the fourcc the
                    // descriptor carries -- the swap is the contract's, not the buffer's, and
                    // reading GL_BGRA_EXT instead would put it behind an extension for nothing.
                    for px in dst[to..to + row].as_chunks_mut::<4>().0 {
                        px.swap(0, 2);
                    }
                    got = y as u32 + 1;
                }
            }
        } else {
            eprintln!(
                "[virglrs] vrend: {handle:?}: its exported descriptor imports but will not \
                 attach ({status:#x}); nothing was read"
            );
        }
        self.gl.bind_framebuffer(GL_FRAMEBUFFER, None);
        self.gl.delete_framebuffer(fb);
        self.gl.delete_texture(name);
        self.gl.drain_errors();
        (got > 0).then_some(got)
    }

    /// A share of that surface, for a holder outside vrend -- a venus context importing this
    /// resource, which must keep the surface alive rather than name it. See
    /// [`resource::Resource::surface_share`].
    pub fn resource_surface_share(
        &mut self,
        handle: ResourceHandle,
    ) -> Option<Arc<dyn surface::Held>> {
        self.resources.sync().get(&handle)?.resource()?.surface_share()
    }

    /// Answer a classic context fence: make it true that the GL work has run, and retire it.
    ///
    /// Retirement is queued behind the work rather than taken here, so this returns as soon as the
    /// fence is *taken* -- the caller is holding the renderer, and waiting under it is what made
    /// one heavy client slow down every other context.
    pub fn fence_context(
        &mut self,
        ctx: ClassicCtx,
        ring: RingIdx,
        id: FenceId,
        guest: &dyn Guest,
    ) {
        let ctx = ctx.id();
        // With no waiter there is no queue to retire behind, so the fence is answered inline --
        // the way this renderer did before there was one. Taking a sync and dropping it unwaited
        // would retire the fence early, which is the whole hazard this path exists to prevent.
        let pictures = self.decodes_in_flight(Some(ctx));
        if self.waiter.is_none() {
            pictures.iter().for_each(|p| p.wait());
            self.finish_contexts(&[ctx]);
            self.answer_parked(guest);
            self.fences.retire_context(ctx, ring, id);
            return;
        }
        let queries = self.queries_before_fence(Some(ctx), guest);
        let answer = self.take_fence(Some(ctx));
        if self.debug.enabled(super::debug::Switch::Fence) {
            eprintln!(
                "[virglrs] fence: context ctx={ctx:?} ring={ring:?} id={} answer={}",
                id.0,
                answer.name()
            );
        }
        let ticket = self.in_flight_ticket(Some(ctx), &answer);
        let w = self.waiter.as_ref().expect("checked just above");
        w.retire_context(Owed { pictures, fence: answer, ticket, queries }, ctx, ring, id);
    }

    /// Answer a present fence for work a classic context queued: make it true that the GL work has
    /// run, and retire it as a present.
    ///
    /// [`Self::fence_context`] with the ring taken away. The work waited for is the same -- one
    /// sync per GL queue that context could have drawn on -- and only who is told differs, because
    /// a present fence answers the VMM about a resource rather than the guest about a stream.
    pub fn present_fence(&mut self, ctx: ClassicCtx, id: FenceId, guest: &dyn Guest) {
        let ctx = ctx.id();
        // Same reasoning as `fence_context`: with no waiter there is no queue to retire behind, so
        // the work is finished inline rather than the fence being retired unwaited.
        let pictures = self.decodes_in_flight(Some(ctx));
        if self.waiter.is_none() {
            pictures.iter().for_each(|p| p.wait());
            self.finish_contexts(&[ctx]);
            self.answer_parked(guest);
            self.fences.retire_present(id);
            return;
        }
        let queries = self.queries_before_fence(Some(ctx), guest);
        let answer = self.take_fence(Some(ctx));
        if self.debug.enabled(super::debug::Switch::Fence) {
            eprintln!("[virglrs] fence: present ctx={ctx:?} id={} answer={}", id.0, answer.name());
        }
        let ticket = self.in_flight_ticket(Some(ctx), &answer);
        let w = self.waiter.as_ref().expect("checked just above");
        w.retire_present(Owed { pictures, fence: answer, ticket, queries }, id);
    }

    /// Answer a fence on the legacy global ring, which names its context from outside.
    ///
    /// `on` is the context whose work the fence is for. `None` -- or a context this renderer does
    /// not have -- means it cannot be attributed to one, and the fence is answered by its place in
    /// the waiter's queue instead; see [`Self::take_fence`].
    pub fn fence_global(&mut self, on: Option<ClassicCtx>, id: ClientFenceId, guest: &dyn Guest) {
        let on = on.map(ClassicCtx::id);
        let pictures = self.decodes_in_flight(on);
        if self.waiter.is_none() {
            pictures.iter().for_each(|p| p.wait());
            self.finish_all();
            self.answer_parked(guest);
            self.fences.retire_global(id);
            return;
        }
        let queries = self.queries_before_fence(on, guest);
        let answer = self.take_fence(on);
        if self.debug.enabled(super::debug::Switch::Fence) {
            eprintln!("[virglrs] fence: global id={} on={on:?} answer={}", id.0, answer.name());
        }
        let ticket = self.in_flight_ticket(on, &answer);
        let w = self.waiter.as_ref().expect("checked just above");
        w.retire_global(Owed { pictures, fence: answer, ticket, queries }, id);
    }

    /// Whether a fence about to be taken for `on` covers a query whose result is not written yet,
    /// and so must hold until the render thread has answered it; see [`waiter::Pump`].
    ///
    /// Where nobody has promised to serve the pump, holding would hang the fence, so the work is
    /// finished here and the queries answered now instead. That is correct but costs this thread
    /// a GPU wait, which is what the waiter exists to avoid.
    ///
    /// A fence that names no context -- or one this renderer does not have -- covers everyone's
    /// queries, as it covers everyone's work, and is always answered here: nothing it waits on
    /// says the work has run.
    fn queries_before_fence(&mut self, on: Option<ContextId>, guest: &dyn Guest) -> bool {
        let unanswered = match on.and_then(|ctx| self.contexts.get(&ctx)) {
            Some(ctx) => ctx.has_unanswered_queries(),
            None => self.contexts.values().any(Context::has_unanswered_queries),
        };
        if !unanswered {
            return false;
        }
        // Only a fence on a context we have is answered by syncs on its work; any other is
        // `Ordered` (see `decide_fence`), and a poll after it could find the work still running.
        let on = on.filter(|ctx| self.contexts.contains_key(ctx));
        if on.is_some() && self.waiter.as_ref().is_some_and(|w| w.pump().subscribed()) {
            return true;
        }
        match on {
            Some(ctx) => self.finish_contexts(&[ctx]),
            None => self.finish_all(),
        }
        self.answer_parked(guest);
        false
    }

    /// Answer every parked query whose result is ready: `vrend_renderer_check_queries`.
    fn answer_parked(&mut self, guest: &dyn Guest) {
        let waiting: Vec<ContextId> = self
            .contexts
            .iter()
            .filter(|(_, ctx)| ctx.has_unanswered_queries())
            .map(|(id, _)| *id)
            .collect();
        for id in waiting {
            let (mut host, contexts) = self.split(id, guest);
            if let Some(ctx) = contexts.get_mut(&id) {
                ctx.answer_parked(&mut host);
            }
        }
    }

    /// [`crate::renderer::Renderer::poll`]: answer the parked queries the waiter is holding
    /// fences for, and let those fences go.
    pub fn poll(&mut self, guest: &dyn Guest) {
        let Some(pump) = self.waiter.as_ref().map(|w| Arc::clone(w.pump())) else {
            return;
        };
        pump.serve(|| self.answer_parked(guest));
    }

    /// [`crate::renderer::Renderer::poll_descriptor`]. `None` with no waiter: every fence is then
    /// answered inline, and there is never anything to poll for.
    pub fn poll_descriptor(&self) -> std::io::Result<Option<std::os::fd::OwnedFd>> {
        self.waiter.as_ref().map(|w| w.pump().descriptor()).transpose()
    }

    /// [`crate::renderer::Renderer::settle_video`]: every decode thread idle, every picture
    /// delivered.
    pub fn settle_video(&mut self) {
        // The threads first, context by context: each codec's newest decode landing means its
        // thread has nothing left, and only then is every picture there to deliver.
        for ctx in self.contexts.values() {
            ctx.decodes_in_flight().for_each(|landing| landing.wait());
        }
        if !self.unsettled.any() {
            return;
        }
        // A per-plane picture is uploaded in whatever context is current, and this is a call
        // from the VMM with no context of its own: ctx0 is the renderer's.
        self.switch_ctx0();
        for slot in self.resources.sync().values() {
            if let Some(texture) = slot.resource().and_then(Resource::texture) {
                texture.settle(&self.gl, super::video::pending::Wait::Block);
            }
        }
        self.gl.flush();
    }

    /// The hardware decodes `on` has in flight, which a fence created now must not retire ahead
    /// of: before decodes ran on their own threads, an END_FRAME was finished before any fence
    /// after it could be taken, and the guest kernel signals everything at or below a delivered
    /// id. One per codec, since a codec's decodes land in order.
    ///
    /// A fence that names no context -- or one this renderer does not have -- waits for every
    /// context's, the way its inline answer finishes every context's GL work
    /// ([`Self::finish_all`]). Nothing says whose work it covers, so it is taken to cover all of
    /// it; a decode thread is idle between frames, so that is rarely more than one picture a
    /// codec.
    fn decodes_in_flight(&self, on: Option<ContextId>) -> Vec<Arc<super::video::pending::Landing>> {
        match on.and_then(|ctx| self.contexts.get(&ctx)) {
            Some(ctx) => ctx.decodes_in_flight().collect(),
            None => self.contexts.values().flat_map(Context::decodes_in_flight).collect(),
        }
    }

    /// How to answer the fence for the work `on` has queued: a sync on each GL queue that work
    /// could be on, or [`Answer::Ordered`] for a fence with no work of its own.
    ///
    /// **A fence that names no context is not a fence over everything.** It rides the same Global
    /// ring as every context-named fence, and the waiter's queue is FIFO, so retiring it behind
    /// the queue already places it after every render fenced before it. Finishing every context
    /// here bought nothing over that and cost a full drain of every GL queue, on the thread that
    /// services virtio-gpu for all of them -- measured as two thirds of all fences under a
    /// texture-churning workload, every one of them a `RESOURCE_CREATE_3D` that had rendered
    /// nothing. The C never did this either: its global fence takes a sync on ctx0, a context that
    /// never draws.
    fn take_fence(&mut self, on: Option<ContextId>) -> Answer {
        let began = self.tally.mark();
        let answer = self.decide_fence(on);
        self.tally.fence(&answer, began);
        answer
    }

    /// How this fence is answered. Wrapped by [`Self::take_fence`], which is the only caller: the
    /// decision has four exits and an instrument that must be remembered at each of them is one
    /// that will be missing from the fifth.
    fn decide_fence(&mut self, on: Option<ContextId>) -> Answer {
        if self.fence_finish {
            self.finish_all();
            return Answer::Ordered;
        }
        // Nothing names a context we have: either the Global ring named none, or it named one that
        // is gone or belongs to venus. Either way there is no work of ours to wait on, and the
        // queue's order is the answer.
        let Some(id) = on.filter(|id| self.contexts.contains_key(id)) else {
            return Answer::Ordered;
        };
        // One sync per queue, and NOT one sync. A sync covers the context it was taken on and
        // nothing else -- `Context::gl_contexts` says so -- so taking one on whichever sub-context
        // happened to be current answered the fence while a sibling sub-context's renders were
        // still queued, and ctx0's uploads with them. This is exactly the set `finish_contexts`
        // finishes, which is the set it has to be: the fence path exists to replace that finish,
        // and a replacement that covers less is not one.
        let mut syncs = Vec::new();
        let mut refused = false;
        // `hop` is the position in the walk, ctx0 included at the end, because what the walk costs
        // per position is the question the hop line exists to answer.
        let mut hop = 0usize;
        if let Some(ctx) = self.contexts.get(&id) {
            for (sub, gl_ctx) in ctx.gl_contexts() {
                let began = self.tally.mark();
                self.current
                    .switch_to(&self.winsys, gl_ctx, GlContext::Sub(id, sub))
                    .expect("a sub-context's GL context exists");
                let switched = self.tally.mark();
                let taken = self.gl.fence();
                let marks = began.zip(switched).zip(self.tally.mark()).map(|((b, s), t)| (b, s, t));
                self.tally.fence_hop(hop, marks);
                hop += 1;
                match taken {
                    Some(f) => syncs.push(f),
                    None => {
                        refused = true;
                        break;
                    }
                }
            }
        }
        // ctx0 last, which also leaves it current -- where `finish_contexts` leaves it.
        //
        // **This sync is free, measured, and the measurement is worth keeping** because it looks
        // expensive and is not. It costs ctx0 an `eglMakeCurrent` the embedder backing does not
        // dedup, a flush that is synchronous on this share group, and a reset of `Current`'s single
        // `BoundProgram` slot -- so the next batch's first draw pays a `glUseProgram` it would have
        // skipped. All three together are bounded under ~2%: A/B'd 2026-09-10 on limina's vkmark
        // vehicle (four boots, legs alternated so a host drift is absorbed, guest-CPU and llvmpipe
        // controls held), skipping it measured +0.7% and +1.7% against its neighbouring leg, both
        // inside that vehicle's noise. So it is not the ~5% this was suspected of, and a 1-2% cost
        // is not excluded. Do not re-derive the hypothesis from the shape of the code -- and note
        // that `us/cmd` cannot price this, because the tally's submit window does not contain
        // `take_fence` (see [`tally`]); the fence line's own timer is what to read.
        if !refused {
            let began = self.tally.mark();
            self.switch_ctx0();
            let switched = self.tally.mark();
            let taken = self.gl.fence();
            let marks = began.zip(switched).zip(self.tally.mark()).map(|((b, s), t)| (b, s, t));
            self.tally.fence_hop(hop, marks);
            match taken {
                Some(f) => syncs.push(f),
                None => refused = true,
            }
        }
        if refused {
            // The driver refused a sync for work we know was queued, so it has to be waited for
            // the expensive way -- but only on the contexts that could hold it, never on every
            // one. What was already taken is spent first: a `Fence` aborts on drop rather than
            // leaking a driver allocation, so dropping the partial set would take the process
            // down on the path that exists to recover.
            for f in syncs {
                self.gl.fence_delete(f);
            }
            // Counted apart from the free ordering, which answers with the same `Answer`: this one
            // is a full finish of every context that could hold the work.
            self.tally.fence_drained();
            self.finish_contexts(&[id]);
            return Answer::Ordered;
        }
        assert!(!syncs.is_empty(), "a fence answered by no sync at all would retire early");
        Answer::Syncs(syncs)
    }

    /// `vrend_renderer_resource_sync_iosurface`: make a surface-backed resource's contents whole
    /// before the surface is presented. The texture's storage *is* the surface, so there is
    /// nothing to copy -- only the renders queued into it to complete, since the present that
    /// follows reads the bytes on another queue. `false` for a resource with no surface, which
    /// is the caller's cue to read the pixels back instead.
    ///
    /// The renders live on the queue of whichever sub-context drew them, and a finish waits for
    /// one context's queue only -- so this finishes the contexts the guest kernel has this
    /// resource attached to, and then ctx0, where this renderer's own blits and transfers run.
    ///
    /// **`attached` is who may reach the resource, not who has written it**, and the difference is
    /// real: `virtio_gpu_gem_object_close` sends CONTEXT_DETACH_RESOURCE with no fence wait, so a
    /// client that renders, hands the buffer on and closes its handle is gone from the set while
    /// its draws are still queued. What makes the present sound is not this finish but the guest:
    /// `virtio_gpu_plane_prepare_fb` calls `drm_gem_plane_helper_prepare_fb`, so the atomic commit
    /// that sends the flush has already waited the scanout's fences -- and since `create_fence`
    /// those retire only once the render they name has run. This is belt to that guest's braces,
    /// for a consumer that skips them.
    ///
    /// **It must not finish anything else, because this is the present path.** It runs on every
    /// page-flip, so finishing every context would make each repaint of the desktop wait for the
    /// heaviest client's frame -- a compositor's cursor update paying for an unrelated WebGL
    /// canvas. The pinned C finishes the context that last rendered and then ctx0, for the same
    /// reason and by a shorter road: it reads whichever context happens to be current, which is
    /// the implicit-global habit this renderer does not keep.
    ///
    /// A resource attached to nothing has no such set, and is finished the old way rather than
    /// early: presenting a frame that has not been rendered is worse than presenting it late.
    pub fn resource_sync_surface(
        &mut self,
        handle: ResourceHandle,
        attached: &[ContextId],
    ) -> bool {
        if self.resource_surface(handle).is_none() {
            return false;
        }
        self.tally.present();
        if attached.is_empty() {
            // Reachable when the last context holding it was destroyed and the VMM flushes it
            // anyway -- a compositor that died. Said out loud because the same branch is where a
            // bookkeeping regression would land, and falling back to the slow path is a thing that
            // reads green.
            eprintln!(
                "[virglrs] vrend: {handle:?} is presented but attached to no context; \
                 finishing every context for it"
            );
            self.finish_all();
        } else {
            self.finish_contexts(attached);
        }
        true
    }

    /// Wait for `which` and ctx0 to have executed what was queued on them.
    ///
    /// The bounded form of [`Vrend::finish_all`], for a caller that knows whose work it needs.
    fn finish_contexts(&mut self, which: &[ContextId]) {
        for id in which {
            let Some(ctx) = self.contexts.get(id) else { continue };
            for (sub, gl_ctx) in ctx.gl_contexts() {
                self.current
                    .switch_to(&self.winsys, gl_ctx, GlContext::Sub(*id, sub))
                    .expect("a sub-context's GL context exists");
                self.gl.finish();
            }
        }
        self.switch_ctx0();
        self.gl.finish();
    }

    /// Wait for every GL context this renderer owns to have executed what was queued on it.
    ///
    /// The renders live on the queue of whichever sub-context drew them, and a finish waits for
    /// one context's queue only -- so a caller that cannot say whose work it needs has to finish
    /// them all. A caller that *can* say wants [`Vrend::finish_contexts`]: this one is the
    /// fallback, and it is far too expensive to sit on a path that runs per frame.
    ///
    /// Finishing ctx0 alone is not a substitute, whatever it costs: ctx0 never draws, and the
    /// harness caught that reading the frame before last off a scanout.
    ///
    /// "Every context" means every *guest* context and ctx0. The blitter holds a GL context of its
    /// own ([`blitter::Blitter`]) and is in neither this nor [`Vrend::finish_contexts`], so a blit
    /// into a surface-backed destination is waited for by neither -- which predates the bounding
    /// and is not fixed by it.
    pub fn finish_all(&mut self) {
        for (id, ctx) in &self.contexts {
            for (sub, gl_ctx) in ctx.gl_contexts() {
                self.current
                    .switch_to(&self.winsys, gl_ctx, GlContext::Sub(*id, sub))
                    .expect("a sub-context's GL context exists");
                self.gl.finish();
            }
        }
        self.switch_ctx0();
        self.gl.finish();
    }

    /// Delete the host side of every resource whose owner has let it go.
    ///
    /// Nothing calls this to destroy a particular resource, and that is the point: what a handle
    /// is owed was settled when the renderer's table dropped its [`resource::Claim`], and this is
    /// only the place with the GL context that half needs. So it is called wherever ctx0 can be
    /// made current -- before a create charges for storage, and once per batch -- and does
    /// nothing at all when nothing is owed, which is the common case.
    ///
    /// An untyped slot owns only a share of someone else's storage: dropping it is the whole of
    /// its teardown and it needs no context, so it goes with the rest and asks for nothing.
    fn sweep_condemned(&mut self) {
        let parked = self.resources.take_parked();
        if parked.is_empty() {
            return;
        }
        self.switch_ctx0();
        for slot in parked {
            if let resource::Slot::Resource(res) = slot
                && let Some(still_attached) = res.destroy(&self.gl)
            {
                self.doomed.push(still_attached);
            }
        }
        self.sweep_doomed();
    }

    /// Delete the parked texture storage nothing holds any more. Called where ctx0 is current.
    fn sweep_doomed(&mut self) {
        let mut i = 0;
        while i < self.doomed.len() {
            if Arc::strong_count(&self.doomed[i]) == 1 {
                let t = self.doomed.swap_remove(i);
                Arc::into_inner(t).expect("the only share").destroy(&self.gl);
            } else {
                i += 1;
            }
        }
    }

    pub fn resource(&mut self, handle: ResourceHandle) -> Option<&Resource> {
        self.settle(handle);
        self.resources.sync().get(&handle)?.resource()
    }

    /// The guest attached pages to a resource: a host-memory buffer pays them whatever they are
    /// owed (`vrend_pipe_resource_attach_iov`).
    ///
    /// A freshly created resource owes nothing, and that is what keeps this from racing the
    /// guest -- see [`resource::Shadow`].
    pub fn resource_attached(&mut self, handle: ResourceHandle, pages: &Iov<'_>) {
        self.tally.attached(pages.entries());
        if let Some(Resource { storage: resource::Storage::Host(shadow), .. }) =
            self.resources.sync().get_mut(&handle).and_then(resource::Slot::resource_mut)
            && !shadow.mirror_into(pages)
        {
            eprintln!(
                "[virglrs] resource {handle}: the attached pages are smaller than the buffer"
            );
        }
    }

    /// The guest is detaching a resource's pages: a host-memory buffer pulls them back first
    /// (`vrend_pipe_resource_detach_iov`).
    pub fn resource_detaching(&mut self, handle: ResourceHandle, pages: &Iov<'_>) {
        if let Some(Resource { storage: resource::Storage::Host(shadow), .. }) =
            self.resources.sync().get_mut(&handle).and_then(resource::Slot::resource_mut)
        {
            // The pages are about to go away, so what they hold survives only here -- and is
            // owed back to whatever pages arrive next.
            let ok = pages.copy_out(0, shadow.bytes_mut());
            shadow.unmirrored();
            if !ok {
                eprintln!(
                    "[virglrs] resource {handle}: the detached pages are smaller than the buffer"
                );
            }
        }
    }

    // ---- transfers ----

    /// A transfer on the API path (`virgl_renderer_transfer_{read,write}_iov`): on the named
    /// context, or on ctx0 for the VMM's own.
    ///
    /// `own` is the resource's attached pages, `pages` the ones the transfer names -- the same
    /// ones when the caller gave none.
    pub fn transfer(
        &mut self,
        ctx: Option<ClassicCtx>,
        handle: ResourceHandle,
        own: Option<&Iov<'_>>,
        pages: transfer::Through<'_, '_>,
        info: &Info,
    ) -> Result<(), transfer::Error> {
        match ctx.map(ClassicCtx::id) {
            Some(id) => {
                if !self.contexts.contains_key(&id) {
                    return Err(transfer::Error::NoPages);
                }
                // The context's current sub-context owns the GL context to run on.
                let (mut host, contexts) = self.split(id, &NoGuest);
                contexts[&id].make_current(&mut host);
            }
            None => self.switch_ctx0(),
        }
        if pages.is_empty() {
            return Err(transfer::Error::NoPages);
        }
        self.settle(handle);
        let began = self.tally.mark();
        let res = self
            .resources
            .sync()
            .get_mut(&handle)
            .and_then(resource::Slot::resource_mut)
            .ok_or(transfer::Error::NoPages)?;
        let bytes = transfer::box_bytes(res, info);
        let r = match pages {
            transfer::Through::ToHost(pages) => transfer::write(
                &self.gl,
                self.current.program(),
                &self.formats,
                &mut self.staging,
                res,
                own,
                &pages,
                info,
            ),
            transfer::Through::ToGuest(pages) => transfer::read(
                &self.gl,
                self.current.program(),
                &self.features,
                &self.formats,
                &mut self.staging,
                res,
                own,
                pages,
                info,
            ),
        };
        self.tally.transfer(began, tally::TransferDoor::Api, bytes);
        r
    }
}

/// A guest that answers nothing: for a switch of GL context, which asks nothing of the guest.
struct NoGuest;

impl Guest for NoGuest {
    fn attached(&self, _: ContextId, _: ResourceHandle) -> bool {
        false
    }

    fn pages(&self, _: ContextId, _: ResourceHandle) -> Option<Iov<'_>> {
        None
    }

    fn blob_pixels(&self, _: ContextId, _: ResourceHandle) -> Option<PixelSource<'_>> {
        None
    }
}

/// The GL a context speaks, read back from it -- `epoxy_is_desktop_gl` and `epoxy_gl_version` --
/// or why vrend will not run on it. A desktop context must not be read as GLES: `4.6 (Core
/// Profile)` parses as a plausible "GLES 4.6", and every probe after it would answer about an API
/// the renderer is not translating for.
///
/// `profile` is the context's `GL_CONTEXT_PROFILE_MASK`, asked only of a desktop context.
fn host_api(
    version_string: &str,
    profile: impl FnOnce() -> GLenum,
    asked: HostGl,
) -> Result<Api, UnservedGl> {
    let version = parse_version(version_string);
    if version_string.starts_with("OpenGL ES ") {
        return Ok(Api::Gles(version));
    }
    if asked != HostGl::Desktop {
        return Err(UnservedGl::NotAskedFor);
    }
    if version < 33 {
        return Err(UnservedGl::TooOld);
    }
    // Asked of the context rather than read off the string, which only some vendors spell out.
    if profile() & GL_CONTEXT_CORE_PROFILE_BIT == 0 {
        return Err(UnservedGl::Compatibility);
    }
    Ok(Api::Gl(version))
}

/// `epoxy_gl_version` for a version string: "OpenGL ES 3.1 Mesa ..." and "3.1 Mesa ..." are 31.
fn parse_version(s: &str) -> u32 {
    let rest = s.strip_prefix("OpenGL ES ").unwrap_or(s);
    let mut it = rest.split(|c: char| !c.is_ascii_digit()).filter(|p| !p.is_empty());
    let major: u32 = it.next().and_then(|p| p.parse().ok()).unwrap_or(0);
    let minor: u32 = it.next().and_then(|p| p.parse().ok()).unwrap_or(0);
    major * 10 + minor
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vrend::pipe::ShaderStage;
    use crate::vrend::proto::ObjectHandle;

    #[test]
    fn a_blob_is_never_published_past_the_resource_backing_it() {
        // The size the guest asks to publish and the width it allocated are one fact sent twice,
        // and the guest is the only thing that can make them disagree. Exactly is the ordinary
        // case, and less is the guest publishing part of what it made.
        publishable(0x21000, 0x21000).expect("exactly what was allocated");
        publishable(0x1000, 0x21000).expect("part of what was allocated");

        // More is the guest asking the VMM to map whatever the driver put after the buffer into
        // its address space. There is no repair for it: mapping less than was asked reports
        // success for a mapping that was not requested, and the guest indexes past the end of it.
        assert_eq!(
            publishable(0x22000, 0x21000),
            Err(ClaimRefused::Oversize { asked: 0x22000, allocated: 0x21000 }),
            "a blob larger than its resource is refused, not trimmed to fit"
        );
    }

    /// The probed table on the live host, for the formats the classic corpus creates. Needs a
    /// GPU: the zink-on-KosmicKrisp environment on Darwin, Mesa's own on Linux.
    #[test]
    fn the_host_table_for_the_corpus_formats() {
        let _display = crate::vrend::one_display_at_a_time();
        struct Discard;
        impl crate::fence::FenceSink for Discard {
            fn context_fence(&mut self, _: ContextId, _: RingIdx, _: FenceId) {}
            fn present_fence(&mut self, _: FenceId) {}

            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        // Declared first so it outlives the renderer: the fence waiter retires through this as it
        // drains, which happens while `v` is dropping.
        let retire = crate::fence::Retirement::start(
            Box::new(Discard),
            crate::vrend::debug::Switches::default(),
        );
        let v = Vrend::new(
            Config::default(),
            &crate::budget::Budget::with_cap(None, false),
            retire.handle(),
            None,
            crate::vrend::resource::Condemned::default(),
            crate::vrend::debug::Traces::default(),
            crate::vrend::debug::Switches::default(),
        )
        .expect("vrend comes up");
        let present: Vec<&str> = v.features.present().map(|f| f.name()).collect();
        eprintln!("features: {}", present.join(" "));
        for raw in [1, 2, 20, 48, 49, 64, 65, 67, 131, 134, 177, 227] {
            let f = super::super::proto::Format::from_wire(raw).unwrap();
            eprintln!("{raw:>4} {:<24} {:?}", f.name(), v.formats.get(f));
        }
    }

    /// A host that cannot make a multisample array texture advertises no multisampling, in both
    /// places a guest reads it.
    ///
    /// The capset has no per-target bit, so advertising the 2D form alone would tell a guest a
    /// format multisamples and then refuse its array form -- and a refused create reaches no
    /// guest. The host here has the array form, so the host without it is this one with the
    /// feature withdrawn, probed again against the same live driver.
    #[test]
    fn without_multisample_arrays_no_multisampling_is_advertised() {
        let _display = crate::vrend::one_display_at_a_time();
        struct Discard;
        impl crate::fence::FenceSink for Discard {
            fn context_fence(&mut self, _: ContextId, _: RingIdx, _: FenceId) {}
            fn present_fence(&mut self, _: FenceId) {}

            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        let retire = crate::fence::Retirement::start(
            Box::new(Discard),
            crate::vrend::debug::Switches::default(),
        );
        let v = Vrend::new(
            Config::default(),
            &crate::budget::Budget::with_cap(None, false),
            retire.handle(),
            None,
            crate::vrend::resource::Condemned::default(),
            crate::vrend::debug::Traces::default(),
            crate::vrend::debug::Switches::default(),
        )
        .expect("vrend comes up");
        let multisampled = |c: &caps::CapsV2| {
            (0..super::super::proto::FORMAT_MAX)
                .filter_map(super::super::proto::Format::from_wire)
                .filter(|f| *f != super::super::proto::Format::NONE)
                .filter(|f| c.supported_multisample_formats.has(*f))
                .count()
        };

        // The control: this host multisamples, so a zero below is the withdrawal and not the host.
        assert!(v.features.multisample_textures(), "the host this runs on has the array form");
        assert!(v.caps.v1.max_samples > 1, "and advertises more than one sample");
        assert!(multisampled(&v.caps) > 0, "and at least one multisampling format");

        // Probed again from the same driver, as `Vrend::new` probes it, and then withdrawn.
        let mut without = Features::probe(v.features.api(), v.gl.extensions());
        without.reconcile(&v.gl);
        assert!(without.multisample_textures(), "the re-probe sees what the first probe saw");
        without.clear(Feature::storage_multisample_2d_array);
        let table = Table::probe(&v.gl, &without);
        let caps = caps::CapsV2::probe(&v.gl, &without, &v.limits, &table, None);
        assert_eq!(caps.v1.max_samples, 1, "no sample count above one");
        assert_eq!(multisampled(&caps), 0, "no format multisamples");
        assert_eq!(caps.sample_locations, [0; 8], "and no sample positions for counts not offered");
        assert!(table.entries().all(|e| !e.can_multisample), "the table agrees with the caps");
    }

    /// What a cursor readback will and will not answer for, against a live driver.
    ///
    /// The refusals are the interesting half: this is reached with nothing but a resource handle,
    /// so without the ceiling and the 2D rule it would be a way to read any resource back through
    /// a call that is supposed to be about a pointer. The pixels themselves are scored by a guest
    /// -- a desktop with a visible pointer -- because that is what the answer is for.
    #[test]
    fn only_a_cursor_shaped_resource_reads_back_as_one() {
        let _display = crate::vrend::one_display_at_a_time();
        struct Discard;
        impl crate::fence::FenceSink for Discard {
            fn context_fence(&mut self, _: ContextId, _: RingIdx, _: FenceId) {}
            fn present_fence(&mut self, _: FenceId) {}

            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        let retire = crate::fence::Retirement::start(
            Box::new(Discard),
            crate::vrend::debug::Switches::default(),
        );
        let mut v = Vrend::new(
            Config::default(),
            &crate::budget::Budget::with_cap(None, false),
            retire.handle(),
            None,
            crate::vrend::resource::Condemned::default(),
            crate::vrend::debug::Traces::default(),
            crate::vrend::debug::Switches::default(),
        )
        .expect("vrend comes up");

        // B8G8R8A8_UNORM, which is what a cursor plane is everywhere it exists.
        let bgra = super::super::proto::Format::from_wire(1).expect("B8G8R8A8_UNORM");
        let tex = |w: u32, h: u32| resource::Args {
            target: TextureTarget::Texture2d,
            format: bgra,
            bind: resource::Bind(1 << 1),
            width: w,
            height: h,
            depth: 1,
            array_size: 1,
            last_level: 0,
            nr_samples: 0,
            flags: resource::ResourceFlags(0),
        };
        let handle = |n: u32| ResourceHandle::new(n).expect("a resource handle is non-zero");

        v.resource_create(handle(1), tex(64, 64)).expect("a cursor-sized texture");
        let cursor = v.cursor_contents(handle(1)).expect("reads back");
        assert_eq!((cursor.width, cursor.height), (64, 64));
        // The buffer is the image's size and not a row short of it: the extent and the bytes are
        // handed over together precisely so a caller cannot be told the wrong shape for them.
        assert_eq!(cursor.pixels.len(), 64 * 64 * 4, "four bytes a pixel, every row present");

        // Past the ceiling. A scanout is this shape, and it is not a cursor.
        v.resource_create(handle(2), tex(256, 256)).expect("an ordinary texture");
        assert!(v.cursor_contents(handle(2)).is_none(), "larger than any cursor plane");

        // A handle nothing holds. Not an error to a VMM -- it means no pointer this frame.
        assert!(v.cursor_contents(handle(3)).is_none(), "nothing holds this handle");
    }

    /// A context with one timestamp query, whose result goes to sixteen bytes of guest pages, and
    /// a recorder of the context fences that retire.
    struct QueryRig {
        // Before `retire`: the waiter retires through it as it drains, so it must go first.
        v: Vrend,
        retire: crate::fence::Retirement,
        retired: std::sync::mpsc::Receiver<u64>,
        guest: WholePages,
        page: Box<[u8; 16]>,
        ctx: ClassicCtx,
        query: ObjectHandle,
    }

    /// Every resource attached, over the same sixteen bytes of pages: one whole query result.
    struct WholePages([crate::abi::GuestIov; 1]);

    impl Guest for WholePages {
        fn attached(&self, _: ContextId, _: ResourceHandle) -> bool {
            true
        }
        fn pages(&self, _: ContextId, _: ResourceHandle) -> Option<Iov<'_>> {
            Some(Iov::new(&self.0))
        }
        fn blob_pixels(&self, _: ContextId, _: ResourceHandle) -> Option<PixelSource<'_>> {
            None
        }
    }

    impl QueryRig {
        fn new() -> QueryRig {
            struct Recorder(std::sync::mpsc::Sender<u64>);
            impl crate::fence::FenceSink for Recorder {
                fn context_fence(&mut self, _: ContextId, _: RingIdx, f: FenceId) {
                    let _ = self.0.send(f.0);
                }
                fn present_fence(&mut self, _: FenceId) {}
                fn global_fence(&mut self, _: ClientFenceId) {}
            }
            let (tx, retired) = std::sync::mpsc::channel();
            let retire = crate::fence::Retirement::start(
                Box::new(Recorder(tx)),
                crate::vrend::debug::Switches::default(),
            );
            let mut v = Vrend::new(
                Config::default(),
                &crate::budget::Budget::with_cap(None, false),
                retire.handle(),
                None,
                crate::vrend::resource::Condemned::default(),
                crate::vrend::debug::Traces::default(),
                crate::vrend::debug::Switches::default(),
            )
            .expect("vrend comes up");
            assert!(v.waiter.is_some(), "the premise: fences go through the waiter");
            assert!(v.features.has(Feature::timer_query), "the premise: timer queries");
            let mut page = Box::new([0u8; 16]);
            let guest = WholePages([crate::abi::GuestIov {
                base: crate::abi::VmmPtr(page.as_mut_ptr().cast()),
                len: page.len(),
            }]);
            let res = ResourceHandle::new(1).expect("a resource handle is non-zero");
            v.resource_create(
                res,
                resource::Args {
                    target: TextureTarget::Buffer,
                    format: super::super::proto::Format::from_wire(64).expect("R8_UNORM"),
                    bind: resource::Bind::CUSTOM,
                    width: 16,
                    height: 1,
                    depth: 1,
                    array_size: 1,
                    last_level: 0,
                    nr_samples: 0,
                    flags: resource::ResourceFlags(0),
                },
            )
            .expect("a host-memory buffer");
            let ctx = ClassicCtx::for_test(ContextId::new(1).expect("a context id"));
            v.context_create(ctx, &guest).expect("a context");
            let query = ObjectHandle::new(1).expect("non-zero");
            QueryRig { v, retire, retired, guest, page, ctx, query }
        }

        /// What mesa sends at end-query: the end, and the one request for the result, in one
        /// batch with nothing flushed between them -- so the result is not ready when asked.
        fn end_and_ask(&mut self) {
            use crate::vrend::encode::encode;
            use crate::vrend::pipe::QueryType;
            use crate::vrend::proto::{Command, Object, QueryCreate};
            let resource = ResourceHandle::new(1).expect("non-zero");
            let create = QueryCreate { kind: QueryType::Timestamp, index: 0, offset: 0, resource };
            let mut wire = Vec::new();
            let handle = self.query;
            encode(&Command::CreateObject { handle, object: Object::Query(create) }, &mut wire);
            encode(&Command::EndQuery(handle), &mut wire);
            encode(&Command::GetQueryResult { query: handle, wait: false }, &mut wire);
            self.v.submit(self.ctx, &wire, &self.guest).expect("the context").expect("recorded");
            assert!(
                self.v.contexts[&self.ctx.id()].has_unanswered_queries(),
                "the premise: the driver had the result at once, so nothing was parked and this \
                 test cannot tell a parked query from one answered on the spot"
            );
            assert_eq!(self.state(), 0, "nothing is written for a result not ready");
        }

        /// The `virgl_host_query_state` word in the guest's pages: 1 is DONE.
        fn state(&self) -> u32 {
            u32::from_le_bytes(self.page[..4].try_into().expect("four bytes"))
        }

        fn finish(mut self) {
            self.v.context_destroy(self.ctx, &self.guest);
            drop(self.v);
            drop(self.retire);
        }
    }

    /// A query whose result was not ready when the guest asked is answered before the fence behind
    /// the ask retires, by a poll the waiter holds that fence for.
    ///
    /// The guest asks once, at end-query, then waits for the fence and reads the buffer straight
    /// from its pages. A fence that retires first hands it a result that is not there, and a
    /// guest that waited (KWin's render-time query) re-reads forever.
    #[test]
    fn a_parked_query_is_answered_by_a_poll_before_its_fence_retires() {
        use std::io::Read;
        let _display = crate::vrend::one_display_at_a_time();
        let mut rig = QueryRig::new();
        let bell = rig.v.poll_descriptor().expect("a dup").expect("a waiter, so a descriptor");
        let bell = std::os::unix::net::UnixStream::from(bell);
        bell.set_nonblocking(true).expect("nonblocking");
        rig.end_and_ask();
        rig.v.fence_context(rig.ctx, RingIdx(0), FenceId(7), &rig.guest);

        // Held, however long the GPU takes: only the render thread can read the result.
        let early = rig.retired.recv_timeout(std::time::Duration::from_millis(500));
        assert!(early.is_err(), "the fence retired with the query still unanswered");
        assert_eq!(rig.state(), 0, "and nothing answered it behind the poll's back");
        // Read rather than peeked, which is unstable: the poll serves what was asked whether or
        // not the byte is still there to drain.
        assert!(matches!((&bell).read(&mut [0u8; 1]), Ok(1)), "the descriptor asks for a poll");

        rig.v.poll(&rig.guest);
        let retired = rig.retired.recv_timeout(std::time::Duration::from_secs(10));
        assert_eq!(retired, Ok(7), "the poll lets the fence go");
        assert_eq!(rig.state(), 1, "VIRGL_QUERY_STATE_DONE is in the pages");
        let after = (&bell).read(&mut [0u8; 1]).map_err(|e| e.kind());
        assert_eq!(after, Err(std::io::ErrorKind::WouldBlock), "and nothing asks for another");
        rig.finish();
    }

    /// A fence that names no context is answered by its place in the queue, not by syncs on the
    /// query's work, so a poll behind it could find the result still unready: the work is finished
    /// and the query answered when that fence is taken, even with a poll on offer.
    #[test]
    fn a_fence_naming_no_context_answers_a_parked_query_when_it_is_taken() {
        let _display = crate::vrend::one_display_at_a_time();
        let mut rig = QueryRig::new();
        let _bell = rig.v.poll_descriptor().expect("a dup").expect("a waiter, so a descriptor");
        rig.end_and_ask();
        rig.v.fence_global(None, ClientFenceId(7), &rig.guest);
        assert_eq!(rig.state(), 1, "VIRGL_QUERY_STATE_DONE is in the pages before the fence goes");
        rig.finish();
    }

    /// With nobody to poll, the same query is answered when the fence is taken, by finishing the
    /// work there: slower, but a fence that waited for a poll nobody makes would never retire.
    #[test]
    fn without_a_poll_a_parked_query_is_answered_when_its_fence_is_taken() {
        let _display = crate::vrend::one_display_at_a_time();
        let mut rig = QueryRig::new();
        rig.end_and_ask();
        rig.v.fence_context(rig.ctx, RingIdx(0), FenceId(7), &rig.guest);
        assert_eq!(rig.state(), 1, "VIRGL_QUERY_STATE_DONE is in the pages before the fence goes");
        let retired = rig.retired.recv_timeout(std::time::Duration::from_secs(10));
        assert_eq!(retired, Ok(7), "and the fence retires without a poll");
        rig.finish();
    }

    /// Both classic fence entry points, driven end to end against a live driver and a live waiter.
    ///
    /// What this is for is the half of the fence path no unit test of [`Answer`] can reach: a
    /// `Fence` aborts on drop rather than being leaked, so a caller that takes a sync and then
    /// loses it takes the process with it -- and the only way to that is through these two
    /// functions, with a waiter running and a context the fence can be attributed to. Asserting on
    /// the retirements is secondary; surviving the call is the test.
    #[test]
    fn both_classic_fence_paths_spend_every_sync_they_take() {
        let _display = crate::vrend::one_display_at_a_time();
        struct Recorder(std::sync::mpsc::Sender<u64>);
        impl crate::fence::FenceSink for Recorder {
            fn context_fence(&mut self, _: ContextId, _: RingIdx, f: FenceId) {
                let _ = self.0.send(f.0);
            }
            fn present_fence(&mut self, _: FenceId) {}

            fn global_fence(&mut self, f: ClientFenceId) {
                let _ = self.0.send(u64::from(f.0));
            }
        }
        let (tx, rx) = std::sync::mpsc::channel();
        let retire = crate::fence::Retirement::start(
            Box::new(Recorder(tx)),
            crate::vrend::debug::Switches::default(),
        );
        let mut v = Vrend::new(
            Config::default(),
            &crate::budget::Budget::with_cap(None, false),
            retire.handle(),
            None,
            crate::vrend::resource::Condemned::default(),
            crate::vrend::debug::Traces::default(),
            crate::vrend::debug::Switches::default(),
        )
        .expect("vrend comes up");
        assert!(
            v.waiter.is_some(),
            "without a waiter no sync is ever taken and this tests nothing"
        );

        let ctx = ClassicCtx::for_test(ContextId::new(1).expect("a context id"));
        v.context_create(ctx, &NoGuest).expect("a context");

        // Each of the three shapes a classic fence comes in: named by a context, named by the
        // global ring, and naming nothing at all.
        v.fence_context(ctx, RingIdx(0), FenceId(11), &NoGuest);
        v.fence_global(Some(ctx), ClientFenceId(22), &NoGuest);
        v.fence_global(None, ClientFenceId(33), &NoGuest);

        // Dropped before the assertions: the waiter drains on the way out, so this is what makes
        // every fence above have been retired by the time they are read.
        v.context_destroy(ctx, &NoGuest);
        drop(v);
        // Three, each with its own deadline, and then nothing: draining until a timeout would make
        // every run of this test pay that timeout.
        let got: Vec<u64> = (0..3)
            .map(|_| rx.recv_timeout(std::time::Duration::from_secs(10)).expect("a fence retires"))
            .collect();
        assert_eq!(got, vec![11, 22, 33], "all three retire, and in the order they were taken");
        assert!(rx.try_recv().is_err(), "and nothing else was retired");
    }

    /// A fence covers every GL queue its context could have rendered on, not whichever sub-context
    /// happened to be current.
    ///
    /// Each sub-context has a command queue of its own, so one sync answered the fence while a
    /// sibling's renders were still outstanding -- and the guest was told work was done that was
    /// not. The count is what pins it: with one sync per queue this is the number of sub-contexts
    /// plus ctx0, and a build that takes one sync answers 1 whatever the context looks like.
    #[test]
    fn a_fence_covers_every_sub_context_and_ctx0() {
        let _display = crate::vrend::one_display_at_a_time();
        // Nothing asserts on retirement here; what is asserted is the shape of the answer.
        struct Ignore;
        impl crate::fence::FenceSink for Ignore {
            fn context_fence(&mut self, _: ContextId, _: RingIdx, _: FenceId) {}
            fn present_fence(&mut self, _: FenceId) {}

            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        let retire = crate::fence::Retirement::start(
            Box::new(Ignore),
            crate::vrend::debug::Switches::default(),
        );
        let mut v = Vrend::new(
            Config::default(),
            &crate::budget::Budget::with_cap(None, false),
            retire.handle(),
            None,
            crate::vrend::resource::Condemned::default(),
            crate::vrend::debug::Traces::default(),
            crate::vrend::debug::Switches::default(),
        )
        .expect("vrend comes up");

        let ctx = ClassicCtx::for_test(ContextId::new(1).expect("a context id"));
        v.context_create(ctx, &NoGuest).expect("a context");

        // `cmd | obj << 8 | len << 16`, then the payload -- see `proto`. Two more sub-contexts, so
        // the answer has to grow: a context always has sub-context 0 already.
        for sub in [7u32, 9] {
            v.submit(
                ctx,
                &[crate::vrend::proto::Cmd::CreateSubCtx as u32 | (1 << 16), sub],
                &NoGuest,
            )
            .expect("the context takes the batch")
            .expect("a sub-context is created");
        }

        let answer = v.decide_fence(Some(ctx.id()));
        let Answer::Syncs(syncs) = answer else {
            panic!("a context with queued sub-contexts is answered by syncs, got {}", answer.name())
        };
        assert_eq!(syncs.len(), 4, "sub-contexts 0, 7 and 9, and ctx0");
        // Spent by hand: nothing retired these, and a `Fence` aborts on drop.
        for f in syncs {
            v.gl.fence_delete(f);
        }

        v.context_destroy(ctx, &NoGuest);
    }

    /// A rasterizer bind marks the shader dirty on its own, with nothing else changing.
    ///
    /// Flat shading, two-sided colour and polygon stipple are shader work on this host, so the
    /// rasterizer is a shader-key input; the C marks the key dirty on every rasterizer bind. A
    /// port that only stores the state draws the next triangle through the program selected
    /// for the previous rasterizer. `flatshade.score` pins the pixels; this pins the mark.
    #[test]
    fn binding_a_rasterizer_marks_the_shader_dirty() {
        let _display = crate::vrend::one_display_at_a_time();
        struct Ignore;
        impl crate::fence::FenceSink for Ignore {
            fn context_fence(&mut self, _: ContextId, _: RingIdx, _: FenceId) {}
            fn present_fence(&mut self, _: FenceId) {}

            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        let retire = crate::fence::Retirement::start(
            Box::new(Ignore),
            crate::vrend::debug::Switches::default(),
        );
        let mut v = Vrend::new(
            Config::default(),
            &crate::budget::Budget::with_cap(None, false),
            retire.handle(),
            None,
            crate::vrend::resource::Condemned::default(),
            crate::vrend::debug::Traces::default(),
            crate::vrend::debug::Switches::default(),
        )
        .expect("vrend comes up");
        let ctx = ClassicCtx::for_test(ContextId::new(1).expect("a context id"));
        v.context_create(ctx, &NoGuest).expect("a context");
        assert!(!v.contexts[&ctx.id()].shader_dirty(), "a fresh context has nothing to reselect");

        // `cmd | obj << 8 | len << 16`, then the payload -- see `proto`. A rasterizer that does
        // nothing but let a triangle through: front-CCW and the half-pixel centre, point size and
        // line width 1.0, and no offset.
        use crate::vrend::proto::{Cmd, ObjectType};
        let rasterizer = ObjectType::Rasterizer as u32;
        let handle = 7u32;
        let create = [
            Cmd::CreateObject as u32 | rasterizer << 8 | 9 << 16,
            handle,
            (1 << 15) | (1 << 29),
            1.0f32.to_bits(),
            0,
            0,
            1.0f32.to_bits(),
            0,
            0,
            0,
        ];
        v.submit(ctx, &create, &NoGuest).expect("the context takes the batch").expect("created");
        assert!(!v.contexts[&ctx.id()].shader_dirty(), "creating an object binds nothing");

        let bind = [Cmd::BindObject as u32 | rasterizer << 8 | 1 << 16, handle];
        v.submit(ctx, &bind, &NoGuest).expect("the context takes the batch").expect("bound");
        assert!(v.contexts[&ctx.id()].shader_dirty(), "the bind alone marks the shader dirty");

        v.context_destroy(ctx, &NoGuest);
    }

    /// A timestamp query records the GPU's clock at its END_QUERY, and its result comes back
    /// whole: eight bytes, as the C writes it for a timer query.
    ///
    /// A timestamp has no begin. A query name that nothing ever issued is not a query object,
    /// so reading one back is a GL error, and that poisons the context the guest asked from.
    #[test]
    fn a_timestamp_query_is_recorded_and_read_back_in_eight_bytes() {
        use crate::vrend::encode::encode;
        use crate::vrend::pipe::QueryType;
        use crate::vrend::proto::{Command, Object, QueryCreate};
        let _display = crate::vrend::one_display_at_a_time();
        struct Discard;
        impl crate::fence::FenceSink for Discard {
            fn context_fence(&mut self, _: ContextId, _: RingIdx, _: FenceId) {}
            fn present_fence(&mut self, _: FenceId) {}

            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        struct AllAttached;
        impl Guest for AllAttached {
            fn attached(&self, _: ContextId, _: ResourceHandle) -> bool {
                true
            }
            fn pages(&self, _: ContextId, _: ResourceHandle) -> Option<Iov<'_>> {
                None
            }
            fn blob_pixels(&self, _: ContextId, _: ResourceHandle) -> Option<PixelSource<'_>> {
                None
            }
        }
        let retire = crate::fence::Retirement::start(
            Box::new(Discard),
            crate::vrend::debug::Switches::default(),
        );
        let mut v = Vrend::new(
            Config::default(),
            &crate::budget::Budget::with_cap(None, false),
            retire.handle(),
            None,
            crate::vrend::resource::Condemned::default(),
            crate::vrend::debug::Traces::default(),
            crate::vrend::debug::Switches::default(),
        )
        .expect("vrend comes up");
        assert!(v.features.has(Feature::timer_query), "the premise: this driver has timer queries");

        // The query's answer goes into host memory: a CUSTOM buffer, as mesa makes one.
        let res = ResourceHandle::new(1).expect("a resource handle is non-zero");
        v.resource_create(
            res,
            resource::Args {
                target: TextureTarget::Buffer,
                format: super::super::proto::Format::from_wire(64).expect("R8_UNORM"),
                bind: resource::Bind::CUSTOM,
                width: 16,
                height: 1,
                depth: 1,
                array_size: 1,
                last_level: 0,
                nr_samples: 0,
                flags: resource::ResourceFlags(0),
            },
        )
        .expect("a host-memory buffer");
        let ctx = ClassicCtx::for_test(ContextId::new(1).expect("a context id"));
        v.context_create(ctx, &AllAttached).expect("a context");

        let query = crate::vrend::proto::ObjectHandle::new(1).expect("non-zero");
        let create = QueryCreate { kind: QueryType::Timestamp, index: 0, offset: 0, resource: res };
        let mut wire = Vec::new();
        encode(&Command::CreateObject { handle: query, object: Object::Query(create) }, &mut wire);
        encode(&Command::EndQuery(query), &mut wire);
        v.submit(ctx, &wire, &AllAttached).expect("the context is here").expect("recorded");
        // Ready by construction, so one read is the whole answer.
        v.finish_all();
        let mut wire = Vec::new();
        encode(&Command::GetQueryResult { query, wait: true }, &mut wire);
        v.submit(ctx, &wire, &AllAttached)
            .expect("the context is here")
            .expect("a recorded timestamp reads back");

        let answer = match &v.resources.sync().get(&res).and_then(resource::Slot::resource) {
            Some(r) => match &r.storage {
                resource::Storage::Host(shadow) => shadow.bytes()[..16].to_vec(),
                _ => panic!("a CUSTOM buffer is host memory"),
            },
            None => panic!("the buffer is here"),
        };
        let word = |at: usize| u32::from_le_bytes(answer[at..at + 4].try_into().expect("4 bytes"));
        assert_eq!(word(0), 1, "VIRGL_QUERY_STATE_DONE");
        assert_eq!(word(4), 8, "a timer query's result is 64 bits wide");
        let result = u64::from_le_bytes(answer[8..16].try_into().expect("8 bytes"));
        assert_ne!(result, 0, "the GPU's clock was recorded");

        v.context_destroy(ctx, &AllAttached);
    }

    /// A compute dispatch runs, from the wire's grid and from an indirect buffer's.
    ///
    /// The shader numbers every texel it reaches, so a grid read wrongly -- or an indirect
    /// dispatch that took the wire's zeros, or the buffer's words from the wrong offset -- leaves
    /// texels the reads below disagree about. Then a dispatch whose grid would run past its
    /// buffer is refused by name rather than handed to GL.
    #[test]
    fn a_dispatch_runs_its_grid_from_the_wire_or_its_buffer() {
        use crate::vrend::encode::encode;
        use crate::vrend::pipe::ImageAccess;
        use crate::vrend::proto::{
            Box3, Command, Object, ShaderChunk, ShaderCreate, ShaderImage, ShaderKind, Transfer,
        };
        let _display = crate::vrend::one_display_at_a_time();
        struct Discard;
        impl crate::fence::FenceSink for Discard {
            fn context_fence(&mut self, _: ContextId, _: RingIdx, _: FenceId) {}
            fn present_fence(&mut self, _: FenceId) {}

            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        struct AllAttached;
        impl Guest for AllAttached {
            fn attached(&self, _: ContextId, _: ResourceHandle) -> bool {
                true
            }
            fn pages(&self, _: ContextId, _: ResourceHandle) -> Option<Iov<'_>> {
                None
            }
            fn blob_pixels(&self, _: ContextId, _: ResourceHandle) -> Option<PixelSource<'_>> {
                None
            }
        }
        let retire = crate::fence::Retirement::start(
            Box::new(Discard),
            crate::vrend::debug::Switches::default(),
        );
        let mut v = Vrend::new(
            Config::default(),
            &crate::budget::Budget::with_cap(None, false),
            retire.handle(),
            None,
            crate::vrend::resource::Condemned::default(),
            crate::vrend::debug::Traces::default(),
            crate::vrend::debug::Switches::default(),
        )
        .expect("vrend comes up");
        assert!(v.features.has(Feature::compute_shader), "the premise: this driver has compute");

        const SIDE: u32 = 16;
        let r32f = super::super::proto::Format::from_wire(28).expect("R32_FLOAT");
        let handle = |n: u32| ResourceHandle::new(n).expect("a resource handle is non-zero");
        let (whole, top, args) = (handle(1), handle(2), handle(3));
        let tex = resource::Args {
            target: TextureTarget::Texture2d,
            format: r32f,
            bind: resource::Bind(1 << 3),
            width: SIDE,
            height: SIDE,
            depth: 1,
            array_size: 1,
            last_level: 0,
            nr_samples: 0,
            flags: resource::ResourceFlags(0),
        };
        v.resource_create(whole, tex).expect("a texture");
        v.resource_create(top, tex).expect("a texture");
        let buffer = resource::Args {
            target: TextureTarget::Buffer,
            format: super::super::proto::Format::from_wire(64).expect("R8_UNORM"),
            bind: resource::Bind(1 << 8),
            width: 32,
            height: 1,
            ..tex
        };
        v.resource_create(args, buffer).expect("a command-args buffer");
        let ctx = ClassicCtx::for_test(ContextId::new(1).expect("a context id"));
        v.context_create(ctx, &AllAttached).expect("a context");

        let cs = tgsi_words(
            "COMP\nPROPERTY CS_FIXED_BLOCK_WIDTH 8\nPROPERTY CS_FIXED_BLOCK_HEIGHT 8\n\
             PROPERTY CS_FIXED_BLOCK_DEPTH 1\nDCL SV[0], THREAD_ID\nDCL SV[1], BLOCK_ID\n\
             DCL IMAGE[0], 2D, PIPE_FORMAT_R32_FLOAT, WR\nDCL TEMP[0..1]\n\
             IMM[0] UINT32 { 8, 16, 1, 0 }\n\
             0: UMAD TEMP[0].xy, SV[1].xyyy, IMM[0].xxxx, SV[0].xyyy\n\
             1: UMAD TEMP[1].x, TEMP[0].yyyy, IMM[0].yyyy, TEMP[0].xxxx\n\
             2: UADD TEMP[1].x, TEMP[1].xxxx, IMM[0].zzzz\n\
             3: U2F TEMP[1].x, TEMP[1].xxxx\n\
             4: STORE IMAGE[0], TEMP[0], TEMP[1].xxxx, 2D, PIPE_FORMAT_R32_FLOAT\n\
             5: END\n",
        );
        let shader = ObjectHandle::new(1).expect("non-zero");
        let image = |resource| Command::SetShaderImages {
            stage: ShaderStage::Compute,
            start_slot: 0,
            images: vec![Some(ShaderImage {
                format: r32f,
                access: ImageAccess::Write,
                layer_offset: 0,
                level_size: 0,
                resource,
            })],
        };
        // The grid at word one, so the dispatch has to honour the offset to read it.
        let grid = [0xdead, 2, 1, 1];
        let region = Box3 { x: 0, y: 0, z: 0, width: 16, height: 1, depth: 1 };
        let transfer =
            Transfer { resource: args, level: 0, usage: 0, stride: 0, layer_stride: 0, region };
        let mut wire = Vec::new();
        for cmd in [
            Command::CreateObject {
                handle: shader,
                object: Object::Shader(ShaderCreate {
                    stage: ShaderStage::Compute,
                    chunk: ShaderChunk::New { total_bytes: cs.len() as u32 * 4 },
                    num_tokens: 300,
                    kind: ShaderKind::Compute { req_local_mem: 0 },
                    text: &cs,
                }),
            },
            Command::BindShader { stage: ShaderStage::Compute, handle: Some(shader) },
            Command::ResourceInlineWrite { transfer, data: &grid },
            image(whole),
            Command::LaunchGrid {
                block: [8, 8, 1],
                grid: [2, 2, 1],
                indirect: None,
                indirect_offset: 0,
            },
            image(top),
            Command::LaunchGrid {
                block: [8, 8, 1],
                grid: [0; 3],
                indirect: Some(args),
                indirect_offset: 4,
            },
            Command::MemoryBarrier((1 << 14) - 1),
        ] {
            encode(&cmd, &mut wire);
        }
        v.submit(ctx, &wire, &AllAttached).expect("the context is here").expect("both dispatch");

        let texels = |v: &mut Vrend, res| -> Vec<f32> {
            let read = v.cursor_contents(res).expect("a small 2D texture reads back");
            read.pixels.chunks(4).map(|b| f32::from_ne_bytes(b.try_into().expect("4"))).collect()
        };
        let number = |x: u32, y: u32| (x + SIDE * y + 1) as f32;
        let read = texels(&mut v, whole);
        for (y, x) in (0..SIDE).flat_map(|y| (0..SIDE).map(move |x| (y, x))) {
            assert_eq!(read[(y * SIDE + x) as usize], number(x, y), "texel ({x}, {y})");
        }
        let read = texels(&mut v, top);
        for (y, x) in (0..SIDE).flat_map(|y| (0..SIDE).map(move |x| (y, x))) {
            let want = if y < 8 { number(x, y) } else { 0.0 };
            assert_eq!(read[(y * SIDE + x) as usize], want, "texel ({x}, {y}) of the 2x1 grid");
        }

        // Words 6 to 8 of an eight-word buffer: the last two are not there.
        let mut wire = Vec::new();
        let past = Command::LaunchGrid {
            block: [8, 8, 1],
            grid: [0; 3],
            indirect: Some(args),
            indirect_offset: 24,
        };
        encode(&past, &mut wire);
        assert!(
            v.submit(ctx, &wire, &AllAttached).expect("the context is here").is_err(),
            "a grid past its buffer is refused"
        );

        v.context_destroy(ctx, &AllAttached);
    }

    /// A query result the guest's pages are too short to take stays owed to them: the shadow
    /// holds it, and the next attach of pages that can hold it writes it there. Marked as
    /// delivered instead, it is lost -- the attach writes nothing to pages it believes agree.
    #[test]
    fn a_query_result_the_pages_cannot_take_stays_owed() {
        use crate::vrend::encode::encode;
        use crate::vrend::pipe::QueryType;
        use crate::vrend::proto::{Command, Object, QueryCreate};
        let _display = crate::vrend::one_display_at_a_time();
        struct Discard;
        impl crate::fence::FenceSink for Discard {
            fn context_fence(&mut self, _: ContextId, _: RingIdx, _: FenceId) {}
            fn present_fence(&mut self, _: FenceId) {}

            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        /// Every resource attached, over the same eight bytes of pages: half a query result.
        struct ShortPages([crate::abi::GuestIov; 1]);
        impl Guest for ShortPages {
            fn attached(&self, _: ContextId, _: ResourceHandle) -> bool {
                true
            }
            fn pages(&self, _: ContextId, _: ResourceHandle) -> Option<Iov<'_>> {
                Some(Iov::new(&self.0))
            }
            fn blob_pixels(&self, _: ContextId, _: ResourceHandle) -> Option<PixelSource<'_>> {
                None
            }
        }
        let pages = |buf: &mut [u8]| {
            [crate::abi::GuestIov {
                base: crate::abi::VmmPtr(buf.as_mut_ptr().cast()),
                len: buf.len(),
            }]
        };
        let mut short = [0u8; 8];
        let guest = ShortPages(pages(&mut short));
        let retire = crate::fence::Retirement::start(
            Box::new(Discard),
            crate::vrend::debug::Switches::default(),
        );
        let mut v = Vrend::new(
            Config::default(),
            &crate::budget::Budget::with_cap(None, false),
            retire.handle(),
            None,
            crate::vrend::resource::Condemned::default(),
            crate::vrend::debug::Traces::default(),
            crate::vrend::debug::Switches::default(),
        )
        .expect("vrend comes up");
        assert!(v.features.has(Feature::timer_query), "the premise: this driver has timer queries");
        let res = ResourceHandle::new(1).expect("a resource handle is non-zero");
        v.resource_create(
            res,
            resource::Args {
                target: TextureTarget::Buffer,
                format: super::super::proto::Format::from_wire(64).expect("R8_UNORM"),
                bind: resource::Bind::CUSTOM,
                width: 16,
                height: 1,
                depth: 1,
                array_size: 1,
                last_level: 0,
                nr_samples: 0,
                flags: resource::ResourceFlags(0),
            },
        )
        .expect("a host-memory buffer");
        let ctx = ClassicCtx::for_test(ContextId::new(1).expect("a context id"));
        v.context_create(ctx, &guest).expect("a context");
        let query = crate::vrend::proto::ObjectHandle::new(1).expect("non-zero");
        let create = QueryCreate { kind: QueryType::Timestamp, index: 0, offset: 0, resource: res };
        let mut wire = Vec::new();
        encode(&Command::CreateObject { handle: query, object: Object::Query(create) }, &mut wire);
        encode(&Command::EndQuery(query), &mut wire);
        v.submit(ctx, &wire, &guest).expect("the context is here").expect("recorded");
        v.finish_all();
        let mut wire = Vec::new();
        encode(&Command::GetQueryResult { query, wait: true }, &mut wire);
        v.submit(ctx, &wire, &guest).expect("the context is here").expect("read back");
        v.context_destroy(ctx, &guest);

        let mut whole = [0u8; 16];
        let entries = pages(&mut whole);
        let slot = v.resources.sync().get_mut(&res).and_then(resource::Slot::resource_mut);
        let Some(resource::Storage::Host(shadow)) = slot.map(|r| &mut r.storage) else {
            panic!("a CUSTOM buffer is host memory");
        };
        assert!(shadow.mirror_into(&Iov::new(&entries)), "sixteen bytes of pages hold it");
        assert_eq!(whole[..4], 1u32.to_le_bytes(), "VIRGL_QUERY_STATE_DONE reaches the pages");
    }

    /// A CUSTOM buffer is host memory the guest sized, so it is in the ledger at that size.
    ///
    /// Its width is whatever the create said -- mesa's video bitstream buffers are CUSTOM and
    /// grow with the stream -- and without the charge the one thing that can say which host
    /// allocation is growing would be blind to it.
    #[test]
    fn a_custom_buffer_is_charged_for_the_bytes_it_holds() {
        let _display = crate::vrend::one_display_at_a_time();
        struct Discard;
        impl crate::fence::FenceSink for Discard {
            fn context_fence(&mut self, _: ContextId, _: RingIdx, _: FenceId) {}
            fn present_fence(&mut self, _: FenceId) {}

            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        let retire = crate::fence::Retirement::start(
            Box::new(Discard),
            crate::vrend::debug::Switches::default(),
        );
        let budget = crate::budget::Budget::with_cap(None, false);
        let mut v = Vrend::new(
            Config::default(),
            &budget,
            retire.handle(),
            None,
            crate::vrend::resource::Condemned::default(),
            crate::vrend::debug::Traces::default(),
            crate::vrend::debug::Switches::default(),
        )
        .expect("vrend comes up");
        let before = budget.classic();
        let res = ResourceHandle::new(1).expect("a resource handle is non-zero");
        v.resource_create(
            res,
            resource::Args {
                target: TextureTarget::Buffer,
                format: super::super::proto::Format::from_wire(64).expect("R8_UNORM"),
                bind: resource::Bind::CUSTOM,
                width: 65536,
                height: 1,
                depth: 1,
                array_size: 1,
                last_level: 0,
                nr_samples: 0,
                flags: resource::ResourceFlags(0),
            },
        )
        .expect("a host-memory buffer");
        assert_eq!(budget.classic() - before, 65536, "the ledger holds the buffer's width");
    }

    /// A blob whose typing is refused keeps the exporter's share: the handle stays untyped with
    /// the storage it was attached with, as if the SET_TYPE had never come.
    #[cfg(target_os = "macos")]
    #[test]
    fn a_refused_set_type_keeps_the_exporters_storage() {
        use crate::vrend::encode::encode;
        use crate::vrend::proto::Command;
        let _display = crate::vrend::one_display_at_a_time();
        struct Discard;
        impl crate::fence::FenceSink for Discard {
            fn context_fence(&mut self, _: ContextId, _: RingIdx, _: FenceId) {}
            fn present_fence(&mut self, _: FenceId) {}

            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        struct AllAttached;
        impl Guest for AllAttached {
            fn attached(&self, _: ContextId, _: ResourceHandle) -> bool {
                true
            }
            fn pages(&self, _: ContextId, _: ResourceHandle) -> Option<Iov<'_>> {
                None
            }
            fn blob_pixels(&self, _: ContextId, _: ResourceHandle) -> Option<PixelSource<'_>> {
                None
            }
        }
        let retire = crate::fence::Retirement::start(
            Box::new(Discard),
            crate::vrend::debug::Switches::default(),
        );
        let mut v = Vrend::new(
            Config::default(),
            &crate::budget::Budget::with_cap(None, false),
            retire.handle(),
            None,
            crate::vrend::resource::Condemned::default(),
            crate::vrend::debug::Traces::default(),
            crate::vrend::debug::Switches::default(),
        )
        .expect("vrend comes up");
        let ctx = ClassicCtx::for_test(ContextId::new(1).expect("a context id"));
        v.context_create(ctx, &AllAttached).expect("a context");

        let surface = crate::metal::Surface::scanout(64, 8, surface::PixelFormat::Bgra, 256)
            .expect("the system minted a surface");
        let held: Arc<dyn surface::Held> = Arc::new(surface);
        let res = ResourceHandle::new(1).expect("a resource handle is non-zero");
        v.resource_attach_blob(ctx, res, Some(surface::Adoptable::Exported(Arc::clone(&held))));

        // A zero width describes no image, so the typing is refused before anything is adopted.
        let mut wire = Vec::new();
        encode(
            &Command::PipeResourceSetType {
                resource: res,
                format: super::super::proto::Format::from_wire(1).expect("B8G8R8A8_UNORM"),
                bind: 0,
                width: 0,
                height: 8,
                usage: 0,
                modifier: 0,
                planes: vec![crate::vrend::proto::Plane { stride: 256, offset: 0 }],
            },
            &mut wire,
        );
        let refused = v.submit(ctx, &wire, &AllAttached).expect("the context is here");
        assert!(
            matches!(
                refused,
                Err(crate::vrend::context::Fault::RefusedResource {
                    why: resource::Refusal::ZeroWidth,
                    ..
                })
            ),
            "the typing reaches the adopt and is refused there: {refused:?}"
        );

        let kept = match v.resources.sync().get(&res) {
            Some(resource::Slot::Untyped(u)) => u.surface().map(Arc::clone),
            Some(resource::Slot::Resource(_)) => panic!("a refused typing typed the handle"),
            None => panic!("a refused typing deleted the handle"),
        };
        let kept = kept.expect("the slot still holds the exporter's share");
        assert!(Arc::ptr_eq(&kept, &held), "and it is the same share");
        v.context_destroy(ctx, &AllAttached);
    }

    /// A texture no framebuffer can read reads back on desktop GL through
    /// `glGetCompressedTexImage`, into pages other than the ones it was written from. GLES has no
    /// such read, and refuses: the control that says the desktop answer came from the texture and
    /// not from the pages it was uploaded from. A compressed format is never read through a
    /// framebuffer, and its blocks come back as they were stored.
    #[test]
    fn a_texture_no_framebuffer_reads_reads_back_on_desktop_gl() {
        let _display = crate::vrend::one_display_at_a_time();
        struct Discard;
        impl crate::fence::FenceSink for Discard {
            fn context_fence(&mut self, _: ContextId, _: RingIdx, _: FenceId) {}
            fn present_fence(&mut self, _: FenceId) {}
            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        let up = |host_gl| {
            let retire = crate::fence::Retirement::start(
                Box::new(Discard),
                crate::vrend::debug::Switches::default(),
            );
            let v = Vrend::new(
                Config { host_gl, ..Config::default() },
                &crate::budget::Budget::with_cap(None, false),
                retire.handle(),
                None,
                crate::vrend::resource::Condemned::default(),
                crate::vrend::debug::Traces::default(),
                crate::vrend::debug::Switches::default(),
            )
            .expect("vrend comes up");
            (v, retire)
        };
        // RGTC is core in desktop GL 3.0, so a desktop driver stores it rather than emulating it
        // -- an emulated format is decompressed on upload, and reads back as whatever the driver
        // makes of that.
        let unreadable = |v: &Vrend, f: crate::vrend::proto::Format| {
            v.formats.get(f).is_some_and(|e| e.bindings.sampler_view) && f.name() == "RGTC1_UNORM"
        };
        let format = {
            let (gles, _g) = up(HostGl::Gles);
            let (desk, _d) = up(HostGl::Desktop);
            (0..crate::vrend::proto::FORMAT_MAX)
                .filter_map(crate::vrend::proto::Format::from_wire)
                .find(|&f| unreadable(&gles, f) && unreadable(&desk, f))
                .expect("both flavours sample RGTC1")
        };
        let read_back = |host_gl| {
            let (mut v, _retire) = up(host_gl);
            let desc = format.describe().expect("described");
            // Two blocks by two, so the box has rows and columns of blocks to get in order.
            let (w, h) = (8u32, 8u32);
            let stride = desc.stride(w) as u32;
            let layer = stride * desc.blocks_high(h);
            let res = ResourceHandle::new(1).expect("non-zero");
            v.resource_create(
                res,
                resource::Args {
                    target: TextureTarget::Texture2d,
                    format,
                    bind: resource::Bind::SAMPLER_VIEW,
                    width: w,
                    height: h,
                    depth: 1,
                    array_size: 1,
                    last_level: 0,
                    nr_samples: 0,
                    flags: resource::ResourceFlags(0),
                },
            )
            .expect("a texture");
            let info = transfer::Info {
                level: 0,
                stride,
                layer_stride: layer,
                offset: 0,
                region: crate::vrend::proto::Box3 {
                    x: 0,
                    y: 0,
                    z: 0,
                    width: w as i32,
                    height: h as i32,
                    depth: 1,
                },
                synchronized: false,
            };
            let n = layer as usize;
            let mut written: Vec<u8> = (0..n).map(|b| (b as u8).wrapping_mul(37)).collect();
            let from = [crate::abi::GuestIov {
                base: crate::abi::VmmPtr(written.as_mut_ptr().cast()),
                len: written.len(),
            }];
            let from = Iov::new(&from);
            v.transfer(None, res, Some(&from), transfer::Through::ToHost(from.source()), &info)
                .expect("the upload");
            let mut read = vec![0xa5u8; n];
            let into = [crate::abi::GuestIov {
                base: crate::abi::VmmPtr(read.as_mut_ptr().cast()),
                len: read.len(),
            }];
            let into = Iov::new(&into);
            let r = v.transfer(None, res, Some(&from), transfer::Through::ToGuest(&into), &info);
            (r.is_ok(), read == written)
        };
        assert_eq!(read_back(HostGl::Gles), (false, false), "GLES cannot read {}", format.name());
        assert_eq!(read_back(HostGl::Desktop), (true, true), "desktop GL reads {}", format.name());
    }

    /// A cursor whose rows are not a multiple of four bytes reads back as written, on a renderer
    /// that has read nothing back before. GL's default pack alignment is 4, so a read that
    /// leaves the pack state to whoever ran before it gets 3-byte rows padded to 4: offsets
    /// nothing here knows, and on a host without bounded reads, bytes past the buffer.
    #[test]
    fn an_unaligned_cursor_reads_back_on_a_fresh_renderer_on_either_flavour() {
        let _display = crate::vrend::one_display_at_a_time();
        struct Discard;
        impl crate::fence::FenceSink for Discard {
            fn context_fence(&mut self, _: ContextId, _: RingIdx, _: FenceId) {}
            fn present_fence(&mut self, _: FenceId) {}
            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        let r8 = (0..crate::vrend::proto::FORMAT_MAX)
            .filter_map(crate::vrend::proto::Format::from_wire)
            .find(|f| f.name() == "R8_UNORM")
            .expect("R8_UNORM is a wire format");
        for host_gl in [HostGl::Gles, HostGl::Desktop] {
            let retire = crate::fence::Retirement::start(
                Box::new(Discard),
                crate::vrend::debug::Switches::default(),
            );
            let mut v = Vrend::new(
                Config { host_gl, ..Config::default() },
                &crate::budget::Budget::with_cap(None, false),
                retire.handle(),
                None,
                crate::vrend::resource::Condemned::default(),
                crate::vrend::debug::Traces::default(),
                crate::vrend::debug::Switches::default(),
            )
            .expect("vrend comes up");
            let (w, h) = (3u32, 3u32);
            let res = ResourceHandle::new(1).expect("non-zero");
            v.resource_create(
                res,
                resource::Args {
                    target: TextureTarget::Texture2d,
                    format: r8,
                    bind: resource::Bind::SAMPLER_VIEW,
                    width: w,
                    height: h,
                    depth: 1,
                    array_size: 1,
                    last_level: 0,
                    nr_samples: 0,
                    flags: resource::ResourceFlags(0),
                },
            )
            .expect("a texture");
            let info = transfer::Info {
                level: 0,
                stride: w,
                layer_stride: w * h,
                offset: 0,
                region: crate::vrend::proto::Box3 {
                    x: 0,
                    y: 0,
                    z: 0,
                    width: w as i32,
                    height: h as i32,
                    depth: 1,
                },
                synchronized: false,
            };
            let mut written: Vec<u8> = (1..=(w * h) as u8).map(|b| b.wrapping_mul(29)).collect();
            let from = [crate::abi::GuestIov {
                base: crate::abi::VmmPtr(written.as_mut_ptr().cast()),
                len: written.len(),
            }];
            let from = Iov::new(&from);
            v.transfer(None, res, Some(&from), transfer::Through::ToHost(from.source()), &info)
                .expect("the upload");
            let cursor = v.cursor_contents(res);
            assert_eq!(
                cursor.map(|c| c.pixels),
                Some(written),
                "{host_gl:?}: the cursor reads back as written"
            );
        }
    }

    /// No planar YUV format is advertised as multisampled, on either flavour. Its RGBA8 storage
    /// would make a multisample texture, but one is only ever a render target, and the planes
    /// are only sampled.
    #[test]
    fn no_planar_yuv_format_is_advertised_multisampled_on_either_flavour() {
        let _display = crate::vrend::one_display_at_a_time();
        struct Discard;
        impl crate::fence::FenceSink for Discard {
            fn context_fence(&mut self, _: ContextId, _: RingIdx, _: FenceId) {}
            fn present_fence(&mut self, _: FenceId) {}
            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        for host_gl in [HostGl::Gles, HostGl::Desktop] {
            let retire = crate::fence::Retirement::start(
                Box::new(Discard),
                crate::vrend::debug::Switches::default(),
            );
            let v = Vrend::new(
                Config { host_gl, ..Config::default() },
                &crate::budget::Budget::with_cap(None, false),
                retire.handle(),
                None,
                crate::vrend::resource::Condemned::default(),
                crate::vrend::debug::Traces::default(),
                crate::vrend::debug::Switches::default(),
            )
            .expect("vrend comes up");
            let planar: Vec<_> = (0..crate::vrend::proto::FORMAT_MAX)
                .filter_map(crate::vrend::proto::Format::from_wire)
                .filter(|&f| super::super::video::guest_planes(f) > 1)
                .collect();
            assert!(!planar.is_empty(), "the premise: the wire has planar formats");
            let multisampled: Vec<&str> = planar
                .iter()
                .filter(|&&f| v.formats.get(f).is_some_and(|e| e.can_multisample))
                .map(|f| f.name())
                .collect();
            assert!(multisampled.is_empty(), "{host_gl:?}: multisampled planes: {multisampled:?}");
        }
    }

    /// A blue-first cursor reads back in the byte order it was written, on either flavour. A
    /// GLES host stores B8G8R8A8 as RGBA and swaps on upload, so a read that does not swap back
    /// hands the VMM a pointer with red and blue exchanged.
    #[test]
    fn a_bgra_cursor_reads_back_in_the_order_it_was_written_on_either_flavour() {
        let _display = crate::vrend::one_display_at_a_time();
        struct Discard;
        impl crate::fence::FenceSink for Discard {
            fn context_fence(&mut self, _: ContextId, _: RingIdx, _: FenceId) {}
            fn present_fence(&mut self, _: FenceId) {}
            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        let bgra = super::super::proto::Format::from_wire(1).expect("B8G8R8A8_UNORM");
        for host_gl in [HostGl::Gles, HostGl::Desktop] {
            let retire = crate::fence::Retirement::start(
                Box::new(Discard),
                crate::vrend::debug::Switches::default(),
            );
            let mut v = Vrend::new(
                Config { host_gl, ..Config::default() },
                &crate::budget::Budget::with_cap(None, false),
                retire.handle(),
                None,
                crate::vrend::resource::Condemned::default(),
                crate::vrend::debug::Traces::default(),
                crate::vrend::debug::Switches::default(),
            )
            .expect("vrend comes up");
            let (w, h) = (4u32, 4u32);
            let res = ResourceHandle::new(1).expect("non-zero");
            v.resource_create(
                res,
                resource::Args {
                    target: TextureTarget::Texture2d,
                    format: bgra,
                    bind: resource::Bind::SAMPLER_VIEW,
                    width: w,
                    height: h,
                    depth: 1,
                    array_size: 1,
                    last_level: 0,
                    nr_samples: 0,
                    flags: resource::ResourceFlags(0),
                },
            )
            .expect("a texture");
            let info = transfer::Info {
                level: 0,
                stride: w * 4,
                layer_stride: w * h * 4,
                offset: 0,
                region: crate::vrend::proto::Box3 {
                    x: 0,
                    y: 0,
                    z: 0,
                    width: w as i32,
                    height: h as i32,
                    depth: 1,
                },
                synchronized: false,
            };
            // Blue, green, red, alpha: four bytes no swap leaves where they were.
            let mut written: Vec<u8> = [0x10u8, 0x40, 0xc0, 0xff].repeat((w * h) as usize);
            let from = [crate::abi::GuestIov {
                base: crate::abi::VmmPtr(written.as_mut_ptr().cast()),
                len: written.len(),
            }];
            let from = Iov::new(&from);
            v.transfer(None, res, Some(&from), transfer::Through::ToHost(from.source()), &info)
                .expect("the upload");
            let cursor = v.cursor_contents(res).expect("a small 2D texture reads back");
            assert_eq!(&cursor.pixels[..4], &written[..4], "{host_gl:?}: blue first, as written");
            assert_eq!(cursor.pixels, written, "{host_gl:?}");
        }
    }

    /// A cursor no framebuffer can read reads back on desktop GL, as the C reads every cursor
    /// there: through `glGetTexImage`, whose compressed form hands the blocks back as stored.
    /// GLES has only the framebuffer, and answers nothing -- the control that says the desktop
    /// answer took the other road.
    #[test]
    fn a_cursor_no_framebuffer_reads_reads_back_on_desktop_gl() {
        let _display = crate::vrend::one_display_at_a_time();
        struct Discard;
        impl crate::fence::FenceSink for Discard {
            fn context_fence(&mut self, _: ContextId, _: RingIdx, _: FenceId) {}
            fn present_fence(&mut self, _: FenceId) {}
            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        let rgtc1 = (0..crate::vrend::proto::FORMAT_MAX)
            .filter_map(crate::vrend::proto::Format::from_wire)
            .find(|f| f.name() == "RGTC1_UNORM")
            .expect("RGTC1_UNORM is a wire format");
        let read_back = |host_gl| {
            let retire = crate::fence::Retirement::start(
                Box::new(Discard),
                crate::vrend::debug::Switches::default(),
            );
            let mut v = Vrend::new(
                Config { host_gl, ..Config::default() },
                &crate::budget::Budget::with_cap(None, false),
                retire.handle(),
                None,
                crate::vrend::resource::Condemned::default(),
                crate::vrend::debug::Traces::default(),
                crate::vrend::debug::Switches::default(),
            )
            .expect("vrend comes up");
            let desc = rgtc1.describe().expect("described");
            // Two blocks by two, so the rows of blocks have an order to keep.
            let (w, h) = (8u32, 8u32);
            let stride = desc.stride(w) as u32;
            let layer = stride * desc.blocks_high(h);
            let res = ResourceHandle::new(1).expect("non-zero");
            v.resource_create(
                res,
                resource::Args {
                    target: TextureTarget::Texture2d,
                    format: rgtc1,
                    bind: resource::Bind::SAMPLER_VIEW,
                    width: w,
                    height: h,
                    depth: 1,
                    array_size: 1,
                    last_level: 0,
                    nr_samples: 0,
                    flags: resource::ResourceFlags(0),
                },
            )
            .expect("a texture");
            let info = transfer::Info {
                level: 0,
                stride,
                layer_stride: layer,
                offset: 0,
                region: crate::vrend::proto::Box3 {
                    x: 0,
                    y: 0,
                    z: 0,
                    width: w as i32,
                    height: h as i32,
                    depth: 1,
                },
                synchronized: false,
            };
            let mut written: Vec<u8> = (0..layer).map(|b| (b as u8).wrapping_mul(37)).collect();
            let from = [crate::abi::GuestIov {
                base: crate::abi::VmmPtr(written.as_mut_ptr().cast()),
                len: written.len(),
            }];
            let from = Iov::new(&from);
            v.transfer(None, res, Some(&from), transfer::Through::ToHost(from.source()), &info)
                .expect("the upload");
            v.cursor_contents(res).map(|c| c.pixels == written)
        };
        assert_eq!(read_back(HostGl::Gles), None, "GLES cannot read an RGTC1 cursor");
        assert_eq!(read_back(HostGl::Desktop), Some(true), "desktop GL reads its blocks back");
    }

    /// A sampler view of a texture buffer names the buffer target, and is made where the host has
    /// texture buffers -- elsewhere the caps offer none and the range is refused; a view naming
    /// the buffer target over a texture is refused. Neither may reach the texture-target table,
    /// which has no entry for a buffer.
    #[test]
    fn a_buffer_sampler_view_is_made_over_a_buffer_and_refused_over_a_texture() {
        use crate::vrend::encode::encode;
        use crate::vrend::pipe::Swizzle;
        use crate::vrend::proto::{Command, Object, SamplerView};
        let _display = crate::vrend::one_display_at_a_time();
        struct Discard;
        impl crate::fence::FenceSink for Discard {
            fn context_fence(&mut self, _: ContextId, _: RingIdx, _: FenceId) {}
            fn present_fence(&mut self, _: FenceId) {}
            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        struct AllAttached;
        impl Guest for AllAttached {
            fn attached(&self, _: ContextId, _: ResourceHandle) -> bool {
                true
            }
            fn pages(&self, _: ContextId, _: ResourceHandle) -> Option<Iov<'_>> {
                None
            }
            fn blob_pixels(&self, _: ContextId, _: ResourceHandle) -> Option<PixelSource<'_>> {
                None
            }
        }
        let r8 = super::super::proto::Format::from_wire(64).expect("R8_UNORM");
        for host_gl in [HostGl::Gles, HostGl::Desktop] {
            let retire = crate::fence::Retirement::start(
                Box::new(Discard),
                crate::vrend::debug::Switches::default(),
            );
            let mut v = Vrend::new(
                Config { host_gl, ..Config::default() },
                &crate::budget::Budget::with_cap(None, false),
                retire.handle(),
                None,
                crate::vrend::resource::Condemned::default(),
                crate::vrend::debug::Traces::default(),
                crate::vrend::debug::Switches::default(),
            )
            .expect("vrend comes up");
            let (buffer, texture) = (
                ResourceHandle::new(1).expect("non-zero"),
                ResourceHandle::new(2).expect("non-zero"),
            );
            v.resource_create(buffer, buffer_args(resource::Bind::SAMPLER_VIEW, 64))
                .expect("a texture buffer");
            v.resource_create(
                texture,
                resource::Args {
                    target: TextureTarget::Texture2d,
                    format: r8,
                    bind: resource::Bind::SAMPLER_VIEW,
                    width: 8,
                    height: 8,
                    depth: 1,
                    array_size: 1,
                    last_level: 0,
                    nr_samples: 0,
                    flags: resource::ResourceFlags(0),
                },
            )
            .expect("a texture");
            let tbo = v.features.has(Feature::arb_or_gles_ext_texture_buffer);
            // A context each, so that the first refusal cannot be what refuses the second.
            for (n, resource) in [(1, buffer), (2, texture)] {
                let ctx = ClassicCtx::for_test(ContextId::new(n).expect("a context id"));
                v.context_create(ctx, &AllAttached).expect("a context");
                let mut wire = Vec::new();
                encode(
                    &Command::CreateObject {
                        handle: ObjectHandle::new(n).expect("non-zero"),
                        object: Object::SamplerView(SamplerView {
                            resource,
                            format: r8,
                            target: TextureTarget::Buffer,
                            first_element_or_layers: 0,
                            last_element_or_levels: 63,
                            swizzle: [Swizzle::X, Swizzle::Y, Swizzle::Z, Swizzle::W],
                        }),
                    },
                    &mut wire,
                );
                let ran = v.submit(ctx, &wire, &AllAttached).expect("the context is here");
                let as_wanted = match (resource == buffer, tbo) {
                    (true, true) => ran.is_ok(),
                    (true, false) => matches!(ran, Err(Fault::OutOfRange { .. })),
                    (false, _) => {
                        matches!(ran, Err(Fault::IllegalResource { handle, .. }) if handle == texture)
                    }
                };
                assert!(
                    as_wanted,
                    "{host_gl:?} (texture buffers: {tbo}): view of {resource:?}: {ran:?}"
                );
            }
        }
    }

    /// A blit's boxes are the guest's own, and a gallium box may run backwards or off its texture,
    /// so the blit cannot ask that they lie inside the resource. A coordinate or an end far past
    /// any texture is refused; anything short of that, however stretched, is served or refused
    /// without the coordinate arithmetic overflowing -- which, with overflow checks on, as in a
    /// dev build of any consumer, aborts the process.
    #[test]
    fn a_blit_box_far_past_any_texture_is_refused_and_one_short_of_it_is_survived() {
        use crate::vrend::encode::encode;
        use crate::vrend::pipe::TexFilter;
        use crate::vrend::proto::{Blit, BlitTarget, Box3, Command, Scissor};
        let _display = crate::vrend::one_display_at_a_time();
        struct Discard;
        impl crate::fence::FenceSink for Discard {
            fn context_fence(&mut self, _: ContextId, _: RingIdx, _: FenceId) {}
            fn present_fence(&mut self, _: FenceId) {}
            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        struct AllAttached;
        impl Guest for AllAttached {
            fn attached(&self, _: ContextId, _: ResourceHandle) -> bool {
                true
            }
            fn pages(&self, _: ContextId, _: ResourceHandle) -> Option<Iov<'_>> {
                None
            }
            fn blob_pixels(&self, _: ContextId, _: ResourceHandle) -> Option<PixelSource<'_>> {
                None
            }
        }
        let format = |n| super::super::proto::Format::from_wire(n).expect("a wire format");
        // A source no framebuffer can hold, so the blit takes the blitter's shader, which is where
        // a stretched box is scaled.
        let (rgb9e5, rgba) = (format(125), format(67));
        let far = 1 << 25;
        let near = 1 << 23;
        let r = |x, y, width, height| Box3 { x, y, z: 0, width, height, depth: 1 };
        // (source box, destination box, refused as out of range)
        let cases = [
            (r(i32::MAX - 4, 0, 8, 8), r(0, 0, 8, 8), true),
            (r(0, i32::MIN, 8, -8), r(0, 0, 8, 8), true),
            (r(0, 0, 8, 8), r(far, 0, 8, 8), true),
            (r(0, 0, 8, 8), r(0, 0, 8, -far), true),
            (r(-near, -near, 1, 1), r(0, 0, near, near), false),
            (r(15, 15, -near, -near), r(near, near, -1, -1), false),
        ];
        for host_gl in [HostGl::Gles, HostGl::Desktop] {
            let retire = crate::fence::Retirement::start(
                Box::new(Discard),
                crate::vrend::debug::Switches::default(),
            );
            let mut v = Vrend::new(
                Config { host_gl, ..Config::default() },
                &crate::budget::Budget::with_cap(None, false),
                retire.handle(),
                None,
                crate::vrend::resource::Condemned::default(),
                crate::vrend::debug::Traces::default(),
                crate::vrend::debug::Switches::default(),
            )
            .expect("vrend comes up");
            let (src, dst) = (
                ResourceHandle::new(1).expect("non-zero"),
                ResourceHandle::new(2).expect("non-zero"),
            );
            let sampled = resource::Bind::SAMPLER_VIEW;
            let target = resource::Bind(resource::Bind::RENDER_TARGET.0 | sampled.0);
            for (handle, format, bind) in [(src, rgb9e5, sampled), (dst, rgba, target)] {
                v.resource_create(
                    handle,
                    resource::Args {
                        target: TextureTarget::Texture2d,
                        format,
                        bind,
                        width: 16,
                        height: 16,
                        depth: 1,
                        array_size: 1,
                        last_level: 0,
                        nr_samples: 0,
                        flags: resource::ResourceFlags(0),
                    },
                )
                .expect("a texture");
            }
            for (n, (from, to, refused)) in cases.into_iter().enumerate() {
                // A context each, so that one refusal cannot be what refuses the next.
                let ctx = ClassicCtx::for_test(ContextId::new(n as u32 + 1).expect("an id"));
                v.context_create(ctx, &AllAttached).expect("a context");
                let mut wire = Vec::new();
                encode(
                    &Command::Blit(Blit {
                        mask: 0xf,
                        filter: TexFilter::Nearest,
                        scissor_enable: false,
                        render_condition_enable: false,
                        alpha_blend: false,
                        scissor: Scissor { minx: 0, miny: 0, maxx: 0, maxy: 0 },
                        dst: BlitTarget { resource: dst, level: 0, format: rgba, region: to },
                        src: BlitTarget { resource: src, level: 0, format: rgb9e5, region: from },
                    }),
                    &mut wire,
                );
                let ran = v.submit(ctx, &wire, &AllAttached).expect("the context is here");
                if refused {
                    assert!(
                        matches!(ran, Err(Fault::OutOfRange { .. })),
                        "{host_gl:?}: {from:?} -> {to:?}: {ran:?}"
                    );
                }
            }
        }
    }

    /// A blit into a layered destination lands at the layers its box names. The source here is one
    /// no framebuffer can hold, so the blit draws through the blitter's shader one destination
    /// slice at a time, and each slice is counted from the box's first: a 3D destination's box
    /// starts at a slice of its own, and an array box deeper than one layer walks its layers. A
    /// cube source's box names a face, and the face is what lands.
    #[test]
    fn a_blit_through_the_blitter_lands_at_the_destination_box_s_layers() {
        use crate::vrend::encode::encode;
        use crate::vrend::pipe::TexFilter;
        use crate::vrend::proto::{Blit, BlitTarget, Box3, Command, Scissor};
        let _display = crate::vrend::one_display_at_a_time();
        struct Discard;
        impl crate::fence::FenceSink for Discard {
            fn context_fence(&mut self, _: ContextId, _: RingIdx, _: FenceId) {}
            fn present_fence(&mut self, _: FenceId) {}
            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        struct AllAttached;
        impl Guest for AllAttached {
            fn attached(&self, _: ContextId, _: ResourceHandle) -> bool {
                true
            }
            fn pages(&self, _: ContextId, _: ResourceHandle) -> Option<Iov<'_>> {
                None
            }
            fn blob_pixels(&self, _: ContextId, _: ResourceHandle) -> Option<PixelSource<'_>> {
                None
            }
        }
        let named = |name: &str| {
            (0..crate::vrend::proto::FORMAT_MAX)
                .filter_map(crate::vrend::proto::Format::from_wire)
                .find(|f| f.name() == name)
                .expect("a wire format")
        };
        let (rgb9e5, rgba) = (named("R9G9B9E5_FLOAT"), named("R8G8B8A8_UNORM"));
        // An RGB9E5 texel of exponent 16 reads its mantissas over 256, so 256 is 1.0.
        let texel = |r: u32, g: u32, b: u32| (r | g << 9 | b << 18 | 16 << 27).to_le_bytes();
        let (red, green) = (texel(256, 0, 0), texel(0, 256, 0));
        let (w, h) = (4u32, 4u32);
        let layer_bytes = (w * h * 4) as usize;
        let region = |z, depth| Box3 { x: 0, y: 0, z, width: w as i32, height: h as i32, depth };
        let info = |z, depth| transfer::Info {
            level: 0,
            stride: w * 4,
            layer_stride: w * h * 4,
            offset: 0,
            region: region(z, depth),
            synchronized: false,
        };
        // (source target, source layers, destination target, destination depth, destination
        // layers, the blit's source box, its destination box, the destination's layers after it)
        let cases = [
            (
                TextureTarget::Texture2d,
                vec![red],
                TextureTarget::Texture3d,
                4,
                1,
                region(0, 1),
                region(2, 1),
                [None, None, Some(red), None],
            ),
            (
                TextureTarget::Array2d,
                vec![red, green],
                TextureTarget::Array2d,
                1,
                4,
                region(0, 2),
                region(1, 2),
                [None, Some(red), Some(green), None],
            ),
            // A cube face is sampled by the direction that points at it: +Z and -Y, each the
            // only red face of its cube.
            (
                TextureTarget::Cube,
                vec![green, green, green, green, red, green],
                TextureTarget::Array2d,
                1,
                4,
                region(4, 1),
                region(0, 1),
                [Some(red), None, None, None],
            ),
            (
                TextureTarget::Cube,
                vec![green, green, green, red, green, green],
                TextureTarget::Array2d,
                1,
                4,
                region(3, 1),
                region(2, 1),
                [None, None, Some(red), None],
            ),
        ];
        for host_gl in [HostGl::Gles, HostGl::Desktop] {
            for (n, (src_target, src_layers, dst_target, depth, array_size, from, to, want)) in
                cases.iter().cloned().enumerate()
            {
                let retire = crate::fence::Retirement::start(
                    Box::new(Discard),
                    crate::vrend::debug::Switches::default(),
                );
                let mut v = Vrend::new(
                    Config { host_gl, ..Config::default() },
                    &crate::budget::Budget::with_cap(None, false),
                    retire.handle(),
                    None,
                    crate::vrend::resource::Condemned::default(),
                    crate::vrend::debug::Traces::default(),
                    crate::vrend::debug::Switches::default(),
                )
                .expect("vrend comes up");
                let ctx = ClassicCtx::for_test(ContextId::new(1).expect("a context id"));
                v.context_create(ctx, &AllAttached).expect("a context");
                let (src, dst) = (
                    ResourceHandle::new(1).expect("non-zero"),
                    ResourceHandle::new(2).expect("non-zero"),
                );
                let sampled = resource::Bind::SAMPLER_VIEW;
                let rendered = resource::Bind(resource::Bind::RENDER_TARGET.0 | sampled.0);
                let mut src_bytes: Vec<u8> =
                    src_layers.iter().flat_map(|t| t.repeat((w * h) as usize)).collect();
                let src_count = src_layers.len() as u32;
                let dst_count = depth.max(array_size);
                let mut dst_bytes = vec![0u8; layer_bytes * dst_count as usize];
                for (handle, target, format, bind, depth, array_size, bytes, count) in [
                    (src, src_target, rgb9e5, sampled, 1, src_count, &mut src_bytes, src_count),
                    (dst, dst_target, rgba, rendered, depth, array_size, &mut dst_bytes, dst_count),
                ] {
                    v.resource_create(
                        handle,
                        resource::Args {
                            target,
                            format,
                            bind,
                            width: w,
                            height: h,
                            depth,
                            array_size,
                            last_level: 0,
                            nr_samples: 0,
                            flags: resource::ResourceFlags(0),
                        },
                    )
                    .expect("a texture");
                    let from = [crate::abi::GuestIov {
                        base: crate::abi::VmmPtr(bytes.as_mut_ptr().cast()),
                        len: bytes.len(),
                    }];
                    let from = Iov::new(&from);
                    // A layer at a time: a cube takes one face per upload.
                    for z in 0..count {
                        let info = transfer::Info {
                            offset: (z as usize * layer_bytes) as u64,
                            ..info(z as i32, 1)
                        };
                        v.transfer(
                            None,
                            handle,
                            Some(&from),
                            transfer::Through::ToHost(from.source()),
                            &info,
                        )
                        .expect("the upload");
                    }
                }
                let mut wire = Vec::new();
                encode(
                    &Command::Blit(Blit {
                        mask: 0xf,
                        filter: TexFilter::Nearest,
                        scissor_enable: false,
                        render_condition_enable: false,
                        alpha_blend: false,
                        scissor: Scissor { minx: 0, miny: 0, maxx: 0, maxy: 0 },
                        dst: BlitTarget { resource: dst, level: 0, format: rgba, region: to },
                        src: BlitTarget { resource: src, level: 0, format: rgb9e5, region: from },
                    }),
                    &mut wire,
                );
                let ran = v.submit(ctx, &wire, &AllAttached).expect("the context is here");
                assert!(ran.is_ok(), "{host_gl:?} case {n}: {ran:?}");
                let mut read = vec![0xa5u8; layer_bytes * dst_count as usize];
                let into = [crate::abi::GuestIov {
                    base: crate::abi::VmmPtr(read.as_mut_ptr().cast()),
                    len: read.len(),
                }];
                let into = Iov::new(&into);
                v.transfer(
                    None,
                    dst,
                    None,
                    transfer::Through::ToGuest(&into),
                    &info(0, dst_count as i32),
                )
                .expect("the destination reads back");
                for (layer, want) in want.iter().enumerate() {
                    let pixel = match want {
                        Some(t) if *t == red => [0xff, 0, 0, 0xff],
                        Some(_) => [0, 0xff, 0, 0xff],
                        None => [0; 4],
                    };
                    let got = &read[layer * layer_bytes..(layer + 1) * layer_bytes];
                    assert!(
                        got.as_chunks::<4>().0.iter().all(|p| *p == pixel),
                        "{host_gl:?} case {n}: layer {layer} should be {pixel:?}, starts {:?}",
                        &got[..4]
                    );
                }
                v.context_destroy(ctx, &AllAttached);
            }
        }
    }

    /// A transfer's box is the guest's own, and the renderer sizes it before it checks it against
    /// the resource. A box whose bytes overflow any integer is refused like any other box outside
    /// the resource, not left to overflow: with overflow checks on, which a dev build of any
    /// consumer has, that overflow aborts the process.
    #[test]
    fn a_transfer_box_too_large_to_size_is_refused() {
        use crate::vrend::encode::encode;
        use crate::vrend::proto::{Box3, Command, Transfer};
        let _display = crate::vrend::one_display_at_a_time();
        struct Discard;
        impl crate::fence::FenceSink for Discard {
            fn context_fence(&mut self, _: ContextId, _: RingIdx, _: FenceId) {}
            fn present_fence(&mut self, _: FenceId) {}
            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        struct AllAttached;
        impl Guest for AllAttached {
            fn attached(&self, _: ContextId, _: ResourceHandle) -> bool {
                true
            }
            fn pages(&self, _: ContextId, _: ResourceHandle) -> Option<Iov<'_>> {
                None
            }
            fn blob_pixels(&self, _: ContextId, _: ResourceHandle) -> Option<PixelSource<'_>> {
                None
            }
        }
        let retire = crate::fence::Retirement::start(
            Box::new(Discard),
            crate::vrend::debug::Switches::default(),
        );
        let mut v = Vrend::new(
            Config::default(),
            &crate::budget::Budget::with_cap(None, false),
            retire.handle(),
            None,
            crate::vrend::resource::Condemned::default(),
            crate::vrend::debug::Traces::default(),
            crate::vrend::debug::Switches::default(),
        )
        .expect("vrend comes up");
        let ctx = ClassicCtx::for_test(ContextId::new(1).expect("a context id"));
        v.context_create(ctx, &AllAttached).expect("a context");
        let rgba = super::super::proto::Format::from_wire(67).expect("R8G8B8A8_UNORM");
        assert_eq!(rgba.name(), "R8G8B8A8_UNORM");
        let res = ResourceHandle::new(1).expect("non-zero");
        v.resource_create(
            res,
            resource::Args {
                target: TextureTarget::Texture2d,
                format: rgba,
                bind: resource::Bind::SAMPLER_VIEW,
                width: 16,
                height: 16,
                depth: 1,
                array_size: 1,
                last_level: 0,
                nr_samples: 0,
                flags: resource::ResourceFlags(0),
            },
        )
        .expect("a texture");
        // Each extent alone, then all three: a row of 2^31 four-byte texels overflows a u32, and
        // the three together overflow a u64.
        let max = i32::MAX;
        for (width, height, depth) in [(max, 1, 1), (1, max, max), (max, max, max)] {
            let mut wire = Vec::new();
            encode(
                &Command::ResourceInlineWrite {
                    transfer: Transfer {
                        resource: res,
                        level: 0,
                        usage: 0,
                        stride: 0,
                        layer_stride: 0,
                        region: Box3 { x: 0, y: 0, z: 0, width, height, depth },
                    },
                    data: &[0; 4],
                },
                &mut wire,
            );
            let ran = v.submit(ctx, &wire, &AllAttached).expect("the context is here");
            assert!(
                matches!(ran, Err(Fault::Transfer { error: transfer::Error::BoxOutOfRange, .. })),
                "{width}x{height}x{depth}: {ran:?}"
            );
        }
    }

    /// A copy between two textures no framebuffer can hold, on a driver without
    /// `glCopyImageSubData`, lands on desktop GL: the source level is read with
    /// `glGetCompressedTexImage` and the box is written into the destination. The box is one
    /// block, taken from the source's second column and put in the destination's second row,
    /// so a copy that ignored either origin -- the C's own fallback reads from the level's start
    /// -- puts the wrong block in the wrong place. GLES has no such read, and with no guest pages
    /// to copy through, refuses.
    #[test]
    fn a_copy_between_unrenderable_textures_lands_on_desktop_gl() {
        use crate::vrend::encode::encode;
        use crate::vrend::proto::Command;
        let _display = crate::vrend::one_display_at_a_time();
        struct Discard;
        impl crate::fence::FenceSink for Discard {
            fn context_fence(&mut self, _: ContextId, _: RingIdx, _: FenceId) {}
            fn present_fence(&mut self, _: FenceId) {}
            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        struct AllAttached;
        impl Guest for AllAttached {
            fn attached(&self, _: ContextId, _: ResourceHandle) -> bool {
                true
            }
            fn pages(&self, _: ContextId, _: ResourceHandle) -> Option<Iov<'_>> {
                None
            }
            fn blob_pixels(&self, _: ContextId, _: ResourceHandle) -> Option<PixelSource<'_>> {
                None
            }
        }
        let rgtc1 = (0..crate::vrend::proto::FORMAT_MAX)
            .filter_map(crate::vrend::proto::Format::from_wire)
            .find(|f| f.name() == "RGTC1_UNORM")
            .expect("RGTC1_UNORM is a wire format");
        let copy = |host_gl| {
            let retire = crate::fence::Retirement::start(
                Box::new(Discard),
                crate::vrend::debug::Switches::default(),
            );
            let mut v = Vrend::new(
                Config { host_gl, ..Config::default() },
                &crate::budget::Budget::with_cap(None, false),
                retire.handle(),
                None,
                crate::vrend::resource::Condemned::default(),
                crate::vrend::debug::Traces::default(),
                crate::vrend::debug::Switches::default(),
            )
            .expect("vrend comes up");
            // The copy-image road takes any same-format copy, so it is withdrawn: what is left is
            // a driver older than GL 4.3, which is where the C reaches its fallback.
            v.features.clear(Feature::copy_image);
            let ctx = ClassicCtx::for_test(ContextId::new(1).expect("a context id"));
            v.context_create(ctx, &AllAttached).expect("a context");
            let desc = rgtc1.describe().expect("described");
            let (w, h) = (8u32, 8u32);
            let stride = desc.stride(w) as u32;
            let layer = stride * desc.blocks_high(h);
            let block = desc.block_bytes() as usize;
            let info = transfer::Info {
                level: 0,
                stride,
                layer_stride: layer,
                offset: 0,
                region: crate::vrend::proto::Box3 {
                    x: 0,
                    y: 0,
                    z: 0,
                    width: w as i32,
                    height: h as i32,
                    depth: 1,
                },
                synchronized: false,
            };
            let mut src_bytes: Vec<u8> =
                (0..layer).map(|b| (b as u8).wrapping_mul(37) | 1).collect();
            let mut dst_bytes = vec![0u8; layer as usize];
            for (n, bytes) in [(1, &mut src_bytes), (2, &mut dst_bytes)] {
                let res = ResourceHandle::new(n).expect("non-zero");
                v.resource_create(
                    res,
                    resource::Args {
                        target: TextureTarget::Texture2d,
                        format: rgtc1,
                        bind: resource::Bind::SAMPLER_VIEW,
                        width: w,
                        height: h,
                        depth: 1,
                        array_size: 1,
                        last_level: 0,
                        nr_samples: 0,
                        flags: resource::ResourceFlags(0),
                    },
                )
                .expect("a texture");
                let from = [crate::abi::GuestIov {
                    base: crate::abi::VmmPtr(bytes.as_mut_ptr().cast()),
                    len: bytes.len(),
                }];
                let from = Iov::new(&from);
                v.transfer(None, res, Some(&from), transfer::Through::ToHost(from.source()), &info)
                    .expect("the upload");
            }
            let (src, dst) = (
                ResourceHandle::new(1).expect("non-zero"),
                ResourceHandle::new(2).expect("non-zero"),
            );
            let mut wire = Vec::new();
            encode(
                &Command::ResourceCopyRegion {
                    dst,
                    dst_level: 0,
                    dst_x: 0,
                    dst_y: 4,
                    dst_z: 0,
                    src,
                    src_level: 0,
                    src_region: crate::vrend::proto::Box3 {
                        x: 4,
                        y: 0,
                        z: 0,
                        width: 4,
                        height: 4,
                        depth: 1,
                    },
                },
                &mut wire,
            );
            let ran = v.submit(ctx, &wire, &AllAttached).expect("the context is here");
            if ran.is_err() {
                return None;
            }
            let mut read = vec![0xa5u8; layer as usize];
            let into = [crate::abi::GuestIov {
                base: crate::abi::VmmPtr(read.as_mut_ptr().cast()),
                len: read.len(),
            }];
            let into = Iov::new(&into);
            v.transfer(None, dst, None, transfer::Through::ToGuest(&into), &info)
                .expect("the destination reads back");
            let mut want = vec![0u8; layer as usize];
            // Source block (1, 0) is bytes [block, 2 * block); destination block (0, 1) starts a
            // row of blocks down, at `stride`.
            let row = stride as usize;
            want[row..row + block].copy_from_slice(&src_bytes[block..2 * block]);
            Some(read == want)
        };
        assert_eq!(copy(HostGl::Gles), None, "GLES has no pages here to copy RGTC1 through");
        assert_eq!(copy(HostGl::Desktop), Some(true), "desktop GL copies the one block");
    }

    /// A copy between two textures no framebuffer can hold, on a driver without
    /// `glCopyImageSubData`, lands on either flavour: desktop GL reads the source back, and GLES,
    /// which cannot, copies through the guest's pages, which the guest keeps whole behind exactly
    /// such a texture. The copy is a box out of the source's base level into the destination's
    /// second level at another place, so a copy that took either level's offset, pitch or origin
    /// from the wrong side lands the wrong texels. What is scored is the destination sampled in a
    /// draw, against the same draw of the copy's result uploaded directly -- and, on GLES, the
    /// destination's pages, which a later copy out of it reads.
    #[test]
    fn a_copy_between_unrenderable_textures_samples_as_the_copied_image() {
        use crate::vrend::pipe::TransferDirection;
        use crate::vrend::pipe::{CompareFunc, MipFilter, Swizzle, TexFilter, TexWrap};
        use crate::vrend::proto::{Box3, Command, Object, SamplerState, SamplerView, Transfer};
        const SAMPLE_FS: &str = "FRAG\nDCL IN[0], POSITION, LINEAR\nDCL OUT[0], COLOR\n\
                                 DCL SAMP[0]\nDCL SVIEW[0], 2D, FLOAT\nDCL TEMP[0]\n\
                                 IMM[0] FLT32 { 0.0625, 0.0625, 0.0, 0.0 }\n  \
                                 0: MUL TEMP[0], IN[0], IMM[0]\n  \
                                 1: TEX OUT[0], TEMP[0], SAMP[0], 2D\n  2: END\n";
        let rgb9e5 = (0..crate::vrend::proto::FORMAT_MAX)
            .filter_map(crate::vrend::proto::Format::from_wire)
            .find(|f| f.name() == "R9G9B9E5_FLOAT")
            .expect("R9G9B9E5_FLOAT is a wire format");
        // Mantissas over an exponent of 16 are m / 256: exact, and distinct per texel.
        let texel = |r: u32, g: u32, b: u32| (r | g << 9 | b << 18 | 16 << 27).to_le_bytes();
        // Level 0 is 16x16 and level 1 8x8, four bytes a texel, tight and in order.
        let (base, level1) = (16 * 16 * 4, 8 * 8 * 4);
        let mut src = vec![0u8; base + level1];
        for (n, px) in src.chunks_mut(4).enumerate() {
            let (i, j) = (n as u32 % 16, n as u32 / 16);
            px.copy_from_slice(&texel(32 * (i % 8), 32 * (j % 8), 32 * ((i + j) % 8)));
        }
        let mut dst = vec![0u8; base + level1];
        for px in dst.chunks_mut(4) {
            px.copy_from_slice(&texel(0, 0, 255));
        }
        // Source texels (4..8, 2..6) of level 0 land at (2..6, 4..8) of level 1.
        let mut copied = dst.clone();
        for j in 0..4 {
            let from = ((2 + j) * 16 + 4) * 4;
            let to = base + ((4 + j) * 8 + 2) * 4;
            copied[to..to + 16].copy_from_slice(&src[from..from + 16]);
        }
        let draw = |host_gl, copy: bool| {
            let (s, d) = (
                ResourceHandle::new(10).expect("non-zero"),
                ResourceHandle::new(11).expect("non-zero"),
            );
            let src_pages = std::cell::RefCell::new(src.clone());
            let dst_pages =
                std::cell::RefCell::new(if copy { dst.clone() } else { copied.clone() });
            let args = resource::Args {
                target: TextureTarget::Texture2d,
                format: rgb9e5,
                bind: resource::Bind::SAMPLER_VIEW,
                width: 16,
                height: 16,
                depth: 1,
                array_size: 1,
                last_level: 1,
                nr_samples: 0,
                flags: resource::ResourceFlags(0),
            };
            let upload = |resource, level: u32, size: i32, offset: u32| Command::Transfer3d {
                transfer: Transfer {
                    resource,
                    level,
                    usage: 0,
                    stride: 0,
                    layer_stride: 0,
                    region: Box3 { x: 0, y: 0, z: 0, width: size, height: size, depth: 1 },
                },
                offset,
                direction: TransferDirection::ToHost,
            };
            let o = |n: u32| ObjectHandle::new(n).expect("non-zero");
            let mut before = vec![
                upload(s, 0, 16, 0),
                upload(s, 1, 8, base as u32),
                upload(d, 0, 16, 0),
                upload(d, 1, 8, base as u32),
            ];
            if copy {
                before.push(Command::ResourceCopyRegion {
                    dst: d,
                    dst_level: 1,
                    dst_x: 2,
                    dst_y: 4,
                    dst_z: 0,
                    src: s,
                    src_level: 0,
                    src_region: Box3 { x: 4, y: 2, z: 0, width: 4, height: 4, depth: 1 },
                });
            }
            before.extend([
                Command::CreateObject {
                    handle: o(20),
                    object: Object::SamplerView(SamplerView {
                        resource: d,
                        format: rgb9e5,
                        target: TextureTarget::Texture2d,
                        first_element_or_layers: 0,
                        // Level 1 alone.
                        last_element_or_levels: 1 | 1 << 8,
                        swizzle: [Swizzle::X, Swizzle::Y, Swizzle::Z, Swizzle::W],
                    }),
                },
                Command::CreateObject {
                    handle: o(21),
                    object: Object::SamplerState(SamplerState {
                        wrap_s: TexWrap::ClampToEdge,
                        wrap_t: TexWrap::ClampToEdge,
                        wrap_r: TexWrap::ClampToEdge,
                        min_img_filter: TexFilter::Nearest,
                        min_mip_filter: MipFilter::None,
                        mag_img_filter: TexFilter::Nearest,
                        compare_mode: false,
                        compare_func: CompareFunc::LessEqual,
                        seamless_cube_map: false,
                        max_anisotropy: 0,
                        lod_bias: 0.0,
                        min_lod: 0.0,
                        max_lod: 0.0,
                        border_color: [0; 4],
                    }),
                },
                Command::SetSamplerViews {
                    stage: ShaderStage::Fragment,
                    start_slot: 0,
                    views: vec![Some(o(20))],
                },
                Command::BindSamplerStates {
                    stage: ShaderStage::Fragment,
                    start_slot: 0,
                    states: vec![Some(o(21))],
                },
            ]);
            let more = More {
                resources: vec![(s, args), (d, args)],
                before,
                pages: vec![(s, &src_pages), (d, &dst_pages)],
                // The copy-image road takes any same-format copy, so it is withdrawn: what is
                // left is a driver older than GL 4.3 or GLES 3.2, which is where the C reaches
                // its fallback.
                withdrawn: vec![Feature::copy_image],
                ..More::default()
            };
            let pixels = draw_over_target(OneDraw {
                host_gl,
                format: "R8G8B8A8_UNORM",
                clear: [0.0; 4],
                fs: Some(SAMPLE_FS),
                vs: None,
                consts: &[],
                pipeline: None,
                more: Some(&more),
                logicop: None,
                tess: None,
            })
            .expect("the draw runs")
            .expect("no tessellation asked for");
            (pixels, dst_pages.into_inner())
        };
        for host_gl in [HostGl::Gles, HostGl::Desktop] {
            let (direct, _) = draw(host_gl, false);
            let (through_copy, pages) = draw(host_gl, true);
            assert!(
                direct.chunks(4).any(|p| p != &direct[..4]),
                "{host_gl:?}: the premise: the copied box shows in the sampled level"
            );
            assert!(direct == through_copy, "{host_gl:?}: the copy samples as the image it made");
            if host_gl == HostGl::Gles {
                assert!(pages == copied, "GLES: the destination's pages hold the copy");
            }
        }
    }

    /// Desktop GL stores a 1D texture as one, and a 1D array as a 2D texture whose rows are its
    /// layers; GLES has no 1D textures and stores both as 2D. Either way a box written to every
    /// layer reads back as written -- a layer uploaded as the wrong row, or as a 2D image of a
    /// target that is not one, does not.
    #[test]
    fn a_1d_texture_is_one_on_desktop_gl_and_round_trips_on_both() {
        let _display = crate::vrend::one_display_at_a_time();
        struct Discard;
        impl crate::fence::FenceSink for Discard {
            fn context_fence(&mut self, _: ContextId, _: RingIdx, _: FenceId) {}
            fn present_fence(&mut self, _: FenceId) {}
            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        let format = (0..crate::vrend::proto::FORMAT_MAX)
            .filter_map(crate::vrend::proto::Format::from_wire)
            .find(|f| f.name() == "R8G8B8A8_UNORM")
            .expect("a wire format by that name");
        let round_trip = |host_gl, target, layers: u32| {
            let retire = crate::fence::Retirement::start(
                Box::new(Discard),
                crate::vrend::debug::Switches::default(),
            );
            let mut v = Vrend::new(
                Config { host_gl, ..Config::default() },
                &crate::budget::Budget::with_cap(None, false),
                retire.handle(),
                None,
                crate::vrend::resource::Condemned::default(),
                crate::vrend::debug::Traces::default(),
                crate::vrend::debug::Switches::default(),
            )
            .expect("vrend comes up");
            let w = 8u32;
            let res = ResourceHandle::new(1).expect("non-zero");
            v.resource_create(
                res,
                resource::Args {
                    target,
                    format,
                    bind: resource::Bind(
                        resource::Bind::SAMPLER_VIEW.0 | resource::Bind::RENDER_TARGET.0,
                    ),
                    width: w,
                    height: 1,
                    depth: 1,
                    array_size: layers,
                    last_level: 0,
                    nr_samples: 0,
                    flags: resource::ResourceFlags(0),
                },
            )
            .expect("a texture");
            let gl_target = v
                .resources
                .sync()
                .get(&res)
                .and_then(|e| e.resource())
                .and_then(|r| r.texture())
                .map(|t| t.target)
                .expect("a texture's storage");
            let stride = w * 4;
            let info = transfer::Info {
                level: 0,
                stride,
                layer_stride: stride,
                offset: 0,
                region: crate::vrend::proto::Box3 {
                    x: 0,
                    y: 0,
                    z: 0,
                    width: w as i32,
                    height: 1,
                    depth: layers as i32,
                },
                synchronized: false,
            };
            let n = (stride * layers) as usize;
            let mut written: Vec<u8> = (0..n).map(|b| (b as u8).wrapping_mul(37)).collect();
            let from = [crate::abi::GuestIov {
                base: crate::abi::VmmPtr(written.as_mut_ptr().cast()),
                len: written.len(),
            }];
            let from = Iov::new(&from);
            v.transfer(None, res, Some(&from), transfer::Through::ToHost(from.source()), &info)
                .expect("the upload");
            let mut read = vec![0xa5u8; n];
            let into = [crate::abi::GuestIov {
                base: crate::abi::VmmPtr(read.as_mut_ptr().cast()),
                len: read.len(),
            }];
            let into = Iov::new(&into);
            v.transfer(None, res, Some(&from), transfer::Through::ToGuest(&into), &info)
                .expect("the readback");
            (gl_target, read == written)
        };
        use crate::vrend::gl::gles::{
            GL_TEXTURE_1D, GL_TEXTURE_1D_ARRAY, GL_TEXTURE_2D, GL_TEXTURE_2D_ARRAY,
        };
        use TextureTarget::{Array1d, Texture1d};
        assert_eq!(round_trip(HostGl::Gles, Texture1d, 1), (GL_TEXTURE_2D, true));
        assert_eq!(round_trip(HostGl::Gles, Array1d, 3), (GL_TEXTURE_2D_ARRAY, true));
        assert_eq!(round_trip(HostGl::Desktop, Texture1d, 1), (GL_TEXTURE_1D, true));
        assert_eq!(round_trip(HostGl::Desktop, Array1d, 3), (GL_TEXTURE_1D_ARRAY, true));
    }

    /// A desktop context is refused unless desktop GL was asked for, and then only a core profile
    /// at 3.3 or later. QEMU hands one over by default, and until the desktop leg is whole that
    /// must stay a refusal it can report, not a renderer running GLES code on desktop GL.
    #[test]
    fn a_desktop_context_is_taken_only_when_asked_for() {
        let core = || GL_CONTEXT_CORE_PROFILE_BIT;
        let compat = || 0;
        let es = "OpenGL ES 3.2 Mesa 26.0.0";
        let desk = "4.6 (Core Profile) Mesa 26.0.0";
        assert_eq!(host_api(es, core, HostGl::Gles), Ok(Api::Gles(32)));
        assert_eq!(host_api(es, core, HostGl::Desktop), Ok(Api::Gles(32)), "what arrived rules");
        assert_eq!(host_api(desk, core, HostGl::Gles), Err(UnservedGl::NotAskedFor));
        assert_eq!(host_api(desk, core, HostGl::Desktop), Ok(Api::Gl(46)));
        assert_eq!(host_api(desk, compat, HostGl::Desktop), Err(UnservedGl::Compatibility));
        assert_eq!(host_api("3.2 Mesa", core, HostGl::Desktop), Err(UnservedGl::TooOld));
    }

    #[test]
    fn the_version_string_parses_the_way_epoxy_reads_it() {
        assert_eq!(parse_version("OpenGL ES 3.1 Mesa 26.0.0"), 31);
        assert_eq!(parse_version("OpenGL ES 3.2 Mesa 26.0.0-devel (git-abc)"), 32);
        assert_eq!(parse_version("4.6 (Core Profile) Mesa 25.2.0"), 46);
        assert_eq!(parse_version("4.6.0 NVIDIA 550.54"), 46);
        assert_eq!(parse_version(""), 0);
    }

    /// A journal reaches a classic context only between `replay_begin` and `replay_end`. Handed
    /// to a live context, it would be replayed over what the guest has built since.
    #[test]
    fn a_journal_fed_to_a_live_classic_context_is_refused() {
        use crate::vrend::context::NOT_REPLAYING;
        let _display = crate::vrend::one_display_at_a_time();
        struct Discard;
        impl crate::fence::FenceSink for Discard {
            fn context_fence(&mut self, _: ContextId, _: RingIdx, _: FenceId) {}
            fn present_fence(&mut self, _: FenceId) {}

            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        let retire = crate::fence::Retirement::start(
            Box::new(Discard),
            crate::vrend::debug::Switches::default(),
        );
        let mut v = Vrend::new(
            Config::default(),
            &crate::budget::Budget::with_cap(None, false),
            retire.handle(),
            None,
            crate::vrend::resource::Condemned::default(),
            crate::vrend::debug::Traces::default(),
            crate::vrend::debug::Switches::default(),
        )
        .expect("vrend comes up");
        let ctx = ClassicCtx::for_test(ContextId::new(1).expect("a context id"));
        v.context_create(ctx, &NoGuest).expect("a context");
        // `CREATE_SUB_CTX 1`: the smallest thing a journal retains, so this one is not empty.
        v.submit(ctx, &[(1 << 16) | 29, 1], &NoGuest)
            .expect("the context is here")
            .expect("a sub-context");
        let journal = v.journal_export(ctx).expect("a live context exports its journal");
        let entries = |bytes: &[u8]| crate::vrend::journal::parse(bytes).map(|e| e.len());
        let retained = entries(&journal).expect("this renderer reads its own journal");
        assert!(retained > 0, "a context with a sub-context retains something");

        assert_eq!(
            v.journal_restore(ctx, &journal),
            Err(ReplayRefused::JournalRefused(NOT_REPLAYING)),
            "handed outside a replay span, the journal is refused"
        );
        assert_eq!(
            v.replay_upto(ctx, &NoGuest, Seq(u64::MAX)),
            Err(ReplayRefused::NotReplaying),
            "and nothing is fed"
        );

        v.context_destroy(ctx, &NoGuest);
        v.context_create(ctx, &NoGuest).expect("a fresh context to rebuild");
        v.replay_begin(ctx).expect("the context is here");
        assert_eq!(
            v.journal_restore(ctx, &journal),
            Ok(retained),
            "inside the span the same journal is taken"
        );
        v.replay_upto(ctx, &NoGuest, Seq(u64::MAX)).expect("and fed");
        v.replay_end(ctx).expect("the context is here");
        let rebuilt = v.journal_export(ctx).expect("the rebuilt context exports its journal");
        assert_eq!(
            entries(&rebuilt),
            Ok(retained),
            "what was fed is what the rebuilt context retains"
        );
        v.context_destroy(ctx, &NoGuest);
    }

    /// A retained command this renderer cannot frame back poisons the rebuilt context, and the
    /// feed says so and stops, rather than reporting a clean rebuild of a context that will refuse
    /// everything the guest sends next.
    #[test]
    fn a_replay_that_poisons_its_context_stops_and_says_so() {
        use crate::vrend::journal::{Entry, Step, serialize};
        use std::borrow::Cow;
        let _display = crate::vrend::one_display_at_a_time();
        struct Discard;
        impl crate::fence::FenceSink for Discard {
            fn context_fence(&mut self, _: ContextId, _: RingIdx, _: FenceId) {}
            fn present_fence(&mut self, _: FenceId) {}

            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        let retire = crate::fence::Retirement::start(
            Box::new(Discard),
            crate::vrend::debug::Switches::default(),
        );
        let mut v = Vrend::new(
            Config::default(),
            &crate::budget::Budget::with_cap(None, false),
            retire.handle(),
            None,
            crate::vrend::resource::Condemned::default(),
            crate::vrend::debug::Traces::default(),
            crate::vrend::debug::Switches::default(),
        )
        .expect("vrend comes up");
        let ctx = ClassicCtx::for_test(ContextId::new(1).expect("a context id"));
        v.context_create(ctx, &NoGuest).expect("a context");

        // A header promising nine dwords with none behind it: nothing frames it. Then a
        // `CREATE_SUB_CTX` that would be fine on its own, which the feed must not reach.
        let torn = vec![vec![crate::vrend::proto::Cmd::CreateObject as u32 | 9 << 16]];
        let fine = vec![vec![(1 << 16) | 29, 1]];
        let journal = serialize(&[
            Entry { seq: Seq(1), step: Step::Feed { sub: 0, chunks: Cow::Borrowed(&torn) } },
            Entry { seq: Seq(2), step: Step::Feed { sub: 0, chunks: Cow::Borrowed(&fine) } },
        ]);
        v.replay_begin(ctx).expect("the context is here");
        assert_eq!(v.journal_restore(ctx, &journal), Ok(2));
        assert_eq!(
            v.replay_upto(ctx, &NoGuest, Seq(u64::MAX)),
            Err(ReplayRefused::Poisoned),
            "the rebuild reports the context it poisoned"
        );
        v.replay_end(ctx).expect("the context is here");
        v.context_destroy(ctx, &NoGuest);
    }

    /// A command a rebuild drops leaves nothing behind for the guest's first live submit to
    /// answer for: not the fault, and not a GL error it left in the context's GL queue. Either
    /// would refuse every batch the guest sends after the restore.
    #[test]
    fn a_command_a_replay_drops_does_not_refuse_the_next_live_submit() {
        use crate::vrend::encode::encode;
        use crate::vrend::journal::{Entry, Step, serialize};
        use crate::vrend::pipe::Swizzle;
        use crate::vrend::proto::{Command, Object, ObjectType, SamplerView};
        use std::borrow::Cow;
        let _display = crate::vrend::one_display_at_a_time();
        struct Discard;
        impl crate::fence::FenceSink for Discard {
            fn context_fence(&mut self, _: ContextId, _: RingIdx, _: FenceId) {}
            fn present_fence(&mut self, _: FenceId) {}

            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        let retire = crate::fence::Retirement::start(
            Box::new(Discard),
            crate::vrend::debug::Switches::default(),
        );
        let mut v = Vrend::new(
            Config::default(),
            &crate::budget::Budget::with_cap(None, false),
            retire.handle(),
            None,
            crate::vrend::resource::Condemned::default(),
            crate::vrend::debug::Traces::default(),
            crate::vrend::debug::Switches::default(),
        )
        .expect("vrend comes up");
        let ctx = ClassicCtx::for_test(ContextId::new(1).expect("a context id"));
        v.context_create(ctx, &NoGuest).expect("a context");

        let o = |n: u32| ObjectHandle::new(n).expect("non-zero");
        let wire = |c: Command<'_>| {
            let mut w = Vec::new();
            encode(&c, &mut w);
            vec![w]
        };
        // Destroying a handle nothing holds runs, and touches nothing.
        let harmless = || Command::DestroyObject { kind: ObjectType::SamplerView, handle: o(9) };
        // A view of a resource that is gone, as a journal entry whose referent died is.
        let stale = Command::CreateObject {
            handle: o(5),
            object: Object::SamplerView(SamplerView {
                resource: ResourceHandle::new(28).expect("non-zero"),
                format: super::super::proto::Format::from_wire(1).expect("B8G8R8A8_UNORM"),
                target: TextureTarget::Texture2d,
                first_element_or_layers: 0,
                last_element_or_levels: 0,
                swizzle: [Swizzle::X, Swizzle::Y, Swizzle::Z, Swizzle::W],
            }),
        };
        let (first, second) = (wire(harmless()), wire(stale));
        let journal = serialize(&[
            Entry { seq: Seq(1), step: Step::Feed { sub: 0, chunks: Cow::Borrowed(&first) } },
            Entry { seq: Seq(2), step: Step::Feed { sub: 0, chunks: Cow::Borrowed(&second) } },
        ]);
        v.replay_begin(ctx).expect("the context is here");
        assert_eq!(v.journal_restore(ctx, &journal), Ok(2));
        v.replay_upto(ctx, &NoGuest, Seq(1)).expect("the first entry is fed");
        // What the stale command would leave had it got as far as GL before its fault: an error in
        // the queue of the GL context it ran in, which the first entry left current.
        v.gl.bind_texture(0xDEAD, None);
        v.replay_upto(ctx, &NoGuest, Seq(u64::MAX)).expect("a dropped command does not poison");
        let dropped = v.contexts.get_mut(&ctx.id()).expect("the context").replay_end();
        assert_eq!(dropped, 1, "the stale view is dropped");

        let live = wire(harmless()).concat();
        assert_eq!(
            v.submit(ctx, &live, &NoGuest).expect("the context is here"),
            Ok(()),
            "the guest's first batch after the restore is accepted"
        );
        v.context_destroy(ctx, &NoGuest);
    }

    /// A second `replay_begin` keeps the span open with the journal the first was handed,
    /// rather than starting over with an empty one.
    #[test]
    fn a_classic_replay_begun_twice_keeps_the_journal_it_was_handed() {
        let _display = crate::vrend::one_display_at_a_time();
        struct Discard;
        impl crate::fence::FenceSink for Discard {
            fn context_fence(&mut self, _: ContextId, _: RingIdx, _: FenceId) {}
            fn present_fence(&mut self, _: FenceId) {}

            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        let retire = crate::fence::Retirement::start(
            Box::new(Discard),
            crate::vrend::debug::Switches::default(),
        );
        let mut v = Vrend::new(
            Config::default(),
            &crate::budget::Budget::with_cap(None, false),
            retire.handle(),
            None,
            crate::vrend::resource::Condemned::default(),
            crate::vrend::debug::Traces::default(),
            crate::vrend::debug::Switches::default(),
        )
        .expect("vrend comes up");
        let ctx = ClassicCtx::for_test(ContextId::new(1).expect("a context id"));
        v.context_create(ctx, &NoGuest).expect("a context");
        v.submit(ctx, &[(1 << 16) | 29, 1], &NoGuest)
            .expect("the context is here")
            .expect("a sub-context");
        let journal = v.journal_export(ctx).expect("a live context exports its journal");
        let entries = |bytes: &[u8]| crate::vrend::journal::parse(bytes).map(|e| e.len());
        let retained = entries(&journal).expect("this renderer reads its own journal");
        assert!(retained > 0, "a context with a sub-context retains something");
        v.context_destroy(ctx, &NoGuest);

        v.context_create(ctx, &NoGuest).expect("a fresh context to rebuild");
        v.replay_begin(ctx).expect("the context is here");
        assert_eq!(v.journal_restore(ctx, &journal), Ok(retained));
        v.replay_begin(ctx).expect("the context is here");
        v.replay_upto(ctx, &NoGuest, Seq(u64::MAX)).expect("the context is replaying");
        v.replay_end(ctx).expect("the context is here");
        let rebuilt = v.journal_export(ctx).expect("the rebuilt context exports its journal");
        assert_eq!(
            entries(&rebuilt),
            Ok(retained),
            "the journal the first replay_begin was handed is what the rebuild fed"
        );
        v.context_destroy(ctx, &NoGuest);
    }

    /// A sampler view or sampler state destroyed while bound leaves its slot live, and its handle
    /// is free for the guest's next create. A rebuild must bind what the slots hold, not what the
    /// last command named: replaying the command binds nothing past the dead view, and binds a
    /// reused handle into a slot that was empty.
    #[test]
    fn a_rebuild_binds_what_the_units_hold_not_what_was_last_sent() {
        use crate::vrend::encode::encode;
        use crate::vrend::pipe::{
            CompareFunc, MipFilter, ShaderStage, Swizzle, TexFilter, TexWrap,
        };
        use crate::vrend::proto::{Command, Object, ObjectType, SamplerState, SamplerView};
        let _display = crate::vrend::one_display_at_a_time();
        struct Discard;
        impl crate::fence::FenceSink for Discard {
            fn context_fence(&mut self, _: ContextId, _: RingIdx, _: FenceId) {}
            fn present_fence(&mut self, _: FenceId) {}

            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        /// Every resource reachable from every context, as a VMM that attached them would say.
        struct AllAttached;
        impl Guest for AllAttached {
            fn attached(&self, _: ContextId, _: ResourceHandle) -> bool {
                true
            }
            fn pages(&self, _: ContextId, _: ResourceHandle) -> Option<Iov<'_>> {
                None
            }
            fn blob_pixels(&self, _: ContextId, _: ResourceHandle) -> Option<PixelSource<'_>> {
                None
            }
        }
        let retire = crate::fence::Retirement::start(
            Box::new(Discard),
            crate::vrend::debug::Switches::default(),
        );
        let mut v = Vrend::new(
            Config::default(),
            &crate::budget::Budget::with_cap(None, false),
            retire.handle(),
            None,
            crate::vrend::resource::Condemned::default(),
            crate::vrend::debug::Traces::default(),
            crate::vrend::debug::Switches::default(),
        )
        .expect("vrend comes up");
        let bgra = super::super::proto::Format::from_wire(1).expect("B8G8R8A8_UNORM");
        let res = ResourceHandle::new(1).expect("a resource handle is non-zero");
        v.resource_create(
            res,
            resource::Args {
                target: TextureTarget::Texture2d,
                format: bgra,
                bind: resource::Bind(1 << 3),
                width: 16,
                height: 16,
                depth: 1,
                array_size: 1,
                last_level: 0,
                nr_samples: 0,
                flags: resource::ResourceFlags(0),
            },
        )
        .expect("a texture");
        let o = |n: u32| crate::vrend::proto::ObjectHandle::new(n).expect("non-zero");
        let view = |swizzle| {
            Object::SamplerView(SamplerView {
                resource: res,
                format: bgra,
                target: TextureTarget::Texture2d,
                first_element_or_layers: 0,
                last_element_or_levels: 0,
                swizzle,
            })
        };
        let state = |wrap| {
            Object::SamplerState(SamplerState {
                wrap_s: wrap,
                wrap_t: wrap,
                wrap_r: wrap,
                min_img_filter: TexFilter::Linear,
                min_mip_filter: MipFilter::None,
                mag_img_filter: TexFilter::Linear,
                compare_mode: false,
                compare_func: CompareFunc::LessEqual,
                seamless_cube_map: false,
                max_anisotropy: 0,
                lod_bias: 0.0,
                min_lod: 0.0,
                max_lod: 0.0,
                border_color: [0; 4],
            })
        };
        let rgba = [Swizzle::X, Swizzle::Y, Swizzle::Z, Swizzle::W];
        let stage = ShaderStage::Fragment;
        let mut wire = Vec::new();
        for c in [
            Command::CreateObject { handle: o(5), object: view(rgba) },
            Command::CreateObject { handle: o(6), object: view(rgba) },
            Command::SetSamplerViews { stage, start_slot: 0, views: vec![Some(o(5)), Some(o(6))] },
            Command::CreateObject { handle: o(7), object: state(TexWrap::Repeat) },
            Command::CreateObject { handle: o(8), object: state(TexWrap::ClampToEdge) },
            Command::BindSamplerStates {
                stage,
                start_slot: 0,
                states: vec![Some(o(7)), Some(o(8))],
            },
            // Destroyed while bound, then the handles reused for objects nothing binds.
            Command::DestroyObject { kind: ObjectType::SamplerView, handle: o(5) },
            Command::DestroyObject { kind: ObjectType::SamplerState, handle: o(7) },
            Command::CreateObject {
                handle: o(5),
                object: view([Swizzle::Z, Swizzle::Y, Swizzle::X, Swizzle::W]),
            },
            Command::CreateObject { handle: o(7), object: state(TexWrap::MirrorRepeat) },
        ] {
            encode(&c, &mut wire);
        }
        let ctx = ClassicCtx::for_test(ContextId::new(1).expect("a context id"));
        v.context_create(ctx, &AllAttached).expect("a context");
        v.submit(ctx, &wire, &AllAttached).expect("the context is here").expect("accepted");
        let live = v.contexts[&ctx.id()].bound_units(stage);
        assert_eq!(
            live,
            (vec![(1, o(6))], vec![(0, o(8))]),
            "the dead view left its slot; the dead state's slot closed up"
        );

        let journal = v.journal_export(ctx).expect("a live context exports its journal");
        v.context_destroy(ctx, &AllAttached);
        v.context_create(ctx, &AllAttached).expect("a fresh context to rebuild");
        v.replay_begin(ctx).expect("the context is here");
        v.journal_restore(ctx, &journal).expect("the journal is taken");
        v.replay_upto(ctx, &AllAttached, Seq(u64::MAX)).expect("and fed");
        let dropped = v.contexts.get_mut(&ctx.id()).expect("the context").replay_end();
        assert_eq!(dropped, 0, "a rebuild of what the context holds drops nothing");
        assert_eq!(v.contexts[&ctx.id()].bound_units(stage), live, "and binds what it held");
        v.context_destroy(ctx, &AllAttached);
    }

    /// A surface with samples its texture lacks is the guest's
    /// `glFramebufferTexture2DMultisampleEXT`, which it is offered whenever the host has the
    /// extension. Such a host renders into it and resolves into the texture; refusing it there
    /// poisoned the first context that took the offer. A host without the extension never made
    /// it, and says which feature a guest asking anyway lacks.
    #[test]
    fn a_multisampled_surface_renders_into_its_texture_where_the_host_offers_it() {
        use crate::vrend::encode::encode;
        use crate::vrend::proto::{Command, Object, Surface};
        let _display = crate::vrend::one_display_at_a_time();
        struct Discard;
        impl crate::fence::FenceSink for Discard {
            fn context_fence(&mut self, _: ContextId, _: RingIdx, _: FenceId) {}
            fn present_fence(&mut self, _: FenceId) {}

            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        struct AllAttached;
        impl Guest for AllAttached {
            fn attached(&self, _: ContextId, _: ResourceHandle) -> bool {
                true
            }
            fn pages(&self, _: ContextId, _: ResourceHandle) -> Option<Iov<'_>> {
                None
            }
            fn blob_pixels(&self, _: ContextId, _: ResourceHandle) -> Option<PixelSource<'_>> {
                None
            }
        }
        let retire = crate::fence::Retirement::start(
            Box::new(Discard),
            crate::vrend::debug::Switches::default(),
        );
        let mut v = Vrend::new(
            Config::default(),
            &crate::budget::Budget::with_cap(None, false),
            retire.handle(),
            None,
            crate::vrend::resource::Condemned::default(),
            crate::vrend::debug::Traces::default(),
            crate::vrend::debug::Switches::default(),
        )
        .expect("vrend comes up");
        let offered = v.features.has(Feature::implicit_msaa);
        let bgra = super::super::proto::Format::from_wire(1).expect("B8G8R8A8_UNORM");
        let res = ResourceHandle::new(1).expect("a resource handle is non-zero");
        v.resource_create(
            res,
            resource::Args {
                target: TextureTarget::Texture2d,
                format: bgra,
                bind: resource::Bind(
                    resource::Bind::RENDER_TARGET.0 | resource::Bind::SAMPLER_VIEW.0,
                ),
                width: 16,
                height: 16,
                depth: 1,
                array_size: 1,
                last_level: 0,
                nr_samples: 0,
                flags: resource::ResourceFlags(0),
            },
        )
        .expect("a texture");
        let o = |n: u32| crate::vrend::proto::ObjectHandle::new(n).expect("non-zero");
        let red = [1.0f32.to_bits(), 0, 0, 1.0f32.to_bits()];
        let mut wire = Vec::new();
        for c in [
            Command::CreateObject {
                handle: o(1),
                object: Object::Surface(Surface {
                    resource: res,
                    format: bgra,
                    first_element_or_level: 0,
                    last_element_or_layers: 0,
                    samples: 4,
                }),
            },
            Command::SetFramebufferState { zsurf: None, cbufs: vec![Some(o(1))] },
            Command::Clear { buffers: 1 << 2, color: red, depth: 0.0, stencil: 0 },
        ] {
            encode(&c, &mut wire);
        }
        let ctx = ClassicCtx::for_test(ContextId::new(1).expect("a context id"));
        v.context_create(ctx, &AllAttached).expect("a context");
        let ran = v.submit(ctx, &wire, &AllAttached).expect("the context is here");
        if offered {
            ran.expect("a host that offers the extension renders into the surface");
            let cursor = v.cursor_contents(res).expect("a 16x16 2D texture reads back");
            // Red in B8G8R8A8's byte order: blue, green, red, alpha.
            assert!(
                cursor.pixels.as_chunks::<4>().0.iter().all(|p| *p == [0, 0, 0xff, 0xff]),
                "the clear resolved into the texture: {:?}",
                &cursor.pixels[..4]
            );
        } else {
            assert!(
                matches!(ran, Err(Fault::NoFeature { feature: Feature::implicit_msaa, .. })),
                "a host without the extension names it: {ran:?}"
            );
        }
        v.context_destroy(ctx, &AllAttached);
    }

    /// What [`draw_over_target`] draws: a clear of a 16x16 target to `clear`, then -- when there
    /// is a fragment shader -- one triangle covering it, through the evaluation stage when `tess`
    /// names the levels `SET_TESS_STATE` sends. `vs` replaces the pass-through vertex shader,
    /// `consts` sets a stage's first constant, and `pipeline`, when given, is told whether the
    /// host serves separable stages and whether the draw ran a program pipeline.
    struct OneDraw<'a> {
        host_gl: HostGl,
        format: &'a str,
        clear: [f32; 4],
        vs: Option<&'a str>,
        fs: Option<&'a str>,
        consts: &'a [(ShaderStage, [f32; 4])],
        logicop: Option<crate::vrend::pipe::LogicOp>,
        tess: Option<f32>,
        pipeline: Option<&'a std::cell::Cell<Option<PipelineSeen>>>,
        more: Option<&'a More<'a>>,
    }

    /// What a draw needs beyond one vertex buffer and one target: resources made before the
    /// stream, commands sent ahead of the draw, and further rounds of commands each followed by
    /// the same draw again. The target read back is the last draw's.
    ///
    /// `indirect` draws every time through an indirect buffer instead; `after` is sent once the
    /// last draw is; and each of `read_back` -- a buffer and a length -- is read into `read` once
    /// everything ran and a fence has answered any query the stream left waiting. `pages` are the
    /// guest's pages behind a resource, held for the whole run, and `withdrawn` the features the
    /// renderer is made to lack. `stream_output` is the vertex shader's, and `draw` replaces the
    /// one triangle drawn. `texel_limit` lowers the renderer's texture-buffer texel limit, which
    /// GL's own is far past anything a test wants to allocate.
    #[derive(Default)]
    struct More<'a> {
        stream_output: Option<crate::vrend::proto::StreamOutput>,
        draw: Option<crate::vrend::proto::Draw>,
        pages: Vec<(ResourceHandle, &'a std::cell::RefCell<Vec<u8>>)>,
        withdrawn: Vec<Feature>,
        resources: Vec<(ResourceHandle, resource::Args)>,
        before: Vec<crate::vrend::proto::Command<'a>>,
        redraws: Vec<Vec<crate::vrend::proto::Command<'a>>>,
        indirect: Option<crate::vrend::proto::IndirectDraw>,
        after: Vec<crate::vrend::proto::Command<'a>>,
        read_back: Vec<(ResourceHandle, u32)>,
        read: Option<&'a std::cell::RefCell<Vec<Vec<u8>>>>,
        texel_limit: Option<u32>,
    }

    /// Draw `d` into a fresh target and read the target back. `Ok(None)` is a host without
    /// tessellation asked to tessellate, which refuses the evaluation shader as it never told the
    /// guest it could.
    fn draw_over_target(d: OneDraw<'_>) -> Result<Option<Vec<u8>>, Fault> {
        use crate::vrend::encode::encode;
        use crate::vrend::pipe::{LogicOp, PrimType};
        use crate::vrend::proto::{
            BlendState, Box3, Command, Draw, Object, ObjectType, RtBlend, ShaderChunk,
            ShaderCreate, ShaderKind, StreamOutput, Surface, TessDraw, Transfer, VertexBuffer,
            VertexElement, Viewport,
        };
        let _display = crate::vrend::one_display_at_a_time();
        struct Discard;
        impl crate::fence::FenceSink for Discard {
            fn context_fence(&mut self, _: ContextId, _: RingIdx, _: FenceId) {}
            fn present_fence(&mut self, _: FenceId) {}

            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        struct AllAttached(Vec<(ResourceHandle, [crate::abi::GuestIov; 1])>);
        impl Guest for AllAttached {
            fn attached(&self, _: ContextId, _: ResourceHandle) -> bool {
                true
            }
            fn pages(&self, _: ContextId, handle: ResourceHandle) -> Option<Iov<'_>> {
                self.0.iter().find(|(h, _)| *h == handle).map(|(_, iov)| Iov::new(iov))
            }
            fn blob_pixels(&self, _: ContextId, _: ResourceHandle) -> Option<PixelSource<'_>> {
                None
            }
        }
        let mut held: Vec<_> = d
            .more
            .map_or(&[][..], |m| &m.pages)
            .iter()
            .map(|(h, p)| (*h, p.borrow_mut()))
            .collect();
        let guest = AllAttached(
            held.iter_mut()
                .map(|(h, p)| {
                    let iov = crate::abi::GuestIov {
                        base: crate::abi::VmmPtr(p.as_mut_ptr().cast()),
                        len: p.len(),
                    };
                    (*h, [iov])
                })
                .collect(),
        );
        let retire = crate::fence::Retirement::start(
            Box::new(Discard),
            crate::vrend::debug::Switches::default(),
        );
        let mut v = Vrend::new(
            Config { host_gl: d.host_gl, ..Config::default() },
            &crate::budget::Budget::with_cap(None, false),
            retire.handle(),
            None,
            crate::vrend::resource::Condemned::default(),
            crate::vrend::debug::Traces::default(),
            crate::vrend::debug::Switches::default(),
        )
        .expect("vrend comes up");
        for &f in d.more.map_or(&[][..], |m| &m.withdrawn) {
            v.features.clear(f);
        }
        if let Some(limit) = d.more.and_then(|m| m.texel_limit) {
            v.limits.max_texture_buffer_size = limit;
        }
        let tessellates = v.features.has(Feature::tessellation);
        let format = |n| super::super::proto::Format::from_wire(n).expect("a known format");
        let target_format = (0..crate::vrend::proto::FORMAT_MAX)
            .filter_map(crate::vrend::proto::Format::from_wire)
            .find(|f| f.name() == d.format)
            .expect("a wire format by that name");
        let target = ResourceHandle::new(1).expect("a resource handle is non-zero");
        let vertices = ResourceHandle::new(2).expect("a resource handle is non-zero");
        let texture = resource::Args {
            target: TextureTarget::Texture2d,
            format: target_format,
            bind: resource::Bind(resource::Bind::RENDER_TARGET.0 | resource::Bind::SAMPLER_VIEW.0),
            width: 16,
            height: 16,
            depth: 1,
            array_size: 1,
            last_level: 0,
            nr_samples: 0,
            flags: resource::ResourceFlags(0),
        };
        v.resource_create(target, texture).expect("a texture");
        v.resource_create(
            vertices,
            resource::Args {
                target: TextureTarget::Buffer,
                format: format(64),
                bind: resource::Bind::VERTEX_BUFFER,
                width: 48,
                height: 1,
                depth: 1,
                array_size: 1,
                last_level: 0,
                nr_samples: 0,
                flags: resource::ResourceFlags(0),
            },
        )
        .expect("a vertex buffer");
        for (handle, args) in d.more.map_or(&[][..], |m| &m.resources) {
            v.resource_create(*handle, *args).expect("a resource the draw asked for");
        }
        // One triangle over the whole target, as clip-space xyzw.
        let corners: Vec<u32> =
            [[-1.0f32, -1.0, 0.0, 1.0], [3.0, -1.0, 0.0, 1.0], [-1.0, 3.0, 0.0, 1.0]]
                .iter()
                .flatten()
                .map(|f| f.to_bits())
                .collect();
        let o = |n: u32| ObjectHandle::new(n).expect("non-zero");
        let (vs, tes, fs) = (
            tgsi_words(d.vs.unwrap_or(PASS_VS)),
            tgsi_words(PASS_TES),
            tgsi_words(d.fs.unwrap_or(RED_FS)),
        );
        let consts: Vec<(ShaderStage, Vec<u32>)> =
            d.consts.iter().map(|(s, c)| (*s, c.map(f32::to_bits).to_vec())).collect();
        let vs_stream_output = d.more.and_then(|m| m.stream_output.clone()).unwrap_or_default();
        fn shader<'t>(stage: ShaderStage, text: &'t [u32], so: &StreamOutput) -> Object<'t> {
            let stream_output = match stage {
                ShaderStage::Vertex => so.clone(),
                _ => StreamOutput::default(),
            };
            Object::Shader(ShaderCreate {
                stage,
                chunk: ShaderChunk::New { total_bytes: text.len() as u32 * 4 },
                num_tokens: 300,
                kind: ShaderKind::Graphics { stream_output },
                text,
            })
        }
        let writes = RtBlend { equation: None, colormask: 0xf };
        let mut commands = vec![
            Command::ResourceInlineWrite {
                transfer: Transfer {
                    resource: vertices,
                    level: 0,
                    usage: 0,
                    stride: 0,
                    layer_stride: 0,
                    region: Box3 { x: 0, y: 0, z: 0, width: 48, height: 1, depth: 1 },
                },
                data: &corners,
            },
            Command::CreateObject {
                handle: o(1),
                object: shader(ShaderStage::Vertex, &vs, &vs_stream_output),
            },
            Command::CreateObject {
                handle: o(3),
                object: shader(ShaderStage::Fragment, &fs, &vs_stream_output),
            },
            Command::BindShader { stage: ShaderStage::Vertex, handle: Some(o(1)) },
            Command::BindShader { stage: ShaderStage::Fragment, handle: Some(o(3)) },
        ];
        if d.tess.is_some() {
            commands.extend([
                Command::CreateObject {
                    handle: o(2),
                    object: shader(ShaderStage::TessEval, &tes, &vs_stream_output),
                },
                Command::BindShader { stage: ShaderStage::TessEval, handle: Some(o(2)) },
            ]);
        }
        commands.extend([
            Command::CreateObject {
                handle: o(4),
                object: Object::VertexElements(vec![VertexElement {
                    src_offset: 0,
                    instance_divisor: 0,
                    vertex_buffer_index: 0,
                    src_format: format(31),
                }]),
            },
            Command::BindObject { kind: ObjectType::VertexElements, handle: Some(o(4)) },
            Command::SetVertexBuffers(vec![VertexBuffer {
                stride: 16,
                offset: 0,
                resource: Some(vertices),
            }]),
            Command::CreateObject {
                handle: o(5),
                object: Object::Blend(BlendState {
                    independent_blend_enable: false,
                    logicop_enable: d.logicop.is_some(),
                    dither: false,
                    alpha_to_coverage: false,
                    alpha_to_one: false,
                    logicop_func: d.logicop.unwrap_or(LogicOp::Copy),
                    rt: [writes; 8],
                }),
            },
            Command::BindObject { kind: ObjectType::Blend, handle: Some(o(5)) },
            Command::CreateObject {
                handle: o(6),
                object: Object::Surface(Surface {
                    resource: target,
                    format: target_format,
                    first_element_or_level: 0,
                    last_element_or_layers: 0,
                    samples: 0,
                }),
            },
            Command::SetFramebufferState { zsurf: None, cbufs: vec![Some(o(6))] },
            Command::SetViewportState {
                start_slot: 0,
                viewports: vec![Viewport { scale: [8.0, 8.0, 0.5], translate: [8.0, 8.0, 0.5] }],
            },
            Command::Clear {
                buffers: 1 << 2,
                color: d.clear.map(f32::to_bits),
                depth: 0.0,
                stencil: 0,
            },
        ]);
        if let Some(level) = d.tess {
            commands.push(Command::SetTessState([level; 6]));
        }
        for (stage, data) in &consts {
            commands.push(Command::SetConstantBuffer { stage: *stage, index: 0, data });
        }
        let draw = d.fs.is_some();
        let indirect = d.more.and_then(|m| m.indirect);
        let replaced = d.more.and_then(|m| m.draw);
        let draw_vbo = || {
            if let Some(draw) = replaced {
                return Command::DrawVbo(draw);
            }
            Command::DrawVbo(Draw {
                start: 0,
                count: 3,
                mode: if d.tess.is_some() { PrimType::Patches } else { PrimType::Triangles },
                indexed: false,
                instance_count: 1,
                index_bias: 0,
                start_instance: 0,
                primitive_restart: false,
                restart_index: 0,
                min_index: 0,
                max_index: 2,
                count_from_so: None,
                tess: match d.tess {
                    Some(_) => Some(TessDraw { vertices_per_patch: 3, drawid: 0 }),
                    // The wire carries the tessellation words ahead of the indirect ones.
                    None => indirect.map(|_| TessDraw { vertices_per_patch: 0, drawid: 0 }),
                },
                indirect,
            })
        };
        if let Some(m) = d.more {
            commands.extend(m.before.iter().cloned());
        }
        commands.extend(draw.then(draw_vbo));
        for round in d.more.map_or(&[][..], |m| &m.redraws) {
            commands.extend(round.iter().cloned());
            commands.push(draw_vbo());
        }
        if let Some(m) = d.more {
            commands.extend(m.after.iter().cloned());
        }
        let mut wire = Vec::new();
        for c in &commands {
            encode(c, &mut wire);
        }
        let ctx = ClassicCtx::for_test(ContextId::new(1).expect("a context id"));
        v.context_create(ctx, &guest).expect("a context");
        let ran = v.submit(ctx, &wire, &guest).expect("the context is here");
        let out = match ran {
            Err(Fault::Shader { cmd: crate::vrend::proto::Cmd::CreateObject, .. })
                if d.tess.is_some() && !tessellates =>
            {
                Ok(None)
            }
            Err(e) => Err(e),
            Ok(()) => {
                // Asked of the sub-context's GL context, which the draw left current.
                if let Some(p) = d.pipeline {
                    let bound =
                        v.gl.get_integer(crate::vrend::gl::gles::GL_PROGRAM_PIPELINE_BINDING);
                    let served = v.shader_cfg.serves_separable();
                    p.set(Some(PipelineSeen { served, bound: bound != 0 }));
                }
                if let Some(m) = d.more
                    && let Some(read) = m.read
                {
                    // A fence naming no context answers every query left waiting.
                    v.fence_global(None, ClientFenceId(1), &guest);
                    for &(handle, len) in &m.read_back {
                        let mut bytes = vec![0xa5u8; len as usize];
                        let into = [crate::abi::GuestIov {
                            base: crate::abi::VmmPtr(bytes.as_mut_ptr().cast()),
                            len: bytes.len(),
                        }];
                        let into = Iov::new(&into);
                        let info = transfer::Info {
                            level: 0,
                            stride: 0,
                            layer_stride: 0,
                            offset: 0,
                            region: Box3 {
                                x: 0,
                                y: 0,
                                z: 0,
                                width: len as i32,
                                height: 1,
                                depth: 1,
                            },
                            synchronized: false,
                        };
                        v.transfer(None, handle, None, transfer::Through::ToGuest(&into), &info)
                            .expect("a buffer the draw asked for reads back");
                        read.borrow_mut().push(bytes);
                    }
                }
                let cursor = v.cursor_contents(target).expect("a 16x16 2D texture reads back");
                Ok(Some(cursor.pixels))
            }
        };
        v.context_destroy(ctx, &guest);
        drop(guest);
        drop(held);
        out
    }

    /// Whether a fresh renderer on `host_gl` has `feature`: what a test of a feature the host
    /// may lack asks first, so that it scores the feature where it is and the refusal where not.
    fn host_has(host_gl: HostGl, feature: Feature) -> bool {
        let _display = crate::vrend::one_display_at_a_time();
        struct Discard;
        impl crate::fence::FenceSink for Discard {
            fn context_fence(&mut self, _: ContextId, _: RingIdx, _: FenceId) {}
            fn present_fence(&mut self, _: FenceId) {}
            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        let retire = crate::fence::Retirement::start(
            Box::new(Discard),
            crate::vrend::debug::Switches::default(),
        );
        let v = Vrend::new(
            Config { host_gl, ..Config::default() },
            &crate::budget::Budget::with_cap(None, false),
            retire.handle(),
            None,
            crate::vrend::resource::Condemned::default(),
            crate::vrend::debug::Traces::default(),
            crate::vrend::debug::Switches::default(),
        )
        .expect("vrend comes up");
        v.features.has(feature)
    }

    /// A buffer of `width` bytes: host memory for `Bind::CUSTOM`, a GL buffer otherwise.
    /// A renderer of `host_gl`, brought up to ask what it offers.
    fn renderer(host_gl: HostGl) -> Vrend {
        struct Discard;
        impl crate::fence::FenceSink for Discard {
            fn context_fence(&mut self, _: ContextId, _: RingIdx, _: FenceId) {}
            fn present_fence(&mut self, _: FenceId) {}
            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        let retire = crate::fence::Retirement::start(
            Box::new(Discard),
            crate::vrend::debug::Switches::default(),
        );
        Vrend::new(
            Config { host_gl, ..Config::default() },
            &crate::budget::Budget::with_cap(None, false),
            retire.handle(),
            None,
            crate::vrend::resource::Condemned::default(),
            crate::vrend::debug::Traces::default(),
            crate::vrend::debug::Switches::default(),
        )
        .expect("vrend comes up")
    }

    /// Whether a renderer of `host_gl` offers the guest `format` for sampling.
    fn offered(host_gl: HostGl, format: crate::vrend::proto::Format) -> bool {
        let _display = crate::vrend::one_display_at_a_time();
        renderer(host_gl).caps().v1.sampler.has(format)
    }

    /// Whether a renderer of `host_gl` has `feature`.
    fn serves(host_gl: HostGl, feature: Feature) -> bool {
        let _display = crate::vrend::one_display_at_a_time();
        renderer(host_gl).features.has(feature)
    }

    fn buffer_args(bind: resource::Bind, width: u32) -> resource::Args {
        resource::Args {
            target: TextureTarget::Buffer,
            format: super::super::proto::Format::from_wire(64).expect("R8_UNORM"),
            bind,
            width,
            height: 1,
            depth: 1,
            array_size: 1,
            last_level: 0,
            nr_samples: 0,
            flags: resource::ResourceFlags(0),
        }
    }

    /// `RESOURCE_INLINE_WRITE` of `data` at the start of buffer `resource`.
    fn inline_write(resource: ResourceHandle, data: &[u32]) -> crate::vrend::proto::Command<'_> {
        use crate::vrend::proto::{Box3, Command, Transfer};
        Command::ResourceInlineWrite {
            transfer: Transfer {
                resource,
                level: 0,
                usage: 0,
                stride: 0,
                layer_stride: 0,
                region: Box3 {
                    x: 0,
                    y: 0,
                    z: 0,
                    width: data.len() as i32 * 4,
                    height: 1,
                    depth: 1,
                },
            },
            data,
        }
    }

    /// Pipeline-statistics, stream-out overflow and per-stream queries count where the host has
    /// them, as the C counts them: the one triangle submits three vertices and one primitive,
    /// with no stream-out bound nothing overflowed, and with no geometry shader every primitive
    /// is generated on stream 0 and none on stream 1. A statistic past the table is refused,
    /// and a host without the queries -- GLES has none of them -- refuses them, as its caps never
    /// offered them.
    #[test]
    fn pipeline_statistics_and_overflow_queries_count_where_the_host_has_them() {
        use crate::vrend::pipe::QueryType;
        use crate::vrend::proto::{Command, Object, QueryCreate};
        let h = |n| ResourceHandle::new(n).expect("non-zero");
        let o = |n| ObjectHandle::new(n).expect("non-zero");
        // VIRGL_STAT_QUERY_IA_VERTICES and _IA_PRIMITIVES, then the overflow of any stream.
        let queries = |stat0: u16| {
            [
                (o(20), h(10), QueryType::PipelineStatistics, stat0),
                (o(21), h(11), QueryType::PipelineStatistics, 1),
                (o(22), h(12), QueryType::SoOverflowAnyPredicate, 0),
                (o(23), h(13), QueryType::PrimitivesGenerated, 0),
                (o(24), h(14), QueryType::PrimitivesGenerated, 1),
            ]
        };
        let run = |host_gl, stat0| {
            let read = std::cell::RefCell::new(Vec::new());
            let qs = queries(stat0);
            let more = More {
                resources: qs
                    .iter()
                    .map(|&(_, r, _, _)| (r, buffer_args(resource::Bind::CUSTOM, 16)))
                    .collect(),
                before: qs
                    .iter()
                    .flat_map(|&(q, resource, kind, index)| {
                        let create = QueryCreate { kind, index, offset: 0, resource };
                        [
                            Command::CreateObject { handle: q, object: Object::Query(create) },
                            Command::BeginQuery(q),
                        ]
                    })
                    .collect(),
                after: qs
                    .iter()
                    .flat_map(|&(q, ..)| {
                        [Command::EndQuery(q), Command::GetQueryResult { query: q, wait: true }]
                    })
                    .collect(),
                read_back: qs.iter().map(|&(_, r, _, _)| (r, 16)).collect(),
                read: Some(&read),
                ..Default::default()
            };
            let ran = draw_over_target(OneDraw {
                host_gl,
                format: "R8G8B8A8_UNORM",
                clear: [0.0; 4],
                fs: Some(RED_FS),
                vs: None,
                consts: &[],
                pipeline: None,
                more: Some(&more),
                logicop: None,
                tess: None,
            });
            ran.map(|_| read.into_inner())
        };
        for host_gl in [HostGl::Gles, HostGl::Desktop] {
            let has = host_has(host_gl, Feature::pipeline_statistics_query)
                && host_has(host_gl, Feature::transform_feedback_overflow_query)
                && host_has(host_gl, Feature::transform_feedback3);
            if !has {
                let ran = run(host_gl, 0);
                assert!(
                    matches!(ran, Err(Fault::NoFeature { .. })),
                    "{host_gl:?}: a host without the queries refuses them: {ran:?}"
                );
                continue;
            }
            let read = run(host_gl, 0).expect("the draw runs");
            let word = |b: &[u8], at: usize| u32::from_le_bytes(b[at..at + 4].try_into().unwrap());
            let result = |b: &[u8]| u64::from_le_bytes(b[8..16].try_into().unwrap());
            for (b, want) in read.iter().zip([3, 1, 0, 1, 0]) {
                assert_eq!(word(b, 0), 1, "{host_gl:?}: VIRGL_QUERY_STATE_DONE");
                assert_eq!(result(b), want, "{host_gl:?}: the counted value");
            }
            let past = run(host_gl, 11);
            assert!(
                matches!(past, Err(Fault::OutOfRange { .. })),
                "{host_gl:?}: no twelfth statistic: {past:?}"
            );
        }
    }

    /// A query buffer object takes an occlusion query's result, and whether it is available,
    /// into a GL buffer at the offsets the guest names, where the host has query buffers: the
    /// triangle covers all 256 samples of the 16x16 target. A host without them refuses.
    #[test]
    fn a_query_result_lands_in_a_buffer_where_the_host_has_query_buffers() {
        use crate::vrend::pipe::{QueryType, QueryValueType};
        use crate::vrend::proto::{Command, Object, QueryCreate};
        let (state, qbo) = (
            ResourceHandle::new(10).expect("non-zero"),
            ResourceHandle::new(11).expect("non-zero"),
        );
        let q = ObjectHandle::new(20).expect("non-zero");
        let run = |host_gl| {
            let read = std::cell::RefCell::new(Vec::new());
            let ask = |offset, index| Command::GetQueryResultQbo {
                query: q,
                buffer: qbo,
                wait: true,
                result_type: QueryValueType::U32,
                offset,
                index,
            };
            let zeros = [0u32; 4];
            let create = QueryCreate {
                kind: QueryType::OcclusionCounter,
                index: 0,
                offset: 0,
                resource: state,
            };
            let more = More {
                resources: vec![
                    (state, buffer_args(resource::Bind::CUSTOM, 16)),
                    (qbo, buffer_args(resource::Bind::VERTEX_BUFFER, 16)),
                ],
                before: vec![
                    inline_write(qbo, &zeros),
                    Command::CreateObject { handle: q, object: Object::Query(create) },
                    Command::BeginQuery(q),
                ],
                after: vec![Command::EndQuery(q), ask(4, 0), ask(8, -1)],
                read_back: vec![(qbo, 16)],
                read: Some(&read),
                ..Default::default()
            };
            draw_over_target(OneDraw {
                host_gl,
                format: "R8G8B8A8_UNORM",
                clear: [0.0; 4],
                fs: Some(RED_FS),
                vs: None,
                consts: &[],
                pipeline: None,
                more: Some(&more),
                logicop: None,
                tess: None,
            })
            .map(|_| read.into_inner())
        };
        for host_gl in [HostGl::Gles, HostGl::Desktop] {
            let ran = run(host_gl);
            if !host_has(host_gl, Feature::qbo) {
                assert!(
                    matches!(ran, Err(Fault::NoFeature { feature: Feature::qbo, .. })),
                    "{host_gl:?}: a host without query buffers refuses: {ran:?}"
                );
                continue;
            }
            let read = ran.expect("the draw runs");
            let word = |at: usize| u32::from_le_bytes(read[0][at..at + 4].try_into().unwrap());
            assert_eq!(word(0), 0, "{host_gl:?}: nothing before the offset");
            assert_eq!(word(4), 256, "{host_gl:?}: every sample of the target passed");
            assert_eq!(word(8), 1, "{host_gl:?}: and the result was available");
        }
    }

    /// An indirect draw with a count buffer draws as many of its commands as the buffer says,
    /// where the host has indirect parameters: a count of zero leaves the target as cleared,
    /// one draws the triangle. A host without them refuses the draw.
    #[test]
    fn an_indirect_draw_count_is_read_from_its_buffer_where_the_host_has_one() {
        use crate::vrend::proto::IndirectDraw;
        let (commands, count) = (
            ResourceHandle::new(10).expect("non-zero"),
            ResourceHandle::new(11).expect("non-zero"),
        );
        // DrawArraysIndirectCommand: three vertices, one instance, from the first.
        let command = [3u32, 1, 0, 0];
        let run = |host_gl, n: u32| {
            let n = [n];
            let more = More {
                resources: vec![
                    (commands, buffer_args(resource::Bind::VERTEX_BUFFER, 16)),
                    (count, buffer_args(resource::Bind::VERTEX_BUFFER, 4)),
                ],
                before: vec![inline_write(commands, &command), inline_write(count, &n)],
                indirect: Some(IndirectDraw {
                    resource: commands,
                    offset: 0,
                    stride: 16,
                    draw_count: 1,
                    draw_count_offset: 0,
                    draw_count_resource: Some(count),
                }),
                ..Default::default()
            };
            draw_over_target(OneDraw {
                host_gl,
                format: "R8G8B8A8_UNORM",
                clear: [0.0; 4],
                fs: Some(RED_FS),
                vs: None,
                consts: &[],
                pipeline: None,
                more: Some(&more),
                logicop: None,
                tess: None,
            })
        };
        for host_gl in [HostGl::Gles, HostGl::Desktop] {
            if !host_has(host_gl, Feature::indirect_params) {
                let ran = run(host_gl, 1);
                assert!(
                    matches!(ran, Err(Fault::NoFeature { feature: Feature::indirect_params, .. })),
                    "{host_gl:?}: a host without indirect parameters refuses: {ran:?}"
                );
                continue;
            }
            let none = run(host_gl, 0).expect("the draw runs").expect("no tessellation");
            assert!(none.iter().all(|b| *b == 0), "{host_gl:?}: a count of zero draws nothing");
            let one = run(host_gl, 1).expect("the draw runs").expect("no tessellation");
            assert!(
                one.chunks(4).all(|p| p == [0xff, 0, 0, 0xff]),
                "{host_gl:?}: a count of one draws the triangle: {:?}",
                &one[..4]
            );
        }
    }

    /// An indexed draw under transform feedback on a GLES host that refuses one -- GLES before 3.2
    /// without `OES_geometry_shader` -- captures the vertices its indices name, in their order,
    /// and primitive restart cuts it where the index says. The host's own feature is withdrawn,
    /// so a GLES that has it takes the same road. The capture is the triangle's corners in the
    /// order the indices give, which no draw of the vertex buffer in its own order produces.
    ///
    /// The vertices are gathered into buffers the guest's count and strides size, so a draw that
    /// would gather past the bounds is refused before anything is allocated for it.
    #[test]
    fn an_indexed_draw_under_transform_feedback_captures_its_indexed_vertices() {
        use crate::vrend::context::deindex;
        use crate::vrend::pipe::PrimType;
        use crate::vrend::proto::{
            Command, Draw, IndexBuffer, IndexType, Object, SoBuffer, SoOutput, StreamOutput,
            StreamoutTarget, VertexBuffer,
        };
        let (indices, captured) = (
            ResourceHandle::new(10).expect("non-zero"),
            ResourceHandle::new(11).expect("non-zero"),
        );
        let target = ObjectHandle::new(20).expect("non-zero");
        // The vertex buffer the harness draws: three corners, xyzw each.
        let corners = [[-1.0f32, -1.0, 0.0, 1.0], [3.0, -1.0, 0.0, 1.0], [-1.0, 3.0, 0.0, 1.0]];
        // `index_list` is U32 indices written into the index buffer; `unwritten` instead draws
        // that many U8 indices out of a buffer left as made, and `stride` is the vertex buffer's.
        let run = |mode, index_list: &[u32], restart, unwritten: Option<u32>, stride: u32| {
            let (index_type, count, index_bytes) = match unwritten {
                Some(n) => (IndexType::U8, n, n),
                None => (IndexType::U32, index_list.len() as u32, 4 * index_list.len() as u32),
            };
            let read = std::cell::RefCell::new(Vec::new());
            let capture_bytes = 6 * 16;
            let more = More {
                stream_output: Some(StreamOutput {
                    stride: [4, 0, 0, 0],
                    outputs: vec![SoOutput {
                        register_index: 0,
                        start_component: 0,
                        num_components: 4,
                        output_buffer: SoBuffer::from_wire(0).expect("buffer 0"),
                        dst_offset: 0,
                        stream: 0,
                    }],
                }),
                resources: vec![
                    (indices, buffer_args(resource::Bind::INDEX_BUFFER, index_bytes)),
                    (captured, buffer_args(resource::Bind::STREAM_OUTPUT, capture_bytes)),
                ],
                before: [
                    (!index_list.is_empty()).then(|| inline_write(indices, index_list)),
                    Some(inline_write(captured, &[0u32; 24])),
                ]
                .into_iter()
                .flatten()
                .chain([
                    Command::SetVertexBuffers(vec![VertexBuffer {
                        stride,
                        offset: 0,
                        resource: Some(ResourceHandle::new(2).expect("the harness's vertices")),
                    }]),
                    Command::CreateObject {
                        handle: target,
                        object: Object::StreamoutTarget(StreamoutTarget {
                            resource: captured,
                            buffer_offset: 0,
                            buffer_size: capture_bytes,
                        }),
                    },
                    Command::SetStreamoutTargets { append_bitmask: 0, targets: vec![Some(target)] },
                    Command::SetIndexBuffer(Some(IndexBuffer {
                        resource: indices,
                        index_type,
                        offset: 0,
                    })),
                ])
                .collect(),
                draw: Some(Draw {
                    start: 0,
                    count,
                    mode,
                    indexed: true,
                    instance_count: 1,
                    index_bias: 0,
                    start_instance: 0,
                    primitive_restart: restart,
                    restart_index: u32::MAX,
                    min_index: 0,
                    max_index: 2,
                    count_from_so: None,
                    tess: None,
                    indirect: None,
                }),
                after: vec![Command::SetStreamoutTargets { append_bitmask: 0, targets: vec![] }],
                read_back: vec![(captured, capture_bytes)],
                read: Some(&read),
                withdrawn: vec![Feature::geometry_shader],
                ..Default::default()
            };
            draw_over_target(OneDraw {
                host_gl: HostGl::Gles,
                format: "R8G8B8A8_UNORM",
                clear: [0.0; 4],
                fs: Some(RED_FS),
                vs: None,
                consts: &[],
                pipeline: None,
                more: Some(&more),
                logicop: None,
                tess: None,
            })
            .map(|_| read.into_inner().remove(0))
        };
        let floats = |bytes: &[u8]| -> Vec<[f32; 4]> {
            bytes
                .as_chunks::<16>()
                .0
                .iter()
                .map(|v| {
                    std::array::from_fn(|i| {
                        f32::from_le_bytes(v[i * 4..i * 4 + 4].try_into().unwrap())
                    })
                })
                .collect()
        };
        let (a, b, c) = (corners[0], corners[1], corners[2]);
        let got = run(PrimType::Triangles, &[2, 0, 1, 1, 2, 0], false, None, 16)
            .expect("the draw is served");
        assert_eq!(floats(&got), [c, a, b, b, c, a], "in index order");
        // A strip cut in two by a restart is two triangles, each from its own run.
        let got = run(PrimType::TriangleStrip, &[0, 1, 2, u32::MAX, 2, 1, 0], true, None, 16)
            .expect("the draw is served");
        assert_eq!(floats(&got), [a, b, c, c, b, a], "one triangle per run");
        // One vertex past the count bound; and, at a count inside it, a stride that takes one
        // binding's gathered vertices past theirs.
        let past_count = deindex::MAX_VERTICES + 1;
        let wide = 32;
        let past_bytes = (deindex::MAX_GATHERED / wide) as u32 + 1;
        assert!(past_bytes <= deindex::MAX_VERTICES, "the second draw passes the first bound");
        for (count, stride, what) in [
            (past_count, 16, "a de-indexed draw's vertex count"),
            (past_bytes, wide as u32, "a de-indexed draw's gathered vertices"),
        ] {
            let ran = run(PrimType::Points, &[], false, Some(count), stride);
            assert!(
                matches!(ran, Err(Fault::OutOfRange { what: w, .. }) if w == what),
                "{count} vertices of stride {stride}: {ran:?}"
            );
        }
    }

    /// On a desktop host, a texture of each of these formats reads back the bytes uploaded into
    /// it, whole and after a sub-box upload. Desktop GL clamps a read from a normalized buffer
    /// to [0, 1] unless told not to, which zeroes every negative snorm value; and it uploads
    /// through a packed type a GLES host has no use for.
    #[test]
    fn a_desktop_texture_reads_back_what_was_uploaded() {
        let _display = crate::vrend::one_display_at_a_time();
        struct Discard;
        impl crate::fence::FenceSink for Discard {
            fn context_fence(&mut self, _: ContextId, _: RingIdx, _: FenceId) {}
            fn present_fence(&mut self, _: FenceId) {}
            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        let retire = crate::fence::Retirement::start(
            Box::new(Discard),
            crate::vrend::debug::Switches::default(),
        );
        let mut v = Vrend::new(
            Config { host_gl: HostGl::Desktop, ..Config::default() },
            &crate::budget::Budget::with_cap(None, false),
            retire.handle(),
            None,
            crate::vrend::resource::Condemned::default(),
            crate::vrend::debug::Traces::default(),
            crate::vrend::debug::Switches::default(),
        )
        .expect("vrend comes up");
        let info = |x: i32, y: i32, w: i32, h: i32, stride: u32| transfer::Info {
            level: 0,
            stride,
            layer_stride: 0,
            offset: 0,
            region: crate::vrend::proto::Box3 { x, y, z: 0, width: w, height: h, depth: 1 },
            synchronized: false,
        };
        let iov = |bytes: &mut [u8]| crate::abi::GuestIov {
            base: crate::abi::VmmPtr(bytes.as_mut_ptr().cast()),
            len: bytes.len(),
        };
        let formats =
            ["R8_SNORM", "R8G8B8A8_SNORM", "R16G16B16A16_SNORM", "RGTC1_SNORM", "B2G3R3_UNORM"];
        for (n, name) in formats.into_iter().enumerate() {
            let format = (0..crate::vrend::proto::FORMAT_MAX)
                .filter_map(crate::vrend::proto::Format::from_wire)
                .find(|f| f.name() == name)
                .expect("a wire format");
            let d = format.describe().expect("described");
            let res = ResourceHandle::new(n as u32 + 1).expect("non-zero");
            let (w, h) = (64u32, 32u32);
            v.resource_create(
                res,
                resource::Args {
                    target: TextureTarget::Texture2d,
                    format,
                    bind: resource::Bind::SAMPLER_VIEW,
                    width: w,
                    height: h,
                    depth: 1,
                    array_size: 1,
                    last_level: 0,
                    nr_samples: 0,
                    flags: resource::ResourceFlags(0),
                },
            )
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
            let stride = d.stride(w) as u32;
            let upload = |v: &mut Vrend, bytes: &mut [u8], i: &transfer::Info| {
                let from = [iov(bytes)];
                let from = Iov::new(&from);
                v.transfer(None, res, Some(&from), transfer::Through::ToHost(from.source()), i)
                    .unwrap_or_else(|e| panic!("{name}: upload {e:?}"));
            };
            let read = |v: &mut Vrend| {
                let mut read = vec![0xa5u8; (stride * d.blocks_high(h)) as usize];
                let into = [iov(&mut read)];
                let into = Iov::new(&into);
                v.transfer(
                    None,
                    res,
                    None,
                    transfer::Through::ToGuest(&into),
                    &info(0, 0, w as i32, h as i32, stride),
                )
                .unwrap_or_else(|e| panic!("{name}: read {e:?}"));
                read
            };
            // Every byte value, so a negative snorm and every bit of a packed one is in there.
            let mut want: Vec<u8> = (0..stride * d.blocks_high(h))
                .map(|i| (i as u8).wrapping_mul(29).wrapping_add(7))
                .collect();
            upload(&mut v, &mut want.clone(), &info(0, 0, w as i32, h as i32, stride));
            assert!(read(&mut v) == want, "{name}: whole");
            let (x, y, sw, sh) = (8u32, 4u32, 16u32, 8u32);
            let row = d.stride(sw) as usize;
            let mut sub: Vec<u8> = (0..row * d.blocks_high(sh) as usize)
                .map(|i| (i as u8).wrapping_mul(13).wrapping_add(200))
                .collect();
            upload(&mut v, &mut sub, &info(x as i32, y as i32, sw as i32, sh as i32, row as u32));
            for r in 0..d.blocks_high(sh) as usize {
                let at = (d.blocks_high(y) as usize + r) * stride as usize + d.stride(x) as usize;
                want[at..at + row].copy_from_slice(&sub[r * row..(r + 1) * row]);
            }
            assert!(read(&mut v) == want, "{name}: after a sub-box");
        }
    }

    /// Whether a host serves separable stages, and whether a draw ran a program pipeline.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    struct PipelineSeen {
        served: bool,
        bound: bool,
    }

    /// Separable stages run as a pipeline of their own programs on desktop GL with pipelines,
    /// as the C runs them, and are linked whole elsewhere: GLES serves no separable program, and
    /// a desktop host without pipelines never advertised them. Either way each stage's constant
    /// reaches its own stage: green from the vertex shader through a generic, red added in the
    /// fragment shader, so yellow -- where a constant written into the wrong stage's program
    /// leaves one of the two out.
    #[test]
    fn separable_stages_draw_as_a_pipeline_on_desktop_gl_and_linked_whole_on_gles() {
        const SEPARABLE_VS: &str = "VERT\nPROPERTY SEPARABLE_PROGRAM 1\nDCL IN[0]\n\
                                    DCL OUT[0], POSITION\nDCL OUT[1], GENERIC[0]\nDCL CONST[0]\n  \
                                    0: MOV OUT[0], IN[0]\n  1: MOV OUT[1], CONST[0]\n  2: END\n";
        const SEPARABLE_FS: &str = "FRAG\nPROPERTY SEPARABLE_PROGRAM 1\n\
                                    DCL IN[0], GENERIC[0], PERSPECTIVE\nDCL OUT[0], COLOR\n\
                                    DCL CONST[0]\n  0: ADD OUT[0], IN[0], CONST[0]\n  1: END\n";
        for host_gl in [HostGl::Gles, HostGl::Desktop] {
            let pipeline = std::cell::Cell::new(None);
            let pixels = draw_over_target(OneDraw {
                host_gl,
                format: "R8G8B8A8_UNORM",
                vs: Some(SEPARABLE_VS),
                fs: Some(SEPARABLE_FS),
                consts: &[
                    (ShaderStage::Vertex, [0.0, 1.0, 0.0, 0.5]),
                    (ShaderStage::Fragment, [1.0, 0.0, 0.0, 0.5]),
                ],
                clear: [0.0; 4],
                logicop: None,
                tess: None,
                pipeline: Some(&pipeline),
                more: None,
            })
            .expect("the draw runs")
            .expect("no tessellation asked for");
            assert!(
                pixels.as_chunks::<4>().0.iter().all(|p| *p == [0xff, 0xff, 0, 0xff]),
                "{host_gl:?}: both stages' constants reached the target: {:?}",
                &pixels[..4]
            );
            let seen = pipeline.get().expect("the draw ran");
            assert_eq!(seen.bound, seen.served, "{host_gl:?}: a pipeline where it is served");
            if host_gl == HostGl::Gles {
                assert!(!seen.served, "GLES serves no separable stage");
            }
            eprintln!("{host_gl:?}: separable stages served: {}", seen.served);
        }
    }

    /// Separable stages sampling through different sampler types draw as one pipeline. A stage's
    /// sampler units are its own program's state, so a pipeline made from stages other
    /// pipelines drew with starts with the units those draws wrote: here the two-sampler vertex
    /// stage's second 2D sampler and the cube sampler both sit at unit 1. Mesa refuses a pipeline
    /// whose active samplers of different types share a unit other than 0, so a pipeline
    /// validated before its draw writes the units it really uses is refused, and the context
    /// poisoned, for a draw that is valid.
    #[test]
    fn separable_stages_sampling_different_targets_draw_as_one_pipeline() {
        use crate::vrend::pipe::{CompareFunc, MipFilter, Swizzle, TexFilter, TexWrap};
        use crate::vrend::proto::{Command, Object, SamplerState, SamplerView};
        use crate::vrend::proto::{ShaderChunk, ShaderCreate, ShaderKind, StreamOutput};
        const VS_ONE: &str = "VERT\nPROPERTY SEPARABLE_PROGRAM 1\nDCL IN[0]\n\
                              DCL OUT[0], POSITION\nDCL OUT[1], GENERIC[0]\nDCL SAMP[0]\n\
                              DCL SVIEW[0], 2D, FLOAT\nDCL TEMP[0]\n\
                              IMM[0] FLT32 { 0.5, 0.5, 0.0, 0.0 }\n  0: MOV OUT[0], IN[0]\n  \
                              1: TXL TEMP[0], IMM[0], SAMP[0], 2D\n  2: MOV OUT[1], TEMP[0]\n  \
                              3: END\n";
        const VS_TWO: &str = "VERT\nPROPERTY SEPARABLE_PROGRAM 1\nDCL IN[0]\n\
                              DCL OUT[0], POSITION\nDCL OUT[1], GENERIC[0]\nDCL SAMP[0]\n\
                              DCL SAMP[1]\nDCL SVIEW[0], 2D, FLOAT\nDCL SVIEW[1], 2D, FLOAT\n\
                              DCL TEMP[0..1]\nIMM[0] FLT32 { 0.5, 0.5, 0.0, 0.0 }\n  \
                              0: MOV OUT[0], IN[0]\n  1: TXL TEMP[0], IMM[0], SAMP[0], 2D\n  \
                              2: TXL TEMP[1], IMM[0], SAMP[1], 2D\n  \
                              3: ADD OUT[1], TEMP[0], TEMP[1]\n  4: END\n";
        const FS_CUBE: &str = "FRAG\nPROPERTY SEPARABLE_PROGRAM 1\n\
                               DCL IN[0], GENERIC[0], PERSPECTIVE\nDCL OUT[0], COLOR\n\
                               DCL SAMP[0]\nDCL SVIEW[0], CUBE, FLOAT\nDCL TEMP[0]\n\
                               IMM[0] FLT32 { 1.0, 0.0, 0.0, 1.0 }\n  \
                               0: TEX TEMP[0], IMM[0], SAMP[0], CUBE\n  \
                               1: ADD TEMP[0], TEMP[0], IN[0]\n  2: ADD OUT[0], TEMP[0], IMM[0]\n  \
                               3: END\n";
        const FS_RED: &str = "FRAG\nPROPERTY SEPARABLE_PROGRAM 1\nDCL OUT[0], COLOR\n\
                              IMM[0] FLT32 { 1.0, 0.0, 0.0, 1.0 }\n  0: MOV OUT[0], IMM[0]\n  \
                              1: END\n";
        let o = |n: u32| ObjectHandle::new(n).expect("non-zero");
        let bgra = super::super::proto::Format::from_wire(1).expect("B8G8R8A8_UNORM");
        let (flat, cube) = (
            ResourceHandle::new(10).expect("a resource handle is non-zero"),
            ResourceHandle::new(11).expect("a resource handle is non-zero"),
        );
        let texture = |target, array_size| resource::Args {
            target,
            format: bgra,
            bind: resource::Bind::SAMPLER_VIEW,
            width: 16,
            height: 16,
            depth: 1,
            array_size,
            last_level: 0,
            nr_samples: 0,
            flags: resource::ResourceFlags(0),
        };
        let view = |resource, target, last_layer: u32| {
            Object::SamplerView(SamplerView {
                resource,
                format: bgra,
                target,
                first_element_or_layers: last_layer << 16,
                last_element_or_levels: 0,
                swizzle: [Swizzle::X, Swizzle::Y, Swizzle::Z, Swizzle::W],
            })
        };
        let state = Object::SamplerState(SamplerState {
            wrap_s: TexWrap::ClampToEdge,
            wrap_t: TexWrap::ClampToEdge,
            wrap_r: TexWrap::ClampToEdge,
            min_img_filter: TexFilter::Nearest,
            min_mip_filter: MipFilter::None,
            mag_img_filter: TexFilter::Nearest,
            compare_mode: false,
            compare_func: CompareFunc::LessEqual,
            seamless_cube_map: false,
            max_anisotropy: 0,
            lod_bias: 0.0,
            min_lod: 0.0,
            max_lod: 0.0,
            border_color: [0; 4],
        });
        fn shader(handle: ObjectHandle, stage: ShaderStage, text: &[u32]) -> Command<'_> {
            Command::CreateObject {
                handle,
                object: Object::Shader(ShaderCreate {
                    stage,
                    chunk: ShaderChunk::New { total_bytes: text.len() as u32 * 4 },
                    num_tokens: 300,
                    kind: ShaderKind::Graphics { stream_output: StreamOutput::default() },
                    text,
                }),
            }
        }
        let (vs_two, fs_red) = (tgsi_words(VS_TWO), tgsi_words(FS_RED));
        let (vs, fs) = (ShaderStage::Vertex, ShaderStage::Fragment);
        let more = More {
            resources: vec![
                (flat, texture(TextureTarget::Texture2d, 1)),
                (cube, texture(TextureTarget::Cube, 6)),
            ],
            before: vec![
                Command::CreateObject {
                    handle: o(20),
                    object: view(flat, TextureTarget::Texture2d, 0),
                },
                Command::CreateObject { handle: o(21), object: view(cube, TextureTarget::Cube, 5) },
                Command::CreateObject { handle: o(22), object: state },
                shader(o(23), vs, &vs_two),
                shader(o(24), fs, &fs_red),
                Command::SetSamplerViews { stage: vs, start_slot: 0, views: vec![Some(o(20)); 2] },
                Command::BindSamplerStates {
                    stage: vs,
                    start_slot: 0,
                    states: vec![Some(o(22)); 2],
                },
                Command::SetSamplerViews { stage: fs, start_slot: 0, views: vec![Some(o(21))] },
                Command::BindSamplerStates { stage: fs, start_slot: 0, states: vec![Some(o(22))] },
            ],
            // The first draw puts the cube sampler at unit 1, after the one-sampler vertex
            // stage's; the second puts the two-sampler vertex stage's second sampler there; the
            // third pairs those two stages.
            redraws: vec![
                vec![
                    Command::BindShader { stage: vs, handle: Some(o(23)) },
                    Command::BindShader { stage: fs, handle: Some(o(24)) },
                ],
                vec![Command::BindShader { stage: fs, handle: Some(o(3)) }],
            ],
            ..Default::default()
        };
        for host_gl in [HostGl::Gles, HostGl::Desktop] {
            let pipeline = std::cell::Cell::new(None);
            let pixels = draw_over_target(OneDraw {
                host_gl,
                format: "R8G8B8A8_UNORM",
                vs: Some(VS_ONE),
                fs: Some(FS_CUBE),
                consts: &[],
                clear: [0.0; 4],
                logicop: None,
                tess: None,
                pipeline: Some(&pipeline),
                more: Some(&more),
            })
            .unwrap_or_else(|e| panic!("{host_gl:?}: the draws run: {e:?}"))
            .expect("no tessellation asked for");
            assert!(
                pixels.as_chunks::<4>().0.iter().all(|p| p[0] == 0xff),
                "{host_gl:?}: the cube-sampling fragment stage's red reached the target: {:?}",
                &pixels[..4]
            );
            let seen = pipeline.get().expect("the draws ran");
            assert_eq!(seen.bound, seen.served, "{host_gl:?}: a pipeline where it is served");
            eprintln!("{host_gl:?}: separable stages served: {}", seen.served);
        }
    }

    /// A fragment shader drawing into an integer target writes the integer bits it holds. Its
    /// colour output has to be declared integer: zink on KosmicKrisp converts a float output to
    /// the target's integers, which turns every bit pattern into zero. Written from an
    /// immediate, a constant, a flat varying and an image load, each typed output also has to
    /// compile, which wants the bit casts a plain `#version 150` lacks.
    #[test]
    fn an_integer_target_takes_the_bits_the_shader_writes() {
        use crate::vrend::pipe::ImageAccess;
        use crate::vrend::proto::{Box3, Command, ShaderImage, Transfer};
        const IMM: &str = "FRAG\nDCL OUT[0], COLOR\n\
                           IMM[0] UINT32 { 1065353216, 1073741824, 1077936128, 1082130432 }\n  \
                           0: MOV OUT[0], IMM[0]\n  1: END\n";
        // The fragment stage's only integer is its output's cast.
        const VS: &str = "VERT\nDCL IN[0]\nDCL OUT[0], POSITION\nDCL OUT[1], GENERIC[0]\n\
                          IMM[0] UINT32 { 1065353216, 1073741824, 1077936128, 1082130432 }\n  \
                          0: MOV OUT[0], IN[0]\n  1: MOV OUT[1], IMM[0]\n  2: END\n";
        const VARYING: &str = "FRAG\nDCL IN[0], GENERIC[0], CONSTANT\nDCL OUT[0], COLOR\n  \
                               0: MOV OUT[0], IN[0]\n  1: END\n";
        const CONST: &str = "FRAG\nDCL OUT[0], COLOR\nDCL CONST[0]\n  0: MOV OUT[0], CONST[0]\n  \
                             1: END\n";
        const LOAD: &str = "FRAG\nDCL OUT[0], COLOR\n\
                            DCL IMAGE[0], 2D, PIPE_FORMAT_R32G32B32A32_UINT, WR\n\
                            IMM[0] INT32 { 0, 0, 0, 0 }\n  \
                            0: LOAD OUT[0], IMAGE[0], IMM[0], 2D, PIPE_FORMAT_R32G32B32A32_UINT\n  \
                            1: END\n";
        let rgba32ui = (0..crate::vrend::proto::FORMAT_MAX)
            .filter_map(crate::vrend::proto::Format::from_wire)
            .find(|f| f.name() == "R32G32B32A32_UINT")
            .expect("a wire format");
        let image = ResourceHandle::new(10).expect("a resource handle is non-zero");
        let texel = [1.0f32, 2.0, 3.0, 4.0].map(f32::to_bits);
        let more = More {
            resources: vec![(
                image,
                resource::Args {
                    target: TextureTarget::Texture2d,
                    format: rgba32ui,
                    bind: resource::Bind::SAMPLER_VIEW,
                    width: 1,
                    height: 1,
                    depth: 1,
                    array_size: 1,
                    last_level: 0,
                    nr_samples: 0,
                    flags: resource::ResourceFlags(0),
                },
            )],
            before: vec![
                Command::ResourceInlineWrite {
                    transfer: Transfer {
                        resource: image,
                        level: 0,
                        usage: 0,
                        stride: 16,
                        layer_stride: 0,
                        region: Box3 { x: 0, y: 0, z: 0, width: 1, height: 1, depth: 1 },
                    },
                    data: &texel,
                },
                Command::SetShaderImages {
                    stage: ShaderStage::Fragment,
                    start_slot: 0,
                    images: vec![Some(ShaderImage {
                        format: rgba32ui,
                        access: ImageAccess::Read,
                        layer_offset: 0,
                        level_size: 0,
                        resource: image,
                    })],
                },
            ],
            ..Default::default()
        };
        let consts = [(ShaderStage::Fragment, texel.map(f32::from_bits))];
        let want: Vec<u8> = texel.iter().flat_map(|w| w.to_le_bytes()).collect();
        for host_gl in [HostGl::Gles, HostGl::Desktop] {
            for (what, vs, fs) in [
                ("immediate", None, IMM),
                ("constant", None, CONST),
                ("varying", Some(VS), VARYING),
                ("image load", None, LOAD),
            ] {
                let pixels = draw_over_target(OneDraw {
                    host_gl,
                    format: "R32G32B32A32_UINT",
                    vs,
                    fs: Some(fs),
                    consts: &consts,
                    clear: [0.0; 4],
                    logicop: None,
                    tess: None,
                    pipeline: None,
                    more: Some(&more),
                })
                .unwrap_or_else(|e| panic!("{host_gl:?}: the {what} draws: {e:?}"))
                .expect("no tessellation asked for");
                assert!(
                    pixels.chunks(16).all(|p| p == want),
                    "{host_gl:?}: the {what}'s bits reached the target: {:?}",
                    &pixels[..16]
                );
            }
        }
    }

    /// A texture image's level is the low byte of the wire's level word. Mesa sends a
    /// `pipe_image_view`'s union as it lies, and above the 8-bit level the word holds whatever the
    /// guest left there: `0xffff_0000` from a stock guest's piglit. Taken whole, that is level
    /// -65536, which GL refuses to bind.
    #[test]
    fn an_image_binds_at_the_level_in_the_low_byte() {
        use crate::vrend::pipe::ImageAccess;
        use crate::vrend::proto::{Box3, Command, ShaderImage, Transfer};
        const FS: &str = "FRAG\nDCL OUT[0], COLOR\n\
                          DCL IMAGE[0], 2D, PIPE_FORMAT_R8G8B8A8_UNORM, WR\n\
                          DCL TEMP[0]\nIMM[0] INT32 { 3, 2, 0, 0 }\n  \
                          0: LOAD TEMP[0], IMAGE[0], IMM[0], 2D, PIPE_FORMAT_R8G8B8A8_UNORM\n  \
                          1: MOV OUT[0], TEMP[0]\n  2: END\n";
        let rgba8 = super::super::proto::Format::from_wire(67).expect("R8G8B8A8_UNORM");
        let image = ResourceHandle::new(10).expect("a resource handle is non-zero");
        let red = [0xff00_00ffu32; 16];
        let more = More {
            resources: vec![(
                image,
                resource::Args {
                    target: TextureTarget::Texture2d,
                    format: rgba8,
                    bind: resource::Bind::SAMPLER_VIEW,
                    width: 4,
                    height: 4,
                    depth: 1,
                    array_size: 1,
                    last_level: 0,
                    nr_samples: 0,
                    flags: resource::ResourceFlags(0),
                },
            )],
            before: vec![
                Command::ResourceInlineWrite {
                    transfer: Transfer {
                        resource: image,
                        level: 0,
                        usage: 0,
                        stride: 16,
                        layer_stride: 0,
                        region: Box3 { x: 0, y: 0, z: 0, width: 4, height: 4, depth: 1 },
                    },
                    data: &red,
                },
                Command::SetShaderImages {
                    stage: ShaderStage::Fragment,
                    start_slot: 0,
                    images: vec![Some(ShaderImage {
                        format: rgba8,
                        access: ImageAccess::Read,
                        layer_offset: 0,
                        level_size: 0xffff_0000,
                        resource: image,
                    })],
                },
            ],
            ..Default::default()
        };
        for host_gl in [HostGl::Gles, HostGl::Desktop] {
            let pixels = draw_over_target(OneDraw {
                host_gl,
                format: "R8G8B8A8_UNORM",
                vs: None,
                fs: Some(FS),
                consts: &[],
                clear: [0.0; 4],
                logicop: None,
                tess: None,
                pipeline: None,
                more: Some(&more),
            })
            .unwrap_or_else(|e| panic!("{host_gl:?}: the load draws: {e:?}"))
            .expect("no tessellation asked for");
            assert!(
                pixels.as_chunks::<4>().0.iter().all(|p| *p == [0xff, 0, 0, 0xff]),
                "{host_gl:?}: the image's red was loaded: {:?}",
                &pixels[..4]
            );
        }
    }

    /// An image slot the guest empties is empty to the next draw: GL keeps a unit's image until
    /// something else is bound there, so a draw that binds only the slots holding views leaves
    /// an emptied one reading, and adding into, the image an earlier draw bound. A load from an
    /// empty unit reads zero.
    #[test]
    fn an_emptied_image_slot_reads_nothing() {
        use crate::vrend::pipe::ImageAccess;
        use crate::vrend::proto::{Box3, Command, ShaderImage, Transfer};
        const FS: &str = "FRAG\nDCL OUT[0], COLOR\n\
                          DCL IMAGE[0], 2D, PIPE_FORMAT_R8G8B8A8_UNORM, WR\n\
                          DCL TEMP[0]\nIMM[0] INT32 { 0, 0, 0, 0 }\n  \
                          0: LOAD TEMP[0], IMAGE[0], IMM[0], 2D, PIPE_FORMAT_R8G8B8A8_UNORM\n  \
                          1: MOV OUT[0], TEMP[0]\n  2: END\n";
        let rgba8 = super::super::proto::Format::from_wire(67).expect("R8G8B8A8_UNORM");
        let image = ResourceHandle::new(10).expect("a resource handle is non-zero");
        let red = [0xff00_00ffu32; 16];
        let images = |view: Option<ShaderImage>| Command::SetShaderImages {
            stage: ShaderStage::Fragment,
            start_slot: 0,
            images: vec![view],
        };
        let more = More {
            resources: vec![(
                image,
                resource::Args {
                    target: TextureTarget::Texture2d,
                    format: rgba8,
                    bind: resource::Bind::SAMPLER_VIEW,
                    width: 4,
                    height: 4,
                    depth: 1,
                    array_size: 1,
                    last_level: 0,
                    nr_samples: 0,
                    flags: resource::ResourceFlags(0),
                },
            )],
            before: vec![
                Command::ResourceInlineWrite {
                    transfer: Transfer {
                        resource: image,
                        level: 0,
                        usage: 0,
                        stride: 16,
                        layer_stride: 0,
                        region: Box3 { x: 0, y: 0, z: 0, width: 4, height: 4, depth: 1 },
                    },
                    data: &red,
                },
                images(Some(ShaderImage {
                    format: rgba8,
                    access: ImageAccess::Read,
                    layer_offset: 0,
                    level_size: 0,
                    resource: image,
                })),
            ],
            redraws: vec![vec![images(None)]],
            ..Default::default()
        };
        for host_gl in [HostGl::Gles, HostGl::Desktop] {
            let pixels = draw_over_target(OneDraw {
                host_gl,
                format: "R8G8B8A8_UNORM",
                vs: None,
                fs: Some(FS),
                consts: &[],
                clear: [0.5; 4],
                logicop: None,
                tess: None,
                pipeline: None,
                more: Some(&more),
            })
            .unwrap_or_else(|e| panic!("{host_gl:?}: the draws run: {e:?}"))
            .expect("no tessellation asked for");
            assert!(
                pixels.as_chunks::<4>().0.iter().all(|p| *p == [0, 0, 0, 0]),
                "{host_gl:?}: the emptied slot read nothing: {:?}",
                &pixels[..4]
            );
        }
    }

    /// A rectangle in a format desktop GL cannot hold as one -- here RGTC1, which a guest picks
    /// for `GL_COMPRESSED_RED` -- is stored as a 2D texture, so the shader has to sample it as
    /// one. Declared as a rectangle sampler, the draw samples nothing; sized or fetched without
    /// a LOD, the 2D sampler does not compile. A sample, a fetch and a size query each draw what
    /// the texture holds, on both flavours.
    #[test]
    fn a_rectangle_stored_as_2d_samples_as_2d() {
        use crate::vrend::pipe::{CompareFunc, MipFilter, Swizzle, TexFilter, TexWrap};
        use crate::vrend::proto::{Box3, Command, Object, SamplerState, SamplerView, Transfer};
        const TEX: &str = "FRAG\nDCL OUT[0], COLOR\nDCL SAMP[0]\nDCL SVIEW[0], RECT, FLOAT\n\
                           DCL TEMP[0]\nIMM[0] FLT32 { 8.0, 8.0, 0.0, 0.0 }\n  \
                           0: TEX TEMP[0], IMM[0], SAMP[0], RECT\n  1: MOV OUT[0], TEMP[0]\n  \
                           2: END\n";
        const TXF: &str = "FRAG\nDCL OUT[0], COLOR\nDCL SAMP[0]\nDCL SVIEW[0], RECT, FLOAT\n\
                           DCL TEMP[0]\nIMM[0] INT32 { 8, 8, 0, 0 }\n  \
                           0: TXF TEMP[0], IMM[0], SAMP[0], RECT\n  1: MOV OUT[0], TEMP[0]\n  \
                           2: END\n";
        // The width, 16, scaled to 1.0.
        const TXQ: &str = "FRAG\nDCL OUT[0], COLOR\nDCL SAMP[0]\nDCL SVIEW[0], RECT, FLOAT\n\
                           DCL TEMP[0]\nIMM[0] INT32 { 0, 0, 0, 0 }\n\
                           IMM[1] FLT32 { 0.0625, 0.0, 0.0, 1.0 }\n  \
                           0: TXQ TEMP[0], IMM[0].xxxx, SAMP[0], RECT\n  \
                           1: I2F TEMP[0], TEMP[0]\n  2: MUL OUT[0], TEMP[0].xxxx, IMM[1]\n  \
                           3: END\n";
        let o = |n: u32| ObjectHandle::new(n).expect("non-zero");
        let rgtc1 = (0..crate::vrend::proto::FORMAT_MAX)
            .filter_map(crate::vrend::proto::Format::from_wire)
            .find(|f| f.name() == "RGTC1_UNORM")
            .expect("a wire format");
        let rect = ResourceHandle::new(10).expect("a resource handle is non-zero");
        // Every 4x4 block is red 255 at both endpoints: full red throughout.
        let blocks: Vec<u32> = [0x0000_ffffu32, 0].repeat(16);
        let more = More {
            resources: vec![(
                rect,
                resource::Args {
                    target: TextureTarget::Rect,
                    format: rgtc1,
                    bind: resource::Bind::SAMPLER_VIEW,
                    width: 16,
                    height: 16,
                    depth: 1,
                    array_size: 1,
                    last_level: 0,
                    nr_samples: 0,
                    flags: resource::ResourceFlags(0),
                },
            )],
            before: vec![
                Command::ResourceInlineWrite {
                    transfer: Transfer {
                        resource: rect,
                        level: 0,
                        usage: 0,
                        stride: 32,
                        layer_stride: 0,
                        region: Box3 { x: 0, y: 0, z: 0, width: 16, height: 16, depth: 1 },
                    },
                    data: &blocks,
                },
                Command::CreateObject {
                    handle: o(20),
                    object: Object::SamplerView(SamplerView {
                        resource: rect,
                        format: rgtc1,
                        target: TextureTarget::Rect,
                        first_element_or_layers: 0,
                        last_element_or_levels: 0,
                        swizzle: [Swizzle::X, Swizzle::Zero, Swizzle::Zero, Swizzle::One],
                    }),
                },
                Command::CreateObject {
                    handle: o(21),
                    object: Object::SamplerState(SamplerState {
                        wrap_s: TexWrap::ClampToEdge,
                        wrap_t: TexWrap::ClampToEdge,
                        wrap_r: TexWrap::ClampToEdge,
                        min_img_filter: TexFilter::Nearest,
                        min_mip_filter: MipFilter::None,
                        mag_img_filter: TexFilter::Nearest,
                        compare_mode: false,
                        compare_func: CompareFunc::LessEqual,
                        seamless_cube_map: false,
                        max_anisotropy: 0,
                        lod_bias: 0.0,
                        min_lod: 0.0,
                        max_lod: 0.0,
                        border_color: [0; 4],
                    }),
                },
                Command::SetSamplerViews {
                    stage: ShaderStage::Fragment,
                    start_slot: 0,
                    views: vec![Some(o(20))],
                },
                Command::BindSamplerStates {
                    stage: ShaderStage::Fragment,
                    start_slot: 0,
                    states: vec![Some(o(21))],
                },
            ],
            ..Default::default()
        };
        for host_gl in [HostGl::Gles, HostGl::Desktop] {
            for (what, fs) in [("sample", TEX), ("fetch", TXF), ("size query", TXQ)] {
                let pixels = draw_over_target(OneDraw {
                    host_gl,
                    format: "R8G8B8A8_UNORM",
                    vs: None,
                    fs: Some(fs),
                    consts: &[],
                    clear: [0.0; 4],
                    logicop: None,
                    tess: None,
                    pipeline: None,
                    more: Some(&more),
                })
                .unwrap_or_else(|e| panic!("{host_gl:?}: the {what} draws: {e:?}"))
                .expect("no tessellation asked for");
                assert!(
                    pixels.as_chunks::<4>().0.iter().all(|p| *p == [0xff, 0, 0, 0xff]),
                    "{host_gl:?}: the {what} draws the rectangle's red: {:?}",
                    &pixels[..4]
                );
            }
        }
    }

    /// Two pipelines sharing a fragment stage number its uniform block differently: after a
    /// vertex stage with a block of its own the fragment block is the second binding, after one
    /// without it is the first. A block's binding lives in the stage's own program, so drawing
    /// with the first pipeline again has to number it again. Left as the second pipeline set it,
    /// the fragment stage reads the vertex stage's block -- white -- instead of its own red.
    #[test]
    fn a_pipeline_found_again_renumbers_its_shared_stages_blocks() {
        use crate::vrend::proto::{Box3, Command, Object, ShaderChunk, ShaderCreate};
        use crate::vrend::proto::{ShaderKind, StreamOutput, Transfer};
        const VS_BLOCK: &str = "VERT\nPROPERTY SEPARABLE_PROGRAM 1\nDCL IN[0]\n\
                                DCL OUT[0], POSITION\nDCL CONST[1][0]\n  \
                                0: MUL OUT[0], IN[0], CONST[1][0]\n  1: END\n";
        const VS_PLAIN: &str = "VERT\nPROPERTY SEPARABLE_PROGRAM 1\nDCL IN[0]\n\
                                DCL OUT[0], POSITION\n  0: MOV OUT[0], IN[0]\n  1: END\n";
        const FS_BLOCK: &str = "FRAG\nPROPERTY SEPARABLE_PROGRAM 1\nDCL OUT[0], COLOR\n\
                                DCL CONST[1][0]\n  0: MOV OUT[0], CONST[1][0]\n  1: END\n";
        let o = |n: u32| ObjectHandle::new(n).expect("non-zero");
        let (vs_ubo, fs_ubo) = (
            ResourceHandle::new(10).expect("a resource handle is non-zero"),
            ResourceHandle::new(11).expect("a resource handle is non-zero"),
        );
        let buffer = resource::Args {
            target: TextureTarget::Buffer,
            format: super::super::proto::Format::from_wire(64).expect("a known format"),
            bind: resource::Bind::CONSTANT_BUFFER,
            width: 16,
            height: 1,
            depth: 1,
            array_size: 1,
            last_level: 0,
            nr_samples: 0,
            flags: resource::ResourceFlags(0),
        };
        let white: Vec<u32> = [1.0f32; 4].map(f32::to_bits).to_vec();
        let red: Vec<u32> = [1.0f32, 0.0, 0.0, 1.0].map(f32::to_bits).to_vec();
        let plain = tgsi_words(VS_PLAIN);
        let write = |resource, data| Command::ResourceInlineWrite {
            transfer: Transfer {
                resource,
                level: 0,
                usage: 0,
                stride: 0,
                layer_stride: 0,
                region: Box3 { x: 0, y: 0, z: 0, width: 16, height: 1, depth: 1 },
            },
            data,
        };
        let ubo = |stage, resource| Command::SetUniformBuffer {
            stage,
            index: 1,
            offset: 0,
            length: 16,
            resource: Some(resource),
        };
        let more = More {
            resources: vec![(vs_ubo, buffer), (fs_ubo, buffer)],
            before: vec![
                write(vs_ubo, &white),
                write(fs_ubo, &red),
                ubo(ShaderStage::Vertex, vs_ubo),
                ubo(ShaderStage::Fragment, fs_ubo),
                Command::CreateObject {
                    handle: o(20),
                    object: Object::Shader(ShaderCreate {
                        stage: ShaderStage::Vertex,
                        chunk: ShaderChunk::New { total_bytes: plain.len() as u32 * 4 },
                        num_tokens: 300,
                        kind: ShaderKind::Graphics { stream_output: StreamOutput::default() },
                        text: &plain,
                    }),
                },
            ],
            // The pipeline without the vertex block, then the first one found again.
            redraws: vec![
                vec![Command::BindShader { stage: ShaderStage::Vertex, handle: Some(o(20)) }],
                vec![Command::BindShader { stage: ShaderStage::Vertex, handle: Some(o(1)) }],
            ],
            ..Default::default()
        };
        for host_gl in [HostGl::Gles, HostGl::Desktop] {
            let pipeline = std::cell::Cell::new(None);
            let pixels = draw_over_target(OneDraw {
                host_gl,
                format: "R8G8B8A8_UNORM",
                vs: Some(VS_BLOCK),
                fs: Some(FS_BLOCK),
                consts: &[],
                clear: [0.0; 4],
                logicop: None,
                tess: None,
                pipeline: Some(&pipeline),
                more: Some(&more),
            })
            .unwrap_or_else(|e| panic!("{host_gl:?}: the draws run: {e:?}"))
            .expect("no tessellation asked for");
            assert!(
                pixels.as_chunks::<4>().0.iter().all(|p| *p == [0xff, 0, 0, 0xff]),
                "{host_gl:?}: the fragment stage read its own block: {:?}",
                &pixels[..4]
            );
            let seen = pipeline.get().expect("the draws ran");
            assert_eq!(seen.bound, seen.served, "{host_gl:?}: a pipeline where it is served");
            eprintln!("{host_gl:?}: separable stages served: {}", seen.served);
        }
    }

    /// An evaluation shader with no control shader is a valid guest pipeline. GLES will not link
    /// it, so a GLES host puts a control stage ahead of it that writes `SET_TESS_STATE`'s levels;
    /// desktop GL links it as bound and takes the levels as its patch defaults. Either way levels
    /// of one draw the patch -- one triangle covering the target -- and levels of zero cull it. A
    /// host without tessellation never told the guest, and refuses the evaluation shader.
    #[test]
    fn an_evaluation_shader_without_a_control_shader_draws_its_patches() {
        for host_gl in [HostGl::Gles, HostGl::Desktop] {
            let draw = |level| {
                draw_over_target(OneDraw {
                    host_gl,
                    format: "R8G8B8A8_UNORM",
                    clear: [0.0; 4],
                    fs: Some(RED_FS),
                    vs: None,
                    consts: &[],
                    pipeline: None,
                    more: None,
                    logicop: None,
                    tess: Some(level),
                })
                .expect("the draw runs")
            };
            let Some(drawn) = draw(1.0) else {
                continue;
            };
            assert!(
                drawn.as_chunks::<4>().0.iter().all(|p| *p == [0xff, 0, 0, 0xff]),
                "{host_gl:?}: the patch covered the target: {:?}",
                &drawn[..4]
            );
            let culled = draw(0.0).expect("tessellates");
            assert!(
                culled.iter().all(|b| *b == 0),
                "{host_gl:?}: levels of zero cull the patch: {:?}",
                &culled[..4]
            );
        }
    }

    /// A logic op is the blender's on desktop GL and the shader's on GLES, which emulates the ops
    /// that need no framebuffer read. `SET` writes every bit whatever the shader output. The
    /// GLES emulation declares its outputs only for a shader that writes every colour buffer, as
    /// gallium marks one that writes `gl_FragColor`, so that is the shader drawn.
    #[test]
    fn a_logic_op_replaces_the_fragment_on_either_flavour() {
        const RED_ALL_CBUFS_FS: &str = "FRAG\nPROPERTY FS_COLOR0_WRITES_ALL_CBUFS 1\n\
                                        DCL OUT[0], COLOR\nIMM[0] FLT32 { 1.0, 0.0, 0.0, 1.0 }\n  \
                                        0: MOV OUT[0], IMM[0]\n  1: END\n";
        for host_gl in [HostGl::Gles, HostGl::Desktop] {
            let pixels = draw_over_target(OneDraw {
                host_gl,
                format: "R8G8B8A8_UNORM",
                clear: [0.0; 4],
                fs: Some(RED_ALL_CBUFS_FS),
                vs: None,
                consts: &[],
                pipeline: None,
                more: None,
                logicop: Some(crate::vrend::pipe::LogicOp::Set),
                tess: None,
            })
            .expect("the draw runs")
            .expect("no tessellation asked for");
            assert!(pixels.iter().all(|b| *b == 0xff), "{host_gl:?}: {:?}", &pixels[..4]);
        }
    }

    /// A logic op applies to a shader that writes its colour buffers one by one, as well as to
    /// one that writes them all: on GLES the shader emulates it whichever way it declares its
    /// outputs, and a depth output declared ahead of the colour is not a colour buffer.
    /// `COPY_INVERTED` of opaque red is cyan at zero alpha, which only the shader's own colour
    /// run through the op produces.
    #[test]
    fn a_logic_op_applies_to_a_shader_that_writes_one_colour_buffer() {
        const DEPTH_THEN_RED_FS: &str = "FRAG\nDCL OUT[0], POSITION\nDCL OUT[1], COLOR\n\
                                         IMM[0] FLT32 { 1.0, 0.0, 0.0, 1.0 }\n  \
                                         0: MOV OUT[0].z, IMM[0].yyyy\n  \
                                         1: MOV OUT[1], IMM[0]\n  2: END\n";
        for host_gl in [HostGl::Gles, HostGl::Desktop] {
            let pixels = draw_over_target(OneDraw {
                host_gl,
                format: "R8G8B8A8_UNORM",
                clear: [0.0; 4],
                fs: Some(DEPTH_THEN_RED_FS),
                vs: None,
                consts: &[],
                pipeline: None,
                more: None,
                logicop: Some(crate::vrend::pipe::LogicOp::CopyInverted),
                tess: None,
            })
            .expect("the draw runs")
            .expect("no tessellation asked for");
            assert!(
                pixels.chunks(4).all(|p| p == [0, 0xff, 0xff, 0]),
                "{host_gl:?}: {:?}",
                &pixels[..4]
            );
        }
    }

    /// A desktop core profile has no alpha textures: an A8 target is stored in red, and what the
    /// shader writes to alpha has to land there. Black at full alpha tells the two apart.
    #[test]
    fn an_alpha_only_target_takes_the_shaders_alpha_on_desktop_gl() {
        const OPAQUE_BLACK_FS: &str = "FRAG\nDCL OUT[0], COLOR\nIMM[0] FLT32 { 0.0, 0.0, 0.0, 1.0 }\n  \
                                       0: MOV OUT[0], IMM[0]\n  1: END\n";
        let pixels = draw_over_target(OneDraw {
            host_gl: HostGl::Desktop,
            format: "A8_UNORM",
            clear: [0.0; 4],
            fs: Some(OPAQUE_BLACK_FS),
            vs: None,
            consts: &[],
            pipeline: None,
            more: None,
            logicop: None,
            tess: None,
        })
        .expect("the draw runs")
        .expect("no tessellation asked for");
        assert_eq!(pixels.len(), 16 * 16, "one byte a pixel");
        assert!(pixels.iter().all(|b| *b == 0xff), "{:?}", &pixels[..4]);
    }

    /// A clear of an alpha-only target on desktop GL clears its red, which stores the alpha, to
    /// the clear's alpha. Black at full alpha tells that from a clear of red to red.
    #[test]
    fn an_alpha_only_target_clears_to_the_clears_alpha_on_desktop_gl() {
        let pixels = draw_over_target(OneDraw {
            host_gl: HostGl::Desktop,
            format: "A8_UNORM",
            clear: [0.0, 0.0, 0.0, 1.0],
            fs: None,
            vs: None,
            consts: &[],
            pipeline: None,
            more: None,
            logicop: None,
            tess: None,
        })
        .expect("the clear runs")
        .expect("no tessellation asked for");
        assert!(pixels.iter().all(|b| *b == 0xff), "{:?}", &pixels[..4]);
    }

    /// A display of ours makes contexts in the GL the caller asked for, and init reads back which
    /// arrived: GLES by default, a core-profile desktop context when desktop is asked for.
    /// The guest is given one image count for the vertex, geometry and tessellation stages, and
    /// uses it in each: a stage that takes fewer fails the guest's link. So the count is no more
    /// than any of those stages takes, as the driver tells it here.
    #[test]
    fn the_image_count_for_the_other_stages_fits_each_of_them() {
        use crate::vrend::gl::gles::*;
        let _display = crate::vrend::one_display_at_a_time();
        struct Discard;
        impl crate::fence::FenceSink for Discard {
            fn context_fence(&mut self, _: ContextId, _: RingIdx, _: FenceId) {}
            fn present_fence(&mut self, _: FenceId) {}
            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        for host_gl in [HostGl::Gles, HostGl::Desktop] {
            let retire = crate::fence::Retirement::start(
                Box::new(Discard),
                crate::vrend::debug::Switches::default(),
            );
            let v = Vrend::new(
                Config { host_gl, ..Config::default() },
                &crate::budget::Budget::with_cap(None, false),
                retire.handle(),
                None,
                crate::vrend::resource::Condemned::default(),
                crate::vrend::debug::Traces::default(),
                crate::vrend::debug::Switches::default(),
            )
            .expect("vrend comes up");
            if !v.features.has(Feature::images) {
                continue;
            }
            let advertised = v.caps().max_shader_image_other_stages;
            let mut stages = vec![("vertex", GL_MAX_VERTEX_IMAGE_UNIFORMS)];
            if v.features.has(Feature::geometry_shader) {
                stages.push(("geometry", GL_MAX_GEOMETRY_IMAGE_UNIFORMS));
            }
            if v.features.has(Feature::tessellation) {
                stages.push(("tessellation control", GL_MAX_TESS_CONTROL_IMAGE_UNIFORMS));
                stages.push(("tessellation evaluation", GL_MAX_TESS_EVALUATION_IMAGE_UNIFORMS));
            }
            for (stage, name) in stages {
                let takes = v.gl.get_integer(name).max(0) as u32;
                eprintln!("{host_gl:?}: the {stage} stage takes {takes} images");
                assert!(
                    advertised <= takes,
                    "{host_gl:?}: {advertised} images offered, the {stage} stage takes {takes}"
                );
            }
        }
    }

    /// A texture buffer larger than the host's texel limit is legal GL, and is sized as the limit.
    /// Under a limit of 4, a view of 8 texels -- through a sampler and through an image -- reads
    /// back 4 texels from the shader, where it was refused and the context poisoned.
    #[test]
    fn a_texture_buffer_past_the_texel_limit_is_sized_as_the_limit() {
        use crate::vrend::pipe::{ImageAccess, Swizzle};
        use crate::vrend::proto::{Command, Object, SamplerView, ShaderImage};
        // The size, 4, scaled to 0.25: red 64.
        const VIEW: &str = "FRAG\nDCL OUT[0], COLOR\nDCL SAMP[0]\nDCL SVIEW[0], BUFFER, FLOAT\n\
                            DCL TEMP[0]\nIMM[0] INT32 { 0, 0, 0, 0 }\n\
                            IMM[1] FLT32 { 0.0625, 0.0, 0.0, 1.0 }\n  \
                            0: TXQ TEMP[0], IMM[0].xxxx, SAMP[0], BUFFER\n  \
                            1: I2F TEMP[0], TEMP[0]\n  2: MUL OUT[0], TEMP[0].xxxx, IMM[1]\n  \
                            3: END\n";
        const IMAGE: &str = "FRAG\nDCL OUT[0], COLOR\n\
                             DCL IMAGE[0], BUFFER, PIPE_FORMAT_R32_FLOAT, WR\nDCL TEMP[0]\n\
                             IMM[0] FLT32 { 0.0625, 0.0, 0.0, 1.0 }\n  \
                             0: RESQ TEMP[0].x, IMAGE[0]\n  1: I2F TEMP[0], TEMP[0]\n  \
                             2: MUL OUT[0], TEMP[0].xxxx, IMM[0]\n  3: END\n";
        let r32f = super::super::proto::Format::from_wire(28).expect("R32_FLOAT");
        let texels = ResourceHandle::new(10).expect("a resource handle is non-zero");
        let view = ObjectHandle::new(20).expect("non-zero");
        let more = More {
            resources: vec![(texels, buffer_args(resource::Bind::SAMPLER_VIEW, 32))],
            before: vec![
                Command::CreateObject {
                    handle: view,
                    object: Object::SamplerView(SamplerView {
                        resource: texels,
                        format: r32f,
                        target: TextureTarget::Buffer,
                        first_element_or_layers: 0,
                        last_element_or_levels: 7,
                        swizzle: [Swizzle::X, Swizzle::Y, Swizzle::Z, Swizzle::W],
                    }),
                },
                Command::SetSamplerViews {
                    stage: ShaderStage::Fragment,
                    start_slot: 0,
                    views: vec![Some(view)],
                },
                Command::SetShaderImages {
                    stage: ShaderStage::Fragment,
                    start_slot: 0,
                    images: vec![Some(ShaderImage {
                        format: r32f,
                        access: ImageAccess::ReadWrite,
                        layer_offset: 0,
                        level_size: 32,
                        resource: texels,
                    })],
                },
            ],
            texel_limit: Some(4),
            ..Default::default()
        };
        for host_gl in [HostGl::Gles, HostGl::Desktop] {
            if !serves(host_gl, Feature::arb_or_gles_ext_texture_buffer) {
                eprintln!("{host_gl:?}: no texture buffers");
                continue;
            }
            for (what, fs) in [("sampler view", VIEW), ("image", IMAGE)] {
                let pixels = draw_over_target(OneDraw {
                    host_gl,
                    format: "R8G8B8A8_UNORM",
                    vs: None,
                    fs: Some(fs),
                    consts: &[],
                    clear: [0.0; 4],
                    logicop: None,
                    tess: None,
                    pipeline: None,
                    more: Some(&more),
                })
                .unwrap_or_else(|e| panic!("{host_gl:?}: the {what} draws: {e:?}"))
                .expect("no tessellation asked for");
                assert!(
                    pixels.as_chunks::<4>().0.iter().all(|p| *p == [64, 0, 0, 255]),
                    "{host_gl:?}: the {what} is sized as the limit: {:?}",
                    &pixels[..4]
                );
            }
        }
    }

    /// A guest samples the RGB32 formats only from texture buffers, so they are offered only
    /// where such a buffer works: here, one is made and sampled wherever RGB32F is offered.
    #[test]
    fn rgb32_is_offered_only_where_its_texture_buffer_samples() {
        use crate::vrend::pipe::Swizzle;
        use crate::vrend::proto::{Box3, Command, Object, SamplerView, Transfer};
        const FS: &str = "FRAG\nDCL OUT[0], COLOR\nDCL SAMP[0]\nDCL SVIEW[0], BUFFER, FLOAT\n\
                          DCL TEMP[0]\nIMM[0] INT32 { 0, 0, 0, 0 }\n  \
                          0: TXF TEMP[0], IMM[0], SAMP[0], BUFFER\n  1: MOV OUT[0], TEMP[0]\n  \
                          2: END\n";
        let rgb32f = (0..crate::vrend::proto::FORMAT_MAX)
            .filter_map(crate::vrend::proto::Format::from_wire)
            .find(|f| f.name() == "R32G32B32_FLOAT")
            .expect("a wire format");
        let texels = ResourceHandle::new(10).expect("a resource handle is non-zero");
        let red = [1.0f32, 0.0, 0.0].map(f32::to_bits);
        let more = More {
            resources: vec![(texels, buffer_args(resource::Bind::SAMPLER_VIEW, 12))],
            before: vec![
                Command::ResourceInlineWrite {
                    transfer: Transfer {
                        resource: texels,
                        level: 0,
                        usage: 0,
                        stride: 0,
                        layer_stride: 0,
                        region: Box3 { x: 0, y: 0, z: 0, width: 12, height: 1, depth: 1 },
                    },
                    data: &red,
                },
                Command::CreateObject {
                    handle: ObjectHandle::new(20).expect("non-zero"),
                    object: Object::SamplerView(SamplerView {
                        resource: texels,
                        format: rgb32f,
                        target: TextureTarget::Buffer,
                        first_element_or_layers: 0,
                        last_element_or_levels: 0,
                        swizzle: [Swizzle::X, Swizzle::Y, Swizzle::Z, Swizzle::W],
                    }),
                },
                Command::SetSamplerViews {
                    stage: ShaderStage::Fragment,
                    start_slot: 0,
                    views: vec![Some(ObjectHandle::new(20).expect("non-zero"))],
                },
            ],
            ..Default::default()
        };
        for host_gl in [HostGl::Gles, HostGl::Desktop] {
            if !offered(host_gl, rgb32f) {
                eprintln!("{host_gl:?}: RGB32F is not offered");
                continue;
            }
            let pixels = draw_over_target(OneDraw {
                host_gl,
                format: "R8G8B8A8_UNORM",
                vs: None,
                fs: Some(FS),
                consts: &[],
                clear: [0.0; 4],
                logicop: None,
                tess: None,
                pipeline: None,
                more: Some(&more),
            })
            .unwrap_or_else(|e| panic!("{host_gl:?}: RGB32F is offered, but its buffer: {e:?}"))
            .expect("no tessellation asked for");
            assert!(
                pixels.as_chunks::<4>().0.iter().all(|p| *p == [0xff, 0, 0, 0xff]),
                "{host_gl:?}: the texture buffer's red was sampled: {:?}",
                &pixels[..4]
            );
        }
    }

    /// A format is offered to the guest only where the driver stores every value of it exactly.
    /// A driver may hold a sized format at another depth -- zink on KosmicKrisp holds 3-3-2 as
    /// 5-6-5 -- and a guest told the host holds it packs values the host then rounds. Each format
    /// offered is made again here and the driver asked what it stores.
    #[test]
    fn every_format_offered_is_stored_exactly() {
        use crate::vrend::gl::gles::GL_TEXTURE_2D;
        let _display = crate::vrend::one_display_at_a_time();
        struct Discard;
        impl crate::fence::FenceSink for Discard {
            fn context_fence(&mut self, _: ContextId, _: RingIdx, _: FenceId) {}
            fn present_fence(&mut self, _: FenceId) {}
            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        for host_gl in [HostGl::Gles, HostGl::Desktop] {
            let retire = crate::fence::Retirement::start(
                Box::new(Discard),
                crate::vrend::debug::Switches::default(),
            );
            let v = Vrend::new(
                Config { host_gl, ..Config::default() },
                &crate::budget::Budget::with_cap(None, false),
                retire.handle(),
                None,
                crate::vrend::resource::Condemned::default(),
                crate::vrend::debug::Traces::default(),
                crate::vrend::debug::Switches::default(),
            )
            .expect("vrend comes up");
            let api = v.features.api();
            if api.is_gles() && !api.gles_at_least(31) {
                continue;
            }
            let (mut offered, mut withheld) = (0, vec![]);
            for f in (0..crate::vrend::proto::FORMAT_MAX)
                .filter_map(crate::vrend::proto::Format::from_wire)
            {
                let Some(entry) = v.formats.get(f) else { continue };
                let Some(d) = f.describe() else { continue };
                if d.is_compressed() || !v.caps().v1.sampler.has(f) && entry.stores_exactly {
                    continue;
                }
                let tex = v.gl.gen_texture();
                v.gl.bind_texture(GL_TEXTURE_2D, Some(tex));
                let g = entry.gl;
                v.gl.tex_image_2d_null(
                    GL_TEXTURE_2D,
                    0,
                    g.internalformat,
                    4,
                    4,
                    g.glformat,
                    g.gltype,
                );
                let held = v.gl.channel_bits(GL_TEXTURE_2D, 0);
                v.gl.bind_texture(GL_TEXTURE_2D, None);
                v.gl.delete_texture(tex);
                let exact = crate::vrend::formats::holds_exactly(d, held);
                if v.caps().v1.sampler.has(f) {
                    assert!(exact, "{host_gl:?}: {} is offered but stored as {held:?}", f.name());
                    offered += 1;
                } else {
                    assert!(!exact, "{host_gl:?}: {} is stored exactly as {held:?}", f.name());
                    withheld.push(f.name());
                }
            }
            eprintln!("{host_gl:?}: {offered} offered, withheld as rounded: {withheld:?}");
        }
    }

    #[test]
    fn the_host_gl_is_the_one_asked_for() {
        let _display = crate::vrend::one_display_at_a_time();
        struct Discard;
        impl crate::fence::FenceSink for Discard {
            fn context_fence(&mut self, _: ContextId, _: RingIdx, _: FenceId) {}
            fn present_fence(&mut self, _: FenceId) {}

            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        let up = |host_gl| {
            let retire = crate::fence::Retirement::start(
                Box::new(Discard),
                crate::vrend::debug::Switches::default(),
            );
            let v = Vrend::new(
                Config { host_gl, ..Config::default() },
                &crate::budget::Budget::with_cap(None, false),
                retire.handle(),
                None,
                crate::vrend::resource::Condemned::default(),
                crate::vrend::debug::Traces::default(),
                crate::vrend::debug::Switches::default(),
            )
            .expect("vrend comes up");
            let caps = v.caps();
            // What the version and the extensions granted, before procs were reconciled.
            let granted = Features::probe(v.features.api(), v.gl.extensions());
            let withdrawn: Vec<_> =
                v.gl.missing_procs()
                    .into_iter()
                    .filter(|(f, _)| granted.has(*f))
                    .map(|(_, name)| name)
                    .collect();
            let bgra = (0..crate::vrend::proto::FORMAT_MAX)
                .filter_map(crate::vrend::proto::Format::from_wire)
                .find(|f| f.name() == "B8G8R8A8_UNORM")
                .and_then(|f| v.formats.get(f))
                .expect("every host has BGRA");
            let bgra = (bgra.gl.glformat, bgra.stores_bgra_as_rgba());
            (
                (v.features.api(), bgra),
                v.gl.table().has_glGetTexImage(),
                caps.capability_bits & caps::cap::HOST_IS_GLES != 0,
                caps.v1.glsl_level,
                withdrawn,
            )
        };
        let ((gles, gles_bgra), _, gles_says_gles, _, _) = up(HostGl::Gles);
        assert!(gles.is_gles(), "GLES unless asked otherwise");
        assert_eq!(
            gles_bgra,
            (super::gl::gles::GL_RGBA, true),
            "GLES keeps BGRA as RGBA and swaps on transfer"
        );
        assert!(gles_says_gles, "and the guest is told so");
        let ((desktop, desktop_bgra), get_tex_image, desktop_says_gles, glsl, withdrawn) =
            up(HostGl::Desktop);
        assert_eq!(
            desktop_bgra,
            (super::gl::gles::GL_BGRA, false),
            "desktop GL's own format groups, where BGRA is native and no transfer swaps it"
        );
        assert!(matches!(desktop, Api::Gl(v) if v >= 33), "a core desktop context: {desktop}");
        assert!(get_tex_image, "and the one table resolves desktop GL's own entry points");
        // The guest picks its GLSL and its fp64 emulation from these two.
        assert!(!desktop_says_gles, "a desktop host does not call itself GLES");
        assert!(glsl >= 330, "a desktop host's GLSL is its GL version's: {glsl}");
        // A feature desktop GL grants is reached through desktop GL's spelling of its procs, not
        // withdrawn for want of the GLES one.
        assert!(withdrawn.is_empty(), "granted on desktop, then withdrawn: {withdrawn:?}");
    }

    /// What the vertex stage holds, and the stage of the shader the table holds under handle 3.
    type ShaderSlot = (Option<Option<ObjectHandle>>, Option<ShaderStage>);

    /// Run `commands` on a fresh context, then rebuild it from its journal, and answer what the
    /// vertex stage holds and what the table holds under handle 3, live and rebuilt, with the
    /// number of commands the rebuild dropped.
    fn shader_slot_across_a_rebuild(
        commands: &[crate::vrend::proto::Command<'_>],
    ) -> (ShaderSlot, ShaderSlot, u64) {
        use crate::vrend::encode::encode;
        let _display = crate::vrend::one_display_at_a_time();
        struct Discard;
        impl crate::fence::FenceSink for Discard {
            fn context_fence(&mut self, _: ContextId, _: RingIdx, _: FenceId) {}
            fn present_fence(&mut self, _: FenceId) {}

            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        let retire = crate::fence::Retirement::start(
            Box::new(Discard),
            crate::vrend::debug::Switches::default(),
        );
        let mut v = Vrend::new(
            Config::default(),
            &crate::budget::Budget::with_cap(None, false),
            retire.handle(),
            None,
            crate::vrend::resource::Condemned::default(),
            crate::vrend::debug::Traces::default(),
            crate::vrend::debug::Switches::default(),
        )
        .expect("vrend comes up");
        let mut wire = Vec::new();
        for c in commands {
            encode(c, &mut wire);
        }
        let ctx = ClassicCtx::for_test(ContextId::new(1).expect("a context id"));
        v.context_create(ctx, &NoGuest).expect("a context");
        v.submit(ctx, &wire, &NoGuest).expect("the context is here").expect("accepted");
        let h = ObjectHandle::new(3).expect("non-zero");
        let seen = |v: &Vrend| {
            let c = &v.contexts[&ctx.id()];
            (c.bound_shader(ShaderStage::Vertex), c.shader_in_table(h))
        };
        let live = seen(&v);

        let journal = v.journal_export(ctx).expect("a live context exports its journal");
        v.context_destroy(ctx, &NoGuest);
        v.context_create(ctx, &NoGuest).expect("a fresh context to rebuild");
        v.replay_begin(ctx).expect("the context is here");
        v.journal_restore(ctx, &journal).expect("the journal is taken");
        v.replay_upto(ctx, &NoGuest, Seq(u64::MAX)).expect("and fed");
        let dropped = v.contexts.get_mut(&ctx.id()).expect("the context").replay_end();
        let rebuilt = seen(&v);
        v.context_destroy(ctx, &NoGuest);
        (live, rebuilt, dropped)
    }

    /// A whole shader of `stage` under handle 3, and the text it is made from.
    fn shader_under_3(stage: ShaderStage, text: &[u32]) -> crate::vrend::proto::Command<'_> {
        use crate::vrend::proto::{
            Command, Object, ShaderChunk, ShaderCreate, ShaderKind, StreamOutput,
        };
        Command::CreateObject {
            handle: ObjectHandle::new(3).expect("non-zero"),
            object: Object::Shader(ShaderCreate {
                stage,
                chunk: ShaderChunk::New { total_bytes: text.len() as u32 * 4 },
                num_tokens: 100,
                kind: ShaderKind::Graphics { stream_output: StreamOutput::default() },
                text,
            }),
        }
    }

    /// TGSI text as the guest sends it: dword-packed, NUL-terminated.
    fn tgsi_words(text: &str) -> Vec<u32> {
        let mut bytes = text.as_bytes().to_vec();
        bytes.push(0);
        bytes.resize(bytes.len().div_ceil(4) * 4, 0);
        bytes.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    const PASS_VS: &str =
        "VERT\nDCL IN[0]\nDCL OUT[0], POSITION\n  0: MOV OUT[0], IN[0]\n  1: END\n";
    /// Triangles, each vertex the patch's corners weighted by the tessellation coordinate.
    const PASS_TES: &str = "TESS_EVAL\nPROPERTY TES_PRIM_MODE 4\nPROPERTY TES_SPACING 2\n\
                            PROPERTY TES_VERTEX_ORDER_CW 0\nPROPERTY TES_POINT_MODE 0\n\
                            DCL IN[][0], POSITION\nDCL SV[0], TESSCOORD\nDCL OUT[0], POSITION\n\
                            DCL TEMP[0]\n  0: MUL TEMP[0], IN[0][0], SV[0].xxxx\n  \
                            1: MAD TEMP[0], IN[1][0], SV[0].yyyy, TEMP[0]\n  \
                            2: MAD OUT[0], IN[2][0], SV[0].zzzz, TEMP[0]\n  3: END\n";
    const RED_FS: &str = "FRAG\nDCL OUT[0], COLOR\nIMM[0] FLT32 { 1.0, 0.0, 0.0, 1.0 }\n  \
                          0: MOV OUT[0], IMM[0]\n  1: END\n";

    /// A shader the guest destroys while it is bound stays bound, as the C's reference keeps it,
    /// and a rebuild must reach the same place: bound, and gone from the table. The bind alone
    /// names a handle the journal no longer creates.
    #[test]
    fn a_shader_destroyed_while_bound_is_rebuilt_bound_and_destroyed() {
        use crate::vrend::proto::{Command, ObjectType};
        let h = ObjectHandle::new(3).expect("non-zero");
        let vs = tgsi_words(PASS_VS);
        let (live, rebuilt, dropped) = shader_slot_across_a_rebuild(&[
            shader_under_3(ShaderStage::Vertex, &vs),
            Command::BindShader { stage: ShaderStage::Vertex, handle: Some(h) },
            Command::DestroyObject { kind: ObjectType::Shader, handle: h },
            // Ignored, as the C ignores it: the handle names no shader now.
            Command::BindShader { stage: ShaderStage::Vertex, handle: Some(h) },
        ]);
        assert_eq!(live, (Some(None), None), "the slot holds the destroyed shader");
        assert_eq!(dropped, 0, "a rebuild of what the context holds drops nothing");
        assert_eq!(rebuilt, live, "and holds what it held");
    }

    /// A create under the handle of a bound shader replaces it in the table and leaves it bound.
    /// The rebuild must free the handle before that create, not after, or it frees the new object.
    #[test]
    fn a_create_over_a_bound_shaders_handle_keeps_both_across_a_rebuild() {
        use crate::vrend::proto::Command;
        let h = ObjectHandle::new(3).expect("non-zero");
        let (vs, fs) = (tgsi_words(PASS_VS), tgsi_words(RED_FS));
        let (live, rebuilt, dropped) = shader_slot_across_a_rebuild(&[
            shader_under_3(ShaderStage::Vertex, &vs),
            Command::BindShader { stage: ShaderStage::Vertex, handle: Some(h) },
            shader_under_3(ShaderStage::Fragment, &fs),
        ]);
        assert_eq!(
            live,
            (Some(None), Some(ShaderStage::Fragment)),
            "the vertex shader stays bound; the handle names the fragment shader"
        );
        assert_eq!(dropped, 0, "a rebuild of what the context holds drops nothing");
        assert_eq!(rebuilt, live, "and holds what it held");
    }

    /// A view names its resource by handle, and the guest may free that handle and reuse it for a
    /// resource of the other kind while the view lives. A view made over a texture carries no
    /// element range -- its dwords are a layer range, here one reading as a plane index -- so
    /// binding it once the handle names a buffer is refused, rather than read as a range nothing
    /// checked.
    #[test]
    fn a_texture_view_is_refused_once_its_handle_names_a_buffer() {
        use crate::vrend::context::Fault;
        use crate::vrend::encode::encode;
        use crate::vrend::pipe::{ShaderStage, Swizzle};
        use crate::vrend::proto::{Cmd, Command, Object, SamplerView};
        let _display = crate::vrend::one_display_at_a_time();
        struct Discard;
        impl crate::fence::FenceSink for Discard {
            fn context_fence(&mut self, _: ContextId, _: RingIdx, _: FenceId) {}
            fn present_fence(&mut self, _: FenceId) {}

            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        struct AllAttached;
        impl Guest for AllAttached {
            fn attached(&self, _: ContextId, _: ResourceHandle) -> bool {
                true
            }
            fn pages(&self, _: ContextId, _: ResourceHandle) -> Option<Iov<'_>> {
                None
            }
            fn blob_pixels(&self, _: ContextId, _: ResourceHandle) -> Option<PixelSource<'_>> {
                None
            }
        }
        let retire = crate::fence::Retirement::start(
            Box::new(Discard),
            crate::vrend::debug::Switches::default(),
        );
        let condemned = crate::vrend::resource::Condemned::default();
        let mut v = Vrend::new(
            Config::default(),
            &crate::budget::Budget::with_cap(None, false),
            retire.handle(),
            None,
            condemned.clone(),
            crate::vrend::debug::Traces::default(),
            crate::vrend::debug::Switches::default(),
        )
        .expect("vrend comes up");
        let bgra = super::super::proto::Format::from_wire(1).expect("B8G8R8A8_UNORM");
        let res = ResourceHandle::new(1).expect("a resource handle is non-zero");
        let args = |target, bind, width| resource::Args {
            target,
            format: bgra,
            bind,
            width,
            height: 1,
            depth: 1,
            array_size: 1,
            last_level: 0,
            nr_samples: 0,
            flags: resource::ResourceFlags(0),
        };
        v.resource_create(res, args(TextureTarget::Texture2d, resource::Bind::SAMPLER_VIEW, 16))
            .expect("a texture");
        let ctx = ClassicCtx::for_test(ContextId::new(1).expect("a context id"));
        v.context_create(ctx, &AllAttached).expect("a context");
        let view = crate::vrend::proto::ObjectHandle::new(5).expect("non-zero");
        let mut wire = Vec::new();
        let object = Object::SamplerView(SamplerView {
            resource: res,
            format: bgra,
            target: TextureTarget::Texture2d,
            // Layers 1 to 0: a plane index on a planar texture, spent on this one. Read as
            // elements, the range ends before it begins.
            first_element_or_layers: 1,
            last_element_or_levels: 0,
            swizzle: [Swizzle::X, Swizzle::Y, Swizzle::Z, Swizzle::W],
        });
        encode(&Command::CreateObject { handle: view, object }, &mut wire);
        v.submit(ctx, &wire, &AllAttached).expect("the context is here").expect("a view");

        // The renderer's table lets the texture go, and the guest reuses its handle for a buffer.
        drop(crate::vrend::resource::Claim::new(res, &condemned));
        v.resource_create(res, args(TextureTarget::Buffer, resource::Bind::SAMPLER_VIEW, 64))
            .expect("a texture buffer under the freed handle");

        wire.clear();
        let stage = ShaderStage::Fragment;
        encode(
            &Command::SetSamplerViews { stage, start_slot: 0, views: vec![Some(view)] },
            &mut wire,
        );
        let refused = v.submit(ctx, &wire, &AllAttached).expect("the context is here");
        assert!(
            matches!(
                refused,
                Err(Fault::IllegalResource { cmd: Cmd::SetSamplerViews, handle }) if handle == res
            ),
            "a view made over a texture does not bind a buffer: {refused:?}"
        );
        v.context_destroy(ctx, &AllAttached);
    }

    /// A read from the control queue -- here the texture a scanout is flushed from -- waits for
    /// the picture decoding into it, and leaves nothing pending behind.
    #[test]
    fn a_control_queue_read_waits_for_the_picture_in_flight() {
        use super::super::video::pending::{Landing, Outcome, Pending, Recipe};

        let _display = crate::vrend::one_display_at_a_time();
        struct Discard;
        impl crate::fence::FenceSink for Discard {
            fn context_fence(&mut self, _: ContextId, _: RingIdx, _: FenceId) {}
            fn present_fence(&mut self, _: FenceId) {}

            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        let retire = crate::fence::Retirement::start(
            Box::new(Discard),
            crate::vrend::debug::Switches::default(),
        );
        let mut v = Vrend::new(
            Config::default(),
            &crate::budget::Budget::with_cap(None, false),
            retire.handle(),
            None,
            crate::vrend::resource::Condemned::default(),
            crate::vrend::debug::Traces::default(),
            crate::vrend::debug::Switches::default(),
        )
        .expect("vrend comes up");

        let handle = ResourceHandle::new(1).expect("a resource handle is non-zero");
        let args = resource::Args {
            target: TextureTarget::Texture2d,
            format: super::super::proto::Format::from_wire(1).expect("B8G8R8A8_UNORM"),
            bind: resource::Bind(1 << 1),
            width: 64,
            height: 64,
            depth: 1,
            array_size: 1,
            last_level: 0,
            nr_samples: 0,
            flags: resource::ResourceFlags(0),
        };
        v.resource_create(handle, args).expect("an ordinary texture");
        let texture = v
            .resources
            .sync()
            .get(&handle)
            .and_then(resource::Slot::resource)
            .and_then(Resource::texture)
            .cloned()
            .expect("a texture resource");

        let landing = Landing::new();
        texture.expect_decode(
            &v.gl,
            Pending::new(Arc::clone(&landing), Recipe::Composite, &v.unsettled),
        );
        let lander = {
            let landing = Arc::clone(&landing);
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(40));
                landing.land(Outcome::Nothing);
            })
        };
        assert!(v.resource_texture(handle).is_some());
        assert!(landing.is_landed(), "the read returned before the picture landed");
        assert!(!v.unsettled.any(), "the read left the picture pending");
        lander.join().expect("the lander finishes");
    }

    /// A batch for a context at the fence-depth bound runs only once one of its fences has
    /// retired -- and not before, however cheap the batch.
    ///
    /// The waiter is held on a picture that lands on another thread, so the context's one fence
    /// stays in flight for a known time. Without the bound the batch runs at once and finds it
    /// still queued; with it, the batch waits for the landing.
    #[test]
    fn a_batch_waits_while_its_context_is_at_the_fence_depth() {
        use super::super::video::pending::{Landing, Outcome};

        let _display = crate::vrend::one_display_at_a_time();
        struct Discard;
        impl crate::fence::FenceSink for Discard {
            fn context_fence(&mut self, _: ContextId, _: RingIdx, _: FenceId) {}
            fn present_fence(&mut self, _: FenceId) {}
            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        let retire = crate::fence::Retirement::start(
            Box::new(Discard),
            crate::vrend::debug::Switches::default(),
        );
        let mut v = Vrend::new(
            Config::default(),
            &crate::budget::Budget::with_cap(None, false),
            retire.handle(),
            None,
            crate::vrend::resource::Condemned::default(),
            crate::vrend::debug::Traces::default(),
            crate::vrend::debug::Switches::default(),
        )
        .expect("vrend comes up");
        assert!(v.waiter.is_some(), "without a waiter nothing is ever in flight");
        v.fence_depth = 1;

        let ctx = ClassicCtx::for_test(ContextId::new(1).expect("a context id"));
        v.context_create(ctx, &NoGuest).expect("a context");

        // Ahead of the context's fence in the waiter's FIFO: nothing behind it retires until it
        // lands.
        let landing = Landing::new();
        v.waiter.as_ref().expect("checked above").retire_global(
            Owed {
                pictures: vec![Arc::clone(&landing)],
                fence: Answer::Ordered,
                ticket: None,
                queries: false,
            },
            ClientFenceId(1),
        );
        v.fence_context(ctx, RingIdx(0), FenceId(2), &NoGuest);
        let gate = v.contexts.get(&ctx.id()).expect("the context").in_flight().clone();
        assert_eq!(gate.queued(), 1, "a fence answered by syncs counts as in flight");

        let hold = std::time::Duration::from_millis(300);
        let lander = {
            let landing = Arc::clone(&landing);
            std::thread::spawn(move || {
                std::thread::sleep(hold);
                landing.land(Outcome::Nothing);
            })
        };
        let began = std::time::Instant::now();
        // An empty batch: whatever it costs, it is not what is being timed.
        v.submit(ctx, &[], &NoGuest)
            .expect("the context takes the batch")
            .expect("an empty batch runs");
        let took = began.elapsed();
        assert_eq!(gate.queued(), 0, "the batch ran while its context was at the bound");
        assert!(
            took >= hold / 2,
            "the batch returned after {took:?}, before the fence could retire"
        );

        lander.join().expect("the lander finishes");
        v.context_destroy(ctx, &NoGuest);
    }
}
