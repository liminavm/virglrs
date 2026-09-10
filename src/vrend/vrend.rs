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
use super::context::{Context, Current, Fault, GlContext, Guest, Host, Pending, Todo};
use super::egl::{self, EglError, Flavour, GlContexts, Version, Winsys};
use super::features::{Feature, Features};
use super::formats::Table;
use super::gl::gles::GL_VERSION;
use super::gl::{self, Gl};
use super::journal::{Census, Seq};
use super::pipe::TextureTarget;
use super::resource::{self, Args, Limits, Refusal, Resource};
use super::shader;
use super::tally;
use super::transfer::{self, Info};
use super::waiter::{self, Answer};
use crate::config::Config;
use crate::decode;
use crate::guest_mem::{Iov, PixelSource};
use crate::ids::{BlobId, ClientFenceId, ContextId, FenceId, ResourceHandle, RingIdx};
use crate::surface;
use std::fmt;
use std::sync::Arc;

/// Why the classic renderer could not come up.
#[derive(Debug)]
pub enum InitError {
    Egl(EglError),
    /// No GLES 3.x context could be made.
    NoContext,
    /// The context that was made is not GLES, carrying its `GL_VERSION`. Only reachable when
    /// something else minted it: this renderer asks its own display for GLES and gets it or
    /// nothing.
    NotGles(String),
}

impl fmt::Display for InitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InitError::Egl(e) => write!(f, "{e}"),
            InitError::NoContext => f.write_str("no GLES 3.x context could be created"),
            InitError::NotGles(v) => write!(
                f,
                "the embedder's GL context is {v}, and this renderer translates for GLES only"
            ),
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
    resources: crate::Map<ResourceHandle, resource::Slot>,
    contexts: crate::Map<ContextId, Context>,
    pub todo: Todo,
    /// What the command path costs per guest command. Inert unless armed -- see
    /// [`tally::Tally`], which says why a profiler cannot answer this.
    tally: tally::Tally,
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
    /// Batches run, ever. The unit a copy of a guest's pages is kept fresh in: within one batch
    /// the guest has had no opportunity to run, so one read serves every draw in it.
    batch: u64,
    /// Classic's handle to the renderer's host-memory ledger -- see [`crate::budget`]. What it
    /// charges is the IOSurfaces this arm mints and nothing else; ordinary GL storage is the
    /// driver's and this process cannot see it.
    budget: crate::budget::Classic,
}

/// The versions tried, newest first -- the GLES rows of the C's `gl_versions` ladder.
const VERSIONS: [Version; 3] = [
    Version { major: 3, minor: 2 },
    Version { major: 3, minor: 1 },
    Version { major: 3, minor: 0 },
];

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
    ) -> Result<Vrend, InitError> {
        let fences_for_inline = fences.clone();
        // An embedder's winsys arrives with ctx0 already made and current, because its display is
        // discovered through that context; ours is opened first and asked for one.
        let (winsys, ctx0, version) = match contexts {
            Some(contexts) => Winsys::embedded(Flavour::Gles, contexts, &VERSIONS)?,
            None => {
                let winsys = Winsys::open(Flavour::Gles)?;
                let mut made = None;
                for v in VERSIONS {
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
        // One fact, read from the configuration once: whether this host's GL needs a flush
        // behind a fence. Both this table and the waiter thread's are made with it.
        let fence_flush = if config.gl_fences_without_draining {
            gl::FenceFlush::Submits
        } else {
            gl::FenceFlush::Needed
        };
        let gl = Gl::new(winsys.gles(), fence_flush);
        let version_string = gl.get_string(GL_VERSION);
        // Whose choice the client API was depends on who minted the context, so it is read back
        // rather than assumed. A desktop-GL context parses as a plausible GLES number and serves
        // nothing: `4.6 (Core Profile)` would be read as "GLES 4.6", and every probe below it
        // would answer about an API this renderer does not translate for.
        if !version_string.starts_with("OpenGL ES ") {
            return Err(InitError::NotGles(version_string));
        }
        let gles_version = parse_gles_version(&version_string);
        let mut features = Features::probe(gles_version, gl.extensions());
        if !winsys.has_gl_colorspace() {
            features.clear(Feature::srgb_write_control);
        }
        features.reconcile(&gl);
        let limits = Limits::query(&gl, &features);
        let shader_cfg = shader::Config::probe(&gl, &features, &limits);
        let formats = Table::probe(&gl, &features);
        let video = config.video.then(decode::Support::probe);
        if let Some(support) = video {
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
        let caps = caps::CapsV2::probe(&gl, &features, &limits, &formats, video.as_ref());
        eprintln!(
            "[virglrs] vrend: {version_string} (gles {gles_version}), {} formats, {} features, \
             {}",
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
                    Gl::new(winsys.gles(), fence_flush),
                    fences,
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
            resources: crate::Map::default(),
            contexts: crate::Map::default(),
            todo: Todo::default(),
            tally: tally::Tally::from_env(),
            blitter: None,
            waiter,
            fences,
            doomed: Vec::new(),
            batch: 0,
            pixels: resource::Refresh::default(),
            budget: crate::budget::Classic::open(budget),
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
        self.current.switch_to(&self.winsys, &self.gl, GlContext::Ctx0, &self.ctx0);
    }

    /// Bind the GL context `which` names.
    ///
    /// `false` when it names something this renderer no longer has: a sub-context of a context that
    /// has been destroyed, which only an embedder-backed winsys can leave a [`Pending`] entry for
    /// (it cannot verify what is current, so it never takes the sync that would have retired the
    /// entry at the departure). Its work went with its GL context and there is nothing to fence.
    fn bind(&mut self, which: GlContext) -> bool {
        let Vrend { winsys, gl, current, ctx0, blitter, contexts, .. } = self;
        let gl_ctx = match which {
            GlContext::Ctx0 => Some(&*ctx0),
            GlContext::Blitter => blitter.as_ref().map(|b| b.context()),
            GlContext::Sub(id, sub) => contexts
                .get(&id)
                .and_then(|c| c.gl_contexts().find(|(s, _)| *s == sub).map(|(_, g)| g)),
        };
        match gl_ctx {
            Some(g) => {
                current.switch_to(winsys, gl, which, g);
                true
            }
            None => false,
        }
    }

    /// The host a context's commands run against, and the contexts beside it: two disjoint
    /// borrows of this renderer, so a context can run against the rest of it.
    fn split<'a>(
        &'a mut self,
        ctx: ContextId,
        guest: &'a dyn Guest,
    ) -> (Host<'a>, &'a mut crate::Map<ContextId, Context>) {
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
            // Neither belongs to a context's commands: the waiter is a thread, and the handle is
            // where a fence goes once answered.
            waiter: _,
            fences: _,
        } = self;
        let host = Host {
            batch: *batch,
            tally,
            budget,
            gl,
            winsys,
            version: *version,
            share: ctx0,
            features,
            formats,
            limits,
            shader_cfg,
            resources,
            pixels,
            guest,
            ctx,
            current,
            todo,
            blitter,
            video: video.as_ref(),
        };
        (host, contexts)
    }

    /// What has been retained for a rebuild: every live context's objects and current state,
    /// plus the type each live blob was given.
    ///
    /// The resources are counted here and not in `Context` because that is where they live -- one
    /// table shared by every context, so no single context can answer for it.
    pub fn journal_census(&self) -> Census {
        let mut c = Census::default();
        for ctx in self.contexts.values() {
            c += ctx.journal_census();
        }
        for wire in self.resource_preamble() {
            c.add_wire(wire.len(), true);
        }
        c
    }

    /// One context's journal, as the bytes the VMM stores and hands back.
    ///
    /// `None` for a context that is not here. An empty journal still serializes: a context that
    /// built nothing is a fact worth restoring accurately, and the alternative -- answering
    /// "no journal" -- is what the VMM reads as "this context is not mine to rebuild".
    pub fn journal_export(&self, id: ContextId) -> Option<Vec<u8>> {
        let ctx = self.contexts.get(&id)?;
        Some(crate::vrend::journal::serialize(&ctx.journal(self.resource_preamble())))
    }

    /// The commands a rebuild must send before anything else: what described each claimed blob,
    /// and what typed each attached one.
    ///
    /// They live on the resource table rather than on a context because one table serves every
    /// context, so no single context can walk it. Describe before type, per resource: a resource
    /// has at most one of the two -- a described one arrives with its shape, an attached one is
    /// told its shape later -- so the order between them never actually arises, and stating it is
    /// cheaper than relying on that staying true.
    fn resource_preamble(&self) -> impl Iterator<Item = &Vec<u32>> {
        self.resources
            .values()
            .filter_map(|s| s.resource())
            .flat_map(|r| r.described_by.as_ref().into_iter().chain(r.typed_by.as_ref()))
    }

    /// Each live context's journal: how many bytes it exports, and how many entries those bytes
    /// read back as.
    ///
    /// The round-trip is the point. A census counts what was retained, which a serializer bug
    /// would leave untouched; parsing our own output back is the cheapest thing that actually
    /// exercises the format on a real world rather than on a fixture we wrote.
    pub fn journal_report(&self) -> Vec<(ContextId, usize, Result<usize, &'static str>)> {
        self.contexts
            .keys()
            .filter_map(|id| {
                let bytes = self.journal_export(*id)?;
                let read_back = crate::vrend::journal::parse(&bytes).map(|e| e.len());
                Some((*id, bytes.len(), read_back))
            })
            .collect()
    }

    /// Begin rebuilding a classic context. `false` if it is not here.
    pub fn replay_begin(&mut self, id: ContextId) -> bool {
        match self.contexts.get_mut(&id) {
            Some(c) => {
                c.replay_begin();
                true
            }
            None => false,
        }
    }

    /// Hand a classic context the journal it will be rebuilt from.
    pub fn journal_restore(&mut self, id: ContextId, bytes: &[u8]) -> Result<usize, &'static str> {
        self.contexts.get_mut(&id).ok_or("no such context")?.replay_restore(bytes)
    }

    /// Feed a classic context's retained commands up to `upto`.
    pub fn replay_upto(&mut self, id: ContextId, guest: &dyn Guest, upto: Seq) -> bool {
        if !self.contexts.contains_key(&id) {
            return false;
        }
        let (mut host, contexts) = self.split(id, guest);
        contexts.get_mut(&id).expect("checked above").replay_upto(&mut host, upto);
        true
    }

    /// Finish rebuilding a classic context, and report what it could not use.
    pub fn replay_end(&mut self, id: ContextId) -> bool {
        match self.contexts.get_mut(&id) {
            Some(c) => {
                c.replay_end();
                true
            }
            None => false,
        }
    }

    // ---- contexts ----

    pub fn context_create(&mut self, id: ContextId, guest: &dyn Guest) -> Result<(), EglError> {
        let (mut host, contexts) = self.split(id, guest);
        let c = Context::new(&mut host)?;
        contexts.insert(id, c);
        Ok(())
    }

    pub fn context_destroy(&mut self, id: ContextId, guest: &dyn Guest) {
        let (mut host, contexts) = self.split(id, guest);
        if let Some(c) = contexts.remove(&id) {
            c.destroy(&mut host);
        }
        self.switch_ctx0();
        // The framebuffers that went with the context were the last holders of any storage the
        // guest freed while they drew into it.
        self.sweep_doomed();
    }

    pub fn has_context(&self, id: ContextId) -> bool {
        self.contexts.contains_key(&id)
    }

    /// Run a batch on a context. `None` for a context this renderer does not have.
    pub fn submit(
        &mut self,
        id: ContextId,
        words: &[u32],
        guest: &dyn Guest,
    ) -> Option<Result<(), Fault>> {
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

    // ---- resources ----

    /// Create the host side of a classic resource, on ctx0.
    pub fn resource_create(&mut self, handle: ResourceHandle, args: Args) -> Result<(), Refusal> {
        assert!(!self.resources.contains_key(&handle), "the renderer checked the handle was free");
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
        self.resources.insert(handle, resource::Slot::Resource(res));
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
        ctx: ContextId,
        blob: BlobId,
        handle: ResourceHandle,
        size: u64,
    ) -> Result<Args, ClaimRefused> {
        assert!(!self.resources.contains_key(&handle), "the renderer checked the handle was free");
        let mut res = self
            .contexts
            .get_mut(&ctx)
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
        self.resources.insert(handle, resource::Slot::Resource(res));
        Ok(args)
    }

    /// Where a claimed resource's buffer is mapped, and how far it runs.
    ///
    /// `None` for every resource that was never published to a guest, which is every ordinary
    /// classic one: the address exists only because [`Self::claim_described`] took it.
    pub fn resource_mapping(&self, handle: ResourceHandle) -> Option<(usize, u64)> {
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
    pub fn resource_attach_blob(
        &mut self,
        handle: ResourceHandle,
        storage: Option<surface::Adoptable>,
    ) {
        self.resources
            .entry(handle)
            .or_insert_with(|| resource::Slot::Untyped(resource::Untyped::new(storage)));
    }

    /// The IOSurface a resource is presented from, if its storage is one. Asked of the resource
    /// every time: the surface goes with the resource, and there is no other place to hold one.
    pub fn resource_surface(&self, handle: ResourceHandle) -> Option<&surface::Surface> {
        self.resources.get(&handle)?.resource()?.surface()
    }

    /// The GL texture a resource's storage is, when its storage is one.
    ///
    /// Asked of the resource every time rather than mirrored anywhere: the name is the texture's
    /// and dies with it, and a copy kept elsewhere would outlive the object it names.
    pub fn resource_texture(&self, handle: ResourceHandle) -> Option<gl::TextureName> {
        Some(self.resources.get(&handle)?.resource()?.texture()?.name)
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
        let res = self.resources.get(&handle)?.resource()?;
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
            vec![0u8; desc.size_2d(desc.stride(res.args.width), res.args.height) as usize];
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
        let (held, width, full_height) = {
            let res = self.resources.get(&handle)?.resource()?;
            (res.surface_share()?, res.args.width, res.args.height)
        };
        let image = match self.winsys.image_from_iosurface(held) {
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
            // Tightly packed, then re-pitched below: `glReadPixels` writes one image at the
            // pack state GL was told, and asking it to write at the caller's stride would make
            // the pack state a second place the row length is decided.
            //
            // `pack_tight` and not the alignment alone. `Gl::read_pixels` bounds its slice
            // against a tightly packed image and says so in its own SAFETY comment -- "with pack
            // row length zero and alignment 1, which every caller sets" -- so a stale
            // `GL_PACK_ROW_LENGTH` left on ctx0 by an earlier readback would have the driver
            // write past the end of `packed` on any host without `glReadnPixelsKHR`. Four fields
            // decide one fact, and this is the one place that names the fact.
            let mut packed = vec![0u8; width as usize * rows as usize * 4];
            self.gl.pack_tight();
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
    pub fn resource_surface_share(&self, handle: ResourceHandle) -> Option<Arc<dyn surface::Held>> {
        self.resources.get(&handle)?.resource()?.surface_share()
    }

    /// Answer a classic context fence: make it true that the GL work has run, and retire it.
    ///
    /// Retirement is queued behind the work rather than taken here, so this returns as soon as the
    /// fence is *taken* -- the caller is holding the renderer, and waiting under it is what made
    /// one heavy client slow down every other context.
    pub fn fence_context(&mut self, ctx: ContextId, ring: RingIdx, id: FenceId) {
        // With no waiter there is no queue to retire behind, so the fence is answered inline --
        // the way this renderer did before there was one. Taking a sync and dropping it unwaited
        // would retire the fence early, which is the whole hazard this path exists to prevent.
        if self.waiter.is_none() {
            self.finish_contexts(&[ctx]);
            self.fences.retire_context(ctx, ring, id);
            return;
        }
        let answer = self.take_fence(Some(ctx));
        if super::debug::enabled(super::debug::Switch::Fence) {
            eprintln!(
                "[virglrs] fence: context ctx={ctx:?} ring={ring:?} id={} answer={}",
                id.0,
                answer.name()
            );
        }
        let w = self.waiter.as_ref().expect("checked just above");
        w.retire_context(answer, ctx, ring, id);
    }

    /// Answer a fence on the legacy global ring, which names its context from outside.
    ///
    /// `on` is the context whose work the fence is for. `None` -- or a context this renderer does
    /// not have -- means it cannot be attributed to one, and the fence is answered by its place in
    /// the waiter's queue instead; see [`Self::take_fence`].
    pub fn fence_global(&mut self, on: Option<ContextId>, id: ClientFenceId) {
        if self.waiter.is_none() {
            self.finish_all();
            self.fences.retire_global(id);
            return;
        }
        let answer = self.take_fence(on);
        if super::debug::enabled(super::debug::Switch::Fence) {
            eprintln!("[virglrs] fence: global id={} on={on:?} answer={}", id.0, answer.name());
        }
        let w = self.waiter.as_ref().expect("checked just above");
        w.retire_global(answer, id);
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
        // `VIRGLRS_FENCE_FINISH=1` puts the old behaviour back -- every context finished inline,
        // on this thread -- so the two can be compared on one build the way the cost of the finish
        // was measured in the first place. Retirement still goes through the waiter's queue, so
        // the comparison changes what the fence costs and not what it means.
        static FORCE_FINISH: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let forced = *FORCE_FINISH
            .get_or_init(|| std::env::var("VIRGLRS_FENCE_FINISH").as_deref() == Ok("1"));
        if forced {
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
        //
        // **Almost none of those syncs are taken here.** One is taken when the thread *leaves* a
        // context, where the departure's release flush drains it anyway, and kept until a fence
        // comes for it (`Current::switch_to`). So this collects what is already waiting and binds
        // only what is not -- in the common case the one context the thread is still on, which
        // costs no switch at all because binding what is bound is a thread-local read. What this
        // replaced walked all five contexts, and the first departure from the loaded one cost a
        // median 589 microseconds of the worker, up to 12 milliseconds.
        let mut syncs = Vec::new();
        let mut refused = false;
        let loaded = self.current.on();
        // Pass one: every sync already taken. No bind, no flush, no walk.
        let mut to_bind: Vec<GlContext> = Vec::new();
        let subs: Vec<GlContext> = match self.contexts.get(&id) {
            Some(ctx) => ctx.gl_contexts().map(|(sub, _)| GlContext::Sub(id, sub)).collect(),
            None => Vec::new(),
        };
        // ctx0, where this renderer's own blits and transfers run -- and the blitter's own context,
        // whose work neither `finish_contexts` nor `finish_all` has ever waited for. That gap is
        // closed here for nothing: the blitter is left at the end of every blit, so a sync over it
        // is already in hand. It is covered for every context rather than only the one the blit was
        // for, because a blit's destination is a resource and a resource is not a context's.
        for which in subs.into_iter().chain([GlContext::Ctx0, GlContext::Blitter]) {
            match self.current.take(which) {
                Some(Pending::Synced(f)) => syncs.push(f),
                // Left without a verifiable context, so nothing names its work: bind and sync it
                // the expensive way. Only an embedder-backed winsys gets here.
                Some(Pending::NeedsSync) => to_bind.push(which),
                // Not left since its last sync, so it has queued nothing since -- unless it is what
                // the thread is still on, which has queued everything and been left by nothing.
                None if which == loaded => to_bind.push(which),
                None => {}
            }
        }
        // Pass two, and the loaded context last: binding anything else leaves it, and a departure
        // takes its sync, so visiting it last is the difference between one bind and two.
        to_bind.sort_by_key(|which| *which == loaded);
        let mut hop = 0usize;
        for which in to_bind {
            let began = self.tally.mark();
            if !self.bind(which) {
                continue;
            }
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
        if syncs.is_empty() {
            // None of these contexts has been left since its last sync and the thread is on none of
            // them, so there is no work of ours here to wait for at all -- the context was named by
            // a fence over something that never reached GL. The queue's order is the answer, which
            // is what it is for a fence that names no context either.
            return Answer::Ordered;
        }
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
    pub fn resource_sync_iosurface(
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
                self.current.switch_to(&self.winsys, &self.gl, GlContext::Sub(*id, sub), gl_ctx);
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
    /// them all. A caller that *can* say wants `Vrend::finish_contexts`: this one is the
    /// fallback, and it is far too expensive to sit on a path that runs per frame.
    ///
    /// Finishing ctx0 alone is not a substitute, whatever it costs: ctx0 never draws, and the
    /// harness caught that reading the frame before last off a scanout.
    ///
    /// "Every context" means every *guest* context and ctx0. The blitter holds a GL context of its
    /// own ([`blitter::Blitter`]) and is in neither this nor `Vrend::finish_contexts`, so a blit
    /// into a surface-backed destination is waited for by neither. A *fence* does cover it -- the
    /// blitter is left at the end of every blit and a departure is where a sync is taken, so one is
    /// already in hand (see [`Current::switch_to`]) -- but these two finishes still do not, and
    /// nothing here closes that.
    pub fn finish_all(&mut self) {
        for (id, ctx) in &self.contexts {
            for (sub, gl_ctx) in ctx.gl_contexts() {
                self.current.switch_to(&self.winsys, &self.gl, GlContext::Sub(*id, sub), gl_ctx);
                self.gl.finish();
            }
        }
        self.switch_ctx0();
        self.gl.finish();
    }

    /// Delete the host side of a resource, on ctx0. A handle this renderer never held is
    /// nothing to delete: the renderer's table also holds resources vrend has no side of.
    pub fn resource_destroy(&mut self, handle: ResourceHandle) {
        // An untyped slot owns only a share of someone else's storage: dropping it is the
        // whole of its teardown, and it needs no GL context to do it.
        if let Some(resource::Slot::Resource(res)) = self.resources.remove(&handle) {
            self.switch_ctx0();
            if let Some(still_attached) = res.destroy(&self.gl) {
                self.doomed.push(still_attached);
            }
            self.sweep_doomed();
        }
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

    pub fn resource(&self, handle: ResourceHandle) -> Option<&Resource> {
        self.resources.get(&handle)?.resource()
    }

    /// The guest attached pages to a resource: a host-memory buffer pays them whatever they are
    /// owed (`vrend_pipe_resource_attach_iov`).
    ///
    /// A freshly created resource owes nothing, and that is what keeps this from racing the
    /// guest -- see [`resource::Shadow`].
    pub fn resource_attached(&mut self, handle: ResourceHandle, pages: &Iov<'_>) {
        if let Some(Resource { storage: resource::Storage::Host(shadow), .. }) =
            self.resources.get_mut(&handle).and_then(resource::Slot::resource_mut)
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
            self.resources.get_mut(&handle).and_then(resource::Slot::resource_mut)
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
        ctx: Option<ContextId>,
        handle: ResourceHandle,
        to_host: bool,
        own: Option<&Iov<'_>>,
        pages: &Iov<'_>,
        info: &Info,
    ) -> Result<(), transfer::Error> {
        match ctx {
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
        let res = self
            .resources
            .get_mut(&handle)
            .and_then(resource::Slot::resource_mut)
            .ok_or(transfer::Error::NoPages)?;
        if to_host {
            transfer::write(&self.gl, self.current.program(), &self.formats, res, own, pages, info)
        } else {
            transfer::read(
                &self.gl,
                self.current.program(),
                &self.features,
                &self.formats,
                res,
                own,
                pages,
                info,
            )
        }
    }
}

impl Drop for Vrend {
    /// Spend the syncs still held for contexts the thread has left.
    ///
    /// A `Fence` aborts on drop rather than leak a driver allocation, so a renderer going away
    /// holding one would take the process down on its way out. ctx0 is bound first because deleting
    /// a sync needs *some* context of its share group current, and ctx0 is the one still standing:
    /// it is a field of this struct, so it is dropped after this runs.
    fn drop(&mut self) {
        self.switch_ctx0();
        let Vrend { gl, current, .. } = self;
        current.spend_all(gl);
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

/// `epoxy_gl_version` for a GLES version string: "OpenGL ES 3.1 Mesa ..." is 31.
fn parse_gles_version(s: &str) -> u32 {
    let rest = s.strip_prefix("OpenGL ES ").unwrap_or(s);
    let mut it = rest.split(|c: char| !c.is_ascii_digit()).filter(|p| !p.is_empty());
    let major: u32 = it.next().and_then(|p| p.parse().ok()).unwrap_or(0);
    let minor: u32 = it.next().and_then(|p| p.parse().ok()).unwrap_or(0);
    major * 10 + minor
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The host property decides whether a fence is flushed behind, and an inverted reading of it
    /// is invisible from inside: both answers take a fence, both hand back a `Fence`, and the
    /// difference is one driver call that shows up as a stall or a hang rather than as an error.
    /// So the mapping is pinned here, where getting it backwards is a failed assert instead of a
    /// slow boot nobody can attribute.
    #[test]
    fn the_host_property_decides_whether_a_fence_is_flushed_behind() {
        let _display = crate::vrend::one_display_at_a_time();
        struct Discard;
        impl crate::fence::FenceSink for Discard {
            fn context_fence(&mut self, _: ContextId, _: RingIdx, _: FenceId) {}
            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        let up = |config| {
            let retire = crate::fence::Retirement::start(Box::new(Discard));
            Vrend::new(config, &crate::budget::Budget::with_cap(None, false), retire.handle(), None)
                .expect("vrend comes up")
        };

        // The default is the answer that is safe on a host nobody has vouched for: take the fence,
        // then flush, because a fence over unsubmitted commands never signals.
        assert_eq!(
            up(Config::default()).gl.fence_flush(),
            gl::FenceFlush::Needed,
            "a host that has not claimed the property gets the flush"
        );
        assert_eq!(
            up(Config { gl_fences_without_draining: true, ..Config::default() }).gl.fence_flush(),
            gl::FenceFlush::Submits,
            "and a host that has claimed it does not, or the drain just moves to the flush"
        );
    }

    /// The fence path's cost is a context switch, and what removes it is that a context is synced
    /// when it is *left* rather than when a fence comes looking for it. Two things have to hold for
    /// that, and neither is visible from a score: the sync has to actually be taken at the
    /// departure, and it has to be taken only while this renderer can *prove* its own context is
    /// current. The proof is `Winsys::still_bound`, and on a host where it answered `false` the
    /// whole mechanism would degrade silently to the walk it replaced -- correct, and slow, with
    /// nothing to say so. So it is asserted here against the live driver.
    #[test]
    fn leaving_a_context_is_what_fences_it() {
        let _display = crate::vrend::one_display_at_a_time();
        struct Discard;
        impl crate::fence::FenceSink for Discard {
            fn context_fence(&mut self, _: ContextId, _: RingIdx, _: FenceId) {}
            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        let retire = crate::fence::Retirement::start(Box::new(Discard));
        let mut v = Vrend::new(
            Config::default(),
            &crate::budget::Budget::with_cap(None, false),
            retire.handle(),
            None,
        )
        .expect("vrend comes up");

        let id = ContextId::new(1).expect("a context id is non-zero");
        v.context_create(id, &NoGuest).expect("a classic context comes up");
        let sub = GlContext::Sub(id, crate::vrend::proto::SubContextId(0));

        // Nothing has been left yet, so nothing is held: an entry exists for exactly those contexts
        // the thread has departed from since their last sync.
        assert!(v.current.take(sub).is_none(), "a context never left holds no sync");

        // Leaving ctx0 takes one over ctx0's queue -- which is the assertion that matters, because
        // `NeedsSync` here would mean `still_bound` could not vouch for what was current and every
        // fence pays a switch again.
        assert!(v.bind(sub), "the context's sub-context binds");
        match v.current.take(GlContext::Ctx0) {
            Some(Pending::Synced(f)) => v.gl.fence_delete(f),
            other => panic!("leaving ctx0 takes a sync over it; got {other:?}"),
        }

        // Going back retires what was held for the context arrived at, because work is about to be
        // queued on it that a sync taken before that work does not cover.
        assert!(v.bind(sub), "binding what is bound is not a departure");
        assert!(v.bind(GlContext::Ctx0), "ctx0 binds");
        assert!(
            v.current.take(GlContext::Ctx0).is_none(),
            "arriving at a context retires the sync taken when it was last left"
        );
        match v.current.take(sub) {
            Some(Pending::Synced(f)) => v.gl.fence_delete(f),
            other => panic!("leaving the sub-context takes a sync over it; got {other:?}"),
        }
    }

    /// A sync is a driver allocation, and what it describes is a sub-context's command queue. When
    /// that queue goes away nothing will ever come back for the sync -- a fence collects only
    /// sub-contexts that still exist -- so the destroy path has to spend it. The guest drives that
    /// path as often as it likes, which is what makes an entry left behind a leak it can mint
    /// without bound rather than a tidiness problem.
    #[test]
    fn destroying_a_context_spends_what_was_held_for_its_sub_contexts() {
        let _display = crate::vrend::one_display_at_a_time();
        struct Discard;
        impl crate::fence::FenceSink for Discard {
            fn context_fence(&mut self, _: ContextId, _: RingIdx, _: FenceId) {}
            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        let retire = crate::fence::Retirement::start(Box::new(Discard));
        let mut v = Vrend::new(
            Config::default(),
            &crate::budget::Budget::with_cap(None, false),
            retire.handle(),
            None,
        )
        .expect("vrend comes up");

        let id = ContextId::new(1).expect("a context id is non-zero");
        v.context_create(id, &NoGuest).expect("a classic context comes up");
        let sub = GlContext::Sub(id, crate::vrend::proto::SubContextId(0));

        // Leave the sub-context, so a sync over it is held, then destroy the context under it.
        assert!(v.bind(sub), "the sub-context binds");
        assert!(v.bind(GlContext::Ctx0), "and is left, which takes a sync over it");
        v.context_destroy(id, &NoGuest);

        assert!(
            v.current.take(sub).is_none(),
            "a destroyed context leaves no sync behind: nothing would ever collect it"
        );
    }

    /// A fence still covers every queue the work could be on, which is the whole reason the walk
    /// existed. Collecting syncs taken at departures instead of taking them here changes where they
    /// come from and must not change the set: a sub-context left earlier is covered by the sync it
    /// was left with, and the context the thread is still on by one taken in place.
    #[test]
    fn a_fence_covers_the_context_left_and_the_one_still_loaded() {
        let _display = crate::vrend::one_display_at_a_time();
        struct Discard;
        impl crate::fence::FenceSink for Discard {
            fn context_fence(&mut self, _: ContextId, _: RingIdx, _: FenceId) {}
            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        let retire = crate::fence::Retirement::start(Box::new(Discard));
        let mut v = Vrend::new(
            Config::default(),
            &crate::budget::Budget::with_cap(None, false),
            retire.handle(),
            None,
        )
        .expect("vrend comes up");

        let id = ContextId::new(1).expect("a context id is non-zero");
        v.context_create(id, &NoGuest).expect("a classic context comes up");
        let sub = GlContext::Sub(id, crate::vrend::proto::SubContextId(0));

        // Work on the sub-context, then leave it for ctx0: the sub-context is now covered by a sync
        // in hand, and ctx0 is what the thread is on.
        assert!(v.bind(sub), "the sub-context binds");
        assert!(v.bind(GlContext::Ctx0), "and the thread goes back to ctx0");

        match v.decide_fence(Some(id)) {
            Answer::Syncs(syncs) => {
                assert_eq!(
                    syncs.len(),
                    2,
                    "the sub-context left and ctx0 still loaded -- one sync each, neither dropped"
                );
                for f in syncs {
                    v.gl.fence_delete(f);
                }
            }
            Answer::Ordered => {
                panic!("work was queued on two contexts; ordering does not cover it")
            }
        }

        // And a second fence with nothing done in between covers nothing of its own: no context has
        // been left since, and the one the thread is on was just synced. The queue's order is the
        // answer, which is what it is for a fence that names no context at all.
        //
        // ctx0 is the exception -- it is what the thread is on, so it is always synced in place.
        match v.decide_fence(Some(id)) {
            Answer::Syncs(syncs) => {
                assert_eq!(syncs.len(), 1, "only the loaded context, fenced where it stands");
                for f in syncs {
                    v.gl.fence_delete(f);
                }
            }
            Answer::Ordered => {}
        }
    }

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
            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        // Declared first so it outlives the renderer: the fence waiter retires through this as it
        // drains, which happens while `v` is dropping.
        let retire = crate::fence::Retirement::start(Box::new(Discard));
        let v = Vrend::new(
            Config::default(),
            &crate::budget::Budget::with_cap(None, false),
            retire.handle(),
            None,
        )
        .expect("vrend comes up");
        let present: Vec<&str> = v.features.present().map(|f| f.name()).collect();
        eprintln!("features: {}", present.join(" "));
        for raw in [1, 2, 20, 48, 49, 64, 65, 67, 131, 134, 177, 227] {
            let f = super::super::proto::Format::from_wire(raw).unwrap();
            eprintln!("{raw:>4} {:<24} {:?}", f.name(), v.formats.get(f));
        }
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
            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        let retire = crate::fence::Retirement::start(Box::new(Discard));
        let mut v = Vrend::new(
            Config::default(),
            &crate::budget::Budget::with_cap(None, false),
            retire.handle(),
            None,
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
            fn global_fence(&mut self, f: ClientFenceId) {
                let _ = self.0.send(u64::from(f.0));
            }
        }
        let (tx, rx) = std::sync::mpsc::channel();
        let retire = crate::fence::Retirement::start(Box::new(Recorder(tx)));
        let mut v = Vrend::new(
            Config::default(),
            &crate::budget::Budget::with_cap(None, false),
            retire.handle(),
            None,
        )
        .expect("vrend comes up");
        assert!(
            v.waiter.is_some(),
            "without a waiter no sync is ever taken and this tests nothing"
        );

        let ctx = ContextId::new(1).expect("a context id");
        v.context_create(ctx, &NoGuest).expect("a context");

        // Each of the three shapes a classic fence comes in: named by a context, named by the
        // global ring, and naming nothing at all.
        v.fence_context(ctx, RingIdx(0), FenceId(11));
        v.fence_global(Some(ctx), ClientFenceId(22));
        v.fence_global(None, ClientFenceId(33));

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
            fn global_fence(&mut self, _: ClientFenceId) {}
        }
        let retire = crate::fence::Retirement::start(Box::new(Ignore));
        let mut v = Vrend::new(
            Config::default(),
            &crate::budget::Budget::with_cap(None, false),
            retire.handle(),
            None,
        )
        .expect("vrend comes up");

        let ctx = ContextId::new(1).expect("a context id");
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

        let answer = v.decide_fence(Some(ctx));
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

    #[test]
    fn the_version_string_parses_the_way_epoxy_reads_it() {
        assert_eq!(parse_gles_version("OpenGL ES 3.1 Mesa 26.0.0"), 31);
        assert_eq!(parse_gles_version("OpenGL ES 3.2 Mesa 26.0.0-devel (git-abc)"), 32);
        assert_eq!(parse_gles_version(""), 0);
    }
}
