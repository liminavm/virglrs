// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! EGL: the winsys the classic renderer draws through.
//!
//! One of the named unsafe modules (CLAUDE.md). libEGL is linked, not dlopened -- the argument is
//! `vulkan.rs`'s, unchanged -- and `eglGetProcAddress` is the one symbol taken from it: every EGL
//! and GLES entry point is resolved through it into a table generated from the Khronos registry.
//!
//! The display is surfaceless: there is no window and never will be, because what a guest sees is
//! an IOSurface a compositor presents (`metal.rs`), and vrend renders into textures that are
//! imported from those. So the platform is `EGL_PLATFORM_SURFACELESS_MESA`, every context is
//! made current with no surface, and the config is a formality the API insists on.
//!
//! What this module hands the rest of vrend is a [`Winsys`] that owns the display, [`Context`]s
//! that cannot outlive it, and a [`Gles`] table loaded once a context is current. Nothing else
//! sees an `EGLDisplay` or an `EGLContext`.

use core::ffi::CStr;
use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

pub(crate) use super::gl::types;
use super::gl::{Gles, ProcAddr};
use crate::surface::{Held, Surface};

#[allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code, clippy::all)]
pub mod proc {
    include!(concat!(env!("OUT_DIR"), "/gl/egl.rs"));
}

use proc::Egl;
use types::*;

// libEGL, linked. `eglGetProcAddress` is the root of every table below it: EGL 1.5 promises it
// answers for core EGL commands as well as extensions, and Mesa answers for GLES through it too.
#[link(name = "EGL")]
unsafe extern "C" {
    fn eglGetProcAddress(name: *const c_char) -> Option<ProcAddr>;
}

fn get_proc(name: &CStr) -> Option<ProcAddr> {
    // SAFETY: `name` is a `&CStr`, so it is NUL-terminated and live for the call; the function
    // takes nothing else.
    unsafe { eglGetProcAddress(name.as_ptr()) }
}

/// The EGL entry points, resolved once. Needs no display: the table is the library's, and a
/// missing entry is a missing extension, which the census below reports by name.
pub fn table() -> Egl {
    // SAFETY: `eglGetProcAddress` answers each name with null or with the address of the EGL
    // command of that name, which is the contract `Egl::load` requires.
    unsafe { Egl::load(&mut get_proc) }
}

/// An EGL call that returned failure, and what `eglGetError` said about it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct EglError {
    pub call: &'static str,
    pub code: EGLint,
}

impl EglError {
    fn name(code: EGLint) -> &'static str {
        match code as u32 {
            proc::EGL_SUCCESS => "EGL_SUCCESS",
            proc::EGL_NOT_INITIALIZED => "EGL_NOT_INITIALIZED",
            proc::EGL_BAD_ACCESS => "EGL_BAD_ACCESS",
            proc::EGL_BAD_ALLOC => "EGL_BAD_ALLOC",
            proc::EGL_BAD_ATTRIBUTE => "EGL_BAD_ATTRIBUTE",
            proc::EGL_BAD_CONFIG => "EGL_BAD_CONFIG",
            proc::EGL_BAD_CONTEXT => "EGL_BAD_CONTEXT",
            proc::EGL_BAD_CURRENT_SURFACE => "EGL_BAD_CURRENT_SURFACE",
            proc::EGL_BAD_DISPLAY => "EGL_BAD_DISPLAY",
            proc::EGL_BAD_MATCH => "EGL_BAD_MATCH",
            proc::EGL_BAD_NATIVE_PIXMAP => "EGL_BAD_NATIVE_PIXMAP",
            proc::EGL_BAD_NATIVE_WINDOW => "EGL_BAD_NATIVE_WINDOW",
            proc::EGL_BAD_PARAMETER => "EGL_BAD_PARAMETER",
            proc::EGL_BAD_SURFACE => "EGL_BAD_SURFACE",
            proc::EGL_CONTEXT_LOST => "EGL_CONTEXT_LOST",
            _ => "an EGL error the registry has no name for",
        }
    }
}

impl fmt::Display for EglError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} failed: {} ({:#x})", self.call, Self::name(self.code), self.code)
    }
}

impl std::error::Error for EglError {}

/// Which client API the winsys binds. Only GLES today; a desktop-GL flavour is a later,
/// separately gated change (`docs/design.md`, P3).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Flavour {
    Gles,
}

/// The GL contexts the renderer runs on, when something else mints them.
///
/// An embedder that already owns a GL stack -- a VMM drawing the guest's scanout into its own
/// window, a compositor hosting the renderer -- cannot simply be handed a texture name from a
/// display of ours: a name means nothing outside the share group it was created in. Implementing
/// this puts the renderer's contexts in *its* group, which is what makes the name it is given
/// nameable.
///
/// It is all of them or none. ctx0, every sub-context, the blitter's and the fence waiter's must
/// come from one factory, or they stop sharing with each other as well.
///
/// **Called only on the renderer's thread.** A GDK context can be made current on the thread that
/// created it and no other, and a windowed backend's is bound to a surface belonging to that
/// thread, so nothing here may be reached from a worker. [`Winsys::thread_display`] answers `None`
/// for a winsys backed by this, which is what stops a second thread from existing to try.
pub trait GlContexts: Send + Sync {
    /// The EGL display the contexts belong to, for the queries and images that need one. It is
    /// the embedder's: already initialised, and not ours to terminate.
    ///
    /// `None` when the embedder cannot name it -- which is not the same as having none. A VMM
    /// whose window toolkit owns the EGL stack may hold no display handle of its own to pass on:
    /// QEMU advertises `get_egl_display` only when it opened an EGL display itself, so its GTK
    /// console offers one and its SDL console does not. The display is then read off the first
    /// context this mints, which is the same display by construction.
    fn display(&self) -> Option<EGLDisplay>;

    /// Mint a context of `version`, in the share group of the ones already minted when `shared`.
    fn create(&self, version: Version, shared: bool) -> Option<EGLContext>;

    /// Bind `ctx` on the calling thread. The embedder is the only thing that knows what is
    /// current, so this is asked every time rather than cached.
    fn make_current(&self, ctx: EGLContext) -> Result<(), EglError>;

    /// Give back a context this minted.
    fn destroy(&self, ctx: EGLContext);
}

/// A client API version to ask `eglCreateContext` for.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Version {
    pub major: u32,
    pub minor: u32,
}

/// The display and everything a context needs to reach it. Shared by every [`Context`] and
/// [`Winsys`] so that a context cannot outlive the display it was created on.
struct Shared {
    egl: Egl,
    display: EGLDisplay,
    backing: Backing,
}

/// Who owns the display and mints the contexts on it.
///
/// The two differ in more than how a context is made: an embedder's display was initialised by
/// the embedder and must not be terminated here, and its context tokens are its own -- under GTK a
/// `GdkGLContext *`, which is not an `EGLContext` and must never be passed to EGL.
enum Backing {
    Own { config: EGLConfig },
    Embedder(Box<dyn GlContexts>),
}

// SAFETY: an `EGLDisplay`, `EGLConfig` or `EGLContext` is a token the library hands out, not
// memory this crate reads: every use goes back through an EGL call, and EGL is specified to be
// callable from any thread, serialising its own state. Which thread a context is *current* on is
// EGL's own bookkeeping (`eglMakeCurrent` binds the calling thread), not a property of the handle,
// so moving the handle between threads asserts nothing. These impls are what lets the renderer
// root, which is `Send`, own the winsys; the make-current discipline is `Vrend`'s.
unsafe impl Send for Shared {}
// SAFETY: as above -- a shared reference grants only the ability to pass the tokens to EGL.
unsafe impl Sync for Shared {}

impl Shared {
    /// Bind `ctx` on the calling thread, with no surface. Binding what is already bound is
    /// nothing, and is asked of EGL rather than remembered.
    ///
    /// Which context a thread has current is EGL's state, not this renderer's, and this process
    /// is not its only writer: a VMM embedding the renderer makes its own context current on this
    /// thread between commands -- QEMU's `-display gtk,gl=on` does it on every scanout -- which
    /// no bookkeeping here can see. A cached answer is therefore a belief a foreign caller can
    /// falsify, and the failure it buys is silent: GL issued into someone else's share group,
    /// where a fresh texture name collides with a live one and the bindings the other context was
    /// relying on are trampled. `eglGetCurrentContext` is a thread-local read and costs far less
    /// than the `eglMakeCurrent` it saves, so the truth is cheaper here than the copy of it.
    fn make_current(&self, ctx: EGLContext) -> Result<(), EglError> {
        // The embedder's tokens are not EGL's, so there is nothing to compare them against and
        // every bind goes through: it is the only thing that knows what its own window has
        // current, and the backends that do this dedup a repeat bind themselves.
        if let Backing::Embedder(contexts) = &self.backing {
            return contexts.make_current(ctx);
        }
        // SAFETY: `eglGetCurrentContext` takes nothing and is defined on any thread.
        if unsafe { self.egl.eglGetCurrentContext()() } == ctx {
            return Ok(());
        }
        // SAFETY: the display is initialised, the context alive on it, and surfaceless contexts
        // are made current with `EGL_NO_SURFACE` twice.
        let ok = unsafe {
            self.egl.eglMakeCurrent()(self.display, proc::EGL_NO_SURFACE, proc::EGL_NO_SURFACE, ctx)
        };
        if ok == proc::EGL_FALSE {
            return Err(self.error("eglMakeCurrent"));
        }
        Ok(())
    }

    fn error(&self, call: &'static str) -> EglError {
        // SAFETY: `eglGetError` takes nothing and is defined on any thread.
        EglError { call, code: unsafe { self.egl.eglGetError()() } }
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        // An embedder's display is the embedder's: it initialised it, other things of its own are
        // on it, and terminating it here would take them with us.
        if matches!(self.backing, Backing::Embedder(_)) {
            return;
        }
        // SAFETY: `display` is the initialised display this struct owns, and every context and
        // image created on it holds an `Arc` of this struct, so all of them are gone by now.
        unsafe { self.egl.eglTerminate()(self.display) };
    }
}

/// The surfaceless EGL display, initialised, with the client API bound and a config chosen.
pub struct Winsys {
    shared: Arc<Shared>,
    flavour: Flavour,
    version: Version,
    extensions: BTreeSet<String>,
}

/// An EGL context on the winsys's display. Destroyed with it; cannot outlive the display.
pub struct Context {
    shared: Arc<Shared>,
    ctx: EGLContext,
}

// SAFETY: `ctx` is an EGL token, for the reason `Shared`'s impl gives; which thread it is current
// on is EGL's bookkeeping, not the handle's.
unsafe impl Send for Context {}

impl Drop for Context {
    fn drop(&mut self) {
        if let Backing::Embedder(contexts) = &self.shared.backing {
            contexts.destroy(self.ctx);
            return;
        }
        // SAFETY: `ctx` was returned by `eglCreateContext` on this display and has not been
        // destroyed, because only this drop destroys it.
        unsafe { self.shared.egl.eglDestroyContext()(self.shared.display, self.ctx) };
    }
}

/// The display, for a thread that owns a [`Context`] and only needs to bind it.
///
/// Handed out by [`Winsys::thread_display`]. It keeps the display alive for as long as the thread
/// holds it, and can do exactly two things -- bind and unbind -- so it cannot become a second
/// owner of anything on the display.
pub struct ThreadDisplay {
    shared: Arc<Shared>,
}

impl ThreadDisplay {
    /// Bind `ctx` on the calling thread, with no surface.
    pub fn make_current(&self, ctx: &Context) -> Result<(), EglError> {
        assert!(Arc::ptr_eq(&ctx.shared, &self.shared), "a context from another display");
        self.shared.make_current(ctx.ctx)
    }

    /// Release whatever context is current on the calling thread.
    pub fn release_current(&self) -> Result<(), EglError> {
        self.shared.make_current(proc::EGL_NO_CONTEXT)
    }
}

/// `EGL_IOSURFACE_LIMINA`: the `eglCreateImageKHR` target limina's Mesa accepts an `IOSurfaceRef`
/// as the client buffer of. The value is the one `egl_dri2.c` defines, and must stay so.
#[cfg(target_os = "macos")]
const EGL_IOSURFACE_LIMINA: EGLenum = 0x3B9A;

/// `EGL_IOSURFACE_PLANE_LIMINA` and `EGL_IOSURFACE_FOURCC_LIMINA`: which plane of a planar
/// surface an image is over, and how that plane's bytes are laid out. Same source as the target
/// above, and the same requirement that the values match.
#[cfg(target_os = "macos")]
const EGL_IOSURFACE_PLANE_LIMINA: EGLint = 0x3B9B;
#[cfg(target_os = "macos")]
const EGL_IOSURFACE_FOURCC_LIMINA: EGLint = 0x3B9C;

/// How one plane of a planar surface is read: the DRM FourCC limina's Mesa names it by.
///
/// A plane of a 4:2:0 biplanar surface is not YUV to the sampler -- it is one or two 8-bit
/// channels, and which one decides the texture format the driver lays over those bytes. The
/// index and the layout travel as one value because neither means anything alone: an index
/// without a layout names bytes with no interpretation, and a layout without an index names no
/// bytes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Plane {
    /// Plane 0 of a 4:2:0 surface: one 8-bit channel, full resolution. `R8`.
    Luma,
    /// Plane 1 of a 4:2:0 biplanar surface: two interleaved 8-bit channels at half resolution.
    /// `GR88`.
    ChromaPair,
}

#[cfg(target_os = "macos")]
impl Plane {
    /// The plane's index within the surface.
    fn index(self) -> EGLint {
        match self {
            Plane::Luma => 0,
            Plane::ChromaPair => 1,
        }
    }

    /// The DRM FourCC for the plane's own layout.
    fn fourcc(self) -> EGLint {
        let code = match self {
            Plane::Luma => u32::from_le_bytes(*b"R8  "),
            Plane::ChromaPair => u32::from_le_bytes(*b"GR88"),
        };
        code as EGLint
    }
}

/// An EGL image over an IOSurface, which is the surface's bytes seen as a GL texture's storage.
///
/// Holds a share of the surface it was made from, so the image cannot outlive what it images:
/// the driver keeps its own reference to the IOSurface, but ours is what keeps the id the
/// compositor was handed naming this surface and not a stranger's minted after it.
///
/// The share is the owner's, whoever that is -- see [`Held`]. An image over a venus allocation's
/// surface keeps that allocation's charge standing for exactly as long as the texture does.
pub struct Image {
    shared: Arc<Shared>,
    image: EGLImageKHR,
    held: Arc<dyn Held>,
}

// SAFETY: `image` is an EGL token, for the reason `Shared`'s impl gives, and the surface is
// `Send + Sync` on its own account.
unsafe impl Send for Image {}
// SAFETY: as above -- a shared reference grants only the ability to pass the token to EGL or GL.
unsafe impl Sync for Image {}

impl Image {
    /// The surface this images.
    pub fn surface(&self) -> &Surface {
        self.held.surface()
    }

    /// A share of the surface this images, for a holder outside the classic side: a venus context
    /// importing this resource keeps the surface alive for as long as it can still reach it,
    /// which is past the classic context that created it and past the resource itself.
    pub fn held(&self) -> Arc<dyn Held> {
        Arc::clone(&self.held)
    }

    /// The token GL binds as texture storage (`GLeglImageOES`). For the GL bindings only, which
    /// take the `Image` by reference and so cannot hold the token past it.
    pub(crate) fn raw(&self) -> EGLImageKHR {
        self.image
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        // SAFETY: `image` was returned by `eglCreateImageKHR` on this display and has not been
        // destroyed, because only this drop destroys it. The surface is released after, by its
        // own drop, so the driver's last look at it (if any) precedes ours.
        unsafe { self.shared.egl.eglDestroyImageKHR()(self.shared.display, self.image) };
    }
}

fn c_str_to_string(p: *const c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    // SAFETY: EGL returns a NUL-terminated string owned by the library, live for the display's
    // lifetime; it is copied out before anything else is called.
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

impl Winsys {
    /// Open the surfaceless display, bind the flavour's API and choose a config.
    ///
    /// Fails, naming the call, when the library has no surfaceless platform, no display can be
    /// initialised (which is what a missing Vulkan ICD looks like from here: zink has nothing to
    /// stand on), or no config renders the API asked for.
    pub fn open(flavour: Flavour) -> Result<Winsys, EglError> {
        let egl = table();
        let client = {
            // SAFETY: `eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS)` is the client-extension
            // query and needs no display.
            let p = unsafe {
                egl.eglQueryString()(proc::EGL_NO_DISPLAY, proc::EGL_EXTENSIONS as EGLint)
            };
            c_str_to_string(p)
        };
        let has = |name: &str| client.split(' ').any(|e| e == name);
        if !has("EGL_EXT_platform_base") || !has("EGL_MESA_platform_surfaceless") {
            return Err(EglError {
                call: "eglGetPlatformDisplay (surfaceless)",
                code: proc::EGL_BAD_PARAMETER as EGLint,
            });
        }
        let get_platform_display = egl
            .try_eglGetPlatformDisplayEXT()
            .expect("EGL_EXT_platform_base is advertised and eglGetPlatformDisplayEXT is absent");
        // SAFETY: the surfaceless platform takes `EGL_DEFAULT_DISPLAY` and no attributes; the
        // extension that defines it was just checked for.
        let display = unsafe {
            get_platform_display(
                proc::EGL_PLATFORM_SURFACELESS_MESA,
                proc::EGL_DEFAULT_DISPLAY,
                core::ptr::null(),
            )
        };
        if display == proc::EGL_NO_DISPLAY {
            let code = unsafe { egl.eglGetError()() };
            return Err(EglError { call: "eglGetPlatformDisplay", code });
        }
        let mut major = 0;
        let mut minor = 0;
        // SAFETY: `display` is the handle just returned; the out-pointers are live locals.
        if unsafe { egl.eglInitialize()(display, &mut major, &mut minor) } == proc::EGL_FALSE {
            let code = unsafe { egl.eglGetError()() };
            return Err(EglError { call: "eglInitialize", code });
        }
        // From here the display is owned, and `Shared`'s drop terminates it on any failure.
        let shared = Arc::new(Shared {
            egl,
            display,
            backing: Backing::Own { config: core::ptr::null_mut() },
        });
        let egl = &shared.egl;

        let extensions: BTreeSet<String> = {
            // SAFETY: `display` is initialised.
            let p = unsafe { egl.eglQueryString()(display, proc::EGL_EXTENSIONS as EGLint) };
            c_str_to_string(p).split(' ').filter(|s| !s.is_empty()).map(str::to_string).collect()
        };

        let (api, renderable) = match flavour {
            Flavour::Gles => (proc::EGL_OPENGL_ES_API, proc::EGL_OPENGL_ES2_BIT),
        };
        // SAFETY: `eglBindAPI` takes an enum and nothing else.
        if unsafe { egl.eglBindAPI()(api) } == proc::EGL_FALSE {
            return Err(shared.error("eglBindAPI"));
        }

        let attribs: [EGLint; 13] = [
            proc::EGL_SURFACE_TYPE as EGLint,
            proc::EGL_PBUFFER_BIT as EGLint,
            proc::EGL_RENDERABLE_TYPE as EGLint,
            renderable as EGLint,
            proc::EGL_RED_SIZE as EGLint,
            1,
            proc::EGL_GREEN_SIZE as EGLint,
            1,
            proc::EGL_BLUE_SIZE as EGLint,
            1,
            proc::EGL_ALPHA_SIZE as EGLint,
            0,
            proc::EGL_NONE as EGLint,
        ];
        let mut config: EGLConfig = core::ptr::null_mut();
        let mut count: EGLint = 0;
        // SAFETY: `attribs` is NONE-terminated, `config` has room for the one config asked for,
        // and `count` is a live local.
        let ok =
            unsafe { egl.eglChooseConfig()(display, attribs.as_ptr(), &mut config, 1, &mut count) };
        if ok == proc::EGL_FALSE || count != 1 {
            return Err(shared.error("eglChooseConfig"));
        }
        // The config is the one field not known at construction; `Arc::get_mut` holds because
        // nothing else has cloned the `Arc` yet.
        let mut shared = shared;
        match &mut Arc::get_mut(&mut shared).expect("no context exists yet").backing {
            Backing::Own { config: slot } => *slot = config,
            Backing::Embedder(_) => unreachable!("this constructor built an owned backing"),
        }

        Ok(Winsys {
            shared,
            flavour,
            version: Version { major: major as u32, minor: minor as u32 },
            extensions,
        })
    }

    /// A winsys over contexts the embedder mints, on the display it already owns.
    ///
    /// Nothing is initialised or configured here: the display is the embedder's, and the config a
    /// context is made with is its business, not ours. What is read from it is what a display can
    /// be asked without owning it -- its version and its extensions.
    ///
    /// The flavour is the embedder's choice too, and is not knowable until a context exists and is
    /// current, so this records what was asked for; [`Winsys::gles`]'s caller is what finds out
    /// what arrived.
    pub fn embedded(
        flavour: Flavour,
        contexts: Box<dyn GlContexts>,
        versions: &[Version],
    ) -> Result<(Winsys, Context, Version), EglError> {
        let egl = table();
        // The first context is minted before the winsys exists because on an embedder that cannot
        // name its display the context is *how* the display is found: bind one, and
        // `eglGetCurrentDisplay` answers with the display it lives on. So a winsys over an
        // embedder never exists without a context of its own, and the return type says that rather
        // than leaving the order to a caller to get right.
        let mut first = None;
        for &v in versions {
            if let Some(ctx) = contexts.create(v, false) {
                first = Some((ctx, v));
                break;
            }
        }
        let (ctx, version_made) = first
            .ok_or(EglError { call: "create_gl_context", code: proc::EGL_BAD_CONTEXT as EGLint })?;
        if let Err(e) = contexts.make_current(ctx) {
            contexts.destroy(ctx);
            return Err(e);
        }
        // An embedder that names no display may still be on EGL, in which case the context just
        // bound is standing on the display we want. It may equally be on GLX, and then there is no
        // EGL display to find and none is needed: this mode runs on the embedder's contexts and
        // its GL, and the C -- which builds no winsys here either -- renders on GLX exactly so.
        let display = match contexts.display() {
            Some(display) => display,
            // SAFETY: `eglGetCurrentDisplay` takes nothing and is defined on any thread. A context
            // the embedder minted is current on this one, so this answers with that context's
            // display, or `EGL_NO_DISPLAY` if it is not an EGL context at all.
            None => unsafe { egl.eglGetCurrentDisplay()() },
        };
        let extensions: BTreeSet<String> = if display == proc::EGL_NO_DISPLAY {
            BTreeSet::new()
        } else {
            // SAFETY: the embedder initialised this display before handing it over.
            let p = unsafe { egl.eglQueryString()(display, proc::EGL_EXTENSIONS as EGLint) };
            c_str_to_string(p).split(' ').filter(|s| !s.is_empty()).map(str::to_string).collect()
        };
        let version = if display == proc::EGL_NO_DISPLAY {
            Version { major: 0, minor: 0 }
        } else {
            // SAFETY: as above.
            let p = unsafe { egl.eglQueryString()(display, proc::EGL_VERSION as EGLint) };
            parse_egl_version(&c_str_to_string(p))
        };
        let shared = Arc::new(Shared { egl, display, backing: Backing::Embedder(contexts) });
        let ctx0 = Context { shared: Arc::clone(&shared), ctx };
        Ok((Winsys { shared, flavour, version, extensions }, ctx0, version_made))
    }

    pub fn flavour(&self) -> Flavour {
        self.flavour
    }

    /// The EGL version the display reports.
    pub fn version(&self) -> Version {
        self.version
    }

    /// Whether the display advertises an extension, by its `EGL_*` name.
    pub fn has_extension(&self, name: &str) -> bool {
        self.extensions.contains(name)
    }

    /// Whether an sRGB drawable can be asked for -- `vrend_winsys_has_gl_colorspace`.
    ///
    /// On a display of ours it is the EGL extension. On an embedder's it is the extension when
    /// there is a display to have asked, and otherwise yes: a winsys with no EGL display is one
    /// whose contexts were configured by someone else, and the C answers the same way for the same
    /// reason (`use_context == CONTEXT_NONE`). Answering no instead would quietly drop
    /// `srgb_write_control` from a host that has it.
    pub fn has_gl_colorspace(&self) -> bool {
        self.shared.display == proc::EGL_NO_DISPLAY || self.has_extension("EGL_KHR_gl_colorspace")
    }

    pub fn extensions(&self) -> impl Iterator<Item = &str> {
        self.extensions.iter().map(String::as_str)
    }

    /// Create a context of the flavour's API at `version`, sharing objects with `shared` if
    /// given.
    pub fn create_context(
        &self,
        version: Version,
        shared: Option<&Context>,
    ) -> Result<Context, EglError> {
        let attribs: [EGLint; 5] = [
            proc::EGL_CONTEXT_MAJOR_VERSION as EGLint,
            version.major as EGLint,
            proc::EGL_CONTEXT_MINOR_VERSION as EGLint,
            version.minor as EGLint,
            proc::EGL_NONE as EGLint,
        ];
        let share = shared.map_or(proc::EGL_NO_CONTEXT, |c| {
            assert!(Arc::ptr_eq(&c.shared, &self.shared), "a share context from another display");
            c.ctx
        });
        if let Backing::Embedder(contexts) = &self.shared.backing {
            // The embedder is told whether to share, not what with: it mints every context of this
            // renderer into one group of its own, so naming one of them would say nothing it does
            // not already know.
            let ctx = contexts.create(version, shared.is_some()).ok_or(EglError {
                call: "create_gl_context",
                code: proc::EGL_BAD_CONTEXT as EGLint,
            })?;
            return Ok(Context { shared: Arc::clone(&self.shared), ctx });
        }
        let config = match &self.shared.backing {
            Backing::Own { config } => *config,
            Backing::Embedder(_) => unreachable!("returned just above"),
        };
        let egl = &self.shared.egl;
        // SAFETY: the display is initialised and the config chosen on it; `attribs` is
        // NONE-terminated; `share` is either no context or one alive on this display.
        let ctx =
            unsafe { egl.eglCreateContext()(self.shared.display, config, share, attribs.as_ptr()) };
        if ctx == proc::EGL_NO_CONTEXT {
            return Err(self.shared.error("eglCreateContext"));
        }
        Ok(Context { shared: Arc::clone(&self.shared), ctx })
    }

    /// Make `ctx` current on this thread, with no surface.
    pub fn make_current(&self, ctx: &Context) -> Result<(), EglError> {
        assert!(Arc::ptr_eq(&ctx.shared, &self.shared), "a context from another display");
        self.shared.make_current(ctx.ctx)
    }

    /// A handle to this display for another thread, which can bind a context it owns and nothing
    /// else.
    ///
    /// Currency is per thread and EGL's own bookkeeping, so a second thread binding its own
    /// context says nothing about what is current here -- which is what lets the fence waiter hold
    /// a context of the share group and wait on it while this thread carries on. It deliberately
    /// cannot create or destroy anything: a thread that could would be a second owner of the
    /// display's objects.
    /// `None` when the embedder mints the contexts: its factory may only be reached from the
    /// renderer's thread, so there is no handle for another one to hold. See [`GlContexts`].
    pub fn thread_display(&self) -> Option<ThreadDisplay> {
        match self.shared.backing {
            Backing::Own { .. } => Some(ThreadDisplay { shared: Arc::clone(&self.shared) }),
            Backing::Embedder(_) => None,
        }
    }

    /// Release whatever context is current on this thread.
    pub fn release_current(&self) -> Result<(), EglError> {
        assert!(
            matches!(self.shared.backing, Backing::Own { .. }),
            "an embedder's contexts are released by the embedder, not through a null token"
        );
        // `EGL_NO_CONTEXT` with no surfaces is the documented way to release.
        self.shared.make_current(proc::EGL_NO_CONTEXT)
    }

    /// An EGL image whose pixels are `surface`'s, for a texture to take as its storage.
    ///
    /// Made against no context: the image belongs to the display, and any context on it may
    /// bind it. Fails, naming the call, when the driver will not import the surface -- the
    /// resource then keeps ordinary GL storage, and the caller says so.
    pub fn image_from_iosurface(&self, held: Arc<dyn Held>) -> Result<Image, EglError> {
        self.image_of_iosurface(held, None)
    }

    /// An EGL image over *one plane* of a planar surface, in that plane's own layout.
    ///
    /// The planes of a composite decode target share one allocation, so each is imaged
    /// separately and the images are what the plane resources take as storage. Each holds its
    /// own share of the surface: the surface outlives whichever plane image is dropped last,
    /// and no plane's image is a view into something already freed.
    pub fn image_from_iosurface_plane(
        &self,
        held: Arc<dyn Held>,
        plane: Plane,
    ) -> Result<Image, EglError> {
        self.image_of_iosurface(held, Some(plane))
    }

    /// No surface can exist on a host that mints none, so this is total rather than refusing:
    /// the caller had to produce one to get here, and the type says it could not have.
    #[cfg(not(target_os = "macos"))]
    fn image_of_iosurface(
        &self,
        held: Arc<dyn Held>,
        _plane: Option<Plane>,
    ) -> Result<Image, EglError> {
        match *held.surface() {}
    }

    #[cfg(target_os = "macos")]
    fn image_of_iosurface(
        &self,
        held: Arc<dyn Held>,
        plane: Option<Plane>,
    ) -> Result<Image, EglError> {
        let egl = &self.shared.egl;
        let attribs = plane.map(|plane| {
            [
                EGL_IOSURFACE_PLANE_LIMINA,
                plane.index(),
                EGL_IOSURFACE_FOURCC_LIMINA,
                plane.fourcc(),
                proc::EGL_NONE as EGLint,
            ]
        });
        // SAFETY: the display is initialised; the target is the one limina's Mesa defines for an
        // `IOSurfaceRef` client buffer, and `surface` is held by the `Image` for as long as the
        // image exists, so the reference passed here outlives every use the driver makes of it.
        // The attribute list, when there is one, is a live local this call outlives and is
        // `EGL_NONE`-terminated; `NULL` is the documented empty list when there is not.
        let image = unsafe {
            egl.eglCreateImageKHR()(
                self.shared.display,
                proc::EGL_NO_CONTEXT,
                EGL_IOSURFACE_LIMINA,
                held.surface().client_buffer(),
                attribs.as_ref().map_or(core::ptr::null(), |a| a.as_ptr()),
            )
        };
        if image.is_null() {
            return Err(self.shared.error("eglCreateImageKHR"));
        }
        Ok(Image { shared: Arc::clone(&self.shared), image, held })
    }

    /// The GLES entry points. Resolved through `eglGetProcAddress`, which for Mesa answers the
    /// same addresses whichever context is current -- dispatch happens behind them -- so one
    /// table serves every context of the display.
    pub fn gles(&self) -> Gles {
        // SAFETY: `eglGetProcAddress` answers each name with null or with the address of the GL
        // command of that name, which is the contract `Gles::load` requires.
        unsafe { Gles::load(&mut get_proc) }
    }
}

/// The `major.minor` at the head of what `eglQueryString(EGL_VERSION)` answers, which is
/// specified to start with it. Zero for anything unreadable: a display that will not say is one we
/// ask nothing of by version.
fn parse_egl_version(s: &str) -> Version {
    let mut it = s.split(|c: char| !c.is_ascii_digit()).filter(|p| !p.is_empty());
    let major = it.next().and_then(|p| p.parse().ok()).unwrap_or(0);
    let minor = it.next().and_then(|p| p.parse().ok()).unwrap_or(0);
    Version { major, minor }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An embedder that hands out tokens of its own and counts what it was asked for. The
    /// display is never dereferenced by anything this exercises, only compared against
    /// `EGL_NO_DISPLAY` and passed to `eglQueryString`, which answers null for a display it does
    /// not know.
    #[derive(Default)]
    struct FakeEmbedder {
        /// Whether this embedder can name its display, which is what decides between the two
        /// ways a winsys finds one. See [`GlContexts::display`].
        nameless: bool,
        minted: std::sync::Mutex<Vec<usize>>,
        destroyed: std::sync::Mutex<Vec<usize>>,
        bound: std::sync::Mutex<Vec<usize>>,
        shared_asked: std::sync::Mutex<Vec<bool>>,
    }

    impl GlContexts for FakeEmbedder {
        fn display(&self) -> Option<EGLDisplay> {
            // Not null, so the winsys accepts it; never dereferenced.
            (!self.nameless).then(core::ptr::dangling_mut)
        }

        fn create(&self, _version: Version, shared: bool) -> Option<EGLContext> {
            let mut minted = self.minted.lock().expect("test");
            // Tokens of the embedder's own choosing, deliberately not EGL contexts.
            let token = 0x1000 + minted.len();
            minted.push(token);
            self.shared_asked.lock().expect("test").push(shared);
            Some(core::ptr::without_provenance_mut(token))
        }

        fn make_current(&self, ctx: EGLContext) -> Result<(), EglError> {
            self.bound.lock().expect("test").push(ctx.addr());
            Ok(())
        }

        fn destroy(&self, ctx: EGLContext) {
            self.destroyed.lock().expect("test").push(ctx.addr());
        }
    }

    /// Every context of an embedder-backed winsys is the embedder's, and goes back to it.
    ///
    /// The point is that it is all of them: a context this renderer minted for itself would be in
    /// a share group of its own, and the texture names it created there would mean nothing to the
    /// embedder that was handed one. Nothing here touches a GPU -- what is being pinned is which
    /// side of the boundary each context came from, which is a fact about the wiring.
    #[test]
    fn an_embedders_contexts_are_all_the_embedders() {
        let fake = Arc::new(FakeEmbedder::default());
        let counts = Arc::clone(&fake);

        struct Lent(Arc<FakeEmbedder>);
        impl GlContexts for Lent {
            fn display(&self) -> Option<EGLDisplay> {
                self.0.display()
            }
            fn create(&self, version: Version, shared: bool) -> Option<EGLContext> {
                self.0.create(version, shared)
            }
            fn make_current(&self, ctx: EGLContext) -> Result<(), EglError> {
                self.0.make_current(ctx)
            }
            fn destroy(&self, ctx: EGLContext) {
                self.0.destroy(ctx);
            }
        }

        let v = Version { major: 3, minor: 2 };
        // ctx0 comes back with the winsys: it is minted and bound to find the display, so it
        // cannot be left for the caller to remember.
        let (winsys, ctx0, made) = Winsys::embedded(Flavour::Gles, Box::new(Lent(fake)), &[v])
            .expect("a display the embedder vouched for");
        assert_eq!(made, v);

        // No second thread can reach a factory that is only callable on this one.
        assert!(winsys.thread_display().is_none());

        let sub = winsys.create_context(v, Some(&ctx0)).expect("and every context beside it");
        winsys.make_current(&sub).expect("bound through the embedder");

        assert_eq!(*counts.minted.lock().expect("test"), vec![0x1000, 0x1001]);
        assert_eq!(*counts.bound.lock().expect("test"), vec![0x1000, 0x1001]);
        // The first has nothing to share with yet; everything after it says so.
        assert_eq!(*counts.shared_asked.lock().expect("test"), vec![false, true]);
        assert!(counts.destroyed.lock().expect("test").is_empty());

        drop(sub);
        drop(ctx0);
        assert_eq!(*counts.destroyed.lock().expect("test"), vec![0x1001, 0x1000]);
    }

    /// An embedder that can name no EGL display still gets a winsys, and it is not a lesser one.
    ///
    /// This is the shape QEMU's SDL console has: contexts on GLX, and so nothing for
    /// `get_egl_display` or `eglGetCurrentDisplay` to answer with. The renderer runs on the
    /// embedder's contexts and its GL, which is what this mode is; the C builds no winsys here
    /// either and renders. What must not happen is the display's absence being read as a host
    /// that cannot do sRGB -- `has_gl_colorspace` follows the C and says yes.
    ///
    /// Nothing is current on this thread and the fake's tokens are not EGL contexts, so
    /// `eglGetCurrentDisplay` answers `EGL_NO_DISPLAY` by specification. That is the case under
    /// test, not a shortcoming of the fake.
    #[test]
    fn an_embedder_that_names_no_display_still_gets_a_winsys() {
        let fake = Arc::new(FakeEmbedder { nameless: true, ..FakeEmbedder::default() });
        let counts = Arc::clone(&fake);

        struct Lent(Arc<FakeEmbedder>);
        impl GlContexts for Lent {
            fn display(&self) -> Option<EGLDisplay> {
                self.0.display()
            }
            fn create(&self, version: Version, shared: bool) -> Option<EGLContext> {
                self.0.create(version, shared)
            }
            fn make_current(&self, ctx: EGLContext) -> Result<(), EglError> {
                self.0.make_current(ctx)
            }
            fn destroy(&self, ctx: EGLContext) {
                self.0.destroy(ctx);
            }
        }

        let v = Version { major: 3, minor: 2 };
        let (winsys, ctx0, _) = Winsys::embedded(Flavour::Gles, Box::new(Lent(fake)), &[v])
            .expect("an embedder with no display to name is still an embedder");

        assert!(winsys.has_gl_colorspace(), "the C says yes with no winsys, and so must this");
        assert_eq!(winsys.extensions().count(), 0, "there was no display to ask");
        // The context minted to look for a display is ctx0, not a probe to be thrown away.
        assert_eq!(*counts.minted.lock().expect("test"), vec![0x1000]);
        assert!(counts.destroyed.lock().expect("test").is_empty());

        drop(ctx0);
        assert_eq!(*counts.destroyed.lock().expect("test"), vec![0x1000]);
    }

    /// libEGL has to be there and answer for every core EGL command through
    /// `eglGetProcAddress`. It needs no display and no GPU: if this fails the link is wrong,
    /// not the driver.
    #[test]
    fn the_library_answers_for_every_core_command() {
        let egl = table();
        for feature in ["EGL_VERSION_1_0", "EGL_VERSION_1_4", "EGL_VERSION_1_5"] {
            assert!(egl.has_all_of(feature), "{feature}: missing {:?}", egl.missing());
        }
        assert!(egl.has_all_of("EGL_EXT_platform_base"));
        assert!(egl.has_all_of("EGL_KHR_image_base"));
    }

    /// A GLES 3.1 context on the surfaceless display, and the whole 3.1 core resolved through
    /// it. Needs the zink-on-KosmicKrisp stack: `VK_ICD_FILENAMES` at the KK ICD and
    /// `MESA_LOADER_DRIVER_OVERRIDE=zink`, the way `harness/replay/vkr-replay.sh` sets them --
    /// so it is opted into rather than run by `cargo test`, which has no GPU.
    #[test]
    #[ignore = "needs the zink-on-KosmicKrisp environment"]
    fn a_gles_31_context_comes_up_surfaceless() {
        let winsys = Winsys::open(Flavour::Gles).expect("the surfaceless display opens");
        assert!(winsys.version() >= Version { major: 1, minor: 4 });
        assert!(
            winsys.has_extension("EGL_KHR_image_base"),
            "{:?}",
            winsys.extensions().collect::<Vec<_>>()
        );
        let ctx =
            winsys.create_context(Version { major: 3, minor: 1 }, None).expect("a 3.1 context");
        winsys.make_current(&ctx).expect("current");
        let gl = winsys.gles();
        for feature in ["GL_ES_VERSION_2_0", "GL_ES_VERSION_3_0", "GL_ES_VERSION_3_1"] {
            assert!(gl.has_all_of(feature), "{feature}: missing {:?}", gl.missing());
        }
        // SAFETY: a context is current on this thread and `GL_VERSION` is a valid name.
        let version =
            c_str_to_string(
                unsafe { gl.glGetString()(super::super::gl::gles::GL_VERSION) } as *const c_char
            );
        assert!(version.starts_with("OpenGL ES 3.1"), "{version}");
        let second = winsys
            .create_context(Version { major: 3, minor: 1 }, Some(&ctx))
            .expect("a shared context");
        winsys.make_current(&second).expect("current");
        winsys.release_current().expect("released");
    }

    /// The two planes of one planar surface import as two images, each in its own layout.
    ///
    /// This is the whole storage model behind a composite decode target: the planes share an
    /// allocation and are reached separately, so if the driver will not take the plane
    /// attributes there is nothing above this that can work. It asks the driver rather than
    /// asserting the attribute values, because the values are only right if Mesa agrees.
    ///
    /// Run it on its own (`--lib each_plane_of -- --ignored`): two displays opened in one
    /// process leave this driver unable to make a shared context, so the ignored tests in this
    /// module fail each other when run together.
    #[test]
    #[ignore = "needs the zink-on-KosmicKrisp environment"]
    fn each_plane_of_a_planar_surface_imports_as_its_own_image() {
        use crate::surface::{PlanarFormat, Surface};

        let winsys = Winsys::open(Flavour::Gles).expect("the surfaceless display opens");
        let surface: Arc<dyn Held> =
            Arc::new(Surface::planar(64, 64, PlanarFormat::BiPlanar420).expect("a planar surface"));
        let id = surface.surface().id();

        let luma = winsys
            .image_from_iosurface_plane(Arc::clone(&surface), Plane::Luma)
            .expect("the driver imports the luma plane");
        let chroma = winsys
            .image_from_iosurface_plane(Arc::clone(&surface), Plane::ChromaPair)
            .expect("the driver imports the chroma plane");

        assert_ne!(luma.raw(), chroma.raw(), "a plane image per plane, not one image twice");
        assert_eq!(luma.surface().id(), id, "both image the surface they were made from");
        assert_eq!(chroma.surface().id(), id);

        // Each holds its own share, so dropping one leaves the other imaging a live surface.
        drop(luma);
        assert_eq!(chroma.surface().id(), id);
    }

    /// A plane image samples the plane it asked for, asked by content.
    ///
    /// The import succeeding proves only that the driver took the attributes, and the image's
    /// own width and height cannot settle it either: they are set from the geometry the import
    /// was handed, so a chroma image reports half-resolution because that is what it was told,
    /// whether or not the plane index reached Metal underneath. Both are measurements of this
    /// side's own input.
    ///
    /// Content is the oracle that cannot lie. Each plane gets a different byte through the CPU,
    /// and each image is read back through a framebuffer. If the index is dropped anywhere
    /// between EGL and Metal, both images are plane 0 and the chroma read returns the luma
    /// pattern.
    #[test]
    #[ignore = "needs the zink-on-KosmicKrisp environment"]
    fn a_plane_image_samples_the_plane_it_asked_for() {
        use super::super::gl::Gl;
        use super::super::gl::gles::{
            GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER, GL_FRAMEBUFFER_COMPLETE, GL_RED, GL_RG,
            GL_TEXTURE_2D, GL_UNSIGNED_BYTE,
        };
        use crate::surface::{PlanarFormat, Surface};

        const LUMA_BYTE: u8 = 0x10;
        const CHROMA_BYTE: u8 = 0x80;

        let winsys = Winsys::open(Flavour::Gles).expect("the surfaceless display opens");
        let ctx =
            winsys.create_context(Version { major: 3, minor: 1 }, None).expect("a 3.1 context");
        winsys.make_current(&ctx).expect("current");
        let gl = Gl::new(winsys.gles());

        let surface = Surface::planar(64, 64, PlanarFormat::BiPlanar420).expect("a planar surface");
        assert!(surface.fill_plane(0, LUMA_BYTE), "the luma plane fills");
        assert!(surface.fill_plane(1, CHROMA_BYTE), "the chroma plane fills");
        let surface: Arc<dyn Held> = Arc::new(surface);

        let read = |plane: Plane, format, w, h| -> Vec<u8> {
            let image = winsys
                .image_from_iosurface_plane(Arc::clone(&surface), plane)
                .expect("the driver imports the plane");
            let texture = gl.gen_texture();
            gl.bind_texture(GL_TEXTURE_2D, Some(texture));
            gl.egl_image_target_texture_2d(GL_TEXTURE_2D, &image);
            let fb = gl.gen_framebuffer();
            gl.bind_framebuffer(GL_FRAMEBUFFER, Some(fb));
            gl.framebuffer_texture_2d(GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, Some(texture), 0);
            assert_eq!(
                gl.check_framebuffer_status(),
                GL_FRAMEBUFFER_COMPLETE,
                "a plane texture is not renderable, so this test cannot read it"
            );
            let mut out = vec![0u8; (w * h) as usize * if format == GL_RG { 2 } else { 1 }];
            assert!(gl.read_pixels(0, 0, w, h, format, GL_UNSIGNED_BYTE, &mut out), "readback");
            gl.bind_framebuffer(GL_FRAMEBUFFER, None);
            out
        };

        let luma = read(Plane::Luma, GL_RED, 64, 64);
        let chroma = read(Plane::ChromaPair, GL_RG, 32, 32);

        assert!(luma.iter().all(|&b| b == LUMA_BYTE), "luma read {:?}", &luma[..8]);
        assert!(
            chroma.iter().all(|&b| b == CHROMA_BYTE),
            "the chroma image did not sample the chroma plane; it read {:?}. A read of \
             {LUMA_BYTE:#04x} means the plane index was dropped between EGL and Metal and both \
             images are plane 0.",
            &chroma[..8]
        );
    }
}
