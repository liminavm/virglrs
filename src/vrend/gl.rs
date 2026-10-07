// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! GLES, as the driver exports it -- and the safe surface the rest of vrend calls it through.
//!
//! One of the named unsafe modules (CLAUDE.md). The table is generated from the Khronos registry
//! by `gl-gen`, for the reason `vulkan.rs` gives: a binding transcribed by hand can disagree with
//! the driver about a parameter, and the disagreement is a stack smash rather than a compile
//! error. Every entry point is resolved through `eglGetProcAddress` ([`super::egl`]), which is
//! what makes the transmute in `Procs::load` sound: EGL promises the address of the function of
//! that name, and the registry gives that name its signature.
//!
//! [`Gl`] is the safe layer over the table, and the only thing the resource, transfer and blit
//! code sees. Each method owns the one hazard its entry point has: a pointer argument is a slice
//! whose length is checked against what GL will read or write, computed here from the format and
//! type the way the driver computes it, with the pixel-store state pinned to tightly packed so
//! there is nothing else for the size to depend on. A call with an argument this crate cannot size
//! is refused before it reaches the driver, never passed through on trust.
//!
//! No method here needs a context to be current for *soundness* -- Mesa dispatches a call with no
//! current context to a stub -- so currency is a correctness property `Vrend` upholds, not a
//! precondition these wrappers encode.

#[allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code, clippy::all)]
pub mod types {
    include!(concat!(env!("OUT_DIR"), "/gl/types.rs"));
}

#[allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code, clippy::all)]
pub mod gles {
    include!(concat!(env!("OUT_DIR"), "/gl/gles.rs"));
}

use core::ffi::CStr;

/// What `glUseProgram` or `glBindProgramPipeline` last left in use on a GL context.
///
/// [`Gl::use_program`] and [`Gl::use_pipeline`] take one and there is nothing else to do with it,
/// so a call site cannot bind a program without recording what it bound -- which is what makes
/// skipping a redundant bind safe rather than a bet on every present and future caller. GL's
/// current program is per-context state, so exactly one of these is ever the right one: the
/// current context's, which [`super::current::Current`] owns and clears on a switch.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct BoundProgram(Option<InUse>);

/// One of the two things a draw runs: a program, or a pipeline of separable ones.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum InUse {
    Program(ProgramName),
    Pipeline(PipelineName),
}

use super::egl::Image;
use super::features::{Api, Feature, Features};

pub use gles::Procs;
use gles::*;
pub use types::*;

/// A fence sync object, owned.
///
/// Unlike every other name in this module this is not a `GLuint` the driver hands out but an
/// opaque pointer, and it owns a driver allocation: dropping it without [`Gl::wait_fence`] leaks
/// that allocation, so `Drop` aborts rather than letting the leak pass. There is exactly one way
/// to make one ([`Gl::fence`]) and one way to spend it ([`Gl::wait_fence`], which consumes it).
///
/// **`Send`, and that is the point.** A sync object belongs to the share group, not to the context
/// that created it, so the spec allows any context of that group -- on any thread -- to wait on it
/// and delete it. That is what lets the fence waiter own a context of its own and wait there,
/// while the thread that took the sync carries on without the renderer lock.
pub struct Fence(GLsync);

// SAFETY: a sync object is a share-group object, not context state: the spec lets any context in
// the share group wait on and delete it, from any thread. The token is opaque and never
// dereferenced here -- it is only ever handed back to the driver -- and `Fence` is neither `Clone`
// nor `Copy`, so exactly one owner can spend it.
unsafe impl Send for Fence {}

impl Drop for Fence {
    fn drop(&mut self) {
        // A sync object dropped on the floor is a driver allocation nothing will ever free, and
        // the fence it stood for is one the guest may still be waiting on. Both are bugs at the
        // site that dropped it, so say so there rather than leaking quietly.
        panic!("a Fence was dropped instead of being spent on Gl::wait_fence");
    }
}

/// A texture name the driver handed out. Never zero: zero is "no texture", and is spelled `None`
/// wherever a binding allows it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TextureName(GLuint);

/// A texture the driver reported as immutable-format, which is the only source `glTextureView`
/// accepts: given any other it puts the context in `GL_INVALID_OPERATION` for the rest of its
/// life. Minted only by [`Gl::immutable_format`], from the driver's answer, so a view can only be
/// asked of a texture the driver itself said would take one.
///
/// Necessary, not sufficient: a texture whose storage is an imported EGL image reports itself
/// immutable and refuses a view all the same. That rule is the resource's (`supports_view`), and
/// is checked beside this one, never instead of it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Immutable(TextureName);

impl Immutable {
    pub fn name(self) -> TextureName {
        self.0
    }

    /// A witness no driver issued, for tests that never call GL with it.
    #[cfg(test)]
    pub const fn unbacked(name: TextureName) -> Immutable {
        Immutable(name)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BufferName(GLuint);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct FramebufferName(GLuint);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct RenderbufferName(GLuint);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct VertexArrayName(GLuint);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SamplerName(GLuint);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TransformFeedbackName(GLuint);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct QueryName(GLuint);

/// A shader object the driver handed out. Never zero.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ShaderName(GLuint);

/// A program object the driver handed out. Never zero.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ProgramName(GLuint);

/// A program pipeline object the driver handed out: separable programs, one a stage. Never zero.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PipelineName(GLuint);

/// A binding point of an indexed buffer target: a uniform block, a shader storage block, an
/// atomic counter buffer, a transform-feedback buffer. The target names the space, so an index
/// means nothing on its own -- and it is not a texture unit, which the draw path counts beside
/// it through the same stages.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BindingPoint(GLuint);

impl BindingPoint {
    /// The first point of a target, where each stage's walk starts.
    pub const FIRST: Self = Self(0);

    pub fn at(index: u32) -> Self {
        Self(index)
    }

    /// The next point, as a stage claims one per block it declared.
    pub fn next(self) -> Self {
        Self(self.0 + 1)
    }

    /// `n` points on, where a stage's blocks start after the ones before it.
    pub fn plus(self, n: u32) -> Self {
        Self(self.0 + n)
    }
}

/// A texture unit: what `glActiveTexture` selects, what a sampler object binds to, and the
/// value a sampler uniform holds. Not a binding point, and not the slot the guest named.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TextureUnit(GLuint);

impl TextureUnit {
    pub const FIRST: Self = Self(0);

    pub fn at(unit: u32) -> Self {
        Self(unit)
    }

    pub fn next(self) -> Self {
        Self(self.0 + 1)
    }

    /// What a sampler uniform is set to, which is the unit's number.
    pub fn uniform_value(self) -> GLint {
        self.0 as GLint
    }
}

/// An image unit: what `glBindImageTexture` binds to. A separate space from [`TextureUnit`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ImageUnit(GLuint);

impl ImageUnit {
    pub fn at(unit: u32) -> Self {
        Self(unit)
    }
}

/// Where a uniform lives in a linked program. GL spells "the program has no such uniform" as
/// -1 and then ignores a write through it, which is a silent no-op three call sites deep; here
/// it is `None` from [`Gl::get_uniform_location`] onwards, and the -1 never exists.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct UniformLocation(GLint);

/// Where a vertex attribute lives in a linked program. GL spells "the program has no such
/// attribute" as -1, and every call taking one then reinterprets it as a huge index; here it is
/// `None` from [`Gl::get_attrib_location`] onwards.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct AttribLocation(GLuint);

impl ShaderName {
    pub fn raw(self) -> GLuint {
        self.0
    }
}

/// The C's `GLvoid const *` offset into a bound buffer, as the draw and indirect calls spell it.
/// The width and signedness of a query result written into a buffer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QueryWord {
    I32,
    U32,
    I64,
    U64,
}

fn offset_ptr(offset: u32) -> *const core::ffi::c_void {
    offset as usize as *const core::ffi::c_void
}

impl TextureName {
    pub fn raw(self) -> GLuint {
        self.0
    }

    /// A name no driver handed out, for tests that never call GL with it.
    #[cfg(test)]
    pub const fn unbacked(id: GLuint) -> TextureName {
        TextureName(id)
    }
}

impl BufferName {
    pub fn raw(self) -> GLuint {
        self.0
    }
}

/// The number of bytes one pixel of `format`/`ty` takes in client memory, tightly packed --
/// `_mesa_bytes_per_pixel`, over the GLES pairs this renderer uploads and reads. `None` for a
/// pair the driver would refuse too, and one the callers refuse first.
pub fn pixel_bytes(format: GLenum, ty: GLenum) -> Option<usize> {
    let components = match format {
        GL_RED | GL_RED_INTEGER | GL_DEPTH_COMPONENT | GL_STENCIL_INDEX | GL_ALPHA
        | GL_LUMINANCE => 1,
        GL_RG | GL_RG_INTEGER | GL_LUMINANCE_ALPHA | GL_DEPTH_STENCIL => 2,
        GL_RGB | GL_RGB_INTEGER => 3,
        GL_RGBA | GL_RGBA_INTEGER | GL_BGRA | GL_BGRA_INTEGER | GL_ABGR_EXT => 4,
        _ => return None,
    };
    let per_component = match ty {
        GL_UNSIGNED_BYTE | GL_BYTE => 1,
        GL_UNSIGNED_SHORT | GL_SHORT | GL_HALF_FLOAT | GL_HALF_FLOAT_OES => 2,
        GL_UNSIGNED_INT | GL_INT | GL_FLOAT => 4,
        // Packed types: the whole pixel in one word, whatever the component count.
        GL_UNSIGNED_BYTE_3_3_2 | GL_UNSIGNED_BYTE_2_3_3_REV => return Some(1),
        GL_UNSIGNED_SHORT_5_6_5
        | GL_UNSIGNED_SHORT_4_4_4_4
        | GL_UNSIGNED_SHORT_5_5_5_1
        | GL_UNSIGNED_SHORT_5_6_5_REV
        | GL_UNSIGNED_SHORT_4_4_4_4_REV
        | GL_UNSIGNED_SHORT_1_5_5_5_REV => return Some(2),
        GL_UNSIGNED_INT_8_8_8_8
        | GL_UNSIGNED_INT_8_8_8_8_REV
        | GL_UNSIGNED_INT_2_10_10_10_REV
        | GL_UNSIGNED_INT_10F_11F_11F_REV
        | GL_UNSIGNED_INT_5_9_9_9_REV
        | GL_UNSIGNED_INT_24_8 => return Some(4),
        GL_FLOAT_32_UNSIGNED_INT_24_8_REV => return Some(8),
        _ => return None,
    };
    Some(components * per_component)
}

/// The bytes a `w`×`h`×`d` image of `format`/`ty` takes tightly packed, or `None` when the pair is
/// unknown or the product overflows.
pub(super) fn image_bytes(
    format: GLenum,
    ty: GLenum,
    w: GLsizei,
    h: GLsizei,
    d: GLsizei,
) -> Option<usize> {
    let px = pixel_bytes(format, ty)?;
    let dim = |v: GLsizei| usize::try_from(v).ok();
    px.checked_mul(dim(w)?)?.checked_mul(dim(h)?)?.checked_mul(dim(d)?)
}

/// What a wrapper behind a feature does when its proc is absent: `Features::reconcile`
/// withdrew every feature whose procs are missing at init, so a caller that checked the
/// feature and still got here found a hole in that table -- a host bug, not a guest one.
fn promised<T>(proc: Option<T>, feature: Feature, name: &str) -> T {
    proc.unwrap_or_else(|| {
        panic!("{name} is behind {}, which reconcile should have withdrawn", feature.name())
    })
}

/// The entry points a feature stands for, each resolved to whichever spelling the driver
/// exports. A resolver is the one place its spellings are listed: the wrapper takes its proc
/// from it, and [`Gl::missing_procs`] asks the same resolver, so `Features::reconcile`
/// withdraws a feature before any wrapper could find its proc absent.
mod procs {
    use super::gles::*;
    use super::{
        Api, Feature, GLbitfield, GLboolean, GLchar, GLeglImageOES, GLenum, GLfloat, GLint,
        GLintptr, GLsizei, GLsizeiptr, GLuint,
    };
    use core::ffi::c_void;

    /// A resolver lists a spelling per API, and asks only the one its table was loaded under: a
    /// driver may answer any name it knows with a stub, so a spelling the API does not have is
    /// not something to fall back to.
    macro_rules! resolver {
        ($name:ident: $sig:ty = gles [$($es:ident),*], gl [$($gl:ident),*]) => {
            pub fn $name(t: &Procs, api: Api) -> Option<$sig> {
                let spellings: &[fn(&Procs) -> Option<$sig>] = match api {
                    Api::Gles(_) => &[$(Procs::$es),*],
                    Api::Gl(_) => &[$(Procs::$gl),*],
                };
                spellings.iter().find_map(|spelling| spelling(t))
            }
        };
    }

    resolver!(color_mask_i: unsafe extern "C" fn(GLuint, GLboolean, GLboolean, GLboolean, GLboolean)
        = gles [try_glColorMaski, try_glColorMaskiEXT, try_glColorMaskiOES], gl [try_glColorMaski]);
    resolver!(enable_i: unsafe extern "C" fn(GLenum, GLuint)
        = gles [try_glEnablei, try_glEnableiEXT, try_glEnableiOES], gl [try_glEnablei]);
    resolver!(disable_i: unsafe extern "C" fn(GLenum, GLuint)
        = gles [try_glDisablei, try_glDisableiEXT, try_glDisableiOES], gl [try_glDisablei]);
    resolver!(blend_func_separate_i: unsafe extern "C" fn(GLuint, GLenum, GLenum, GLenum, GLenum)
        = gles [try_glBlendFuncSeparatei, try_glBlendFuncSeparateiEXT, try_glBlendFuncSeparateiOES], gl [try_glBlendFuncSeparatei]);
    resolver!(blend_equation_separate_i: unsafe extern "C" fn(GLuint, GLenum, GLenum)
        = gles [try_glBlendEquationSeparatei, try_glBlendEquationSeparateiEXT, try_glBlendEquationSeparateiOES], gl [try_glBlendEquationSeparatei]);
    resolver!(min_sample_shading: unsafe extern "C" fn(GLfloat)
        = gles [try_glMinSampleShading, try_glMinSampleShadingOES], gl [try_glMinSampleShading]);
    resolver!(clip_control: unsafe extern "C" fn(GLenum, GLenum)
        = gles [try_glClipControlEXT], gl [try_glClipControl]);
    resolver!(viewport_indexed: unsafe extern "C" fn(GLuint, GLfloat, GLfloat, GLfloat, GLfloat)
        = gles [try_glViewportIndexedfOES], gl [try_glViewportIndexedf]);
    resolver!(scissor_indexed: unsafe extern "C" fn(GLuint, GLint, GLint, GLsizei, GLsizei)
        = gles [try_glScissorIndexedOES], gl [try_glScissorIndexed]);
    resolver!(depth_range_indexed_f: unsafe extern "C" fn(GLuint, GLfloat, GLfloat)
        = gles [try_glDepthRangeIndexedfOES], gl []);
    resolver!(depth_range_indexed_d: unsafe extern "C" fn(GLuint, f64, f64)
        = gles [], gl [try_glDepthRangeIndexed]);
    resolver!(polygon_offset_clamp: unsafe extern "C" fn(GLfloat, GLfloat, GLfloat)
        = gles [try_glPolygonOffsetClampEXT],
          gl [try_glPolygonOffsetClamp, try_glPolygonOffsetClampEXT]);
    resolver!(sampler_parameter_iuiv: unsafe extern "C" fn(GLuint, GLenum, *const GLuint)
        = gles [try_glSamplerParameterIuiv, try_glSamplerParameterIuivEXT, try_glSamplerParameterIuivOES], gl [try_glSamplerParameterIuiv]);
    resolver!(framebuffer_texture: unsafe extern "C" fn(GLenum, GLenum, GLuint, GLint)
        = gles [try_glFramebufferTexture, try_glFramebufferTextureEXT, try_glFramebufferTextureOES], gl [try_glFramebufferTexture]);
    resolver!(framebuffer_texture_3d: unsafe extern "C" fn(GLenum, GLenum, GLenum, GLuint, GLint, GLint)
        = gles [try_glFramebufferTexture3DOES], gl [try_glFramebufferTexture3D]);
    resolver!(framebuffer_texture_2d_multisample: unsafe extern "C" fn(GLenum, GLenum, GLenum, GLuint, GLint, GLsizei)
        = gles [try_glFramebufferTexture2DMultisampleEXT], gl []);
    resolver!(draw_transform_feedback: unsafe extern "C" fn(GLenum, GLuint)
        = gles [], gl [try_glDrawTransformFeedback]);
    resolver!(draw_transform_feedback_instanced: unsafe extern "C" fn(GLenum, GLuint, GLsizei)
        = gles [], gl [try_glDrawTransformFeedbackInstanced]);
    resolver!(texture_view: unsafe extern "C" fn(GLuint, GLenum, GLuint, GLenum, GLuint, GLuint, GLuint, GLuint)
        = gles [try_glTextureViewOES, try_glTextureViewEXT], gl [try_glTextureView]);
    resolver!(egl_image_target_tex_storage: unsafe extern "C" fn(GLenum, GLeglImageOES, *const GLint)
        = gles [try_glEGLImageTargetTexStorageEXT], gl [try_glEGLImageTargetTexStorageEXT]);
    resolver!(egl_image_target_texture_2d: unsafe extern "C" fn(GLenum, GLeglImageOES)
        = gles [try_glEGLImageTargetTexture2DOES], gl [try_glEGLImageTargetTexture2DOES]);
    resolver!(copy_image_sub_data: unsafe extern "C" fn(GLuint, GLenum, GLint, GLint, GLint, GLint, GLuint, GLenum, GLint, GLint, GLint, GLint, GLsizei, GLsizei, GLsizei)
        = gles [try_glCopyImageSubData, try_glCopyImageSubDataEXT, try_glCopyImageSubDataOES], gl [try_glCopyImageSubData]);
    resolver!(tex_buffer: unsafe extern "C" fn(GLenum, GLenum, GLuint)
        = gles [try_glTexBuffer, try_glTexBufferEXT, try_glTexBufferOES], gl [try_glTexBuffer]);
    resolver!(tex_buffer_range: unsafe extern "C" fn(GLenum, GLenum, GLuint, GLintptr, GLsizeiptr)
        = gles [try_glTexBufferRange, try_glTexBufferRangeEXT, try_glTexBufferRangeOES], gl [try_glTexBufferRange]);
    resolver!(clear_tex_sub_image: unsafe extern "C" fn(GLuint, GLint, GLint, GLint, GLint, GLsizei, GLsizei, GLsizei, GLenum, GLenum, *const c_void)
        = gles [try_glClearTexSubImageEXT], gl [try_glClearTexSubImage]);
    resolver!(bind_frag_data_location_indexed: unsafe extern "C" fn(GLuint, GLuint, GLuint, *const GLchar)
        = gles [try_glBindFragDataLocationIndexedEXT], gl [try_glBindFragDataLocationIndexed]);
    resolver!(draw_arrays_instanced_base_instance: unsafe extern "C" fn(GLenum, GLint, GLsizei, GLsizei, GLuint)
        = gles [try_glDrawArraysInstancedBaseInstanceEXT], gl [try_glDrawArraysInstancedBaseInstance]);
    resolver!(draw_elements_instanced_base_instance: unsafe extern "C" fn(GLenum, GLsizei, GLenum, *const c_void, GLsizei, GLuint)
        = gles [try_glDrawElementsInstancedBaseInstanceEXT], gl [try_glDrawElementsInstancedBaseInstance]);
    resolver!(draw_elements_instanced_base_vertex_base_instance: unsafe extern "C" fn(GLenum, GLsizei, GLenum, *const c_void, GLsizei, GLint, GLuint)
        = gles [try_glDrawElementsInstancedBaseVertexBaseInstanceEXT], gl [try_glDrawElementsInstancedBaseVertexBaseInstance]);
    resolver!(multi_draw_arrays_indirect: unsafe extern "C" fn(GLenum, *const c_void, GLsizei, GLsizei)
        = gles [try_glMultiDrawArraysIndirectEXT], gl [try_glMultiDrawArraysIndirect]);
    resolver!(multi_draw_elements_indirect: unsafe extern "C" fn(GLenum, GLenum, *const c_void, GLsizei, GLsizei)
        = gles [try_glMultiDrawElementsIndirectEXT], gl [try_glMultiDrawElementsIndirect]);
    resolver!(multi_draw_arrays_indirect_count: unsafe extern "C" fn(GLenum, *const c_void, GLintptr, GLsizei, GLsizei)
        = gles [], gl [try_glMultiDrawArraysIndirectCount, try_glMultiDrawArraysIndirectCountARB]);
    resolver!(multi_draw_elements_indirect_count: unsafe extern "C" fn(GLenum, GLenum, *const c_void, GLintptr, GLsizei, GLsizei)
        = gles [], gl [try_glMultiDrawElementsIndirectCount, try_glMultiDrawElementsIndirectCountARB]);
    resolver!(bind_program_pipeline: unsafe extern "C" fn(GLuint)
        = gles [try_glBindProgramPipeline, try_glBindProgramPipelineEXT], gl [try_glBindProgramPipeline]);
    resolver!(gen_program_pipelines: unsafe extern "C" fn(GLsizei, *mut GLuint)
        = gles [try_glGenProgramPipelines], gl [try_glGenProgramPipelines]);
    resolver!(delete_program_pipelines: unsafe extern "C" fn(GLsizei, *const GLuint)
        = gles [try_glDeleteProgramPipelines], gl [try_glDeleteProgramPipelines]);
    resolver!(use_program_stages: unsafe extern "C" fn(GLuint, GLbitfield, GLuint)
        = gles [try_glUseProgramStages], gl [try_glUseProgramStages]);
    resolver!(active_shader_program: unsafe extern "C" fn(GLuint, GLuint)
        = gles [try_glActiveShaderProgram], gl [try_glActiveShaderProgram]);
    resolver!(validate_program_pipeline: unsafe extern "C" fn(GLuint)
        = gles [try_glValidateProgramPipeline], gl [try_glValidateProgramPipeline]);
    resolver!(get_program_pipeline_iv: unsafe extern "C" fn(GLuint, GLenum, *mut GLint)
        = gles [try_glGetProgramPipelineiv], gl [try_glGetProgramPipelineiv]);
    resolver!(program_parameter_i: unsafe extern "C" fn(GLuint, GLenum, GLint)
        = gles [try_glProgramParameteri], gl [try_glProgramParameteri]);
    resolver!(tex_storage_3d_multisample: unsafe extern "C" fn(GLenum, GLsizei, GLenum, GLsizei, GLsizei, GLsizei, GLboolean)
        = gles [try_glTexStorage3DMultisample], gl [try_glTexStorage3DMultisample]);
    resolver!(buffer_storage: unsafe extern "C" fn(GLenum, GLsizeiptr, *const c_void, GLbitfield)
        = gles [try_glBufferStorageEXT], gl [try_glBufferStorage]);

    /// One proc a feature stands for: the feature, the name a message gives it, and whether the
    /// driver exported any of its spellings.
    pub type Behind = (Feature, &'static str, fn(&Procs, Api) -> bool);

    /// Every proc a feature stands for, with the feature and the name a message gives it.
    pub const BEHIND: &[Behind] = &[
        (Feature::indep_blend, "glColorMaski", |t, a| color_mask_i(t, a).is_some()),
        (Feature::indep_blend, "glEnablei", |t, a| enable_i(t, a).is_some()),
        (Feature::indep_blend, "glDisablei", |t, a| disable_i(t, a).is_some()),
        (Feature::indep_blend_func, "glBlendFuncSeparatei", |t, a| {
            blend_func_separate_i(t, a).is_some()
        }),
        (Feature::indep_blend_func, "glBlendEquationSeparatei", |t, a| {
            blend_equation_separate_i(t, a).is_some()
        }),
        (Feature::sample_shading, "glMinSampleShading", |t, a| min_sample_shading(t, a).is_some()),
        (Feature::clip_control, "glClipControlEXT", |t, a| clip_control(t, a).is_some()),
        (Feature::viewport_array, "glViewportIndexedf", |t, a| viewport_indexed(t, a).is_some()),
        (Feature::viewport_array, "glScissorIndexed", |t, a| scissor_indexed(t, a).is_some()),
        (Feature::viewport_array, "glDepthRangeIndexed", |t, a| {
            depth_range_indexed_f(t, a).is_some() || depth_range_indexed_d(t, a).is_some()
        }),
        (Feature::polygon_offset_clamp, "glPolygonOffsetClamp", |t, a| {
            polygon_offset_clamp(t, a).is_some()
        }),
        (Feature::sampler_border_colors, "glSamplerParameterIuiv", |t, a| {
            sampler_parameter_iuiv(t, a).is_some()
        }),
        (Feature::geometry_shader, "glFramebufferTexture", |t, a| {
            framebuffer_texture(t, a).is_some()
        }),
        (Feature::texture_3d_attach, "glFramebufferTexture3DOES", |t, a| {
            framebuffer_texture_3d(t, a).is_some()
        }),
        (Feature::implicit_msaa, "glFramebufferTexture2DMultisampleEXT", |t, a| {
            framebuffer_texture_2d_multisample(t, a).is_some()
        }),
        (Feature::texture_view, "glTextureView", |t, a| texture_view(t, a).is_some()),
        (Feature::transform_feedback_draw, "glDrawTransformFeedback", |t, a| {
            draw_transform_feedback(t, a).is_some()
        }),
        (Feature::transform_feedback_instanced, "glDrawTransformFeedbackInstanced", |t, a| {
            draw_transform_feedback_instanced(t, a).is_some()
        }),
        (Feature::egl_image_storage, "glEGLImageTargetTexStorageEXT", |t, a| {
            egl_image_target_tex_storage(t, a).is_some()
        }),
        (Feature::egl_image, "glEGLImageTargetTexture2DOES", |t, a| {
            egl_image_target_texture_2d(t, a).is_some()
        }),
        (Feature::copy_image, "glCopyImageSubData", |t, a| copy_image_sub_data(t, a).is_some()),
        (Feature::arb_or_gles_ext_texture_buffer, "glTexBuffer", |t, a| tex_buffer(t, a).is_some()),
        (Feature::texture_buffer_range, "glTexBufferRange", |t, a| {
            tex_buffer_range(t, a).is_some()
        }),
        (Feature::clear_texture, "glClearTexSubImageEXT", |t, a| {
            clear_tex_sub_image(t, a).is_some()
        }),
        (Feature::dual_src_blend, "glBindFragDataLocationIndexedEXT", |t, a| {
            bind_frag_data_location_indexed(t, a).is_some()
        }),
        (Feature::base_instance, "glDrawArraysInstancedBaseInstanceEXT", |t, a| {
            draw_arrays_instanced_base_instance(t, a).is_some()
        }),
        (Feature::base_instance, "glDrawElementsInstancedBaseInstanceEXT", |t, a| {
            draw_elements_instanced_base_instance(t, a).is_some()
        }),
        (Feature::base_instance, "glDrawElementsInstancedBaseVertexBaseInstanceEXT", |t, a| {
            draw_elements_instanced_base_vertex_base_instance(t, a).is_some()
        }),
        (Feature::multi_draw_indirect, "glMultiDrawArraysIndirectEXT", |t, a| {
            multi_draw_arrays_indirect(t, a).is_some()
        }),
        (Feature::multi_draw_indirect, "glMultiDrawElementsIndirectEXT", |t, a| {
            multi_draw_elements_indirect(t, a).is_some()
        }),
        (Feature::indirect_params, "glMultiDrawArraysIndirectCount", |t, a| {
            multi_draw_arrays_indirect_count(t, a).is_some()
        }),
        (Feature::indirect_params, "glMultiDrawElementsIndirectCount", |t, a| {
            multi_draw_elements_indirect_count(t, a).is_some()
        }),
        (Feature::separate_shader_objects, "glBindProgramPipeline", |t, a| {
            bind_program_pipeline(t, a).is_some()
        }),
        (Feature::separate_shader_objects, "glGenProgramPipelines", |t, a| {
            gen_program_pipelines(t, a).is_some()
        }),
        (Feature::separate_shader_objects, "glDeleteProgramPipelines", |t, a| {
            delete_program_pipelines(t, a).is_some()
        }),
        (Feature::separate_shader_objects, "glUseProgramStages", |t, a| {
            use_program_stages(t, a).is_some()
        }),
        (Feature::separate_shader_objects, "glActiveShaderProgram", |t, a| {
            active_shader_program(t, a).is_some()
        }),
        (Feature::separate_shader_objects, "glValidateProgramPipeline", |t, a| {
            validate_program_pipeline(t, a).is_some()
        }),
        (Feature::separate_shader_objects, "glGetProgramPipelineiv", |t, a| {
            get_program_pipeline_iv(t, a).is_some()
        }),
        (Feature::separate_shader_objects, "glProgramParameteri", |t, a| {
            program_parameter_i(t, a).is_some()
        }),
        (Feature::storage_multisample_2d_array, "glTexStorage3DMultisample", |t, a| {
            tex_storage_3d_multisample(t, a).is_some()
        }),
        (Feature::arb_buffer_storage, "glBufferStorageEXT", |t, a| buffer_storage(t, a).is_some()),
    ];
}

/// `GL_VERSION`, read through a table before a [`Gl`] exists: the answer is what decides which API
/// the `Gl` is built for.
pub fn version_string(t: &Procs) -> String {
    // SAFETY: `glGetString` returns null or a NUL-terminated string owned by the driver, live
    // while the context is; it is copied out immediately.
    let p = unsafe { t.glGetString()(GL_VERSION) };
    if p.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(p.cast::<c_char>()) }.to_string_lossy().into_owned()
}

/// `GL_CONTEXT_PROFILE_MASK`, read as [`version_string`] is. Meaningful on desktop GL only.
pub fn profile_mask(t: &Procs) -> GLenum {
    let mut v: GLint = 0;
    // SAFETY: `GL_CONTEXT_PROFILE_MASK` writes exactly one integer.
    unsafe { t.glGetIntegerv()(GL_CONTEXT_PROFILE_MASK, &mut v) };
    v as GLenum
}

/// The driver's entry points behind a safe surface.
pub struct Gl {
    t: Procs,
    /// The API the context the table was loaded under speaks, which picks each resolver's
    /// spelling.
    api: Api,
    /// Which bounded read entry points this driver serves.
    robust: RobustReads,
}

/// The bounded spellings of the reads -- `glReadnPixels*`, `glGetnTexImage*` -- that this driver
/// serves, chosen as the C chooses them: by what the driver advertises, never by whether a name
/// resolves. Mesa answers every name on either API, and a core name it only stubs writes nothing
/// and reports no error, which reads back as zeros.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RobustReads {
    /// `GL_ARB_robustness`: the `ARB` spellings of all three.
    Arb,
    /// Desktop GL 4.5 without the extension: `glReadnPixels` only.
    Core,
    /// `GL_KHR_robustness` on GLES: `glReadnPixelsKHR` only.
    Khr,
    /// None: the unbounded calls, sized from what the driver reports.
    Unbounded,
}

impl RobustReads {
    /// The C's order in `do_readpixels`.
    pub fn choose(api: Api, features: &Features) -> RobustReads {
        if features.has(Feature::arb_robustness) {
            RobustReads::Arb
        } else if api.gl_at_least(45) {
            RobustReads::Core
        } else if features.has(Feature::gles_khr_robustness) {
            RobustReads::Khr
        } else {
            RobustReads::Unbounded
        }
    }
}

impl Gl {
    /// A table with no bounded reads chosen: [`Gl::reading`] picks them once the features are
    /// probed, which they cannot be before there is a `Gl` to ask.
    pub fn new(t: Procs, api: Api) -> Gl {
        Gl { t, api, robust: RobustReads::Unbounded }
    }

    /// The API this table was resolved for.
    pub fn api(&self) -> Api {
        self.api
    }

    /// The bounded reads this driver serves.
    pub fn reading(self, robust: RobustReads) -> Gl {
        Gl { robust, ..self }
    }

    /// The raw table, for the census of what the driver exports.
    pub fn table(&self) -> &Procs {
        &self.t
    }

    /// The features whose entry points the driver did not hand over, each with the proc it
    /// lacks. `Features::reconcile` withdraws them.
    pub fn missing_procs(&self) -> Vec<(Feature, &'static str)> {
        procs::BEHIND
            .iter()
            .filter(|(_, _, present)| !present(&self.t, self.api))
            .map(|(f, name, _)| (*f, *name))
            .collect()
    }

    // ---- queries ----

    pub fn get_error(&self) -> GLenum {
        // SAFETY: takes nothing.
        unsafe { self.t.glGetError()() }
    }

    /// Drain the error queue, returning the first error or `GL_NO_ERROR`.
    pub fn drain_errors(&self) -> GLenum {
        let first = self.get_error();
        if first != GL_NO_ERROR {
            while self.get_error() != GL_NO_ERROR {}
        }
        first
    }

    pub fn get_integer(&self, name: GLenum) -> GLint {
        let mut v: GLint = 0;
        // SAFETY: every name this crate asks for writes exactly one integer.
        unsafe { self.t.glGetIntegerv()(name, &mut v) };
        v
    }

    /// `GL_VBO_FREE_MEMORY_ATI`: free memory, the largest free block, free auxiliary memory and
    /// its largest free block, in KiB. The one query this crate makes that writes four integers.
    pub fn vbo_free_memory_ati(&self) -> [GLint; 4] {
        let mut v: [GLint; 4] = [0; 4];
        // SAFETY: `GL_VBO_FREE_MEMORY_ATI` writes four integers, and `v` holds four.
        unsafe { self.t.glGetIntegerv()(GL_VBO_FREE_MEMORY_ATI, v.as_mut_ptr()) };
        v
    }

    /// `glGetIntegeri_v`: one integer of an indexed state.
    pub fn get_integer_i(&self, name: GLenum, index: GLuint) -> GLint {
        let mut v: GLint = 0;
        // SAFETY: every indexed name this crate asks for writes exactly one integer.
        unsafe { self.t.glGetIntegeri_v()(name, index, &mut v) };
        v
    }

    pub fn get_float(&self, name: GLenum) -> GLfloat {
        let mut v: GLfloat = 0.0;
        // SAFETY: every name this crate asks for through here writes exactly one float.
        unsafe { self.t.glGetFloatv()(name, &mut v) };
        v
    }

    /// `glGetFloatv` for a two-float range (`GL_ALIASED_POINT_SIZE_RANGE` and its kin).
    pub fn get_float_range(&self, name: GLenum) -> [GLfloat; 2] {
        let mut v: [GLfloat; 2] = [0.0; 2];
        // SAFETY: every name this crate asks for through here writes exactly two floats.
        unsafe { self.t.glGetFloatv()(name, v.as_mut_ptr()) };
        v
    }

    /// `glGetMultisamplefv(GL_SAMPLE_POSITION, index)`: where a sample sits in its pixel.
    pub fn get_sample_position(&self, index: GLuint) -> [GLfloat; 2] {
        let mut v: [GLfloat; 2] = [0.0; 2];
        // SAFETY: `GL_SAMPLE_POSITION` writes exactly two floats.
        unsafe { self.t.glGetMultisamplefv()(GL_SAMPLE_POSITION, index, v.as_mut_ptr()) };
        v
    }

    pub fn get_string(&self, name: GLenum) -> String {
        // SAFETY: `glGetString` returns null or a NUL-terminated string owned by the driver, live
        // while the context is; it is copied out immediately.
        let p = unsafe { self.t.glGetString()(name) };
        if p.is_null() {
            return String::new();
        }
        unsafe { CStr::from_ptr(p.cast::<c_char>()) }.to_string_lossy().into_owned()
    }

    pub fn get_string_i(&self, name: GLenum, index: GLuint) -> String {
        // SAFETY: as `get_string`; an out-of-range index answers null.
        let p = unsafe { self.t.glGetStringi()(name, index) };
        if p.is_null() {
            return String::new();
        }
        unsafe { CStr::from_ptr(p.cast::<c_char>()) }.to_string_lossy().into_owned()
    }

    /// Every extension the context advertises.
    pub fn extensions(&self) -> Vec<String> {
        let n = self.get_integer(GL_NUM_EXTENSIONS).max(0) as GLuint;
        (0..n).map(|i| self.get_string_i(GL_EXTENSIONS, i)).collect()
    }

    // ---- objects ----

    pub fn gen_texture(&self) -> TextureName {
        let mut id: GLuint = 0;
        // SAFETY: room for the one name asked for.
        unsafe { self.t.glGenTextures()(1, &mut id) };
        TextureName(id)
    }

    pub fn delete_texture(&self, tex: TextureName) {
        // SAFETY: one name, read from a live local.
        unsafe { self.t.glDeleteTextures()(1, &tex.0) };
    }

    pub fn bind_texture(&self, target: GLenum, tex: Option<TextureName>) {
        // SAFETY: plain scalars.
        unsafe { self.t.glBindTexture()(target, tex.map_or(0, |t| t.0)) };
    }

    /// Put back a binding read out of GL with [`Gl::get_integer`].
    ///
    /// Raw rather than a [`TextureName`] on purpose: this name came from the driver and may
    /// belong to anything, including zero. Minting a `TextureName` from it would claim it names
    /// a texture this renderer owns, which is the claim the newtype exists to make.
    pub fn bind_texture_name(&self, target: GLenum, name: GLuint) {
        // SAFETY: plain scalars.
        unsafe { self.t.glBindTexture()(target, name) };
    }

    pub fn gen_buffer(&self) -> BufferName {
        let mut id: GLuint = 0;
        // SAFETY: room for the one name asked for.
        unsafe { self.t.glGenBuffers()(1, &mut id) };
        BufferName(id)
    }

    pub fn delete_buffer(&self, buf: BufferName) {
        // SAFETY: one name, read from a live local.
        unsafe { self.t.glDeleteBuffers()(1, &buf.0) };
    }

    pub fn bind_buffer(&self, target: GLenum, buf: Option<BufferName>) {
        // SAFETY: plain scalars.
        unsafe { self.t.glBindBuffer()(target, buf.map_or(0, |b| b.0)) };
    }

    pub fn gen_framebuffer(&self) -> FramebufferName {
        let mut id: GLuint = 0;
        // SAFETY: room for the one name asked for.
        unsafe { self.t.glGenFramebuffers()(1, &mut id) };
        FramebufferName(id)
    }

    pub fn delete_framebuffer(&self, fb: FramebufferName) {
        // SAFETY: one name, read from a live local.
        unsafe { self.t.glDeleteFramebuffers()(1, &fb.0) };
    }

    /// What is bound to `GL_FRAMEBUFFER` right now, `None` for the default one.
    ///
    /// Read back from the driver rather than tracked here: a cached copy would be a second
    /// version of a truth GL already holds, and the two would part company at the first bind
    /// this wrapper did not make.
    pub fn framebuffer_binding(&self) -> Option<FramebufferName> {
        match self.get_integer(GL_FRAMEBUFFER_BINDING) {
            0 => None,
            id => Some(FramebufferName(id as GLuint)),
        }
    }

    pub fn bind_framebuffer(&self, target: GLenum, fb: Option<FramebufferName>) {
        // SAFETY: plain scalars.
        unsafe { self.t.glBindFramebuffer()(target, fb.map_or(0, |f| f.0)) };
    }

    pub fn gen_renderbuffer(&self) -> RenderbufferName {
        let mut id: GLuint = 0;
        // SAFETY: room for the one name asked for.
        unsafe { self.t.glGenRenderbuffers()(1, &mut id) };
        RenderbufferName(id)
    }

    pub fn delete_renderbuffer(&self, rb: RenderbufferName) {
        // SAFETY: one name, read from a live local.
        unsafe { self.t.glDeleteRenderbuffers()(1, &rb.0) };
    }

    // ---- texture allocation ----

    /// `glTexImage2D` with no data: allocate a level and nothing more.
    #[allow(clippy::too_many_arguments)]
    pub fn tex_image_2d_null(
        &self,
        target: GLenum,
        level: GLint,
        internalformat: GLenum,
        w: GLsizei,
        h: GLsizei,
        format: GLenum,
        ty: GLenum,
    ) {
        // SAFETY: a null pixel pointer with no pixel unpack buffer bound reads nothing.
        unsafe {
            self.t.glTexImage2D()(
                target,
                level,
                internalformat as GLint,
                w,
                h,
                0,
                format,
                ty,
                core::ptr::null(),
            )
        };
    }

    /// `glTexImage1D` with no data. Desktop GL only: GLES has no 1D textures.
    pub fn tex_image_1d_null(
        &self,
        level: GLint,
        internalformat: GLenum,
        w: GLsizei,
        format: GLenum,
        ty: GLenum,
    ) {
        assert!(!self.api.is_gles(), "glTexImage1D on a GLES context");
        // SAFETY: a null pixel pointer with no pixel unpack buffer bound reads nothing.
        unsafe {
            self.t.glTexImage1D()(
                GL_TEXTURE_1D,
                level,
                internalformat as GLint,
                w,
                0,
                format,
                ty,
                core::ptr::null(),
            )
        };
    }

    /// `glTexStorage1D`. Desktop GL only.
    pub fn tex_storage_1d(&self, levels: GLsizei, internalformat: GLenum, w: GLsizei) {
        assert!(!self.api.is_gles(), "glTexStorage1D on a GLES context");
        // SAFETY: plain scalars.
        unsafe { self.t.glTexStorage1D()(GL_TEXTURE_1D, levels, internalformat, w) };
    }

    /// `glTexImage3D` with no data.
    #[allow(clippy::too_many_arguments)]
    pub fn tex_image_3d_null(
        &self,
        target: GLenum,
        level: GLint,
        internalformat: GLenum,
        w: GLsizei,
        h: GLsizei,
        d: GLsizei,
        format: GLenum,
        ty: GLenum,
    ) {
        // SAFETY: a null pixel pointer with no pixel unpack buffer bound reads nothing.
        unsafe {
            self.t.glTexImage3D()(
                target,
                level,
                internalformat as GLint,
                w,
                h,
                d,
                0,
                format,
                ty,
                core::ptr::null(),
            )
        };
    }

    pub fn tex_storage_2d(
        &self,
        target: GLenum,
        levels: GLsizei,
        internalformat: GLenum,
        w: GLsizei,
        h: GLsizei,
    ) {
        // SAFETY: plain scalars.
        unsafe { self.t.glTexStorage2D()(target, levels, internalformat, w, h) };
    }

    pub fn tex_storage_3d(
        &self,
        target: GLenum,
        levels: GLsizei,
        internalformat: GLenum,
        w: GLsizei,
        h: GLsizei,
        d: GLsizei,
    ) {
        // SAFETY: plain scalars.
        unsafe { self.t.glTexStorage3D()(target, levels, internalformat, w, h, d) };
    }

    pub fn tex_storage_2d_multisample(
        &self,
        target: GLenum,
        samples: GLsizei,
        internalformat: GLenum,
        w: GLsizei,
        h: GLsizei,
    ) {
        // SAFETY: plain scalars.
        unsafe {
            self.t.glTexStorage2DMultisample()(target, samples, internalformat, w, h, GL_TRUE as _)
        };
    }

    #[allow(clippy::too_many_arguments)]
    pub fn tex_storage_3d_multisample(
        &self,
        target: GLenum,
        samples: GLsizei,
        internalformat: GLenum,
        w: GLsizei,
        h: GLsizei,
        d: GLsizei,
    ) {
        let f = promised(
            procs::tex_storage_3d_multisample(&self.t, self.api),
            Feature::storage_multisample_2d_array,
            "glTexStorage3DMultisample",
        );
        // SAFETY: plain scalars.
        unsafe { f(target, samples, internalformat, w, h, d, GL_TRUE as _) };
    }

    pub fn tex_parameter_i(&self, target: GLenum, name: GLenum, value: GLint) {
        // SAFETY: plain scalars.
        unsafe { self.t.glTexParameteri()(target, name, value) };
    }

    /// `glGetTexParameteriv` for a parameter with one integer answer, on the bound texture.
    pub fn get_tex_parameter_i(&self, target: GLenum, name: GLenum) -> GLint {
        let mut value: GLint = 0;
        // SAFETY: every parameter this is called with has exactly one integer answer, so the
        // driver writes one `GLint` through the pointer, and it points at one.
        unsafe { self.t.glGetTexParameteriv()(target, name, &raw mut value) };
        value
    }

    pub fn pixel_store_i(&self, name: GLenum, value: GLint) {
        // SAFETY: plain scalars.
        unsafe { self.t.glPixelStorei()(name, value) };
    }

    // ---- texture data ----

    /// Upload a tightly packed `w`×`h` image. Refuses, without calling the driver, a slice shorter
    /// than the image or a format/type pair this crate cannot size.
    #[allow(clippy::too_many_arguments)]
    pub fn tex_sub_image_2d(
        &self,
        target: GLenum,
        level: GLint,
        x: GLint,
        y: GLint,
        w: GLsizei,
        h: GLsizei,
        format: GLenum,
        ty: GLenum,
        data: &[u8],
    ) -> bool {
        let Some(need) = image_bytes(format, ty, w, h, 1) else {
            return false;
        };
        if data.len() < need {
            return false;
        }
        // SAFETY: with unpack row length and image height zero and alignment 1 -- which
        // `Gl::unpack_tight` sets and every caller uses -- the driver reads exactly `need` bytes
        // from `data`, and `data` holds at least that many.
        unsafe {
            self.t.glTexSubImage2D()(target, level, x, y, w, h, format, ty, data.as_ptr().cast())
        };
        true
    }

    /// `glTexSubImage1D`. Desktop GL only.
    #[allow(clippy::too_many_arguments)]
    pub fn tex_sub_image_1d(
        &self,
        level: GLint,
        x: GLint,
        w: GLsizei,
        format: GLenum,
        ty: GLenum,
        data: &[u8],
    ) -> bool {
        assert!(!self.api.is_gles(), "glTexSubImage1D on a GLES context");
        let Some(need) = image_bytes(format, ty, w, 1, 1) else {
            return false;
        };
        if data.len() < need {
            return false;
        }
        // SAFETY: with alignment 1 -- which `Gl::unpack_tight` sets and every caller uses -- the
        // driver reads exactly `need` bytes from `data`, and `data` holds at least that many.
        unsafe {
            self.t.glTexSubImage1D()(GL_TEXTURE_1D, level, x, w, format, ty, data.as_ptr().cast())
        };
        true
    }

    /// `glCompressedTexSubImage1D`. Desktop GL only; GL reads exactly `data.len()` bytes.
    pub fn compressed_tex_sub_image_1d(
        &self,
        level: GLint,
        x: GLint,
        w: GLsizei,
        format: GLenum,
        data: &[u8],
    ) -> bool {
        assert!(!self.api.is_gles(), "glCompressedTexSubImage1D on a GLES context");
        let Ok(size) = GLsizei::try_from(data.len()) else {
            return false;
        };
        // SAFETY: the driver reads `size` bytes from `data`, and `size` is `data`'s length.
        unsafe {
            self.t.glCompressedTexSubImage1D()(
                GL_TEXTURE_1D,
                level,
                x,
                w,
                format,
                size,
                data.as_ptr().cast(),
            )
        };
        true
    }

    #[allow(clippy::too_many_arguments)]
    pub fn tex_sub_image_3d(
        &self,
        target: GLenum,
        level: GLint,
        x: GLint,
        y: GLint,
        z: GLint,
        w: GLsizei,
        h: GLsizei,
        d: GLsizei,
        format: GLenum,
        ty: GLenum,
        data: &[u8],
    ) -> bool {
        let Some(need) = image_bytes(format, ty, w, h, d) else {
            return false;
        };
        if data.len() < need {
            return false;
        }
        // SAFETY: as `tex_sub_image_2d`, over `d` tightly packed layers.
        unsafe {
            self.t.glTexSubImage3D()(
                target,
                level,
                x,
                y,
                z,
                w,
                h,
                d,
                format,
                ty,
                data.as_ptr().cast(),
            )
        };
        true
    }

    /// Upload compressed blocks. GL reads exactly `data.len()` bytes -- the length is the
    /// `imageSize` argument -- so there is nothing to check but that the slice fits a `GLsizei`.
    #[allow(clippy::too_many_arguments)]
    pub fn compressed_tex_sub_image_2d(
        &self,
        target: GLenum,
        level: GLint,
        x: GLint,
        y: GLint,
        w: GLsizei,
        h: GLsizei,
        format: GLenum,
        data: &[u8],
    ) -> bool {
        let Ok(size) = GLsizei::try_from(data.len()) else {
            return false;
        };
        // SAFETY: the driver reads `size` bytes from `data`, and `size` is `data`'s length.
        unsafe {
            self.t.glCompressedTexSubImage2D()(
                target,
                level,
                x,
                y,
                w,
                h,
                format,
                size,
                data.as_ptr().cast(),
            )
        };
        true
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compressed_tex_sub_image_3d(
        &self,
        target: GLenum,
        level: GLint,
        x: GLint,
        y: GLint,
        z: GLint,
        w: GLsizei,
        h: GLsizei,
        d: GLsizei,
        format: GLenum,
        data: &[u8],
    ) -> bool {
        let Ok(size) = GLsizei::try_from(data.len()) else {
            return false;
        };
        // SAFETY: as `compressed_tex_sub_image_2d`.
        unsafe {
            self.t.glCompressedTexSubImage3D()(
                target,
                level,
                x,
                y,
                z,
                w,
                h,
                d,
                format,
                size,
                data.as_ptr().cast(),
            )
        };
        true
    }

    /// Read a tightly packed `w`×`h` image out of the read framebuffer into `dst`. Refuses a
    /// destination shorter than the image. Sets the pack state it reads under, so no caller has
    /// to have.
    #[allow(clippy::too_many_arguments)]
    pub fn read_pixels(
        &self,
        x: GLint,
        y: GLint,
        w: GLsizei,
        h: GLsizei,
        format: GLenum,
        ty: GLenum,
        dst: &mut [u8],
    ) -> bool {
        let Some(need) = image_bytes(format, ty, w, h, 1) else {
            return false;
        };
        if dst.len() < need {
            return false;
        }
        self.pack_tight();
        // Desktop GL clamps a read from a normalized buffer to [0, 1] unless told not to, which
        // turns every negative snorm value into zero; the guest's own GL does that clamping where
        // its state asks for it. GLES never clamps a read.
        if !self.api.is_gles() {
            // SAFETY: plain scalars.
            unsafe { self.t.glClampColor()(GL_CLAMP_READ_COLOR, GL_FALSE as GLenum) };
        }
        // Robust readback where the driver has it: the bound is the slice's own length, so
        // whatever the driver believes the image is, it cannot write past `dst`.
        let bounded = match self.robust {
            RobustReads::Arb => self.t.try_glReadnPixelsARB(),
            RobustReads::Core => self.t.try_glReadnPixels(),
            RobustReads::Khr => self.t.try_glReadnPixelsKHR(),
            RobustReads::Unbounded => None,
        };
        if let Some(f) = bounded
            && let Ok(size) = GLsizei::try_from(dst.len())
        {
            // SAFETY: the driver writes at most `size` bytes into `dst`, which holds `size`.
            unsafe { f(x, y, w, h, format, ty, size, dst.as_mut_ptr().cast()) };
            return true;
        }
        // SAFETY: with the tight pack state set above, the driver writes exactly `need` bytes,
        // and `dst` holds at least that.
        unsafe { self.t.glReadPixels()(x, y, w, h, format, ty, dst.as_mut_ptr().cast()) };
        true
    }

    /// `glGetTexImage`: the whole of one level of the texture bound to `target` -- one face, for
    /// a cube-face target -- tightly packed into `dst`. Desktop GL only: GLES reads a texture
    /// through a framebuffer or not at all. The bound is `dst`'s own length under
    /// `GL_ARB_robustness`, and otherwise the level's size as the driver reports it, so a caller
    /// that sized `dst` from a stale idea of the texture gets a refusal, not an overrun. The C
    /// never calls the 4.5 core spelling here, and Mesa answers it without writing anything.
    pub fn get_tex_image(
        &self,
        target: GLenum,
        level: GLint,
        format: GLenum,
        ty: GLenum,
        dst: &mut [u8],
    ) -> bool {
        assert!(!self.api.is_gles(), "glGetTexImage on a GLES context");
        self.pack_tight();
        if self.robust == RobustReads::Arb
            && let Some(f) = self.t.try_glGetnTexImageARB()
            && let Ok(size) = GLsizei::try_from(dst.len())
        {
            // SAFETY: the driver writes at most `size` bytes into `dst`, which holds `size`.
            unsafe { f(target, level, format, ty, size, dst.as_mut_ptr().cast()) };
            return true;
        }
        let (w, h, d) = self.level_size(target, level);
        let Some(need) = image_bytes(format, ty, w, h, d) else {
            return false;
        };
        if dst.len() < need {
            return false;
        }
        // SAFETY: with the tight pack state set above, the driver writes the level's tightly
        // packed image: `need` bytes, measured from the level it will read.
        unsafe { self.t.glGetTexImage()(target, level, format, ty, dst.as_mut_ptr().cast()) };
        true
    }

    /// `glGetCompressedTexImage`: [`Gl::get_tex_image`] for a compressed level, whose blocks
    /// come back as they are stored.
    pub fn get_compressed_tex_image(&self, target: GLenum, level: GLint, dst: &mut [u8]) -> bool {
        assert!(!self.api.is_gles(), "glGetCompressedTexImage on a GLES context");
        if self.robust == RobustReads::Arb
            && let Some(f) = self.t.try_glGetnCompressedTexImageARB()
            && let Ok(size) = GLsizei::try_from(dst.len())
        {
            // SAFETY: the driver writes at most `size` bytes into `dst`, which holds `size`.
            unsafe { f(target, level, size, dst.as_mut_ptr().cast()) };
            return true;
        }
        let need = self.tex_level_parameter(target, level, GL_TEXTURE_COMPRESSED_IMAGE_SIZE);
        if usize::try_from(need).map_or(true, |need| dst.len() < need) {
            return false;
        }
        // SAFETY: the driver writes the level's compressed image, whose size it reported as
        // `need`, and `dst` holds at least that.
        unsafe { self.t.glGetCompressedTexImage()(target, level, dst.as_mut_ptr().cast()) };
        true
    }

    /// The level's width, height and depth as the driver holds them.
    fn level_size(&self, target: GLenum, level: GLint) -> (GLsizei, GLsizei, GLsizei) {
        let get = |pname| self.tex_level_parameter(target, level, pname);
        (get(GL_TEXTURE_WIDTH), get(GL_TEXTURE_HEIGHT), get(GL_TEXTURE_DEPTH))
    }

    /// The bits the driver stores of each of a level's red, green, blue and alpha.
    pub fn channel_bits(&self, target: GLenum, level: GLint) -> [u32; 4] {
        [GL_TEXTURE_RED_SIZE, GL_TEXTURE_GREEN_SIZE, GL_TEXTURE_BLUE_SIZE, GL_TEXTURE_ALPHA_SIZE]
            .map(|p| self.tex_level_parameter(target, level, p).max(0) as u32)
    }

    fn tex_level_parameter(&self, target: GLenum, level: GLint, pname: GLenum) -> GLint {
        let mut v: GLint = 0;
        // SAFETY: every parameter asked here writes exactly one integer.
        unsafe { self.t.glGetTexLevelParameteriv()(target, level, pname, &mut v) };
        v
    }

    /// Upload rows that are padded, rather than tightly packed.
    ///
    /// `row_pixels` is the source stride in *pixels*, which is what `GL_UNPACK_ROW_LENGTH`
    /// wants; a decoded video plane is the one source here whose rows the producer padded. The
    /// unpack state is set and restored around the call, so [`Gl::unpack_tight`] stays the
    /// invariant every other upload relies on.
    ///
    /// `false` if the source is too short for the rectangle, which is a caller that clamped
    /// wrongly rather than anything a guest can reach.
    #[allow(clippy::too_many_arguments)]
    #[must_use = "a refused upload leaves the texture holding the previous frame"]
    pub fn tex_sub_image_2d_padded(
        &self,
        target: GLenum,
        level: GLint,
        x: GLint,
        y: GLint,
        w: GLsizei,
        h: GLsizei,
        format: GLenum,
        ty: GLenum,
        data: &[u8],
        row_pixels: GLsizei,
    ) -> bool {
        // The driver reads `row_pixels` per row for every row but the last, and `w` for it.
        let (Some(stride), Some(last)) = (
            image_bytes(format, ty, row_pixels, h.saturating_sub(1), 1),
            image_bytes(format, ty, w, 1.min(h), 1),
        ) else {
            return false;
        };
        if row_pixels < w || data.len() < stride + last {
            return false;
        }
        self.pixel_store_i(GL_UNPACK_ROW_LENGTH, row_pixels);
        // SAFETY: with row length `row_pixels`, image height and skips zero and alignment 1, the
        // driver reads `stride + last` bytes from `data`, and `data` holds at least that many.
        unsafe {
            self.t.glTexSubImage2D()(target, level, x, y, w, h, format, ty, data.as_ptr().cast())
        };
        self.pixel_store_i(GL_UNPACK_ROW_LENGTH, 0);
        true
    }

    /// Pin the unpack state to tightly packed rows, which is what every upload here assumes.
    pub fn unpack_tight(&self) {
        self.pixel_store_i(GL_UNPACK_ROW_LENGTH, 0);
        self.pixel_store_i(GL_UNPACK_IMAGE_HEIGHT, 0);
        self.pixel_store_i(GL_UNPACK_SKIP_PIXELS, 0);
        self.pixel_store_i(GL_UNPACK_SKIP_ROWS, 0);
        self.pixel_store_i(GL_UNPACK_ALIGNMENT, 1);
    }

    /// Pin the pack state to tightly packed rows, which is what every readback here assumes.
    ///
    /// `GL_PACK_ALIGNMENT` is the load-bearing one and its default is 4, not 1. Readbacks compute
    /// their offsets from the format's own stride, so a row that is not a multiple of four bytes
    /// -- a 127-wide R8 image, say -- would come back padded to a stride nothing else here knows
    /// about: the rows would be read at the wrong offsets, and the last one would be written past
    /// the end of a buffer sized from the unpadded number. So the reads set it themselves rather
    /// than trusting whoever ran on the context before them.
    fn pack_tight(&self) {
        self.pixel_store_i(GL_PACK_ROW_LENGTH, 0);
        self.pixel_store_i(GL_PACK_SKIP_PIXELS, 0);
        self.pixel_store_i(GL_PACK_SKIP_ROWS, 0);
        self.pixel_store_i(GL_PACK_ALIGNMENT, 1);
    }

    // ---- buffers ----

    /// `glBufferData` with no data: allocate the store.
    pub fn buffer_data_null(&self, target: GLenum, size: usize, usage: GLenum) -> bool {
        let Ok(size) = GLsizeiptr::try_from(size) else {
            return false;
        };
        // SAFETY: a null data pointer allocates without reading.
        unsafe { self.t.glBufferData()(target, size, core::ptr::null(), usage) };
        true
    }

    pub fn buffer_storage_null(&self, target: GLenum, size: usize, flags: GLbitfield) -> bool {
        let f = promised(
            procs::buffer_storage(&self.t, self.api),
            Feature::arb_buffer_storage,
            "glBufferStorageEXT",
        );
        let Ok(size) = GLsizeiptr::try_from(size) else {
            return false;
        };
        // SAFETY: a null data pointer allocates without reading.
        unsafe { f(target, size, core::ptr::null(), flags) };
        true
    }

    /// `glBufferData` with contents: allocate the store and fill it in one call.
    pub fn buffer_data(&self, target: GLenum, data: &[u8], usage: GLenum) -> bool {
        let Ok(size) = GLsizeiptr::try_from(data.len()) else {
            return false;
        };
        // SAFETY: the driver reads `size` bytes from `data`, and `size` is `data`'s length.
        unsafe { self.t.glBufferData()(target, size, data.as_ptr().cast(), usage) };
        true
    }

    pub fn buffer_sub_data(&self, target: GLenum, offset: usize, data: &[u8]) -> bool {
        let (Ok(offset), Ok(size)) = (GLintptr::try_from(offset), GLsizeiptr::try_from(data.len()))
        else {
            return false;
        };
        // SAFETY: the driver reads `size` bytes from `data`, and `size` is `data`'s length.
        unsafe { self.t.glBufferSubData()(target, offset, size, data.as_ptr().cast()) };
        true
    }

    /// Map `len` bytes of the bound buffer and *leave it mapped*, answering where the driver put
    /// it.
    ///
    /// The other two mapping helpers unmap before they return, which is what makes them safe to
    /// hand a slice out of. This one does not, and that is the point: it takes the persistent
    /// mapping of a buffer whose whole reason for existing is to be published to a guest, which
    /// writes to it while the host draws from it. A mapping that ended here would have nothing to
    /// publish.
    ///
    /// `flags` must be the flags the store was created with -- `GL_MAP_PERSISTENT_BIT_EXT` is
    /// only accepted on a `glBufferStorage` store that carries it -- so the only honest caller is
    /// one holding those flags rather than re-deriving them.
    ///
    /// An address and not a slice: nothing in this process reads these bytes. The guest does,
    /// through a mapping the VMM makes of this address, and a `&mut [u8]` here would be claiming
    /// exclusive access that the guest is about to contradict. The mapping lasts until the buffer
    /// is deleted, which unmaps it -- so it is good for exactly as long as the resource holding
    /// the buffer is.
    pub fn map_buffer_persistent(
        &self,
        target: GLenum,
        len: usize,
        flags: GLbitfield,
    ) -> Option<usize> {
        let Ok(size) = GLsizeiptr::try_from(len) else {
            return None;
        };
        // SAFETY: plain scalars; the returned pointer is null or addresses `len` bytes of the
        // bound buffer until it is deleted.
        let p = unsafe { self.t.glMapBufferRange()(target, 0, size, flags) };
        (!p.is_null()).then_some(p as usize)
    }

    /// Map `len` bytes of the bound buffer for writing and hand them to `f` as a slice; unmapped
    /// before this returns. `None` if the driver refused the map.
    pub fn map_buffer_write<R>(
        &self,
        target: GLenum,
        offset: usize,
        len: usize,
        flags: GLbitfield,
        f: impl FnOnce(&mut [u8]) -> R,
    ) -> Option<R> {
        let (Ok(off), Ok(size)) = (GLintptr::try_from(offset), GLsizeiptr::try_from(len)) else {
            return None;
        };
        // SAFETY: plain scalars; the returned pointer is null or addresses `len` writable bytes
        // until `glUnmapBuffer`.
        let p = unsafe { self.t.glMapBufferRange()(target, off, size, flags | GL_MAP_WRITE_BIT) };
        if p.is_null() {
            return None;
        }
        // SAFETY: the driver mapped `len` bytes at `p`, exclusively ours until unmapped, which
        // happens after `f` returns and before the slice can be used again.
        let r = f(unsafe { core::slice::from_raw_parts_mut(p.cast::<u8>(), len) });
        unsafe { self.t.glUnmapBuffer()(target) };
        Some(r)
    }

    /// Map `len` bytes of the bound buffer for reading.
    pub fn map_buffer_read<R>(
        &self,
        target: GLenum,
        offset: usize,
        len: usize,
        f: impl FnOnce(&[u8]) -> R,
    ) -> Option<R> {
        let (Ok(off), Ok(size)) = (GLintptr::try_from(offset), GLsizeiptr::try_from(len)) else {
            return None;
        };
        // SAFETY: as `map_buffer_write`, for reading.
        let p = unsafe { self.t.glMapBufferRange()(target, off, size, GL_MAP_READ_BIT) };
        if p.is_null() {
            return None;
        }
        // SAFETY: the driver mapped `len` readable bytes at `p` until unmapped.
        let r = f(unsafe { core::slice::from_raw_parts(p.cast::<u8>(), len) });
        unsafe { self.t.glUnmapBuffer()(target) };
        Some(r)
    }

    // ---- framebuffers ----

    pub fn framebuffer_texture_2d(
        &self,
        attachment: GLenum,
        textarget: GLenum,
        tex: Option<TextureName>,
        level: GLint,
    ) {
        // SAFETY: plain scalars.
        unsafe {
            self.t.glFramebufferTexture2D()(
                GL_FRAMEBUFFER,
                attachment,
                textarget,
                tex.map_or(0, |t| t.0),
                level,
            )
        };
    }

    /// `glFramebufferTexture1D`. Desktop GL only.
    pub fn framebuffer_texture_1d(
        &self,
        attachment: GLenum,
        tex: Option<TextureName>,
        level: GLint,
    ) {
        assert!(!self.api.is_gles(), "glFramebufferTexture1D on a GLES context");
        // SAFETY: plain scalars.
        unsafe {
            self.t.glFramebufferTexture1D()(
                GL_FRAMEBUFFER,
                attachment,
                GL_TEXTURE_1D,
                tex.map_or(0, |t| t.0),
                level,
            )
        };
    }

    pub fn framebuffer_texture_layer(
        &self,
        attachment: GLenum,
        tex: Option<TextureName>,
        level: GLint,
        layer: GLint,
    ) {
        // SAFETY: plain scalars.
        unsafe {
            self.t.glFramebufferTextureLayer()(
                GL_FRAMEBUFFER,
                attachment,
                tex.map_or(0, |t| t.0),
                level,
                layer,
            )
        };
    }

    /// `glFramebufferTexture3DOES`: one slice of a 3D texture.
    pub fn framebuffer_texture_3d(
        &self,
        attachment: GLenum,
        tex: Option<TextureName>,
        level: GLint,
        layer: GLint,
    ) {
        let f = promised(
            procs::framebuffer_texture_3d(&self.t, self.api),
            Feature::texture_3d_attach,
            "glFramebufferTexture3DOES",
        );
        // SAFETY: plain scalars.
        unsafe {
            f(GL_FRAMEBUFFER, attachment, GL_TEXTURE_3D, tex.map_or(0, |t| t.0), level, layer)
        };
    }

    /// `glFramebufferTexture2DMultisampleEXT`: render into `tex` with `samples` samples the
    /// driver keeps to itself and resolves into the texture.
    pub fn framebuffer_texture_2d_multisample(
        &self,
        attachment: GLenum,
        textarget: GLenum,
        tex: Option<TextureName>,
        level: GLint,
        samples: GLsizei,
    ) {
        let f = promised(
            procs::framebuffer_texture_2d_multisample(&self.t, self.api),
            Feature::implicit_msaa,
            "glFramebufferTexture2DMultisampleEXT",
        );
        // SAFETY: plain scalars.
        unsafe { f(GL_FRAMEBUFFER, attachment, textarget, tex.map_or(0, |t| t.0), level, samples) };
    }

    pub fn check_framebuffer_status(&self) -> GLenum {
        // SAFETY: plain scalar.
        unsafe { self.t.glCheckFramebufferStatus()(GL_FRAMEBUFFER) }
    }

    pub fn draw_buffers(&self, bufs: &[GLenum]) {
        let Ok(n) = GLsizei::try_from(bufs.len()) else {
            return;
        };
        // SAFETY: the driver reads `n` enums from `bufs`, which holds `n`.
        unsafe { self.t.glDrawBuffers()(n, bufs.as_ptr()) };
    }

    pub fn read_buffer(&self, src: GLenum) {
        // SAFETY: plain scalar.
        unsafe { self.t.glReadBuffer()(src) };
    }

    /// `glFramebufferTexture`: every layer of a layered texture.
    pub fn framebuffer_texture(&self, attachment: GLenum, tex: Option<TextureName>, level: GLint) {
        let f = promised(
            procs::framebuffer_texture(&self.t, self.api),
            Feature::geometry_shader,
            "glFramebufferTexture",
        );
        // SAFETY: plain scalars.
        unsafe { f(GL_FRAMEBUFFER, attachment, tex.map_or(0, |t| t.0), level) };
    }

    /// `glTextureView`: `view` becomes a view of `levels` levels from `first_level` and
    /// `layers` layers from `first_layer` of `tex`, which only an immutable-format texture can
    /// be -- hence the witness rather than a name.
    #[allow(clippy::too_many_arguments)]
    pub fn texture_view(
        &self,
        view: TextureName,
        target: GLenum,
        tex: Immutable,
        internalformat: GLenum,
        first_level: GLuint,
        levels: GLuint,
        first_layer: GLuint,
        layers: GLuint,
    ) {
        let f = promised(
            procs::texture_view(&self.t, self.api),
            Feature::texture_view,
            "glTextureView",
        );
        // SAFETY: plain scalars.
        unsafe {
            f(view.0, target, tex.0.0, internalformat, first_level, levels, first_layer, layers)
        };
    }

    /// Whether `name`, bound to `target` here, has immutable-format storage: the driver's
    /// answer, as the witness `glTextureView` takes. Asked rather than predicted from the
    /// storage call that was made, so that the flag has one owner.
    pub fn immutable_format(&self, target: GLenum, name: TextureName) -> Option<Immutable> {
        self.bind_texture(target, Some(name));
        (self.get_tex_parameter_i(target, GL_TEXTURE_IMMUTABLE_FORMAT) == GL_TRUE as GLint)
            .then_some(Immutable(name))
    }

    /// `glEGLImageTargetTexStorageEXT`: the bound texture takes `image` as immutable storage.
    pub fn egl_image_target_tex_storage(&self, target: GLenum, image: &Image) {
        let f = promised(
            procs::egl_image_target_tex_storage(&self.t, self.api),
            Feature::egl_image_storage,
            "glEGLImageTargetTexStorageEXT",
        );
        // SAFETY: `image` is a live EGL image on this display, held by the caller across the
        // call; a null attribute list is the documented empty one.
        unsafe { f(target, image.raw(), core::ptr::null()) };
    }

    /// `glEGLImageTargetTexture2DOES`: the bound texture takes `image` as (mutable) storage.
    pub fn egl_image_target_texture_2d(&self, target: GLenum, image: &Image) {
        let f = promised(
            procs::egl_image_target_texture_2d(&self.t, self.api),
            Feature::egl_image,
            "glEGLImageTargetTexture2DOES",
        );
        // SAFETY: as above.
        unsafe { f(target, image.raw()) };
    }

    pub fn framebuffer_parameter_i(&self, name: GLenum, value: GLint) {
        // SAFETY: plain scalars.
        unsafe { self.t.glFramebufferParameteri()(GL_FRAMEBUFFER, name, value) };
    }

    /// `glBlitFramebuffer` from the read framebuffer's rectangle to the draw framebuffer's.
    #[allow(clippy::too_many_arguments)]
    pub fn blit_framebuffer(
        &self,
        src: [GLint; 4],
        dst: [GLint; 4],
        mask: GLbitfield,
        filter: GLenum,
    ) {
        // SAFETY: plain scalars.
        unsafe {
            self.t.glBlitFramebuffer()(
                src[0], src[1], src[2], src[3], dst[0], dst[1], dst[2], dst[3], mask, filter,
            )
        };
    }

    /// `glCopyImageSubData`.
    #[allow(clippy::too_many_arguments)]
    pub fn copy_image_sub_data(
        &self,
        src: TextureName,
        src_target: GLenum,
        src_level: GLint,
        src_origin: [GLint; 3],
        dst: TextureName,
        dst_target: GLenum,
        dst_level: GLint,
        dst_origin: [GLint; 3],
        extent: [GLsizei; 3],
    ) {
        let f = promised(
            procs::copy_image_sub_data(&self.t, self.api),
            Feature::copy_image,
            "glCopyImageSubData",
        );
        // SAFETY: plain scalars.
        unsafe {
            f(
                src.0,
                src_target,
                src_level,
                src_origin[0],
                src_origin[1],
                src_origin[2],
                dst.0,
                dst_target,
                dst_level,
                dst_origin[0],
                dst_origin[1],
                dst_origin[2],
                extent[0],
                extent[1],
                extent[2],
            )
        };
    }

    pub fn copy_buffer_sub_data(
        &self,
        read_offset: usize,
        write_offset: usize,
        size: usize,
    ) -> bool {
        let (Ok(r), Ok(w), Ok(s)) = (
            GLintptr::try_from(read_offset),
            GLintptr::try_from(write_offset),
            GLsizeiptr::try_from(size),
        ) else {
            return false;
        };
        // SAFETY: plain scalars; the copy is between the two bound copy buffers.
        unsafe { self.t.glCopyBufferSubData()(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, r, w, s) };
        true
    }

    /// `glTexBufferRange` on the bound `GL_TEXTURE_BUFFER`, or `glTexBuffer` when `range` is
    /// `None`. A range is bounded by the decoder against `max_texture_buffer_size`, a `u32`,
    /// so it always fits the driver's pointer-sized offset and size.
    pub fn tex_buffer(
        &self,
        internalformat: GLenum,
        buf: BufferName,
        range: Option<(usize, usize)>,
    ) {
        match range {
            Some((offset, size)) => {
                let f = promised(
                    procs::tex_buffer_range(&self.t, self.api),
                    Feature::texture_buffer_range,
                    "glTexBufferRange",
                );
                let offset = GLintptr::try_from(offset).expect("a range the decoder bounded");
                let size = GLsizeiptr::try_from(size).expect("a range the decoder bounded");
                // SAFETY: plain scalars.
                unsafe { f(GL_TEXTURE_BUFFER, internalformat, buf.0, offset, size) };
            }
            None => {
                let f = promised(
                    procs::tex_buffer(&self.t, self.api),
                    Feature::arb_or_gles_ext_texture_buffer,
                    "glTexBuffer",
                );
                // SAFETY: plain scalars.
                unsafe { f(GL_TEXTURE_BUFFER, internalformat, buf.0) };
            }
        }
    }

    /// `glClearTexSubImageEXT` with one pixel of `format`/`ty` as the value. The pair comes
    /// from the format table, which sizes every triple it holds, and the value is the wire's
    /// four words -- sixteen bytes, the largest pixel there is.
    #[allow(clippy::too_many_arguments)]
    pub fn clear_tex_sub_image(
        &self,
        tex: TextureName,
        level: GLint,
        origin: [GLint; 3],
        extent: [GLsizei; 3],
        format: GLenum,
        ty: GLenum,
        value: &[u8],
    ) {
        let f = promised(
            procs::clear_tex_sub_image(&self.t, self.api),
            Feature::clear_texture,
            "glClearTexSubImageEXT",
        );
        let need = pixel_bytes(format, ty).expect("a triple from the format table has a size");
        assert!(value.len() >= need, "a clear value shorter than the pixel");
        // SAFETY: the driver reads one pixel -- `need` bytes -- from `value`, which holds them.
        unsafe {
            f(
                tex.0,
                level,
                origin[0],
                origin[1],
                origin[2],
                extent[0],
                extent[1],
                extent[2],
                format,
                ty,
                value.as_ptr().cast(),
            )
        };
    }

    // ---- fixed-function state ----

    pub fn enable(&self, cap: GLenum) {
        // SAFETY: plain scalar.
        unsafe { self.t.glEnable()(cap) };
    }

    pub fn disable(&self, cap: GLenum) {
        // SAFETY: plain scalar.
        unsafe { self.t.glDisable()(cap) };
    }

    pub fn set_enabled(&self, cap: GLenum, on: bool) {
        if on {
            self.enable(cap);
        } else {
            self.disable(cap);
        }
    }

    pub fn depth_func(&self, func: GLenum) {
        // SAFETY: plain scalar.
        unsafe { self.t.glDepthFunc()(func) };
    }

    pub fn depth_mask(&self, on: bool) {
        // SAFETY: plain scalar.
        unsafe { self.t.glDepthMask()(on as GLboolean) };
    }

    pub fn stencil_op(&self, fail: GLenum, zfail: GLenum, zpass: GLenum) {
        // SAFETY: plain scalars.
        unsafe { self.t.glStencilOp()(fail, zfail, zpass) };
    }

    pub fn stencil_op_separate(&self, face: GLenum, fail: GLenum, zfail: GLenum, zpass: GLenum) {
        // SAFETY: plain scalars.
        unsafe { self.t.glStencilOpSeparate()(face, fail, zfail, zpass) };
    }

    pub fn stencil_func(&self, func: GLenum, reference: GLint, mask: GLuint) {
        // SAFETY: plain scalars.
        unsafe { self.t.glStencilFunc()(func, reference, mask) };
    }

    pub fn stencil_func_separate(
        &self,
        face: GLenum,
        func: GLenum,
        reference: GLint,
        mask: GLuint,
    ) {
        // SAFETY: plain scalars.
        unsafe { self.t.glStencilFuncSeparate()(face, func, reference, mask) };
    }

    pub fn stencil_mask(&self, mask: GLuint) {
        // SAFETY: plain scalar.
        unsafe { self.t.glStencilMask()(mask) };
    }

    pub fn stencil_mask_separate(&self, face: GLenum, mask: GLuint) {
        // SAFETY: plain scalars.
        unsafe { self.t.glStencilMaskSeparate()(face, mask) };
    }

    pub fn color_mask(&self, rgba: [bool; 4]) {
        // SAFETY: plain scalars.
        unsafe {
            self.t.glColorMask()(
                rgba[0] as GLboolean,
                rgba[1] as GLboolean,
                rgba[2] as GLboolean,
                rgba[3] as GLboolean,
            )
        };
    }

    /// `glColorMaski`.
    pub fn color_mask_i(&self, index: GLuint, rgba: [bool; 4]) {
        let f =
            promised(procs::color_mask_i(&self.t, self.api), Feature::indep_blend, "glColorMaski");
        // SAFETY: plain scalars.
        unsafe {
            f(
                index,
                rgba[0] as GLboolean,
                rgba[1] as GLboolean,
                rgba[2] as GLboolean,
                rgba[3] as GLboolean,
            )
        };
    }

    pub fn clear_color(&self, rgba: [f32; 4]) {
        // SAFETY: plain scalars.
        unsafe { self.t.glClearColor()(rgba[0], rgba[1], rgba[2], rgba[3]) };
    }

    /// The depth a clear writes: `glClearDepth`'s double on desktop GL, `glClearDepthf`'s float
    /// on GLES, which has no other.
    pub fn clear_depth(&self, depth: f64) {
        if self.api.is_gles() {
            // SAFETY: plain scalar.
            unsafe { self.t.glClearDepthf()(depth as f32) };
        } else {
            // SAFETY: plain scalar.
            unsafe { self.t.glClearDepth()(depth) };
        }
    }

    pub fn clear_stencil(&self, stencil: GLint) {
        // SAFETY: plain scalar.
        unsafe { self.t.glClearStencil()(stencil) };
    }

    pub fn clear(&self, mask: GLbitfield) {
        // SAFETY: plain scalar.
        unsafe { self.t.glClear()(mask) };
    }

    pub fn clear_buffer_fv(&self, buffer: GLenum, drawbuffer: GLint, value: &[f32; 4]) {
        // SAFETY: the driver reads four floats for `GL_COLOR`, one for `GL_DEPTH`; `value` holds
        // four.
        unsafe { self.t.glClearBufferfv()(buffer, drawbuffer, value.as_ptr()) };
    }

    /// `glLogicOp`. Desktop GL only: GLES has no logic op, and the shader emulates the ops it can.
    pub fn logic_op(&self, op: GLenum) {
        assert!(!self.api.is_gles(), "glLogicOp on a GLES context");
        // SAFETY: a plain enum.
        unsafe { self.t.glLogicOp()(op) };
    }

    pub fn blend_color(&self, rgba: [f32; 4]) {
        // SAFETY: plain scalars.
        unsafe { self.t.glBlendColor()(rgba[0], rgba[1], rgba[2], rgba[3]) };
    }

    pub fn line_width(&self, width: f32) {
        // SAFETY: plain scalar.
        unsafe { self.t.glLineWidth()(width) };
    }

    pub fn polygon_offset(&self, factor: f32, units: f32) {
        // SAFETY: plain scalars.
        unsafe { self.t.glPolygonOffset()(factor, units) };
    }

    /// `glPolygonOffsetClamp`: the core 4.6 name `GL_ARB_polygon_offset_clamp` shares, or
    /// `GL_EXT_polygon_offset_clamp`'s.
    pub fn polygon_offset_clamp(&self, factor: f32, units: f32, clamp: f32) {
        let f = promised(
            procs::polygon_offset_clamp(&self.t, self.api),
            Feature::polygon_offset_clamp,
            "glPolygonOffsetClamp",
        );
        // SAFETY: plain scalars.
        unsafe { f(factor, units, clamp) };
    }

    /// `glPointSize`. Desktop GL only: GLES sizes points from the shader alone.
    pub fn point_size(&self, size: f32) {
        assert!(!self.api.is_gles(), "glPointSize on a GLES context");
        // SAFETY: plain scalar.
        unsafe { self.t.glPointSize()(size) };
    }

    /// `glPointParameteri`. Desktop GL only.
    pub fn point_parameter_i(&self, pname: GLenum, value: GLint) {
        assert!(!self.api.is_gles(), "glPointParameteri on a GLES context");
        // SAFETY: plain scalars.
        unsafe { self.t.glPointParameteri()(pname, value) };
    }

    /// `glPolygonMode`. Desktop GL only.
    pub fn polygon_mode(&self, face: GLenum, mode: GLenum) {
        assert!(!self.api.is_gles(), "glPolygonMode on a GLES context");
        // SAFETY: plain scalars.
        unsafe { self.t.glPolygonMode()(face, mode) };
    }

    /// `glProvokingVertex`. Desktop GL only.
    pub fn provoking_vertex(&self, mode: GLenum) {
        assert!(!self.api.is_gles(), "glProvokingVertex on a GLES context");
        // SAFETY: plain scalar.
        unsafe { self.t.glProvokingVertex()(mode) };
    }

    pub fn cull_face(&self, mode: GLenum) {
        // SAFETY: plain scalar.
        unsafe { self.t.glCullFace()(mode) };
    }

    pub fn front_face(&self, mode: GLenum) {
        // SAFETY: plain scalar.
        unsafe { self.t.glFrontFace()(mode) };
    }

    pub fn sample_mask_i(&self, index: GLuint, mask: GLbitfield) {
        // SAFETY: plain scalars.
        unsafe { self.t.glSampleMaski()(index, mask) };
    }

    /// `glMinSampleShading`.
    pub fn min_sample_shading(&self, value: f32) {
        let f = promised(
            procs::min_sample_shading(&self.t, self.api),
            Feature::sample_shading,
            "glMinSampleShading",
        );
        // SAFETY: plain scalar.
        unsafe { f(value) };
    }

    pub fn viewport(&self, x: GLint, y: GLint, w: GLsizei, h: GLsizei) {
        // SAFETY: plain scalars.
        unsafe { self.t.glViewport()(x, y, w, h) };
    }

    pub fn scissor(&self, x: GLint, y: GLint, w: GLsizei, h: GLsizei) {
        // SAFETY: plain scalars.
        unsafe { self.t.glScissor()(x, y, w, h) };
    }

    /// The depth range: `glDepthRange`'s doubles on desktop GL, `glDepthRangef`'s floats on
    /// GLES, which has no other.
    pub fn depth_range(&self, near: f64, far: f64) {
        if self.api.is_gles() {
            // SAFETY: plain scalars.
            unsafe { self.t.glDepthRangef()(near as f32, far as f32) };
        } else {
            // SAFETY: plain scalars.
            unsafe { self.t.glDepthRange()(near, far) };
        }
    }

    /// `glViewportIndexedf`, `glScissorIndexed` and `glDepthRangeIndexed`: viewport `index` past
    /// the first, behind `Feature::viewport_array` -- core in desktop GL 4.1, and
    /// `GL_OES_viewport_array` on GLES, whose spellings these take there.
    pub fn viewport_indexed(&self, index: GLuint, x: f32, y: f32, w: f32, h: f32) {
        let f = promised(
            procs::viewport_indexed(&self.t, self.api),
            Feature::viewport_array,
            "glViewportIndexedf",
        );
        // SAFETY: plain scalars.
        unsafe { f(index, x, y, w, h) };
    }

    pub fn scissor_indexed(&self, index: GLuint, x: GLint, y: GLint, w: GLsizei, h: GLsizei) {
        let f = promised(
            procs::scissor_indexed(&self.t, self.api),
            Feature::viewport_array,
            "glScissorIndexed",
        );
        // SAFETY: plain scalars.
        unsafe { f(index, x, y, w, h) };
    }

    /// GLES takes the range in single precision (`glDepthRangeIndexedfOES`), desktop GL in
    /// double.
    pub fn depth_range_indexed(&self, index: GLuint, near: f64, far: f64) {
        if let Some(f) = procs::depth_range_indexed_d(&self.t, self.api) {
            // SAFETY: plain scalars.
            unsafe { f(index, near, far) };
            return;
        }
        let f = promised(
            procs::depth_range_indexed_f(&self.t, self.api),
            Feature::viewport_array,
            "glDepthRangeIndexed",
        );
        // SAFETY: plain scalars.
        unsafe { f(index, near as f32, far as f32) };
    }

    /// `glClipControlEXT`.
    pub fn clip_control(&self, origin: GLenum, depth: GLenum) {
        let f = promised(
            procs::clip_control(&self.t, self.api),
            Feature::clip_control,
            "glClipControlEXT",
        );
        // SAFETY: plain scalars.
        unsafe { f(origin, depth) };
    }

    pub fn memory_barrier(&self, barriers: GLbitfield) {
        // SAFETY: plain scalar.
        unsafe { self.t.glMemoryBarrier()(barriers) };
    }

    // ---- vertex arrays ----

    pub fn gen_vertex_array(&self) -> VertexArrayName {
        let mut id: GLuint = 0;
        // SAFETY: room for the one name asked for.
        unsafe { self.t.glGenVertexArrays()(1, &mut id) };
        VertexArrayName(id)
    }

    pub fn delete_vertex_array(&self, vao: VertexArrayName) {
        // SAFETY: one name, read from a live local.
        unsafe { self.t.glDeleteVertexArrays()(1, &vao.0) };
    }

    pub fn bind_vertex_array(&self, vao: Option<VertexArrayName>) {
        // SAFETY: plain scalar.
        unsafe { self.t.glBindVertexArray()(vao.map_or(0, |v| v.0)) };
    }

    pub fn vertex_attrib_format(
        &self,
        index: GLuint,
        size: GLint,
        ty: GLenum,
        normalized: bool,
        offset: GLuint,
    ) {
        // SAFETY: plain scalars.
        unsafe { self.t.glVertexAttribFormat()(index, size, ty, normalized as GLboolean, offset) };
    }

    pub fn vertex_attrib_i_format(&self, index: GLuint, size: GLint, ty: GLenum, offset: GLuint) {
        // SAFETY: plain scalars.
        unsafe { self.t.glVertexAttribIFormat()(index, size, ty, offset) };
    }

    pub fn vertex_attrib_binding(&self, index: GLuint, binding: GLuint) {
        // SAFETY: plain scalars.
        unsafe { self.t.glVertexAttribBinding()(index, binding) };
    }

    pub fn vertex_binding_divisor(&self, binding: GLuint, divisor: GLuint) {
        // SAFETY: plain scalars.
        unsafe { self.t.glVertexBindingDivisor()(binding, divisor) };
    }

    /// `glVertexAttribPointer`: the pre-separate-attribute-format binding, where the format and
    /// the buffer are named in one call. `offset` is a byte offset into the bound `GL_ARRAY_BUFFER`.
    pub fn vertex_attrib_pointer(
        &self,
        index: AttribLocation,
        size: GLint,
        ty: GLenum,
        normalized: bool,
        stride: GLsizei,
        offset: u32,
    ) {
        // SAFETY: the pointer is an offset into the bound buffer, which the driver bounds.
        unsafe {
            self.t.glVertexAttribPointer()(
                index.0,
                size,
                ty,
                normalized as GLboolean,
                stride,
                offset_ptr(offset),
            )
        };
    }

    pub fn enable_vertex_attrib_array_at(&self, index: AttribLocation) {
        // SAFETY: plain scalar.
        unsafe { self.t.glEnableVertexAttribArray()(index.0) };
    }

    pub fn enable_vertex_attrib_array(&self, index: GLuint) {
        // SAFETY: plain scalar.
        unsafe { self.t.glEnableVertexAttribArray()(index) };
    }

    pub fn disable_vertex_attrib_array(&self, index: GLuint) {
        // SAFETY: plain scalar.
        unsafe { self.t.glDisableVertexAttribArray()(index) };
    }

    // ---- samplers ----

    pub fn gen_sampler(&self) -> SamplerName {
        let mut id: GLuint = 0;
        // SAFETY: room for the one name asked for.
        unsafe { self.t.glGenSamplers()(1, &mut id) };
        SamplerName(id)
    }

    pub fn delete_sampler(&self, s: SamplerName) {
        // SAFETY: one name, read from a live local.
        unsafe { self.t.glDeleteSamplers()(1, &s.0) };
    }

    pub fn sampler_parameter_i(&self, s: SamplerName, name: GLenum, value: GLint) {
        // SAFETY: plain scalars.
        unsafe { self.t.glSamplerParameteri()(s.0, name, value) };
    }

    pub fn sampler_parameter_f(&self, s: SamplerName, name: GLenum, value: f32) {
        // SAFETY: plain scalars.
        unsafe { self.t.glSamplerParameterf()(s.0, name, value) };
    }

    /// `glSamplerParameterIuiv` for the border colour.
    pub fn sampler_border_color(&self, s: SamplerName, color: &[GLuint; 4]) {
        let f = promised(
            procs::sampler_parameter_iuiv(&self.t, self.api),
            Feature::sampler_border_colors,
            "glSamplerParameterIuiv",
        );
        // SAFETY: the driver reads four values for `GL_TEXTURE_BORDER_COLOR`; `color` holds four.
        unsafe { f(s.0, GL_TEXTURE_BORDER_COLOR, color.as_ptr()) };
    }

    // ---- transform feedback ----

    pub fn gen_transform_feedback(&self) -> TransformFeedbackName {
        let mut id: GLuint = 0;
        // SAFETY: room for the one name asked for.
        unsafe { self.t.glGenTransformFeedbacks()(1, &mut id) };
        TransformFeedbackName(id)
    }

    pub fn delete_transform_feedback(&self, tf: TransformFeedbackName) {
        // SAFETY: one name, read from a live local.
        unsafe { self.t.glDeleteTransformFeedbacks()(1, &tf.0) };
    }

    pub fn bind_transform_feedback(&self, tf: Option<TransformFeedbackName>) {
        // SAFETY: plain scalar.
        unsafe { self.t.glBindTransformFeedback()(GL_TRANSFORM_FEEDBACK, tf.map_or(0, |t| t.0)) };
    }

    /// `glBeginConditionalRender`: draws and blits are dropped while `query`'s answer says so.
    pub fn begin_conditional_render(&self, query: QueryName, mode: GLenum) {
        // SAFETY: plain scalars.
        unsafe { self.t.glBeginConditionalRender()(query.0, mode) };
    }

    pub fn end_conditional_render(&self) {
        // SAFETY: takes nothing.
        unsafe { self.t.glEndConditionalRender()() };
    }

    /// `glDrawTransformFeedback`: as many vertices as `tf` captured when its capture ended.
    pub fn draw_transform_feedback(&self, mode: GLenum, tf: TransformFeedbackName) {
        let f = promised(
            procs::draw_transform_feedback(&self.t, self.api),
            Feature::transform_feedback_draw,
            "glDrawTransformFeedback",
        );
        // SAFETY: plain scalars.
        unsafe { f(mode, tf.0) };
    }

    /// `glDrawTransformFeedbackInstanced`.
    pub fn draw_transform_feedback_instanced(
        &self,
        mode: GLenum,
        tf: TransformFeedbackName,
        instances: GLsizei,
    ) {
        let f = promised(
            procs::draw_transform_feedback_instanced(&self.t, self.api),
            Feature::transform_feedback_instanced,
            "glDrawTransformFeedbackInstanced",
        );
        // SAFETY: plain scalars.
        unsafe { f(mode, tf.0, instances) };
    }

    pub fn end_transform_feedback(&self) {
        // SAFETY: takes nothing.
        unsafe { self.t.glEndTransformFeedback()() };
    }

    pub fn bind_buffer_base(&self, target: GLenum, index: BindingPoint, buf: Option<BufferName>) {
        // SAFETY: plain scalars.
        unsafe { self.t.glBindBufferBase()(target, index.0, buf.map_or(0, |b| b.0)) };
    }

    pub fn bind_buffer_range(
        &self,
        target: GLenum,
        index: BindingPoint,
        buf: BufferName,
        offset: usize,
        size: usize,
    ) -> bool {
        let (Ok(offset), Ok(size)) = (GLintptr::try_from(offset), GLsizeiptr::try_from(size))
        else {
            return false;
        };
        // SAFETY: plain scalars.
        unsafe { self.t.glBindBufferRange()(target, index.0, buf.0, offset, size) };
        true
    }

    // ---- queries ----

    pub fn gen_query(&self) -> QueryName {
        let mut id: GLuint = 0;
        // SAFETY: room for the one name asked for.
        unsafe { self.t.glGenQueries()(1, &mut id) };
        QueryName(id)
    }

    pub fn delete_query(&self, q: QueryName) {
        // SAFETY: one name, read from a live local.
        unsafe { self.t.glDeleteQueries()(1, &q.0) };
    }

    pub fn begin_query(&self, target: GLenum, q: QueryName) {
        // SAFETY: plain scalars.
        unsafe { self.t.glBeginQuery()(target, q.0) };
    }

    pub fn end_query(&self, target: GLenum) {
        // SAFETY: plain scalar.
        unsafe { self.t.glEndQuery()(target) };
    }

    /// `glBeginQueryIndexed`: a query counting on vertex stream `index`. Desktop GL only, behind
    /// `transform_feedback3`.
    pub fn begin_query_indexed(&self, target: GLenum, index: GLuint, q: QueryName) {
        assert!(!self.api.is_gles(), "glBeginQueryIndexed on a GLES context");
        // SAFETY: plain scalars.
        unsafe { self.t.glBeginQueryIndexed()(target, index, q.0) };
    }

    pub fn end_query_indexed(&self, target: GLenum, index: GLuint) {
        assert!(!self.api.is_gles(), "glEndQueryIndexed on a GLES context");
        // SAFETY: plain scalars.
        unsafe { self.t.glEndQueryIndexed()(target, index) };
    }

    /// `glGetQueryObject*v` with a buffer bound to `GL_QUERY_BUFFER`: the GPU writes `pname` of
    /// query `q` into that buffer at `offset`, as a word of `width`. Desktop GL only, behind `qbo`.
    pub fn query_object_into_buffer(
        &self,
        q: QueryName,
        pname: GLenum,
        width: QueryWord,
        offset: u32,
    ) {
        assert!(!self.api.is_gles(), "glGetQueryObject into a buffer on a GLES context");
        let at = offset_ptr(offset).cast_mut();
        // SAFETY: with a buffer bound to `GL_QUERY_BUFFER`, the pointer is an offset into it,
        // never dereferenced on this side; the driver bounds the write by the buffer's size and
        // raises an error past it.
        unsafe {
            match width {
                QueryWord::I32 => self.t.glGetQueryObjectiv()(q.0, pname, at.cast()),
                QueryWord::U32 => self.t.glGetQueryObjectuiv()(q.0, pname, at.cast()),
                QueryWord::I64 => self.t.glGetQueryObjecti64v()(q.0, pname, at.cast()),
                QueryWord::U64 => self.t.glGetQueryObjectui64v()(q.0, pname, at.cast()),
            }
        }
    }

    /// `glQueryCounterEXT` with `GL_TIMESTAMP_EXT`: record the GPU's clock into `q` once every
    /// command before it has run. A timestamp's only command -- it has no begin and no end.
    pub fn query_timestamp(&self, q: QueryName) {
        // SAFETY: plain scalars.
        unsafe { self.t.glQueryCounterEXT()(q.0, GL_TIMESTAMP_EXT) };
    }

    pub fn get_query_object_uiv(&self, q: QueryName, name: GLenum) -> GLuint {
        let mut v: GLuint = 0;
        // SAFETY: every name asked for writes exactly one integer.
        unsafe { self.t.glGetQueryObjectuiv()(q.0, name, &mut v) };
        v
    }

    /// `glGetQueryObjectui64vEXT`: the 64-bit read a timer query's nanoseconds need.
    pub fn get_query_object_ui64v(&self, q: QueryName, name: GLenum) -> u64 {
        let mut v: GLuint64 = 0;
        // SAFETY: every name asked for writes exactly one 64-bit integer.
        unsafe { self.t.glGetQueryObjectui64vEXT()(q.0, name, &mut v) };
        v
    }

    // ---- shaders ----

    /// `glCreateShader`; `None` when the driver refused the kind.
    pub fn create_shader(&self, kind: GLenum) -> Option<ShaderName> {
        // SAFETY: plain scalar.
        let id = unsafe { self.t.glCreateShader()(kind) };
        (id != 0).then_some(ShaderName(id))
    }

    pub fn delete_shader(&self, shader: ShaderName) {
        // SAFETY: plain scalar.
        unsafe { self.t.glDeleteShader()(shader.0) };
    }

    /// `glShaderSource` and `glCompileShader`, and the driver's log when it refused the source.
    pub fn compile_shader(&self, shader: ShaderName, source: &str) -> Result<(), String> {
        let ptr = source.as_ptr().cast::<GLchar>();
        let len = GLint::try_from(source.len()).expect("a shader's source fits a GLint");
        // SAFETY: one string, with its length given, so the driver reads exactly `source`.
        unsafe { self.t.glShaderSource()(shader.0, 1, &ptr, &len) };
        // SAFETY: plain scalar.
        unsafe { self.t.glCompileShader()(shader.0) };
        let mut status: GLint = 0;
        // SAFETY: `GL_COMPILE_STATUS` writes exactly one integer.
        unsafe { self.t.glGetShaderiv()(shader.0, GL_COMPILE_STATUS, &mut status) };
        if status != 0 {
            return Ok(());
        }
        let mut log = vec![0u8; 65536];
        let mut written: GLsizei = 0;
        // SAFETY: the driver writes at most the capacity given, NUL included, and reports how
        // many bytes it wrote without the NUL.
        unsafe {
            self.t.glGetShaderInfoLog()(
                shader.0,
                log.len() as GLsizei,
                &mut written,
                log.as_mut_ptr().cast::<GLchar>(),
            )
        };
        log.truncate(written.max(0) as usize);
        Err(String::from_utf8_lossy(&log).into_owned())
    }

    // ---- programs ----

    pub fn create_program(&self) -> Option<ProgramName> {
        // SAFETY: no arguments.
        let id = unsafe { self.t.glCreateProgram()() };
        (id != 0).then_some(ProgramName(id))
    }

    /// `glDeleteProgram`, unbinding the program first if it is the one GL has.
    ///
    /// A delete invalidates the name at once, while the object itself lives on as current state,
    /// and Mesa hands out the lowest free name (`util_idalloc_sparse_alloc_range`) -- so the next
    /// `glCreateProgram` can return the name just deleted. A shadow still naming it would then
    /// match the new program and skip its bind, and the uniforms set after it would land on the
    /// deleted one. Unbinding here is what keeps [`BoundProgram`] unable to name something gone,
    /// and taking it by `&mut` is what makes every delete site say so.
    pub fn delete_program(&self, bound: &mut BoundProgram, program: ProgramName) {
        if bound.0 == Some(InUse::Program(program)) {
            self.use_program(bound, None);
        }
        // SAFETY: plain scalar.
        unsafe { self.t.glDeleteProgram()(program.0) };
    }

    pub fn attach_shader(&self, program: ProgramName, shader: ShaderName) {
        // SAFETY: plain scalars.
        unsafe { self.t.glAttachShader()(program.0, shader.0) };
    }

    /// `glLinkProgram`, and the driver's log when the link failed.
    pub fn link_program(&self, program: ProgramName) -> Result<(), String> {
        // SAFETY: plain scalar.
        unsafe { self.t.glLinkProgram()(program.0) };
        let mut status: GLint = 0;
        // SAFETY: `GL_LINK_STATUS` writes exactly one integer.
        unsafe { self.t.glGetProgramiv()(program.0, GL_LINK_STATUS, &mut status) };
        if status != 0 {
            return Ok(());
        }
        let mut log = vec![0u8; 65536];
        let mut written: GLsizei = 0;
        // SAFETY: the driver writes at most the capacity given, NUL included, and reports how
        // many bytes it wrote without the NUL.
        unsafe {
            self.t.glGetProgramInfoLog()(
                program.0,
                log.len() as GLsizei,
                &mut written,
                log.as_mut_ptr().cast::<GLchar>(),
            )
        };
        log.truncate(written.max(0) as usize);
        Err(String::from_utf8_lossy(&log).into_owned())
    }

    /// `glUseProgram`, skipped when `bound` says GL already has that program.
    ///
    /// A bind GL already has is a wasted entry point -- two TLS lookups through Mesa's dispatch
    /// before the driver gets to decide it has nothing to do -- and the draw path re-binds the
    /// same program on most draws. The C re-binds unconditionally (`vrend_use_program`); this does
    /// not, which is only sound because `bound` cannot go stale: see [`BoundProgram`].
    ///
    /// A pipeline in use is unbound first, as the C does: a program in use outranks a pipeline,
    /// but `glUseProgram(0)` would hand the draw back to the pipeline still bound.
    pub fn use_program(&self, bound: &mut BoundProgram, program: Option<ProgramName>) {
        let want = program.map(InUse::Program);
        if bound.0 == want {
            return;
        }
        if let Some(InUse::Pipeline(_)) = bound.0 {
            self.bind_program_pipeline(0);
        }
        // SAFETY: plain scalar; zero is "no program".
        unsafe { self.t.glUseProgram()(program.map_or(0, |p| p.0)) };
        bound.0 = want;
    }

    /// `vrend_use_program` for a pipeline: no program in use, so the pipeline is what draws.
    /// Skipped when `bound` says GL already has it, as [`Gl::use_program`] is.
    pub fn use_pipeline(&self, bound: &mut BoundProgram, pipeline: PipelineName) {
        let want = Some(InUse::Pipeline(pipeline));
        if bound.0 == want {
            return;
        }
        // SAFETY: zero is "no program".
        unsafe { self.t.glUseProgram()(0) };
        self.bind_program_pipeline(pipeline.0);
        bound.0 = want;
    }

    fn bind_program_pipeline(&self, pipeline: GLuint) {
        let f = promised(
            procs::bind_program_pipeline(&self.t, self.api),
            Feature::separate_shader_objects,
            "glBindProgramPipeline",
        );
        // SAFETY: a pipeline name, or zero for none.
        unsafe { f(pipeline) };
    }

    /// `glGenProgramPipelines` for one pipeline.
    pub fn gen_program_pipeline(&self) -> PipelineName {
        let f = promised(
            procs::gen_program_pipelines(&self.t, self.api),
            Feature::separate_shader_objects,
            "glGenProgramPipelines",
        );
        let mut id: GLuint = 0;
        // SAFETY: writes exactly one name.
        unsafe { f(1, &mut id) };
        assert_ne!(id, 0, "the driver names a pipeline it generated");
        PipelineName(id)
    }

    /// `glDeleteProgramPipelines` for one pipeline, out of use first if it is in use, for the
    /// reason [`Gl::delete_program`] gives.
    pub fn delete_program_pipeline(&self, bound: &mut BoundProgram, pipeline: PipelineName) {
        if bound.0 == Some(InUse::Pipeline(pipeline)) {
            self.use_program(bound, None);
        }
        let f = promised(
            procs::delete_program_pipelines(&self.t, self.api),
            Feature::separate_shader_objects,
            "glDeleteProgramPipelines",
        );
        // SAFETY: reads exactly one name.
        unsafe { f(1, &pipeline.0) };
    }

    /// `glProgramParameteri(GL_PROGRAM_SEPARABLE, GL_TRUE)`: the program will be one stage of a
    /// pipeline. Set before the link, which is when it counts.
    pub fn program_separable(&self, program: ProgramName) {
        let f = promised(
            procs::program_parameter_i(&self.t, self.api),
            Feature::separate_shader_objects,
            "glProgramParameteri",
        );
        // SAFETY: plain scalars.
        unsafe { f(program.0, GL_PROGRAM_SEPARABLE, GL_TRUE as GLint) };
    }

    /// `glUseProgramStages`: `program` serves the stages `stages` names in `pipeline`.
    pub fn use_program_stages(
        &self,
        pipeline: PipelineName,
        stages: GLbitfield,
        program: ProgramName,
    ) {
        let f = promised(
            procs::use_program_stages(&self.t, self.api),
            Feature::separate_shader_objects,
            "glUseProgramStages",
        );
        // SAFETY: plain scalars.
        unsafe { f(pipeline.0, stages, program.0) };
    }

    /// `glActiveShaderProgram`: the program in `pipeline` that `glUniform*` writes to.
    pub fn active_shader_program(&self, pipeline: PipelineName, program: ProgramName) {
        let f = promised(
            procs::active_shader_program(&self.t, self.api),
            Feature::separate_shader_objects,
            "glActiveShaderProgram",
        );
        // SAFETY: plain scalars.
        unsafe { f(pipeline.0, program.0) };
    }

    /// `glValidateProgramPipeline` and its `GL_VALIDATE_STATUS`.
    pub fn validate_program_pipeline(&self, pipeline: PipelineName) -> bool {
        let validate = promised(
            procs::validate_program_pipeline(&self.t, self.api),
            Feature::separate_shader_objects,
            "glValidateProgramPipeline",
        );
        let get = promised(
            procs::get_program_pipeline_iv(&self.t, self.api),
            Feature::separate_shader_objects,
            "glGetProgramPipelineiv",
        );
        let mut status: GLint = 0;
        // SAFETY: a plain scalar, then a query that writes exactly one integer.
        unsafe {
            validate(pipeline.0);
            get(pipeline.0, GL_VALIDATE_STATUS, &mut status);
        }
        status != 0
    }

    /// `glGetUniformLocation`; `None` when the program has no such uniform.
    pub fn get_uniform_location(
        &self,
        program: ProgramName,
        name: &str,
    ) -> Option<UniformLocation> {
        let name = std::ffi::CString::new(name).expect("a uniform name has no NUL");
        // SAFETY: a NUL-terminated string, live for the call.
        let loc =
            unsafe { self.t.glGetUniformLocation()(program.0, name.as_ptr().cast::<GLchar>()) };
        (loc >= 0).then_some(UniformLocation(loc))
    }

    /// `glGetAttribLocation`; `None` when the program has no such attribute.
    pub fn get_attrib_location(&self, program: ProgramName, name: &str) -> Option<AttribLocation> {
        let name = std::ffi::CString::new(name).expect("an attribute name has no NUL");
        // SAFETY: a NUL-terminated string, live for the call.
        let loc =
            unsafe { self.t.glGetAttribLocation()(program.0, name.as_ptr().cast::<GLchar>()) };
        u32::try_from(loc).ok().map(AttribLocation)
    }

    /// `glGetUniformBlockIndex`; `None` when the program has no such block.
    pub fn get_uniform_block_index(&self, program: ProgramName, name: &str) -> Option<GLuint> {
        let name = std::ffi::CString::new(name).expect("a block name has no NUL");
        // SAFETY: a NUL-terminated string, live for the call.
        let i =
            unsafe { self.t.glGetUniformBlockIndex()(program.0, name.as_ptr().cast::<GLchar>()) };
        (i != GL_INVALID_INDEX).then_some(i)
    }

    pub fn uniform_block_binding(
        &self,
        program: ProgramName,
        block: GLuint,
        binding: BindingPoint,
    ) {
        // SAFETY: plain scalars.
        unsafe { self.t.glUniformBlockBinding()(program.0, block, binding.0) };
    }

    /// `GL_UNIFORM_BLOCK_DATA_SIZE` of a block.
    pub fn uniform_block_data_size(&self, program: ProgramName, block: GLuint) -> GLint {
        let mut v: GLint = 0;
        // SAFETY: the query writes exactly one integer.
        unsafe {
            self.t.glGetActiveUniformBlockiv()(program.0, block, GL_UNIFORM_BLOCK_DATA_SIZE, &mut v)
        };
        v
    }

    pub fn bind_attrib_location(&self, program: ProgramName, index: GLuint, name: &str) {
        let name = std::ffi::CString::new(name).expect("an attribute name has no NUL");
        // SAFETY: a NUL-terminated string, live for the call.
        unsafe { self.t.glBindAttribLocation()(program.0, index, name.as_ptr().cast::<GLchar>()) };
    }

    /// `glTransformFeedbackVaryings`, interleaved.
    pub fn transform_feedback_varyings(&self, program: ProgramName, varyings: &[String]) {
        let owned: Vec<std::ffi::CString> = varyings
            .iter()
            .map(|v| std::ffi::CString::new(v.as_str()).expect("a varying name has no NUL"))
            .collect();
        let ptrs: Vec<*const GLchar> = owned.iter().map(|c| c.as_ptr().cast::<GLchar>()).collect();
        let count = GLsizei::try_from(ptrs.len()).expect("a varying count fits a GLsizei");
        // SAFETY: `count` NUL-terminated strings, all live for the call.
        unsafe {
            self.t.glTransformFeedbackVaryings()(
                program.0,
                count,
                ptrs.as_ptr(),
                GL_INTERLEAVED_ATTRIBS,
            )
        };
    }

    /// `glBindFragDataLocationIndexedEXT`.
    pub fn bind_frag_data_location_indexed(
        &self,
        program: ProgramName,
        color: GLuint,
        index: GLuint,
        name: &str,
    ) {
        let f = promised(
            procs::bind_frag_data_location_indexed(&self.t, self.api),
            Feature::dual_src_blend,
            "glBindFragDataLocationIndexedEXT",
        );
        let name = std::ffi::CString::new(name).expect("an output name has no NUL");
        // SAFETY: a NUL-terminated string, live for the call.
        unsafe { f(program.0, color, index, name.as_ptr().cast::<GLchar>()) };
    }

    /// `glPatchParameterfv` for the default tessellation levels: four outer, two inner. Desktop
    /// GL only.
    pub fn patch_parameter_fv(&self, pname: GLenum, values: &[f32]) {
        assert!(!self.api.is_gles(), "glPatchParameterfv on a GLES context");
        let need = match pname {
            GL_PATCH_DEFAULT_OUTER_LEVEL => 4,
            GL_PATCH_DEFAULT_INNER_LEVEL => 2,
            _ => panic!("glPatchParameterfv takes no {pname:#x}"),
        };
        assert_eq!(values.len(), need, "the driver reads {need} levels");
        // SAFETY: the driver reads `need` floats for `pname`, and `values` holds exactly that many.
        unsafe { self.t.glPatchParameterfv()(pname, values.as_ptr()) };
    }

    /// `glPrimitiveRestartIndex`. Desktop GL only: GLES restarts at the fixed index alone.
    pub fn primitive_restart_index(&self, index: GLuint) {
        assert!(!self.api.is_gles(), "glPrimitiveRestartIndex on a GLES context");
        // SAFETY: plain scalar.
        unsafe { self.t.glPrimitiveRestartIndex()(index) };
    }

    pub fn uniform_1i(&self, location: UniformLocation, v: GLint) {
        // SAFETY: plain scalars.
        unsafe { self.t.glUniform1i()(location.0, v) };
    }

    pub fn uniform_1iv(&self, location: UniformLocation, v: &[GLint]) {
        let count = GLsizei::try_from(v.len()).expect("a uniform count fits a GLsizei");
        // SAFETY: the driver reads `count` integers from a slice of that length.
        unsafe { self.t.glUniform1iv()(location.0, count, v.as_ptr()) };
    }

    pub fn uniform_4f(&self, location: UniformLocation, v: [f32; 4]) {
        // SAFETY: plain scalars.
        unsafe { self.t.glUniform4f()(location.0, v[0], v[1], v[2], v[3]) };
    }

    /// `glUniform4uiv` over `v`, which holds `count` vectors of four.
    pub fn uniform_4uiv(&self, location: UniformLocation, v: &[u32]) {
        let count = GLsizei::try_from(v.len() / 4).expect("a uniform count fits a GLsizei");
        // SAFETY: the driver reads `count` vectors of four, which the slice holds.
        unsafe { self.t.glUniform4uiv()(location.0, count, v.as_ptr()) };
    }

    pub fn active_texture(&self, unit: TextureUnit) {
        // SAFETY: plain scalar.
        unsafe { self.t.glActiveTexture()(GL_TEXTURE0 + unit.0) };
    }

    pub fn bind_sampler(&self, unit: TextureUnit, sampler: Option<SamplerName>) {
        // SAFETY: plain scalars; zero is "no sampler".
        unsafe { self.t.glBindSampler()(unit.0, sampler.map_or(0, |s| s.0)) };
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bind_image_texture(
        &self,
        unit: ImageUnit,
        texture: Option<TextureName>,
        level: GLint,
        layered: bool,
        layer: GLint,
        access: GLenum,
        format: GLenum,
    ) {
        // SAFETY: plain scalars.
        unsafe {
            self.t.glBindImageTexture()(
                unit.0,
                texture.map_or(0, |t| t.0),
                level,
                layered as GLboolean,
                layer,
                access,
                format,
            )
        };
    }

    pub fn bind_vertex_buffer(
        &self,
        binding: GLuint,
        buf: Option<BufferName>,
        offset: u32,
        stride: u32,
    ) {
        // SAFETY: plain scalars; zero is "no buffer".
        unsafe {
            self.t.glBindVertexBuffer()(
                binding,
                buf.map_or(0, |b| b.0),
                offset as GLintptr,
                stride as GLsizei,
            )
        };
    }

    // ---- blending ----

    pub fn blend_func_separate(
        &self,
        src_rgb: GLenum,
        dst_rgb: GLenum,
        src_a: GLenum,
        dst_a: GLenum,
    ) {
        // SAFETY: plain scalars.
        unsafe { self.t.glBlendFuncSeparate()(src_rgb, dst_rgb, src_a, dst_a) };
    }

    pub fn blend_equation_separate(&self, rgb: GLenum, alpha: GLenum) {
        // SAFETY: plain scalars.
        unsafe { self.t.glBlendEquationSeparate()(rgb, alpha) };
    }

    pub fn blend_equation(&self, mode: GLenum) {
        // SAFETY: plain scalar.
        unsafe { self.t.glBlendEquation()(mode) };
    }

    /// `glBlendFuncSeparatei`.
    pub fn blend_func_separate_i(
        &self,
        buf: GLuint,
        src_rgb: GLenum,
        dst_rgb: GLenum,
        src_a: GLenum,
        dst_a: GLenum,
    ) {
        let f = promised(
            procs::blend_func_separate_i(&self.t, self.api),
            Feature::indep_blend_func,
            "glBlendFuncSeparatei",
        );
        // SAFETY: plain scalars.
        unsafe { f(buf, src_rgb, dst_rgb, src_a, dst_a) };
    }

    /// `glBlendEquationSeparatei`.
    pub fn blend_equation_separate_i(&self, buf: GLuint, rgb: GLenum, alpha: GLenum) {
        let f = promised(
            procs::blend_equation_separate_i(&self.t, self.api),
            Feature::indep_blend_func,
            "glBlendEquationSeparatei",
        );
        // SAFETY: plain scalars.
        unsafe { f(buf, rgb, alpha) };
    }

    /// `glEnablei`/`glDisablei`.
    pub fn set_enabled_i(&self, cap: GLenum, index: GLuint, on: bool) {
        let f = if on {
            promised(procs::enable_i(&self.t, self.api), Feature::indep_blend, "glEnablei")
        } else {
            promised(procs::disable_i(&self.t, self.api), Feature::indep_blend, "glDisablei")
        };
        // SAFETY: plain scalars.
        unsafe { f(cap, index) };
    }

    // ---- draws ----

    pub fn draw_arrays(&self, mode: GLenum, first: GLint, count: GLsizei) {
        // SAFETY: plain scalars; the driver reads the bound arrays, which it bounds itself.
        unsafe { self.t.glDrawArrays()(mode, first, count) };
    }

    pub fn draw_arrays_instanced(
        &self,
        mode: GLenum,
        first: GLint,
        count: GLsizei,
        instances: GLsizei,
    ) {
        // SAFETY: plain scalars.
        unsafe { self.t.glDrawArraysInstanced()(mode, first, count, instances) };
    }

    pub fn draw_arrays_instanced_base_instance(
        &self,
        mode: GLenum,
        first: GLint,
        count: GLsizei,
        instances: GLsizei,
        base_instance: GLuint,
    ) {
        let f = promised(
            procs::draw_arrays_instanced_base_instance(&self.t, self.api),
            Feature::base_instance,
            "glDrawArraysInstancedBaseInstanceEXT",
        );
        // SAFETY: plain scalars.
        unsafe { f(mode, first, count, instances, base_instance) };
    }

    /// `glDrawElements` with the indices at `offset` into the bound element buffer.
    pub fn draw_elements(&self, mode: GLenum, count: GLsizei, ty: GLenum, offset: u32) {
        // SAFETY: an offset into the bound element array buffer, which the driver bounds.
        unsafe { self.t.glDrawElements()(mode, count, ty, offset_ptr(offset)) };
    }

    pub fn draw_range_elements(
        &self,
        mode: GLenum,
        start: GLuint,
        end: GLuint,
        count: GLsizei,
        ty: GLenum,
        offset: u32,
    ) {
        // SAFETY: as `draw_elements`.
        unsafe { self.t.glDrawRangeElements()(mode, start, end, count, ty, offset_ptr(offset)) };
    }

    pub fn draw_elements_base_vertex(
        &self,
        mode: GLenum,
        count: GLsizei,
        ty: GLenum,
        offset: u32,
        base_vertex: GLint,
    ) {
        // SAFETY: as `draw_elements`.
        unsafe {
            self.t.glDrawElementsBaseVertex()(mode, count, ty, offset_ptr(offset), base_vertex)
        };
    }

    #[allow(clippy::too_many_arguments)]
    pub fn draw_range_elements_base_vertex(
        &self,
        mode: GLenum,
        start: GLuint,
        end: GLuint,
        count: GLsizei,
        ty: GLenum,
        offset: u32,
        base_vertex: GLint,
    ) {
        // SAFETY: as `draw_elements`.
        unsafe {
            self.t.glDrawRangeElementsBaseVertex()(
                mode,
                start,
                end,
                count,
                ty,
                offset_ptr(offset),
                base_vertex,
            )
        };
    }

    pub fn draw_elements_instanced(
        &self,
        mode: GLenum,
        count: GLsizei,
        ty: GLenum,
        offset: u32,
        instances: GLsizei,
    ) {
        // SAFETY: as `draw_elements`.
        unsafe { self.t.glDrawElementsInstanced()(mode, count, ty, offset_ptr(offset), instances) };
    }

    pub fn draw_elements_instanced_base_vertex(
        &self,
        mode: GLenum,
        count: GLsizei,
        ty: GLenum,
        offset: u32,
        instances: GLsizei,
        base_vertex: GLint,
    ) {
        // SAFETY: as `draw_elements`.
        unsafe {
            self.t.glDrawElementsInstancedBaseVertex()(
                mode,
                count,
                ty,
                offset_ptr(offset),
                instances,
                base_vertex,
            )
        };
    }

    pub fn draw_elements_instanced_base_instance(
        &self,
        mode: GLenum,
        count: GLsizei,
        ty: GLenum,
        offset: u32,
        instances: GLsizei,
        base_instance: GLuint,
    ) {
        let f = promised(
            procs::draw_elements_instanced_base_instance(&self.t, self.api),
            Feature::base_instance,
            "glDrawElementsInstancedBaseInstanceEXT",
        );
        // SAFETY: as `draw_elements`.
        unsafe { f(mode, count, ty, offset_ptr(offset), instances, base_instance) };
    }

    #[allow(clippy::too_many_arguments)]
    pub fn draw_elements_instanced_base_vertex_base_instance(
        &self,
        mode: GLenum,
        count: GLsizei,
        ty: GLenum,
        offset: u32,
        instances: GLsizei,
        base_vertex: GLint,
        base_instance: GLuint,
    ) {
        let f = promised(
            procs::draw_elements_instanced_base_vertex_base_instance(&self.t, self.api),
            Feature::base_instance,
            "glDrawElementsInstancedBaseVertexBaseInstanceEXT",
        );
        // SAFETY: as `draw_elements`.
        unsafe { f(mode, count, ty, offset_ptr(offset), instances, base_vertex, base_instance) };
    }

    pub fn dispatch_compute(&self, groups: [GLuint; 3]) {
        // SAFETY: plain scalars.
        unsafe { self.t.glDispatchCompute()(groups[0], groups[1], groups[2]) };
    }

    /// `glDispatchComputeIndirect` with the command at `offset` into the bound dispatch buffer.
    pub fn dispatch_compute_indirect(&self, offset: u32) {
        // SAFETY: an offset into the bound dispatch buffer, which the driver bounds.
        unsafe { self.t.glDispatchComputeIndirect()(offset as GLintptr) };
    }

    /// `glDrawArraysIndirect` with the command at `offset` into the bound indirect buffer.
    pub fn draw_arrays_indirect(&self, mode: GLenum, offset: u32) {
        // SAFETY: an offset into the bound indirect buffer, which the driver bounds.
        unsafe { self.t.glDrawArraysIndirect()(mode, offset_ptr(offset)) };
    }

    pub fn draw_elements_indirect(&self, mode: GLenum, ty: GLenum, offset: u32) {
        // SAFETY: as `draw_arrays_indirect`.
        unsafe { self.t.glDrawElementsIndirect()(mode, ty, offset_ptr(offset)) };
    }

    pub fn multi_draw_arrays_indirect(
        &self,
        mode: GLenum,
        offset: u32,
        draw_count: GLsizei,
        stride: GLsizei,
    ) {
        let f = promised(
            procs::multi_draw_arrays_indirect(&self.t, self.api),
            Feature::multi_draw_indirect,
            "glMultiDrawArraysIndirectEXT",
        );
        // SAFETY: as `draw_arrays_indirect`.
        unsafe { f(mode, offset_ptr(offset), draw_count, stride) };
    }

    pub fn multi_draw_elements_indirect(
        &self,
        mode: GLenum,
        ty: GLenum,
        offset: u32,
        draw_count: GLsizei,
        stride: GLsizei,
    ) {
        let f = promised(
            procs::multi_draw_elements_indirect(&self.t, self.api),
            Feature::multi_draw_indirect,
            "glMultiDrawElementsIndirectEXT",
        );
        // SAFETY: as `draw_arrays_indirect`.
        unsafe { f(mode, ty, offset_ptr(offset), draw_count, stride) };
    }

    /// `glMultiDrawArraysIndirectCount`: up to `max_draws` commands from `offset` into the bound
    /// indirect buffer, as many as the count at `count_offset` into the bound parameter buffer.
    pub fn multi_draw_arrays_indirect_count(
        &self,
        mode: GLenum,
        offset: u32,
        count_offset: u32,
        max_draws: GLsizei,
        stride: GLsizei,
    ) {
        let f = promised(
            procs::multi_draw_arrays_indirect_count(&self.t, self.api),
            Feature::indirect_params,
            "glMultiDrawArraysIndirectCount",
        );
        // SAFETY: both offsets are into buffers bound for the call, which the driver bounds.
        unsafe { f(mode, offset_ptr(offset), count_offset as GLintptr, max_draws, stride) };
    }

    pub fn multi_draw_elements_indirect_count(
        &self,
        mode: GLenum,
        ty: GLenum,
        offset: u32,
        count_offset: u32,
        max_draws: GLsizei,
        stride: GLsizei,
    ) {
        let f = promised(
            procs::multi_draw_elements_indirect_count(&self.t, self.api),
            Feature::indirect_params,
            "glMultiDrawElementsIndirectCount",
        );
        // SAFETY: as `multi_draw_arrays_indirect_count`.
        unsafe { f(mode, ty, offset_ptr(offset), count_offset as GLintptr, max_draws, stride) };
    }

    pub fn patch_parameter_i(&self, name: GLenum, value: GLint) {
        // SAFETY: plain scalars.
        unsafe { self.t.glPatchParameteri()(name, value) };
    }

    pub fn begin_transform_feedback(&self, mode: GLenum) {
        // SAFETY: plain scalar.
        unsafe { self.t.glBeginTransformFeedback()(mode) };
    }

    pub fn pause_transform_feedback(&self) {
        // SAFETY: no arguments.
        unsafe { self.t.glPauseTransformFeedback()() };
    }

    pub fn resume_transform_feedback(&self) {
        // SAFETY: no arguments.
        unsafe { self.t.glResumeTransformFeedback()() };
    }

    // ---- misc ----

    /// `glBindProgramPipeline(0)`.
    pub fn bind_program_pipeline_none(&self) {
        self.bind_program_pipeline(0);
    }

    pub fn finish(&self) {
        // SAFETY: takes nothing.
        unsafe { self.t.glFinish()() };
    }

    pub fn flush(&self) {
        // SAFETY: takes nothing.
        unsafe { self.t.glFlush()() };
    }

    /// Take a fence for the work queued on the current context, and flush so it can complete.
    ///
    /// **The flush is not separable from the sync**, which is why this is one call and not two.
    /// A sync object signals when the commands issued before it on its context complete, but a
    /// command that is still sitting in the client-side buffer has not been issued to the GPU at
    /// all: without the flush the sync may never be reached, and whoever waits on it waits
    /// forever. The waiting thread cannot repair this later -- `GL_SYNC_FLUSH_COMMANDS_BIT`
    /// flushes the *waiter's* context, which is not the one holding the work.
    ///
    /// `None` if the driver refused to make one, which is the caller's cue that this fence cannot
    /// be answered by waiting and must be answered some other way.
    pub fn fence(&self) -> Option<Fence> {
        // SAFETY: the condition and flags are the only pair the spec defines for this call.
        let sync = unsafe { self.t.glFenceSync()(GL_SYNC_GPU_COMMANDS_COMPLETE, 0) };
        if sync.is_null() {
            return None;
        }
        self.flush();
        Some(Fence(sync))
    }

    /// Wait once for a fence's work to have run, up to `timeout_ns`.
    ///
    /// Borrows rather than consumes, because one wait is not an answer: `glClientWaitSync` may
    /// report a timeout at an implementation's own cap rather than at the one asked for, so a
    /// caller that means "wait until it is done" has to come back. Spend the fence with
    /// [`Gl::fence_delete`] once the answer is one it will act on.
    ///
    /// Call this on a context of the share group the fence was taken in; it need not be, and
    /// normally is not, the context that took it.
    pub fn fence_wait(&self, fence: &Fence, timeout_ns: u64) -> FenceWait {
        // SAFETY: `sync` came from `glFenceSync` on a context of this share group and is alive --
        // only `fence_delete` deletes one, and it consumes the `Fence`, which is not `Copy`. Zero
        // flags: the flush the fence was created with is what makes the work reachable, and
        // `GL_SYNC_FLUSH_COMMANDS_BIT` would flush this thread's context, not the one holding it.
        let got = unsafe { self.t.glClientWaitSync()(fence.0, 0, timeout_ns) };
        match got {
            GL_ALREADY_SIGNALED | GL_CONDITION_SATISFIED => FenceWait::Signalled,
            GL_TIMEOUT_EXPIRED => FenceWait::Timeout,
            _ => FenceWait::Failed,
        }
    }

    /// Delete a fence. Consumes it: this is what spending one means.
    pub fn fence_delete(&self, fence: Fence) {
        // `Fence` aborts on drop, so take the token out without letting the destructor run.
        let fence = core::mem::ManuallyDrop::new(fence);
        // SAFETY: as `fence_wait` -- alive, of this share group, and nothing reads it after this.
        unsafe { self.t.glDeleteSync()(fence.0) };
    }
}

/// What one [`Gl::fence_wait`] found.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FenceWait {
    /// The work has run.
    Signalled,
    /// Not yet. Says nothing about whether it ever will: come back.
    Timeout,
    /// The driver refused the wait. Not a fence anyone should keep waiting on.
    Failed,
}

#[cfg(test)]
mod tests {
    use super::super::egl::{Flavour, Winsys};
    use super::*;

    /// The bounded reads are chosen by what the driver advertises, in the C's order: Mesa's iris
    /// answers `glGetnCompressedTexImage` and writes nothing, so a name that resolves is no
    /// evidence of a read that works.
    #[test]
    fn the_bounded_reads_are_the_ones_the_driver_advertises() {
        let has = |api, ext: &[&str]| {
            let f = Features::probe(api, ext.iter().map(|e| e.to_string()));
            RobustReads::choose(api, &f)
        };
        assert_eq!(has(Api::Gl(46), &["GL_ARB_robustness"]), RobustReads::Arb);
        assert_eq!(has(Api::Gl(46), &[]), RobustReads::Core, "4.5 made glReadnPixels core");
        assert_eq!(has(Api::Gl(33), &[]), RobustReads::Unbounded);
        assert_eq!(has(Api::Gles(32), &["GL_KHR_robustness"]), RobustReads::Khr);
        assert_eq!(has(Api::Gles(32), &[]), RobustReads::Unbounded, "a GLES version is no gl_ver");
    }

    /// A resolver asks for its API's spellings and no other: a desktop context with only the
    /// GLES name of an entry point has none, and the reverse. A driver may answer a name its API
    /// does not have with a stub, and a stub called in place of the real entry point does nothing
    /// and reports no error.
    #[test]
    fn a_resolver_asks_only_for_its_apis_spelling() {
        let _display = crate::vrend::one_display_at_a_time();
        let winsys = Winsys::open(Flavour::Gles).expect("the surfaceless display opens");
        let only_ext = winsys.procs_without(|n| n == c"glClipControl");
        let only_core = winsys.procs_without(|n| n == c"glClipControlEXT");
        assert!(procs::clip_control(&only_ext, Api::Gles(32)).is_some(), "GLES's own spelling");
        assert!(procs::clip_control(&only_ext, Api::Gl(46)).is_none(), "not desktop GL's");
        assert!(procs::clip_control(&only_core, Api::Gl(46)).is_some(), "desktop GL's own");
        assert!(procs::clip_control(&only_core, Api::Gles(32)).is_none(), "not GLES's");
        // A feature granted on both APIs has a spelling on both: a GLES host advertising
        // `GL_OES_viewport_array` is told viewports past the first are served, and a guest that
        // sets one must reach the driver, not a desktop-only wrapper.
        let all = winsys.procs();
        assert!(procs::viewport_indexed(&all, Api::Gles(32)).is_some(), "OES on GLES");
        assert!(procs::scissor_indexed(&all, Api::Gles(32)).is_some(), "OES on GLES");
        assert!(procs::depth_range_indexed_f(&all, Api::Gles(32)).is_some(), "OES on GLES");
    }
}
