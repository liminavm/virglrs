// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! The classic renderer's shader translator: a guest's TGSI program becomes the GLSL the host
//! compiles, and what the renderer needs to know about it (`vrend_shader.c`).
//!
//! The translation is a faithful port of the C's string emitter, and the text it produces is
//! held to the C's byte for byte (`harness/replay/fixtures/vrend-shaders.txt`): the corpus is
//! the oracle, and a difference in the GLSL is a bug here until the harness says otherwise.
//! A typed intermediate form is a later change, gated on its own.
//!
//! Only the C's GLES leg is ported. This renderer drives a GLES 3.1 context and nothing else
//! (`docs/design.md`), so the C's `use_gles`, `use_core_profile` and
//! `use_explicit_locations` switches are fixed at true, true and false and do not appear in
//! [`Config`]; the desktop branches they guarded are not here. A desktop context is its own gated
//! change, and it brings its branches with it.
//!
//! [`Key`] is what the renderer's state contributes to a translation -- the C's
//! `vrend_shader_key`, which it fills from the bound framebuffer, blend, rasterizer and depth
//! state and the neighbouring stages, and compares whole to find a variant already built. The
//! C's key holds the per-stage part in a union; each stage only ever reads and writes its own
//! member, so the three are separate fields here and compare the same.

pub mod glsl;

use super::features::{Feature, Features};
use super::gl::Gl;
use super::gl::gles::{GL_MAX_TESS_PATCH_COMPONENTS, GL_SHADING_LANGUAGE_VERSION};
pub use super::pipe::slots::{
    MAX_CLIP_OR_CULL_DISTANCES, MAX_COLOR_BUFS, MAX_COMBINED_SSBO_BINDING_POINTS,
    MAX_SHADER_BUFFERS, MAX_SHADER_IMAGES, MAX_SHADER_SAMPLER_VIEWS, MAX_SO_OUTPUTS,
    NUM_CLIP_PLANES, POLYGON_STIPPLE_SIZE,
};
use super::pipe::{CompareFunc, LogicOp, PrimType};
use super::proto::StreamOutput;
use super::resource::Limits;
use super::tgsi::{self, Interpolate, Location, Semantic};

pub use glsl::{Failure, Strings, convert, create_passthrough_tcs};

/// `gl_advanced_blend_mode`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[repr(u32)]
pub enum AdvancedBlend {
    #[default]
    None = 0,
    Multiply,
    Screen,
    Overlay,
    Darken,
    Lighten,
    ColorDodge,
    ColorBurn,
    HardLight,
    SoftLight,
    Difference,
    Exclusion,
    HslHue,
    HslSaturation,
    HslColor,
    HslLuminosity,
    All,
}

impl AdvancedBlend {
    /// The value a `FS_BLEND_EQUATION_ADVANCED` property carries.
    pub fn from_property(v: u32) -> Option<AdvancedBlend> {
        use AdvancedBlend::*;
        const ALL: [AdvancedBlend; 17] = [
            None,
            Multiply,
            Screen,
            Overlay,
            Darken,
            Lighten,
            ColorDodge,
            ColorBurn,
            HardLight,
            SoftLight,
            Difference,
            Exclusion,
            HslHue,
            HslSaturation,
            HslColor,
            HslLuminosity,
            All,
        ];
        ALL.get(v as usize).copied()
    }

    /// `blend_to_name`.
    pub fn layout_name(self) -> &'static str {
        use AdvancedBlend::*;
        match self {
            Multiply => "multiply",
            Screen => "screen",
            Overlay => "overlay",
            Darken => "darken",
            Lighten => "lighten",
            ColorDodge => "colordodge",
            ColorBurn => "colorburn",
            HardLight => "hardlight",
            SoftLight => "softlight",
            Difference => "difference",
            Exclusion => "exclusion",
            HslHue => "hsl_hue",
            HslSaturation => "hsl_saturation",
            HslColor => "hsl_color",
            HslLuminosity => "hsl_luminosity",
            All => "all_equations",
            None => "UNKNOWN",
        }
    }
}

/// The GLSL the translator writes: the C's `use_gles`. Decided by the host's API once, at init.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dialect {
    /// GLSL ES, at the host's own version.
    Es,
    /// Desktop GLSL, core profile, at the lowest version the shader needs.
    Core,
}

/// `vrend_shader_cfg`: what the host's GL can do, as the translator needs to know it. Read once
/// from the probed features when the renderer comes up.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Config {
    pub dialect: Dialect,
    pub glsl_version: u32,
    /// At most eight (`PIPE_MAX_COLOR_BUFS`).
    pub max_draw_buffers: u32,
    pub max_shader_patch_varyings: u32,
    pub has_arrays_of_arrays: bool,
    pub has_gpu_shader5: bool,
    pub has_es31_compat: bool,
    pub has_conservative_depth: bool,
    pub has_dual_src_blend: bool,
    pub has_fbfetch_coherent: bool,
    pub has_cull_distance: bool,
    pub has_nopersective: bool,
    pub has_texture_shadow_lod: bool,
    pub has_vs_layer: bool,
    pub has_vs_viewport_index: bool,
    /// Program pipelines, which a separable stage needs to run as one. The C translates a
    /// separable stage on any desktop host; one without pipelines never advertised them, and
    /// translates and links the stage as any other.
    pub has_separate_shader_objects: bool,
}

impl Config {
    /// The C's `cfg->use_gles`.
    pub fn is_gles(&self) -> bool {
        self.dialect == Dialect::Es
    }

    /// Whether a stage the guest marks separable is translated and run as one: on desktop GL
    /// with program pipelines. GLES is strict about how separable interfaces match -- it
    /// refuses, for one, an input without a matching output -- so the C serves none there.
    pub fn serves_separable(&self) -> bool {
        !self.is_gles() && self.has_separate_shader_objects
    }

    /// The C's `shader_cfg` fill at context creation, from the probed features and limits and
    /// the driver's `GL_SHADING_LANGUAGE_VERSION`.
    pub fn probe(gl: &Gl, features: &Features, limits: &Limits) -> Config {
        let has = |f| features.has(f);
        let dialect = if features.api().is_gles() { Dialect::Es } else { Dialect::Core };
        Config {
            dialect,
            glsl_version: glsl_version(dialect, &gl.get_string(GL_SHADING_LANGUAGE_VERSION)),
            max_draw_buffers: limits.max_draw_buffers,
            max_shader_patch_varyings: if has(Feature::tessellation) {
                gl.get_integer(GL_MAX_TESS_PATCH_COMPONENTS).max(0) as u32 / 4
            } else {
                0
            },
            has_arrays_of_arrays: has(Feature::arrays_of_arrays),
            has_gpu_shader5: has(Feature::gpu_shader5),
            has_es31_compat: has(Feature::gles31_compatibility),
            has_conservative_depth: has(Feature::conservative_depth),
            has_dual_src_blend: has(Feature::dual_src_blend),
            has_fbfetch_coherent: has(Feature::framebuffer_fetch),
            has_cull_distance: has(Feature::cull_distance),
            has_nopersective: has(Feature::shader_noperspective_interpolation),
            has_texture_shadow_lod: has(Feature::texture_shadow_lod),
            has_vs_layer: has(Feature::vs_layer_viewport),
            has_vs_viewport_index: has(Feature::vs_viewport_index),
            has_separate_shader_objects: has(Feature::separate_shader_objects),
        }
    }
}

/// `get_glsl_version`: "OpenGL ES GLSL ES 3.10" is 310 on GLES, and "4.60 NVIDIA" is 460 on
/// desktop GL. The C's `sscanf` skips four words on GLES and none on desktop, and reads
/// `major.minor`; a string that does not fit is version 0, which the C refuses at context
/// creation and the translator would refuse at the first shader.
fn glsl_version(dialect: Dialect, s: &str) -> u32 {
    let skip = match dialect {
        Dialect::Es => 4,
        Dialect::Core => 0,
    };
    let mut words = s.split_whitespace().skip(skip);
    let Some(v) = words.next() else {
        return 0;
    };
    let mut parts = v.split('.');
    let major: u32 = parts.next().and_then(|p| p.parse().ok()).unwrap_or(0);
    let minor: u32 = parts.next().and_then(|p| p.parse().ok()).unwrap_or(0);
    major * 100 + minor
}

/// `vrend_interp_info`: how a fragment shader input is interpolated, as the stage before it
/// must declare its output.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct InterpInfo {
    pub semantic_name: Semantic,
    pub semantic_index: u16,
    pub interpolate: Interpolate,
    pub location: Location,
}

/// `vrend_fs_shader_info`: the fragment shader's demands on the stage feeding it.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct FragmentInfo {
    pub glsl_ver: u32,
    pub has_sample_input: bool,
    pub has_noperspective: bool,
    pub interps: Vec<InterpInfo>,
}

/// `vrend_array`: a run of samplers or images declared as one array.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Array {
    pub first: i32,
    pub array_size: i32,
}

/// `vrend_shader_io_array`: one declared IO array, by semantic.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct IoArray {
    pub name: Semantic,
    pub sid: u32,
    pub size: u32,
    pub array_id: u32,
}

/// `vrend_shader_io_array_info`: the IO arrays a stage declares, at most sixteen.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct IoArrayInfo {
    pub layout: Vec<IoArray>,
}

/// The fragment stage's part of the key (`vrend_shader_key.fs`).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct FsKey {
    pub surface_component_bits: [u8; MAX_COLOR_BUFS],
    pub coord_replace: u32,
    pub swizzle_output_rgb_to_bgr: u8,
    pub needs_manual_srgb_encode_bitmask: u8,
    pub cbufs_are_a8_bitmask: u8,
    pub cbufs_signed_int_bitmask: u8,
    pub cbufs_unsigned_int_bitmask: u8,
    pub logicop_func: Option<LogicOp>,
    pub prim_is_points: bool,
    pub lower_left_origin: bool,
    pub available_color_in_bits: u8,
}

/// The vertex stage's part of the key (`vrend_shader_key.vs`). The C's signed and unsigned
/// attribute masks are only filled under `use_integer`, which this host never sets.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct VsKey {
    pub attrib_zyxw_bitmask: u32,
    pub fog_fixup_mask: u32,
    /// The draw is served as arrays over gathered vertices, which number `gl_VertexID` from
    /// zero; the vertex ID is read from the `virgl_vertex_id` input the draw fills instead.
    pub vertex_id_from_input: bool,
}

/// The name of the input a de-indexed draw passes each vertex's own ID through.
pub const VERTEX_ID_INPUT: &str = "virgl_vertex_id";

/// The location `VERTEX_ID_INPUT` is bound to: the lowest one the vertex shader's own inputs,
/// `attrib_input_mask`, leave free. Fixed at link time and found again at the draw from the same
/// mask, so the two cannot disagree. `None` when the shader uses every location.
pub fn vertex_id_location(attrib_input_mask: u32, max_attribs: u32) -> Option<u32> {
    let free = (!attrib_input_mask).trailing_zeros();
    (free < max_attribs).then_some(free)
}

/// The geometry stage's part of the key (`vrend_shader_key.gs`).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct GsKey {
    pub emit_clip_distance: bool,
}

/// `vrend_shader_key`: everything outside the program that shapes its translation. Two
/// translations of one program with equal keys produce equal GLSL, which is what the variant
/// cache compares.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Key {
    pub out_generic_expected_mask: u64,
    pub out_texcoord_expected_mask: u64,
    pub in_generic_expected_mask: u64,
    pub in_texcoord_expected_mask: u64,
    pub in_patch_expected_mask: u64,
    pub force_invariant_inputs: [u32; 4],
    pub fs_info: FragmentInfo,
    pub in_arrays: IoArrayInfo,
    pub fs: FsKey,
    pub vs: VsKey,
    pub gs: GsKey,
    pub sampler_views_lower_swizzle_mask: [u64; 2],
    pub sampler_views_emulated_rect_mask: [u64; 2],
    pub sampler_views_lower_array_mask: [u64; 2],
    /// Four `pipe_swizzle` values, three bits each.
    pub tex_swizzle: [u16; MAX_SHADER_SAMPLER_VIEWS],
    pub ssbo_binding_offset: u8,
    pub image_binding_offset: u8,
    pub alpha_test: CompareFunc,
    pub num_in_cull: u8,
    pub num_in_clip: u8,
    pub num_out_cull: u8,
    pub num_out_clip: u8,
    pub pstipple_enabled: bool,
    pub add_alpha_test: bool,
    pub color_two_side: bool,
    pub gs_present: bool,
    pub tcs_present: bool,
    pub tes_present: bool,
    pub flatshade: bool,
    pub require_input_arrays: bool,
    pub require_output_arrays: bool,
    pub use_pervertex_in: bool,
    /// The vertex stage writes the rasterizer's point size, `VirglBlock`'s `point_size`, into
    /// `gl_PointSize`: GLES has no fixed point size, only the one the last stage writes.
    pub fixed_point_size: bool,
}

impl Default for Key {
    fn default() -> Key {
        Key {
            out_generic_expected_mask: 0,
            out_texcoord_expected_mask: 0,
            in_generic_expected_mask: 0,
            in_texcoord_expected_mask: 0,
            in_patch_expected_mask: 0,
            force_invariant_inputs: [0; 4],
            fs_info: FragmentInfo::default(),
            in_arrays: IoArrayInfo::default(),
            fs: FsKey::default(),
            vs: VsKey::default(),
            gs: GsKey::default(),
            sampler_views_lower_swizzle_mask: [0; 2],
            sampler_views_emulated_rect_mask: [0; 2],
            sampler_views_lower_array_mask: [0; 2],
            tex_swizzle: [0; MAX_SHADER_SAMPLER_VIEWS],
            ssbo_binding_offset: 0,
            image_binding_offset: 0,
            alpha_test: CompareFunc::Never,
            num_in_cull: 0,
            num_in_clip: 0,
            num_out_cull: 0,
            num_out_clip: 0,
            pstipple_enabled: false,
            add_alpha_test: false,
            color_two_side: false,
            gs_present: false,
            tcs_present: false,
            tes_present: false,
            flatshade: false,
            require_input_arrays: false,
            require_output_arrays: false,
            use_pervertex_in: false,
            fixed_point_size: false,
        }
    }
}

impl Key {
    /// `vrend_shader_sampler_views_mask_get`.
    pub fn view_mask_get(mask: &[u64; 2], index: usize) -> bool {
        (mask[index / 64] >> (index % 64)) & 1 != 0
    }

    /// `vrend_shader_sampler_views_mask_set`.
    pub fn view_mask_set(mask: &mut [u64; 2], index: usize) {
        mask[index / 64] |= 1 << (index % 64);
    }

    /// `vrend_shader_needs_alpha_func`.
    pub fn needs_alpha_func(&self) -> bool {
        self.add_alpha_test && !matches!(self.alpha_test, CompareFunc::Never | CompareFunc::Always)
    }
}

/// `vrend_shader_info`: what a translation tells the renderer about the program, independent
/// of the key.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Info {
    pub invariant_outputs: [u32; 4],
    pub in_generic_emitted_mask: u64,
    pub in_texcoord_emitted_mask: u64,
    pub out_generic_emitted_mask: u64,
    pub out_patch_emitted_mask: u64,
    pub output_arrays: IoArrayInfo,
    pub sampler_arrays: Vec<Array>,
    pub image_arrays: Vec<Array>,
    /// The GLSL name of each stream output, in the stream-output info's order.
    pub so_names: Vec<String>,
    pub so_info: StreamOutput,
    /// Eight cbufs, depth, stencil, samplemask: each a GLSL output index, or -1.
    pub fs_output_layout: [i8; 12],
    pub samplers_used_mask: u32,
    pub images_used_mask: u32,
    pub image_binding_offset: u32,
    pub image_last_binding: i32,
    pub ubo_used_mask: u32,
    pub ssbo_used_mask: u32,
    pub ssbo_binding_offset: u32,
    pub ssbo_last_binding: i32,
    pub shadow_samp_mask: u32,
    pub attrib_input_mask: u32,
    /// `FS_BLEND_EQUATION_ADVANCED`, a bit per [`AdvancedBlend`] mode.
    pub fs_blend_equation_advanced: u32,
    pub fog_input_mask: u32,
    pub fog_output_mask: u32,
    pub num_consts: i32,
    pub num_inputs: i32,
    pub num_outputs: i32,
    pub gs_out_prim: Option<PrimType>,
    pub tes_prim: Option<PrimType>,
    pub out_texcoord_emitted_mask: u8,
    pub ubo_indirect: bool,
    pub tes_point_mode: bool,
    pub gles_use_tex_query_level: bool,
    pub separable_program: bool,
    pub has_input_arrays: bool,
    pub has_output_arrays: bool,
    pub use_pervertex_in: bool,
    pub reads_drawid: bool,
}

impl Info {
    /// `vrend_shader_lookup_sampler_array`: the first sampler of the array `index` sits in, or
    /// -1 when it sits in none.
    pub fn lookup_sampler_array(&self, index: i32) -> i32 {
        self.sampler_arrays
            .iter()
            .find(|a| index >= a.first && index < a.first + a.array_size)
            .map_or(-1, |a| a.first)
    }
}

/// `vrend_shader_query_separable_program`: whether the guest marked `shader` separable and its
/// interface can be matched by location. Generics count down from 31 and patches up from 0, so
/// the two ranges must not meet; and the semantics GLSL matches only by name cannot be
/// guaranteed in every stage of a separable program, so a shader using one is not.
///
/// What a stage's selector says before its first translation: the translation then writes
/// the property alone, as the C's `fill_sinfo` overwrites this.
pub fn query_separable_program(shader: &tgsi::Shader, cfg: &Config) -> bool {
    use tgsi::{File, Processor, Token};
    let mut separable = false;
    let mut unsupported_io = false;
    // The C keeps these in bytes, and a semantic index past one wraps there too.
    let (mut generic_in, mut patch_in, mut generic_out, mut patch_out) = (0u8, 0u8, 0u8, 0u8);
    for token in &shader.tokens {
        match token {
            Token::Property(p) if p.name == tgsi::Property::SeparableProgram => {
                separable = p.data != 0;
            }
            Token::Declaration(d) => {
                // Vertex inputs and fragment outputs are not interfaces.
                if (d.file == File::Input && shader.processor == Processor::Vertex)
                    || (d.file == File::Output && shader.processor == Processor::Fragment)
                {
                    continue;
                }
                let index = d.semantic.index as u8;
                let input = d.file == File::Input;
                match d.semantic.name {
                    Semantic::Patch if input => patch_in = patch_in.max(index),
                    Semantic::Patch => patch_out = patch_out.max(index),
                    Semantic::Generic if input => generic_in = generic_in.max(index),
                    Semantic::Generic => generic_out = generic_out.max(index),
                    Semantic::Color
                    | Semantic::ClipVertex
                    | Semantic::BColor
                    | Semantic::TexCoord
                    | Semantic::Fog => unsupported_io = true,
                    _ => {}
                }
            }
            _ => {}
        }
    }
    let max_varying = glsl::MAX_VARYING;
    let supports = !unsupported_io
        && u32::from(generic_in) + u32::from(patch_in) < max_varying
        && u32::from(generic_out) + u32::from(patch_out) < max_varying
        && u32::from(patch_in) < cfg.max_shader_patch_varyings
        && u32::from(patch_out) < cfg.max_shader_patch_varyings;
    separable && supports
}

/// `vrend_variable_shader_info`: what a translation tells the renderer that depends on the
/// key, so belongs to the variant.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct VarInfo {
    pub fs_info: FragmentInfo,
    pub num_in_clip: u8,
    pub num_in_cull: u8,
    pub num_out_clip: u8,
    pub num_out_cull: u8,
    pub num_ucp: i32,
    pub legacy_color_bits: i32,
}

/// `vrend_shader_samplerreturnconv`: the GLSL sampler prefix for a return type -- a space,
/// not nothing, for a float one, as the C spells it.
pub fn sampler_return_conv(ty: tgsi::ReturnType) -> char {
    match ty {
        tgsi::ReturnType::Sint => 'i',
        tgsi::ReturnType::Uint => 'u',
        _ => ' ',
    }
}

/// `vrend_shader_samplertypeconv`: the GLSL sampler type for a TGSI texture target. GLES has
/// neither one-dimensional nor rectangle samplers, so those targets sample as `2D` there.
pub fn sampler_type_conv(dialect: Dialect, target: tgsi::Texture) -> Option<&'static str> {
    use tgsi::Texture::*;
    let core = dialect == Dialect::Core;
    Some(match target {
        Buffer => "Buffer",
        D1 if core => "1D",
        D1 => "2D",
        D2 => "2D",
        D3 => "3D",
        Cube => "Cube",
        Rect if core => "2DRect",
        Rect => "2D",
        Shadow1d if core => "1DShadow",
        Shadow1d => "2DShadow",
        Shadow2d => "2DShadow",
        ShadowRect if core => "2DRectShadow",
        ShadowRect => "2DShadow",
        Array1d if core => "1DArray",
        Array1d => "2DArray",
        Array2d => "2DArray",
        Shadow1dArray if core => "1DArrayShadow",
        Shadow1dArray => "2DArrayShadow",
        Shadow2dArray => "2DArrayShadow",
        ShadowCube => "CubeShadow",
        Msaa2d => "2DMS",
        Msaa2dArray => "2DMSArray",
        CubeArray => "CubeArray",
        ShadowCubeArray => "CubeArrayShadow",
        Unknown => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_glsl_version_is_read_as_the_c_reads_it() {
        assert_eq!(glsl_version(Dialect::Es, "OpenGL ES GLSL ES 3.10"), 310);
        assert_eq!(glsl_version(Dialect::Es, "OpenGL ES GLSL ES 3.20 (zink)"), 320);
        assert_eq!(glsl_version(Dialect::Es, "3.10"), 0);
        assert_eq!(glsl_version(Dialect::Core, "4.60"), 460);
        assert_eq!(glsl_version(Dialect::Core, "3.30 NVIDIA via Cg compiler"), 330);
    }
}
