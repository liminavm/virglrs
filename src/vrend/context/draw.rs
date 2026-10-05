// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! Draws: the program a draw runs, linked from the variants selected for the bound stages, and
//! everything the C binds around it before the primitives go out (`vrend_draw_vbo`).
//!
//! The order is the C's, because pixels come out of the order GL sees things in: the lazy state
//! first, then the program, then per stage the uniform buffers, the constants, the samplers, the
//! images and the storage buffers, then the vertex layout and the index buffer, and only then
//! the draw call the wire's `pipe_draw_info` names. What the C skips on a clean dirty flag is
//! skipped here on the same flag, since a re-bind is a GL call the C did not make.

use super::*;

/// Which program a sub-context has bound, and where it sits in the program list.
///
/// The two travel together because they must agree: an index alone goes stale the moment
/// [`SubContext::forget_programs_of`] shifts the list under it, and would then bind another
/// program's uniform locations and draw rather than fail. Carried as one value there is nothing to
/// keep in step -- [`SubContext::program_at`] is the only way to spend one, and it checks the
/// serial against the program the index landed on.
///
/// This is also what removes the scans: the six binders below each asked for the program once per
/// stage, around thirty scans per draw for a question settled before the first of them ran.
#[derive(Clone, Copy, Debug)]
pub struct ProgramSlot {
    at: usize,
    serial: ProgramSerial,
}

impl ProgramSlot {
    /// Where this slot lands once the program at `removed`, named `gone`, leaves the list.
    ///
    /// `None` when the program removed is the one this slot named: there is nothing left for it
    /// to name, and the next draw selects again.
    fn after_removing(self, removed: usize, gone: ProgramSerial) -> Option<ProgramSlot> {
        if self.serial == gone {
            return None;
        }
        // Everything after the hole shifts down one; everything before it does not move.
        let at = if self.at > removed { self.at - 1 } else { self.at };
        Some(ProgramSlot { at, ..self })
    }
}

/// `sysval_uniform_block`: what the shaders read through the `VirglBlock` uniform block, laid
/// out as std140 lays it out -- every array element on sixteen bytes.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Sysval {
    pub clip_planes: [[f32; 4]; shader::NUM_CLIP_PLANES],
    pub stipple: [u32; shader::POLYGON_STIPPLE_SIZE],
    pub winsys_adjust_y: f32,
    pub alpha_ref_val: f32,
    pub clip_plane_enabled: f32,
    pub drawid_base: i32,
}

impl Default for Sysval {
    /// The C's zeroed block, with `winsys_adjust_y` at one as the sub-context sets it.
    fn default() -> Sysval {
        Sysval {
            clip_planes: [[0.0; 4]; shader::NUM_CLIP_PLANES],
            stipple: [0; shader::POLYGON_STIPPLE_SIZE],
            winsys_adjust_y: 1.0,
            alpha_ref_val: 0.0,
            clip_plane_enabled: 0.0,
            drawid_base: 0,
        }
    }
}

impl Sysval {
    /// The block's bytes, in the layout the shader declares, written into the caller's buffer.
    ///
    /// Every byte of `out` is written, padding included, so a reused buffer carries nothing over.
    fn write_to(&self, out: &mut [u8; Sysval::SIZE]) {
        let mut at = 0;
        {
            let mut put = |bytes: &[u8]| {
                out[at..at + bytes.len()].copy_from_slice(bytes);
                at += bytes.len();
            };
            for p in &self.clip_planes {
                for c in p {
                    put(&c.to_ne_bytes());
                }
            }
            for s in &self.stipple {
                put(&s.to_ne_bytes());
                put(&[0; 12]);
            }
            put(&self.winsys_adjust_y.to_ne_bytes());
            put(&self.alpha_ref_val.to_ne_bytes());
            put(&self.clip_plane_enabled.to_ne_bytes());
            put(&self.drawid_base.to_ne_bytes());
        }
        debug_assert_eq!(at, Sysval::SIZE);
    }

    const SIZE: usize = shader::NUM_CLIP_PLANES * 16 + shader::POLYGON_STIPPLE_SIZE * 16 + 16;
}

/// How many times a [`Tracked`] value has been reached mutably. Monotonic, so a value changed and
/// changed back still counts as moved -- the comparison is conservative in the safe direction.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Generation(u64);

/// A value that counts the times it was handed out mutably, so a cache of it can be checked
/// against a word instead of against the value.
///
/// The C compares a cookie (`vrend_renderer.c`'s `sysvalue_data_cookie`) where comparing the
/// sysval block itself costs a copy of it and a compare of it on every draw. A cookie someone has
/// to remember to bump would be the bug this renderer is written to avoid, so there is nothing to
/// remember: [`DerefMut`] is the only way to the value, and it bumps. A write that changes nothing
/// still bumps, which costs one redundant upload -- never a stale one.
#[derive(Clone, Copy, Default, Debug)]
pub struct Tracked<T> {
    value: T,
    moved: u64,
}

impl<T> Tracked<T> {
    pub fn new(value: T) -> Tracked<T> {
        Tracked { value, moved: 0 }
    }

    /// Which version of the value this is. Equal generations mean the value has not been reached
    /// mutably since; they do not mean it is unchanged by some other route, because there is none.
    pub fn generation(&self) -> Generation {
        Generation(self.moved)
    }
}

impl<T> std::ops::Deref for Tracked<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.value
    }
}

impl<T> std::ops::DerefMut for Tracked<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.moved += 1;
        &mut self.value
    }
}

/// What `vrend_hw_emit_blend` last told GL, for what it only emits on change.
#[derive(Clone, Copy, Debug)]
pub struct HwBlend {
    pub logicop_enable: bool,
    pub logicop_func: LogicOp,
    pub independent: bool,
    /// `hw_blend_state.rt[i].colormask`: what a clear restores.
    pub colormask: [u8; MAX_COLOR_BUFS],
}

impl Default for HwBlend {
    fn default() -> HwBlend {
        HwBlend {
            logicop_enable: false,
            logicop_func: LogicOp::Clear,
            independent: false,
            colormask: [0xf; MAX_COLOR_BUFS],
        }
    }
}

/// Where a transform feedback object stands, as the draws move it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Xfb {
    /// Bound, not yet begun: the first draw begins it.
    NeedBegin,
    Started,
    Paused,
}

/// What a program links, which is also what a selection looks it up by.
///
/// The C keeps graphics and compute programs on two lists and tells them apart by which list it
/// searched. Here they share one list, so the kind is part of the key: a compute program can never
/// answer a graphics lookup, nor be walked as a draw's stages.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Linkage {
    Graphics {
        /// The variant of each graphics stage, in stage order.
        stages: [Option<VariantId>; 5],
        dual_src: bool,
    },
    Compute(VariantId),
}

impl Linkage {
    /// The last stage a draw walks: the last before the rasterizer, or the fragment stage.
    ///
    /// Only a draw asks, and a draw runs only a program it selected, so a compute program here is
    /// a host invariant broken, not a guest's doing.
    fn last_stage(&self) -> ShaderStage {
        let Linkage::Graphics { stages, .. } = self else {
            panic!("a draw is running a compute program");
        };
        if stages[ShaderStage::TessEval.index()].is_some() {
            ShaderStage::TessEval
        } else if stages[ShaderStage::Geometry.index()].is_some() {
            ShaderStage::Geometry
        } else {
            ShaderStage::Fragment
        }
    }
}

/// What a draw runs: the stages linked into one program, or -- where every stage is separable
/// -- a pipeline of the stages' own programs, as the C's `is_pipeline` chooses.
///
/// A pipeline holds no stage program's name. Each stage's program is its variant's, which owns
/// and deletes it; the pipeline reaches one through the variant its linkage names
/// ([`SubContext::stage_program`], [`Linked::program_in`]), so a name cannot outlive its program
/// here -- a stage that is gone fails the lookup rather than naming a deleted program.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ProgramObject {
    Linked(ProgramName),
    Pipeline { pipeline: PipelineName },
}

impl ProgramObject {
    /// `vrend_set_active_pipeline_stage`: make `program`, a stage of this pipeline, the one
    /// `glUniform*` writes to. A linked program is already that program, so there is nothing
    /// to do.
    pub fn activate(&self, gl: &Gl, program: ProgramName) {
        if let ProgramObject::Pipeline { pipeline, .. } = self {
            gl.active_shader_program(*pipeline, program);
        }
    }

    /// `vrend_use_program`.
    pub fn use_in(&self, gl: &Gl, bound: &mut BoundProgram) {
        match self {
            ProgramObject::Linked(p) => gl.use_program(bound, Some(*p)),
            ProgramObject::Pipeline { pipeline, .. } => gl.use_pipeline(bound, *pipeline),
        }
    }

    /// The program or the pipeline, deleted. A pipeline's stage programs are their variants',
    /// which outlive it and are deleted with them.
    pub fn delete(self, gl: &Gl, bound: &mut BoundProgram) {
        match self {
            ProgramObject::Linked(p) => gl.delete_program(bound, p),
            ProgramObject::Pipeline { pipeline, .. } => gl.delete_program_pipeline(bound, pipeline),
        }
    }
}

/// `vrend_linked_shader_program`: the variants of the bound stages linked into one GL program,
/// and every location the draw path writes through.
pub struct LinkedProgram {
    /// Names this program in the sub-context's list; stable while the program lives.
    pub serial: ProgramSerial,
    pub object: ProgramObject,
    pub linkage: Linkage,
    pub ubo_used_mask: [u32; ShaderStage::COUNT],
    pub samplers_used_mask: [u32; ShaderStage::COUNT],
    pub shadow_samp_mask: [u32; ShaderStage::COUNT],
    /// One location per set bit of `samplers_used_mask`, in bit order.
    pub sampler_locs: [Vec<Option<UniformLocation>>; ShaderStage::COUNT],
    pub shadow_samp_mask_locs: [Vec<Option<UniformLocation>>; ShaderStage::COUNT],
    pub shadow_samp_add_locs: [Vec<Option<UniformLocation>>; ShaderStage::COUNT],
    pub const_location: [Option<UniformLocation>; ShaderStage::COUNT],
    pub num_consts: [usize; ShaderStage::COUNT],
    pub ssbo_used_mask: [u32; ShaderStage::COUNT],
    pub ssbo_binding_offset: [u32; ShaderStage::COUNT],
    pub images_used_mask: [u32; ShaderStage::COUNT],
    /// One location per image slot up to the highest used; `None` where the compiler dropped
    /// the image and the draw has nothing to write.
    pub img_locs: [Vec<Option<UniformLocation>>; ShaderStage::COUNT],
    pub image_binding_offset: [u32; ShaderStage::COUNT],
    pub tex_levels_uniform_id: [Option<UniformLocation>; ShaderStage::COUNT],
    /// The `VirglBlock` binding and the buffer behind it, once a stage declared the block.
    pub virgl_block_bind: Option<BindingPoint>,
    pub sysval_buffer: Option<BufferName>,
    /// The block the buffer holds, so a draw uploads only a block that changed.
    pub sysval_uploaded: Option<Generation>,
    pub reads_drawid: bool,
    pub fs_blend_equation_advanced: u32,
}

impl LinkedProgram {
    /// A program with nothing looked up yet.
    fn new(serial: ProgramSerial, object: ProgramObject, linkage: Linkage) -> LinkedProgram {
        LinkedProgram {
            serial,
            object,
            linkage,
            ubo_used_mask: [0; ShaderStage::COUNT],
            samplers_used_mask: [0; ShaderStage::COUNT],
            shadow_samp_mask: [0; ShaderStage::COUNT],
            sampler_locs: Default::default(),
            shadow_samp_mask_locs: Default::default(),
            shadow_samp_add_locs: Default::default(),
            const_location: [None; ShaderStage::COUNT],
            num_consts: [0; ShaderStage::COUNT],
            ssbo_used_mask: [0; ShaderStage::COUNT],
            ssbo_binding_offset: [0; ShaderStage::COUNT],
            images_used_mask: [0; ShaderStage::COUNT],
            img_locs: Default::default(),
            image_binding_offset: [0; ShaderStage::COUNT],
            tex_levels_uniform_id: [None; ShaderStage::COUNT],
            virgl_block_bind: None,
            sysval_buffer: None,
            sysval_uploaded: None,
            reads_drawid: false,
            fs_blend_equation_advanced: 0,
        }
    }

    /// Whether this program links `variant`.
    pub fn links(&self, variant: VariantId) -> bool {
        match self.linkage {
            Linkage::Graphics { stages, .. } => stages.contains(&Some(variant)),
            Linkage::Compute(v) => v == variant,
        }
    }

    /// `bind_const_locs`, `bind_image_locs` and `bind_ssbo_locs` for one stage.
    fn bind_resource_locs(&mut self, gl: &Gl, features: &Features, l: &Linked<'_>) {
        let s = l.stage.index();
        let id = l.program_in(&self.object);
        let prefix = stage_prefix(l.stage);
        if l.info.num_consts > 0 {
            self.const_location[s] = gl.get_uniform_location(id, &format!("{prefix}const0"));
            self.num_consts[s] = l.info.num_consts as usize;
        }
        let mask = l.info.images_used_mask;
        if (mask != 0 || !l.info.image_arrays.is_empty()) && features.has(Feature::images) {
            let nsamp = (32 - mask.leading_zeros()) as usize;
            let mut locs = vec![None; nsamp];
            if !l.info.image_arrays.is_empty() {
                for arr in &l.info.image_arrays {
                    for j in 0..arr.array_size {
                        let name = format!("{prefix}img{}[{j}]", arr.first);
                        // An image the compiler dropped has no location; the draw skips it.
                        let loc = gl.get_uniform_location(id, &name);
                        let slot = (arr.first + j) as usize;
                        if slot >= locs.len() {
                            locs.resize(slot + 1, None);
                        }
                        locs[slot] = loc;
                    }
                }
            } else {
                for (i, loc) in locs.iter_mut().enumerate() {
                    if mask & (1 << i) != 0 {
                        let name = format!("{prefix}img{i}");
                        *loc = gl.get_uniform_location(id, &name);
                    }
                }
            }
            self.img_locs[s] = locs;
            self.images_used_mask[s] = mask;
            self.image_binding_offset[s] = l.info.image_binding_offset;
        }
        if features.has(Feature::ssbo) {
            self.ssbo_used_mask[s] = l.info.ssbo_used_mask;
            self.ssbo_binding_offset[s] = l.info.ssbo_binding_offset;
        }
    }

    /// `bind_sampler_locs` and `bind_ubo_locs` for one stage, its uniform blocks bound from
    /// `next_ubo_id` on. Answers the first binding after them.
    fn bind_sampler_and_ubo_locs(
        &mut self,
        gl: &Gl,
        (stage, id, info): (ShaderStage, ProgramName, &shader::Info),
        mut next_ubo_id: BindingPoint,
    ) -> BindingPoint {
        let s = stage.index();
        let prefix = stage_prefix(stage);
        self.object.activate(gl, id);
        self.sampler_locs[s].clear();
        self.shadow_samp_mask_locs[s].clear();
        self.shadow_samp_add_locs[s].clear();
        let mut mask = info.samplers_used_mask;
        while mask != 0 {
            let i = mask.trailing_zeros() as i32;
            mask &= mask - 1;
            let name = if !info.sampler_arrays.is_empty() {
                let first = info.lookup_sampler_array(i);
                format!("{prefix}samp{first}[{}]", i - first)
            } else {
                format!("{prefix}samp{i}")
            };
            self.sampler_locs[s].push(gl.get_uniform_location(id, &name));
            let (mask_loc, add_loc) = if info.shadow_samp_mask & (1 << i) != 0 {
                (
                    gl.get_uniform_location(id, &format!("{prefix}shadmask{i}")),
                    gl.get_uniform_location(id, &format!("{prefix}shadadd{i}")),
                )
            } else {
                (None, None)
            };
            self.shadow_samp_mask_locs[s].push(mask_loc);
            self.shadow_samp_add_locs[s].push(add_loc);
        }
        self.samplers_used_mask[s] = info.samplers_used_mask;
        self.shadow_samp_mask[s] = info.shadow_samp_mask;
        let mut mask = info.ubo_used_mask;
        while mask != 0 {
            let i = mask.trailing_zeros();
            mask &= mask - 1;
            let name = if info.ubo_indirect {
                format!("{prefix}ubo[{}]", i.wrapping_sub(1) as i32)
            } else {
                format!("{prefix}ubo{i}")
            };
            if let Some(block) = gl.get_uniform_block_index(id, &name) {
                gl.uniform_block_binding(id, block, next_ubo_id);
            }
            next_ubo_id = next_ubo_id.next();
        }
        self.ubo_used_mask[s] = info.ubo_used_mask;
        next_ubo_id
    }

    /// `rebind_ubo_and_sampler_locs`, with `bind_virgl_block_loc` after it: every stage's
    /// sampler locations and block bindings, numbered through the stages in the C's order, then
    /// the `VirglBlock` at the binding after the last.
    ///
    /// Block bindings are a program's state, and a separable stage's program is shared by every
    /// pipeline it serves -- each numbering it from where the stages before it end -- so a
    /// pipeline found again runs this again when another pipeline bound its stages since.
    fn rebind_ubo_and_sampler_locs(&mut self, gl: &Gl, walk: &[StageBlocks<'_>]) {
        let mut next_ubo_id = BindingPoint::FIRST;
        for stage in walk {
            next_ubo_id = self.bind_sampler_and_ubo_locs(gl, *stage, next_ubo_id);
        }
        self.virgl_block_bind = None;
        for &(_, id, _) in walk {
            let Some(block) = gl.get_uniform_block_index(id, "VirglBlock") else {
                continue;
            };
            let mut created = false;
            if self.virgl_block_bind.is_none() {
                self.virgl_block_bind = Some(next_ubo_id);
                if self.sysval_buffer.is_none() {
                    self.sysval_buffer = Some(gl.gen_buffer());
                    created = true;
                }
            }
            let bind = self.virgl_block_bind.expect("set a moment ago");
            self.object.activate(gl, id);
            gl.uniform_block_binding(id, block, bind);
            let size = gl.uniform_block_data_size(id, block);
            assert!(
                size as usize >= Sysval::SIZE,
                "the VirglBlock the shader declares holds the sysval block"
            );
            if created {
                let buf = self.sysval_buffer.expect("made a moment ago");
                gl.bind_buffer(GL_UNIFORM_BUFFER, Some(buf));
                gl.buffer_data_null(GL_UNIFORM_BUFFER, size as usize, GL_DYNAMIC_DRAW);
                gl.bind_buffer(GL_UNIFORM_BUFFER, None);
            }
        }
    }
}

/// The vertex buffer slots a bind must clear: the ones the hardware still holds past the set
/// the guest has now. `hw` is what the last bind left bound -- not what the last state-set
/// replaced, which is a different number the moment the guest sets twice before drawing.
fn stale_vbo_slots(bound: usize, hw: usize) -> std::ops::Range<usize> {
    bound..hw.max(bound)
}

fn stage_prefix(stage: ShaderStage) -> &'static str {
    match stage {
        ShaderStage::Vertex => "vs",
        ShaderStage::Fragment => "fs",
        ShaderStage::Geometry => "gs",
        ShaderStage::TessCtrl => "tc",
        ShaderStage::TessEval => "te",
        ShaderStage::Compute => "cs",
    }
}

const GRAPHICS_STAGES: [ShaderStage; 5] = [
    ShaderStage::Vertex,
    ShaderStage::TessCtrl,
    ShaderStage::TessEval,
    ShaderStage::Geometry,
    ShaderStage::Fragment,
];

/// Stage order as the C walks it: vertex, fragment, geometry, control, evaluation.
const C_STAGE_ORDER: [ShaderStage; 5] = [
    ShaderStage::Vertex,
    ShaderStage::Fragment,
    ShaderStage::Geometry,
    ShaderStage::TessCtrl,
    ShaderStage::TessEval,
];

fn blend_func(f: BlendFunc) -> GLenum {
    match f {
        BlendFunc::Add => GL_FUNC_ADD,
        BlendFunc::Subtract => GL_FUNC_SUBTRACT,
        BlendFunc::ReverseSubtract => GL_FUNC_REVERSE_SUBTRACT,
        BlendFunc::Min => GL_MIN,
        BlendFunc::Max => GL_MAX,
    }
}

fn blend_factor(f: BlendFactor) -> GLenum {
    use BlendFactor::*;
    match f {
        One => GL_ONE,
        SrcColor => GL_SRC_COLOR,
        SrcAlpha => GL_SRC_ALPHA,
        DstColor => GL_DST_COLOR,
        DstAlpha => GL_DST_ALPHA,
        ConstColor => GL_CONSTANT_COLOR,
        ConstAlpha => GL_CONSTANT_ALPHA,
        Src1Color => GL_SRC1_COLOR_EXT,
        Src1Alpha => GL_SRC1_ALPHA_EXT,
        SrcAlphaSaturate => GL_SRC_ALPHA_SATURATE,
        Zero => GL_ZERO,
        InvSrcColor => GL_ONE_MINUS_SRC_COLOR,
        InvSrcAlpha => GL_ONE_MINUS_SRC_ALPHA,
        InvDstColor => GL_ONE_MINUS_DST_COLOR,
        InvDstAlpha => GL_ONE_MINUS_DST_ALPHA,
        InvConstColor => GL_ONE_MINUS_CONSTANT_COLOR,
        InvConstAlpha => GL_ONE_MINUS_CONSTANT_ALPHA,
        InvSrc1Color => GL_ONE_MINUS_SRC1_COLOR_EXT,
        InvSrc1Alpha => GL_ONE_MINUS_SRC1_ALPHA_EXT,
    }
}

/// `vrend_patch_blend_state`'s patching: the blend state GL is told for the colour targets'
/// formats, and whether a constant factor reads the blend colour's alpha in red. A target
/// without alpha reads its destination alpha as one; an emulated-alpha target blends and masks
/// its alpha in red, the channel that stores it.
fn patch_blend(api: Api, state: &BlendState, targets: &[Option<Format>]) -> (BlendState, bool) {
    let state = *state;
    let mut new_state = state;
    let mut swizzle_blend_color = false;
    let patched = if state.independent_blend_enable { MAX_COLOR_BUFS } else { 1 };
    for i in 0..patched {
        let Some(Some(format)) = targets.get(i).copied() else {
            continue;
        };
        if crate::vrend::formats::is_emulated_alpha(api, format) {
            // The target's alpha is its red: blend and mask the alpha there, and nothing
            // else.
            let rt = state.rt[i];
            if let Some(eq) = rt.equation {
                new_state.rt[i].equation = Some(RtBlendEq {
                    rgb: BlendEq {
                        func: eq.rgb.func,
                        src: conv_a8_blend(eq.alpha.src),
                        dst: conv_a8_blend(eq.alpha.dst),
                    },
                    alpha: BlendEq {
                        func: eq.alpha.func,
                        src: BlendFactor::Zero,
                        dst: BlendFactor::Zero,
                    },
                });
            }
            new_state.rt[i].colormask =
                if rt.colormask & PIPE_MASK_A != 0 { PIPE_MASK_R } else { 0 };
            if let Some(eq) = new_state.rt[i].equation
                && (is_const_blend(eq.rgb.src) || is_const_blend(eq.rgb.dst))
            {
                swizzle_blend_color = true;
            }
            continue;
        }
        let has_alpha = format.describe().is_some_and(|d| d.has_alpha());
        if has_alpha {
            continue;
        }
        let Some(eq) = state.rt[i].equation else {
            continue;
        };
        let dst = [eq.rgb.src, eq.rgb.dst, eq.alpha.src, eq.alpha.dst];
        if !dst.iter().any(|f| is_dst_blend(*f)) {
            continue;
        }
        new_state.rt[i].equation = Some(RtBlendEq {
            rgb: BlendEq {
                func: eq.rgb.func,
                src: conv_dst_blend(eq.rgb.src),
                dst: conv_dst_blend(eq.rgb.dst),
            },
            alpha: BlendEq {
                func: eq.alpha.func,
                src: conv_dst_blend(eq.alpha.src),
                dst: conv_dst_blend(eq.alpha.dst),
            },
        });
    }
    (new_state, swizzle_blend_color)
}

/// `PIPE_MASK_R` and `PIPE_MASK_A` of a colour mask.
const PIPE_MASK_R: u8 = 1;
const PIPE_MASK_A: u8 = 8;

/// `conv_a8_blend`: a destination-alpha factor on an emulated-alpha target, whose alpha is its
/// colour's red.
fn conv_a8_blend(f: BlendFactor) -> BlendFactor {
    match f {
        BlendFactor::DstAlpha => BlendFactor::DstColor,
        BlendFactor::InvDstAlpha => BlendFactor::InvDstColor,
        f => f,
    }
}

fn is_const_blend(f: BlendFactor) -> bool {
    matches!(
        f,
        BlendFactor::ConstColor
            | BlendFactor::ConstAlpha
            | BlendFactor::InvConstColor
            | BlendFactor::InvConstAlpha
    )
}

/// `translate_logicop`.
fn logic_op(op: LogicOp) -> GLenum {
    match op {
        LogicOp::Clear => GL_CLEAR,
        LogicOp::Nor => GL_NOR,
        LogicOp::AndInverted => GL_AND_INVERTED,
        LogicOp::CopyInverted => GL_COPY_INVERTED,
        LogicOp::AndReverse => GL_AND_REVERSE,
        LogicOp::Invert => GL_INVERT,
        LogicOp::Xor => GL_XOR,
        LogicOp::Nand => GL_NAND,
        LogicOp::And => GL_AND,
        LogicOp::Equiv => GL_EQUIV,
        LogicOp::Noop => GL_NOOP,
        LogicOp::OrInverted => GL_OR_INVERTED,
        LogicOp::Copy => GL_COPY,
        LogicOp::OrReverse => GL_OR_REVERSE,
        LogicOp::Or => GL_OR,
        LogicOp::Set => GL_SET,
    }
}

fn is_dst_blend(f: BlendFactor) -> bool {
    matches!(f, BlendFactor::DstAlpha | BlendFactor::InvDstAlpha)
}

fn conv_dst_blend(f: BlendFactor) -> BlendFactor {
    match f {
        BlendFactor::DstAlpha => BlendFactor::One,
        BlendFactor::InvDstAlpha => BlendFactor::Zero,
        f => f,
    }
}

/// `util_blend_state_is_dual`: whether target `i` reads a second colour output.
pub(super) fn blend_is_dual(state: &BlendState, i: usize) -> bool {
    let Some(eq) = state.rt[i].equation else {
        return false;
    };
    use BlendFactor::*;
    let dual = |f| matches!(f, Src1Color | Src1Alpha | InvSrc1Color | InvSrc1Alpha);
    dual(eq.rgb.src) || dual(eq.rgb.dst) || dual(eq.alpha.src) || dual(eq.alpha.dst)
}

/// `get_xfb_mode`.
fn xfb_mode(mode: PrimType) -> GLenum {
    match mode {
        PrimType::Points => GL_POINTS,
        PrimType::Lines
        | PrimType::LineLoop
        | PrimType::LineStrip
        | PrimType::LinesAdjacency
        | PrimType::LineStripAdjacency => GL_LINES,
        _ => GL_TRIANGLES,
    }
}

/// `get_gs_xfb_mode`.
fn gs_xfb_mode(prim: Option<PrimType>) -> GLenum {
    match prim {
        Some(PrimType::Points) => GL_POINTS,
        Some(PrimType::LineStrip) => GL_LINES,
        Some(PrimType::TriangleStrip) => GL_TRIANGLES,
        _ => GL_POINTS,
    }
}

/// `get_tess_xfb_mode`.
fn tess_xfb_mode(prim: Option<PrimType>, point_mode: bool) -> GLenum {
    if point_mode {
        return GL_POINTS;
    }
    match prim {
        Some(PrimType::Quads) | Some(PrimType::Triangles) => GL_TRIANGLES,
        Some(PrimType::Lines) => GL_LINES,
        _ => GL_POINTS,
    }
}

/// Gallium numbers its primitives as GL numbers its modes, and the C passes the one as the
/// other; the quad strip and polygon the desktop enums name are among them.
fn prim_mode(mode: PrimType) -> GLenum {
    debug_assert_eq!(PrimType::Patches.wire(), GL_PATCHES);
    mode.wire() as GLenum
}

/// `get_skip_str`: the `gl_SkipComponents` varying that covers up to four of `skip`.
fn skip_varying(skip: &mut i32) -> Option<&'static str> {
    if *skip < 0 {
        *skip = 0;
        return None;
    }
    let (name, n) = match *skip {
        0 => return None,
        1 => ("gl_SkipComponents1", 1),
        2 => ("gl_SkipComponents2", 2),
        3 => ("gl_SkipComponents3", 3),
        _ => ("gl_SkipComponents4", 4),
    };
    *skip -= n;
    Some(name)
}

/// `set_stream_out_varyings`: the interleaved varying list a stream-output layout asks of GL.
fn set_stream_out_varyings(gl: &Gl, program: ProgramName, info: &shader::Info) {
    let so = &info.so_info;
    if so.outputs.is_empty() {
        return;
    }
    const MAX: usize = shader::MAX_SO_OUTPUTS * 2;
    let mut varyings: Vec<String> = Vec::new();
    let mut last_buffer = 0usize;
    let mut buf_offset = 0i32;
    for (i, out) in so.outputs.iter().enumerate() {
        let buffer = out.output_buffer.index();
        if last_buffer != buffer {
            let mut skip = so.stride[last_buffer] as i32 - buf_offset;
            while skip != 0 && varyings.len() < MAX {
                if let Some(s) = skip_varying(&mut skip) {
                    varyings.push(s.to_string());
                }
            }
            let mut j = last_buffer;
            while j < buffer && varyings.len() < MAX {
                varyings.push("gl_NextBuffer".to_string());
                j += 1;
            }
            last_buffer = buffer;
            buf_offset = 0;
        }
        let mut skip = out.dst_offset as i32 - buf_offset;
        while skip != 0 && varyings.len() < MAX {
            if let Some(s) = skip_varying(&mut skip) {
                varyings.push(s.to_string());
            }
        }
        buf_offset = out.dst_offset as i32 + out.num_components as i32;
        if let Some(name) = info.so_names.get(i)
            && !name.is_empty()
            && varyings.len() < MAX
        {
            varyings.push(name.clone());
        }
    }
    let mut skip = so.stride[last_buffer] as i32 - buf_offset;
    while skip != 0 && varyings.len() < MAX {
        if let Some(s) = skip_varying(&mut skip) {
            varyings.push(s.to_string());
        }
    }
    gl.transform_feedback_varyings(program, &varyings);
}

/// The variant a bound stage's program currently selects, with the program's info.
struct Linked<'a> {
    stage: ShaderStage,
    info: &'a shader::Info,
    variant: &'a Variant,
    gl: ShaderName,
}

impl Linked<'_> {
    /// The program holding this stage's uniforms in `object`: the linked program, or the
    /// variant's own when `object` is a pipeline of separable stages.
    fn program_in(&self, object: &ProgramObject) -> ProgramName {
        match object {
            ProgramObject::Linked(p) => *p,
            ProgramObject::Pipeline { .. } => {
                self.variant.separate.as_ref().expect("a pipeline's stages are separable").program
            }
        }
    }
}

/// A linked program's name for as long as its sub-context lives: minted once and never reused,
/// so a sub-context pointing at one cannot come to mean a later program.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ProgramSerial(u64);

/// A stage as the sampler and block walk sees it: which, the program holding its uniforms, and
/// what it declared.
type StageBlocks<'a> = (ShaderStage, ProgramName, &'a shader::Info);

/// A found pipeline whose stages another pipeline numbered since: the stages and what they
/// declared, to number again. `None` for a program that needs nothing.
fn stale_block_bindings<'a>(
    prog: &LinkedProgram,
    linked: &[Linked<'a>],
) -> Option<Vec<StageBlocks<'a>>> {
    let ProgramObject::Pipeline { .. } = prog.object else {
        return None;
    };
    let last = prog.linkage.last_stage();
    let walk: Vec<&Linked<'a>> = C_STAGE_ORDER
        .iter()
        .filter(|s| s.index() <= last.index())
        .filter_map(|s| linked.iter().find(|l| l.stage == *s))
        .collect();
    let stale = walk.iter().any(|l| {
        l.variant.separate.as_ref().is_some_and(|s| s.blocks_bound_for.get() != Some(prog.serial))
    });
    if !stale {
        return None;
    }
    for l in &walk {
        if let Some(separate) = &l.variant.separate {
            separate.blocks_bound_for.set(Some(prog.serial));
        }
    }
    Some(walk.iter().map(|l| (l.stage, l.program_in(&prog.object), l.info)).collect())
}

impl SubContext {
    /// The next name, taken through a `Cell` so minting one does not need the whole
    /// sub-context: the program being named is built from a borrow of it.
    fn mint_program_serial(&self) -> ProgramSerial {
        let serial = self.next_program_serial.get();
        self.next_program_serial.set(serial + 1);
        ProgramSerial(serial)
    }

    fn program(&self) -> Option<&LinkedProgram> {
        Some(self.program_at(self.prog?))
    }

    /// The program holding `stage`'s uniforms in the program at `at`: the linked program, or for
    /// a pipeline the stage's own, found through the variant that owns it. `None` for a stage a
    /// pipeline does not have.
    ///
    /// The draw that asks selected this program from the bound stages, so the variant its
    /// linkage names is bound and holds its program; not finding it is a host invariant broken.
    fn stage_program(&self, at: ProgramSlot, stage: ShaderStage) -> Option<ProgramName> {
        let prog = self.program_at(at);
        match prog.object {
            ProgramObject::Linked(p) => Some(p),
            ProgramObject::Pipeline { .. } => {
                let Linkage::Graphics { stages, .. } = prog.linkage else {
                    panic!("a compute program is never a pipeline");
                };
                let id = stages[stage.index()]?;
                let variant = self
                    .bound_program(stage)
                    .and_then(|t| t.variants.iter().find(|v| v.id == id))
                    .and_then(|v| v.separate.as_ref())
                    .expect("a pipeline's stages are bound while it draws");
                Some(variant.program)
            }
        }
    }

    /// The current program, to hand to a whole bind pass.
    fn program_slot(&self) -> Option<ProgramSlot> {
        self.prog
    }

    /// The program a slot names.
    ///
    /// The assert is the point of the type. A slot is resolved once and handed to every binder in
    /// the pass, which is what removes the scans -- and a slot that has stopped naming the current
    /// program would not fail, it would bind another program's uniform locations and draw. That is
    /// a host invariant, so it aborts here rather than reaching the screen; the cost is one integer
    /// compare against the scan it replaces.
    fn program_at(&self, at: ProgramSlot) -> &LinkedProgram {
        let prog = &self.programs[at.at];
        assert_eq!(prog.serial, at.serial, "a program slot outlived the program it named");
        prog
    }

    /// The program a slot names, to write to. Checked as [`SubContext::program_at`] is.
    fn program_at_mut(&mut self, at: ProgramSlot) -> &mut LinkedProgram {
        let prog = &mut self.programs[at.at];
        assert_eq!(prog.serial, at.serial, "a program slot outlived the program it named");
        prog
    }

    /// The current variant of each bound graphics stage, compiled -- `None` for a stage that
    /// is bound but has no compiled variant, which is the C's `!current` failure.
    fn linked_stages(&self) -> Result<Vec<Linked<'_>>, ShaderStage> {
        let mut out = Vec::new();
        for stage in GRAPHICS_STAGES {
            if !self.has_stage(stage) {
                continue;
            }
            out.push(self.linked_stage(stage).ok_or(stage)?);
        }
        Ok(out)
    }

    /// The current variant of the stage at `stage`, compiled, or `None` when there is no such
    /// variant.
    fn linked_stage(&self, stage: ShaderStage) -> Option<Linked<'_>> {
        let program = self.bound_program(stage)?;
        let variant = program.variants.first()?;
        Some(Linked { stage, info: &program.info, variant, gl: variant.gl? })
    }

    /// `vrend_destroy_program` for every program linking `variant`, as the C destroys them
    /// with the variant.
    fn forget_programs_of(&mut self, gl: &Gl, bound: &mut BoundProgram, variant: VariantId) {
        let mut i = 0;
        while i < self.programs.len() {
            if self.programs[i].links(variant) {
                let p = self.programs.remove(i);
                self.prog = self.prog.and_then(|slot| slot.after_removing(i, p.serial));
                if let Some(b) = p.sysval_buffer {
                    gl.delete_buffer(b);
                }
                p.object.delete(gl, bound);
            } else {
                i += 1;
            }
        }
    }
}

/// The C's `vrend_shader_destroy`, and the one place a shader leaves a sub-context: the
/// programs linking each variant, then the variant's GL shader.
pub(super) fn release_shader(
    sub: &mut SubContext,
    gl: &Gl,
    bound: &mut BoundProgram,
    shader: Shader,
) {
    if let ShaderText::Whole(p) = shader.text {
        release_variants(sub, gl, bound, p.translated.variants);
    }
}

/// The programs linking each of `variants`, then each variant's GL shader.
pub(super) fn release_variants(
    sub: &mut SubContext,
    gl: &Gl,
    bound: &mut BoundProgram,
    variants: Vec<Variant>,
) {
    for v in variants {
        sub.forget_programs_of(gl, bound, v.id);
        if let Some(separate) = v.separate {
            gl.delete_program(bound, separate.program);
        }
        if let Some(name) = v.gl {
            gl.delete_shader(name);
        }
    }
}

/// The fragment outputs' locations, told to `program` before its link. Answers whether the
/// program is linked for dual-source blending, which it is when the blend asks for it and the
/// shader has the second output to give.
fn bind_fragment_outputs(
    gl: &Gl,
    features: &Features,
    cmd: Cmd,
    program: ProgramName,
    fs: &shader::Info,
    dual_src: bool,
) -> Result<bool, Fault> {
    if fs.num_outputs <= 1 {
        return Ok(false);
    }
    if dual_src {
        if !features.has(Feature::dual_src_blend) {
            return Err(Fault::Shader { cmd, what: "dual-source blending the host lacks" });
        }
        gl.bind_frag_data_location_indexed(program, 0, 0, "fsout_c0");
        gl.bind_frag_data_location_indexed(program, 0, 1, "fsout_c1");
    } else if !features.api().is_gles() && features.has(Feature::dual_src_blend) {
        // GLES shaders carry their output layout themselves; a desktop one is told it here.
        for (i, &layout) in fs.fs_output_layout.iter().enumerate() {
            if i < fs.num_outputs as usize && layout >= 0 {
                let name = format!("fsout_c{layout}");
                gl.bind_frag_data_location_indexed(program, layout as GLuint, 0, &name);
            }
        }
    }
    Ok(dual_src)
}

/// The vertex inputs' locations, told to `program` before its link, where the host binds
/// attributes by location.
fn bind_attrib_locations(gl: &Gl, features: &Features, program: ProgramName, vs: &shader::Info) {
    if features.has(Feature::gles31_vertex_attrib_binding) {
        let mut mask = vs.attrib_input_mask;
        while mask != 0 {
            let i = mask.trailing_zeros();
            mask &= mask - 1;
            gl.bind_attrib_location(program, i, &format!("in_{i}"));
        }
    }
}

/// `vrend_compile_shader`'s separable half and `vrend_link_separable_shader`: a separable
/// stage's own program, made from its compiled shader and linked at once, so every pipeline it
/// serves names a program that is ready. What the link fixes -- stream output, fragment output
/// locations, attribute locations -- is fixed for the stage's life, as the C's is from its first
/// link on.
pub(super) fn link_separable(
    host: &mut Host<'_>,
    cmd: Cmd,
    stage: ShaderStage,
    info: &shader::Info,
    shader: ShaderName,
    dual_src: bool,
) -> Result<ProgramName, Fault> {
    let gl = host.gl;
    let features = host.features;
    let Some(program) = gl.create_program() else {
        return Err(Fault::Shader { cmd, what: "the driver refused a program object" });
    };
    gl.program_separable(program);
    gl.attach_shader(program, shader);
    let fixed = match stage {
        ShaderStage::Vertex | ShaderStage::Geometry | ShaderStage::TessEval => {
            set_stream_out_varyings(gl, program, info);
            if stage == ShaderStage::Vertex {
                bind_attrib_locations(gl, features, program, info);
            }
            Ok(false)
        }
        ShaderStage::Fragment => bind_fragment_outputs(gl, features, cmd, program, info, dual_src),
        _ => Ok(false),
    };
    let linked = fixed.and_then(|_| {
        gl.link_program(program).map_err(|log| {
            eprintln!("[virglrs] vrend: error linking a separable stage:\n{log}");
            Fault::Shader { cmd, what: "a separable stage the driver refused to link" }
        })
    });
    if let Err(e) = linked {
        gl.delete_program(host.current.program(), program);
        return Err(e);
    }
    Ok(program)
}

/// `add_shader_program`: the bound stages' variants linked into one program -- or, where every
/// stage is separable, gathered into a pipeline of their own programs -- with every location
/// the draw path will write through looked up once.
fn add_shader_program(
    host: &mut Host<'_>,
    cmd: Cmd,
    serial: ProgramSerial,
    linked: &[Linked<'_>],
    dual_src: bool,
) -> Result<LinkedProgram, Fault> {
    let gl = host.gl;
    let features = host.features;
    let mut stages = [None; 5];
    for l in linked {
        stages[l.stage.index()] = Some(l.variant.id);
    }
    let by_stage = |s: ShaderStage| linked.iter().find(|l| l.stage == s);
    let vs = by_stage(ShaderStage::Vertex).expect("a program has a vertex stage");
    let fs = by_stage(ShaderStage::Fragment).expect("a program has a fragment stage");
    let gs = by_stage(ShaderStage::Geometry);
    let tes = by_stage(ShaderStage::TessEval);

    // The C's `separable`: every stage's selector says so. A stage's own program exists exactly
    // when it did at its compile on a host with pipelines, so that is what is asked.
    let separable = linked.iter().all(|l| l.variant.separate.is_some());
    let (object, dual_src_linked) = if separable {
        let pipeline = gl.gen_program_pipeline();
        for l in linked {
            let program = l.variant.separate.as_ref().expect("asked a moment ago").program;
            gl.use_program_stages(pipeline, stage_bit(l.stage), program);
        }
        // The C reports a pipeline the driver will not validate and carries on; the report
        // poisons the context, which is what refusing it here does.
        if !gl.validate_program_pipeline(pipeline) {
            gl.delete_program_pipeline(host.current.program(), pipeline);
            return Err(Fault::Shader { cmd, what: "a program pipeline the driver refused" });
        }
        let object = ProgramObject::Pipeline { pipeline };
        (object, dual_src && fs.info.num_outputs > 1)
    } else {
        let Some(id) = gl.create_program() else {
            return Err(Fault::Shader { cmd, what: "the driver refused a program object" });
        };
        for l in linked {
            gl.attach_shader(id, l.gl);
        }
        // The stream-out layout belongs to the last stage before the rasterizer.
        set_stream_out_varyings(gl, id, gs.or(tes).unwrap_or(vs).info);
        let dual_src_linked = match bind_fragment_outputs(gl, features, cmd, id, fs.info, dual_src)
        {
            Ok(d) => d,
            Err(e) => {
                gl.delete_program(host.current.program(), id);
                return Err(e);
            }
        };
        bind_attrib_locations(gl, features, id, vs.info);
        if let Err(log) = gl.link_program(id) {
            gl.delete_program(host.current.program(), id);
            eprintln!("[virglrs] vrend: error linking program:\n{log}");
            for l in linked {
                eprintln!("{}: GLSL:\n{}", stage_prefix(l.stage), l.variant.strings.source());
            }
            return Err(Fault::Shader { cmd, what: "a program the driver refused to link" });
        }
        (ProgramObject::Linked(id), dual_src_linked)
    };

    let linkage = Linkage::Graphics { stages, dual_src: dual_src_linked };
    let last_stage = linkage.last_stage();
    let mut prog = LinkedProgram::new(serial, object, linkage);
    prog.fs_blend_equation_advanced = fs.info.fs_blend_equation_advanced;

    object.use_in(gl, host.current.program());

    // Stage order as the C walks it, vertex through the last stage.
    let walk: Vec<&Linked<'_>> = C_STAGE_ORDER
        .iter()
        .filter(|s| s.index() <= last_stage.index())
        .filter_map(|s| by_stage(*s))
        .collect();
    for l in &walk {
        prog.bind_resource_locs(gl, features, l);
        if l.info.reads_drawid {
            prog.reads_drawid = true;
        }
    }
    let blocks: Vec<StageBlocks<'_>> =
        walk.iter().map(|l| (l.stage, l.program_in(&object), l.info)).collect();
    prog.rebind_ubo_and_sampler_locs(gl, &blocks);
    for l in &walk {
        if let Some(separate) = &l.variant.separate {
            separate.blocks_bound_for.set(Some(serial));
        }
    }

    // The texture-level uniforms the GLES `textureQueryLevels` emulation reads, looked up
    // once here where the C looks them up on every draw.
    for l in &walk {
        if l.info.gles_use_tex_query_level {
            let name = format!("{}_texlod", stage_prefix(l.stage));
            let id = l.program_in(&object);
            prog.tex_levels_uniform_id[l.stage.index()] = gl.get_uniform_location(id, &name);
        }
    }

    Ok(prog)
}

/// The `glUseProgramStages` bit for a stage.
fn stage_bit(stage: ShaderStage) -> GLbitfield {
    match stage {
        ShaderStage::Vertex => GL_VERTEX_SHADER_BIT,
        ShaderStage::Fragment => GL_FRAGMENT_SHADER_BIT,
        ShaderStage::Geometry => GL_GEOMETRY_SHADER_BIT,
        ShaderStage::TessCtrl => GL_TESS_CONTROL_SHADER_BIT,
        ShaderStage::TessEval => GL_TESS_EVALUATION_SHADER_BIT,
        ShaderStage::Compute => panic!("a compute stage is never part of a pipeline"),
    }
}

/// `add_cs_shader_program`: the compute stage's variant linked alone, with every location a
/// dispatch will write through looked up once. A compute program has no stream output, no
/// fragment outputs and no `VirglBlock`, so none of the graphics builder's fixtures apply.
fn add_cs_shader_program(
    host: &mut Host<'_>,
    cmd: Cmd,
    serial: ProgramSerial,
    cs: &Linked<'_>,
) -> Result<LinkedProgram, Fault> {
    let gl = host.gl;
    let Some(id) = gl.create_program() else {
        return Err(Fault::Shader { cmd, what: "the driver refused a program object" });
    };
    gl.attach_shader(id, cs.gl);
    if let Err(log) = gl.link_program(id) {
        gl.delete_program(host.current.program(), id);
        eprintln!("[virglrs] vrend: error linking program:\n{log}");
        eprintln!("cs: GLSL:\n{}", cs.variant.strings.source());
        return Err(Fault::Shader { cmd, what: "a program the driver refused to link" });
    }
    let object = ProgramObject::Linked(id);
    let mut prog = LinkedProgram::new(serial, object, Linkage::Compute(cs.variant.id));
    object.use_in(gl, host.current.program());
    prog.bind_sampler_and_ubo_locs(gl, (cs.stage, id, cs.info), BindingPoint::FIRST);
    prog.bind_resource_locs(gl, host.features, cs);
    Ok(prog)
}

/// What a program selection did, for the draw that asked and the tally that prices it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Selected {
    /// The sub-context's program is not the one it had.
    pub changed: bool,
    /// A variant was translated or compiled, or a program linked: the selection paid the driver,
    /// and its cost is a build's, not a reselect's.
    pub built: bool,
}

impl Context {
    /// `vrend_select_program`, the program half: the variants selected and compiled by
    /// [`Context::select_program`], then the program that links them found or made.
    pub(super) fn select_linked_program(
        &mut self,
        host: &mut Host<'_>,
        cmd: Cmd,
        vertices_per_patch: u32,
    ) -> Result<Selected, Fault> {
        let mut built = self.select_program(host, cmd, vertices_per_patch)?;
        let sub = self.sub();
        let dual_src = sub.blend.as_ref().is_some_and(|b| blend_is_dual(b, 0));
        let linked = match sub.linked_stages() {
            Ok(l) => l,
            Err(_) => return Err(Fault::Shader { cmd, what: "a stage with no compiled variant" }),
        };
        let fs = linked
            .iter()
            .find(|l| l.stage == ShaderStage::Fragment)
            .expect("select_program required a fragment stage");
        let dual_src = dual_src && fs.info.num_outputs > 1;
        let mut stages = [None; 5];
        for l in &linked {
            stages[l.stage.index()] = Some(l.variant.id);
        }
        let want = Linkage::Graphics { stages, dual_src };
        let same = sub.program().is_some_and(|p| p.linkage == want);
        if same {
            // The selection is settled either way; a flag left standing here would run the
            // nine key passes again on every draw of this program.
            self.sub_mut().shader_dirty = false;
            return Ok(Selected { changed: false, built });
        }
        let found = sub
            .programs
            .iter()
            .position(|p| p.linkage == want)
            .map(|at| ProgramSlot { at, serial: sub.programs[at].serial });
        let slot = match found {
            Some(s) => {
                // A pipeline found again re-numbers the blocks of its shared stages when another
                // pipeline numbered them since.
                let stale = stale_block_bindings(sub.program_at(s), &linked).map(|walk| {
                    walk.into_iter().map(|(st, p, i)| (st, p, i.clone())).collect::<Vec<_>>()
                });
                if let Some(walk) = stale {
                    let walk: Vec<StageBlocks<'_>> =
                        walk.iter().map(|(st, p, i)| (*st, *p, i)).collect();
                    let prog = self.sub_mut().program_at_mut(s);
                    prog.object.use_in(host.gl, host.current.program());
                    prog.rebind_ubo_and_sampler_locs(host.gl, &walk);
                }
                s
            }
            None => {
                built = true;
                let serial = sub.mint_program_serial();
                let prog = add_shader_program(host, cmd, serial, &linked, dual_src)?;
                let sub = self.sub_mut();
                sub.programs.push(prog);
                ProgramSlot { at: sub.programs.len() - 1, serial }
            }
        };
        let sub = self.sub_mut();
        let changed = sub.prog.map(|s| s.serial) != Some(slot.serial);
        sub.prog = Some(slot);
        if changed {
            // Every constant buffer and view is re-bound for a new program.
            for stage in [ShaderStage::Vertex, ShaderStage::Fragment] {
                sub.ubos_dirty[stage.index()] = Dirty::all();
                sub.units[stage.index()].mark_all();
            }
        }
        sub.shader_dirty = false;
        Ok(Selected { changed, built })
    }

    /// `vrend_patch_blend_state` and `vrend_hw_emit_blend`: the blend state as bound, patched
    /// for targets without alpha and for alpha emulated in red, told to GL.
    pub(super) fn patch_blend_state(&mut self, host: &mut Host<'_>) {
        let gl = host.gl;
        let features = host.features;
        let api = features.api();
        let sub = self.sub_mut();
        if sub.cbufs.iter().all(Option::is_none) {
            sub.blend_dirty = false;
            return;
        }
        let state = sub.blend.unwrap_or(ZERO_BLEND);
        let mut targets = [None; MAX_COLOR_BUFS];
        for (t, s) in targets.iter_mut().zip(&sub.cbufs) {
            *t = s.as_ref().map(|s| s.format);
        }
        let (new_state, swizzle_blend_color) = patch_blend(api, &state, &targets);

        // vrend_hw_emit_blend
        let mut logicop_changed = false;
        if new_state.logicop_enable != sub.hw_blend.logicop_enable {
            sub.hw_blend.logicop_enable = new_state.logicop_enable;
            logicop_changed = true;
        }
        if new_state.logicop_enable && new_state.logicop_func != sub.hw_blend.logicop_func {
            sub.hw_blend.logicop_func = new_state.logicop_func;
            logicop_changed = true;
        }
        if logicop_changed {
            if api.is_gles() {
                // GLES has no logic op: the shader does it when it can.
                if select::can_emulate_logicop(features, new_state.logicop_func) {
                    sub.shader_dirty = true;
                } else {
                    host.todo.note("a logic op the shader cannot emulate");
                }
            } else if new_state.logicop_enable {
                gl.enable(GL_COLOR_LOGIC_OP);
                gl.logic_op(logic_op(new_state.logicop_func));
            } else {
                gl.disable(GL_COLOR_LOGIC_OP);
            }
        }
        let mask = |m: u8| [m & 1 != 0, m & 2 != 0, m & 4 != 0, m & 8 != 0];
        if new_state.independent_blend_enable
            && features.has(Feature::indep_blend)
            && features.has(Feature::indep_blend_func)
        {
            for i in 0..MAX_COLOR_BUFS {
                let rt = new_state.rt[i];
                if let Some(eq) = rt.equation {
                    if blend_is_dual(&state, i) && !features.has(Feature::dual_src_blend) {
                        eprintln!(
                            "[virglrs] vrend: dual src blend requested but not supported for rt {i}"
                        );
                        continue;
                    }
                    gl.blend_func_separate_i(
                        i as GLuint,
                        blend_factor(eq.rgb.src),
                        blend_factor(eq.rgb.dst),
                        blend_factor(eq.alpha.src),
                        blend_factor(eq.alpha.dst),
                    );
                    gl.blend_equation_separate_i(
                        i as GLuint,
                        blend_func(eq.rgb.func),
                        blend_func(eq.alpha.func),
                    );
                    gl.set_enabled_i(GL_BLEND, i as GLuint, true);
                } else {
                    gl.set_enabled_i(GL_BLEND, i as GLuint, false);
                }
                if rt.colormask != sub.hw_blend.colormask[i] {
                    sub.hw_blend.colormask[i] = rt.colormask;
                    gl.color_mask_i(i as GLuint, mask(rt.colormask));
                }
            }
        } else {
            let rt = new_state.rt[0];
            if let Some(eq) = rt.equation {
                if blend_is_dual(&state, 0) && !features.has(Feature::dual_src_blend) {
                    eprintln!(
                        "[virglrs] vrend: dual src blend requested but not supported for rt 0"
                    );
                }
                gl.blend_func_separate(
                    blend_factor(eq.rgb.src),
                    blend_factor(eq.rgb.dst),
                    blend_factor(eq.alpha.src),
                    blend_factor(eq.alpha.dst),
                );
                gl.blend_equation_separate(blend_func(eq.rgb.func), blend_func(eq.alpha.func));
                gl.enable(GL_BLEND);
            } else {
                gl.disable(GL_BLEND);
            }
            if rt.colormask != sub.hw_blend.colormask[0]
                || (sub.hw_blend.independent && !new_state.independent_blend_enable)
            {
                gl.color_mask(mask(rt.colormask));
                sub.hw_blend.colormask = [rt.colormask; MAX_COLOR_BUFS];
            }
        }
        sub.hw_blend.independent = new_state.independent_blend_enable;
        if features.has(Feature::multisample) {
            gl.set_enabled(GL_SAMPLE_ALPHA_TO_COVERAGE, new_state.alpha_to_coverage);
            // GLES has no alpha-to-one.
            if !api.is_gles() {
                gl.set_enabled(GL_SAMPLE_ALPHA_TO_ONE, new_state.alpha_to_one);
            }
        }
        gl.set_enabled(GL_DITHER, new_state.dither);

        // A constant factor on an emulated-alpha target reads the constant's alpha in red.
        let color = sub.blend_color;
        gl.blend_color(if swizzle_blend_color { [color[3], 0.0, 0.0, 0.0] } else { color });
        sub.blend_dirty = false;
    }

    /// `vrend_draw_bind_ubo_shader`.
    fn draw_bind_ubo(
        sub: &mut SubContext,
        host: &mut Host<'_>,
        at: ProgramSlot,
        stage: ShaderStage,
        mut next_ubo_id: BindingPoint,
    ) -> BindingPoint {
        let gl = host.gl;
        let s = stage.index();
        let mut mask = sub.program_at(at).ubo_used_mask[s];
        let mut dirty = sub.ubos_dirty[s];
        // Nothing dirty is nothing to rebind, whatever is bound. The walk below only narrows this
        // answer, so reaching it first spends a BTree walk per stage per draw to be told what the
        // mask already said.
        if dirty.is_empty() {
            return next_ubo_id.plus(mask.count_ones());
        }
        // The decoder refuses an index past the mask, so every key is a slot it holds.
        let mut used = Dirty::none();
        for slot in sub.ubos[s].keys() {
            used.mark(*slot);
        }
        let update = dirty.intersect(used);
        if update.is_empty() {
            return next_ubo_id.plus(mask.count_ones());
        }
        while mask != 0 {
            let i = mask.trailing_zeros();
            mask &= mask - 1;
            if update.contains(i)
                && let Some(cb) = sub.ubos[s].get(&i)
                && let Some(res) = host.bound_resource(cb.resource)
                && let Storage::Buffer { name, .. } = res.storage
            {
                gl.bind_buffer_range(
                    GL_UNIFORM_BUFFER,
                    next_ubo_id,
                    name,
                    cb.offset as usize,
                    cb.length as usize,
                );
                dirty.unmark(i);
            }
            next_ubo_id = next_ubo_id.next();
        }
        sub.ubos_dirty[s] = dirty;
        next_ubo_id
    }

    /// How many `vec4` constants may be bridged out of a bound buffer.
    ///
    /// A shader shaped this way holds a colour-space matrix or similar -- a handful of vectors,
    /// never hundreds -- so the bound keeps the staging copy small and bounds what one draw does.
    /// A shader wanting more than this is not the shape the bridge exists for, and keeps the
    /// block it was bound as.
    const MAX_BRIDGED_CONSTS: usize = 64;

    /// `vrend_draw_bind_const_shader`: the inline constants, as one `uvec4` array uniform.
    fn draw_bind_const(
        sub: &mut SubContext,
        host: &mut Host<'_>,
        at: ProgramSlot,
        stage: ShaderStage,
        new_program: bool,
    ) {
        let gl = host.gl;
        let s = stage.index();
        let prog = sub.program_at(at);
        let num_consts = prog.num_consts[s];
        let const_location = prog.const_location[s];
        if let Some(loc) = const_location
            && !sub.consts[s].is_empty()
            && sub.shaders[s].is_some()
            && (sub.const_dirty[s] || new_program)
        {
            let n = (num_consts * 4).min(sub.consts[s].len());
            gl.uniform_4uiv(loc, &sub.consts[s][..n]);
            sub.const_dirty[s] = false;
        } else if sub.consts[s].is_empty()
            && let Some(loc) = prog.const_location[s]
            && sub.shaders[s].is_some()
            && (1..=Context::MAX_BRIDGED_CONSTS).contains(&num_consts)
            && let Some(&Ubo { resource, offset, length }) = sub.ubos[s].get(&0)
        {
            // Constant buffer 0 delivered as a resource, feeding a shader that reads plain
            // uniforms.
            //
            // The guest has two unrelated ways to deliver constants and they land in different
            // places: SET_CONSTANT_BUFFER carries them inline, uploaded just above, while
            // SET_UNIFORM_BUFFER names a buffer resource bound as a GL uniform block. Which of
            // the two a shader can read is settled far away, by its TGSI -- a one-dimensional
            // `DCL CONST[0..n]` becomes a plain `uniform uvec4 const0[]` array and never a block
            // -- so a guest that declares them that way and then binds buffer 0 as a resource has
            // a shader whose constants are never written. Nothing rejects the pairing and it
            // reads as zeroes, which is why mesa's vl_compositor, exactly that shape, multiplied
            // every texel by an all-zero colour-space matrix and rendered black.
            //
            // Read from the guest's own pages, and never by mapping the GL buffer. A map on the
            // draw path is a synchronisation point, and a render thread parked in one does not
            // answer the quiesce a suspend waits for -- which shows up as a hung suspend long
            // after a frame that rendered perfectly. The pages are the right source anyway: these
            // constants are CPU-produced and arrive by transfer, so the backing already holds
            // them, and reading it touches no GL state at all.
            //
            // A colour-space matrix that arrives here looking wrong is the guest's, not ours.
            // mesa's graphics compositor uploads a matrix nothing writes after init, so every
            // VA post-processing conversion converts with that seed rather than what the VA
            // frontend computed -- invisible for a decode into NV12 downloaded as BGRA, which
            // is what the seed happens to be, and plainly wrong for an RGB scale, which gets a
            // YUV -> RGB conversion applied to RGB texels. Upstream since mesa 5bc0df5aa; only
            // a driver setting `prefer_compute_for_multimedia` escapes it, and virgl does not.
            // Our guest mesa carries the fix ("vl/compositor: upload the matrix the frontend
            // set, not the init default"); a stock guest still converts with the seed, so
            // measure a colour claim on the tier the guest is actually running.
            let mut bytes = vec![0u8; num_consts * 4 * size_of::<u32>()];
            let take = match length as usize {
                0 => bytes.len(),
                given => bytes.len().min(given),
            };
            let pages = host.guest.pages(host.ctx, resource);
            if pages.is_some_and(|iov| iov.copy_out(u64::from(offset), &mut bytes[..take])) {
                let words: Vec<u32> = bytes
                    .as_chunks::<{ size_of::<u32>() }>()
                    .0
                    .iter()
                    .copied()
                    .map(u32::from_le_bytes)
                    .collect();
                gl.uniform_4uiv(loc, &words);
            }
        }
    }

    /// `vrend_draw_bind_samplers_shader`.
    fn draw_bind_samplers(
        sub: &mut SubContext,
        host: &mut Host<'_>,
        at: ProgramSlot,
        stage: ShaderStage,
        mut next_sampler_id: TextureUnit,
    ) -> TextureUnit {
        let gl = host.gl;
        let max_units = host.limits.max_texture_units;
        let s = stage.index();
        // Borrowed, never cloned. The loop below only reads this sub-context -- its views, its
        // objects, its samplers -- and mutates nothing, so the program's location tables can stay
        // borrowed for the whole of it. Copying them out was three `Vec` allocations per stage per
        // draw, which measured 476 of 5883 samples (8.1%) of the classic command path under a live
        // desktop: the single largest avoidable cost in it, and all of it paid to satisfy a borrow
        // that was never in conflict. The C reaches its equivalents through a pointer
        // (`vrend_renderer.c:5795,5816`) for the same reason.
        let prog = sub.program_at(at);
        let dirty = sub.units[s].dirty();
        let mut mask = prog.samplers_used_mask[s];
        let shadow_mask = prog.shadow_samp_mask[s];
        let sampler_locs = &prog.sampler_locs[s];
        let mask_locs = &prog.shadow_samp_mask_locs[s];
        let add_locs = &prog.shadow_samp_add_locs[s];
        let mut sampler_index = 0usize;
        // On the stack, not the heap: the loop runs once per set bit of a `u32` mask, so there can
        // never be more than 32 of these, and the array's own bound is what enforces it -- an
        // index past it is a violated host invariant and should abort, not grow a buffer.
        // `levels_used` stands in for what the old `Vec`'s length meant, so that a sampler slot the
        // loop skipped still leaves the level already recorded for it alone rather than zeroing it.
        let mut levels_out = [0 as GLint; 32];
        let mut levels_used = 0usize;
        while mask != 0 {
            let i = mask.trailing_zeros();
            mask &= mask - 1;
            let view = sub.units[s].view(i).and_then(|h| match sub.objects.get(&h) {
                Some(Object::SamplerView(v)) => Some(v),
                _ => None,
            });
            // Nothing dirties a texture whose guest pages the guest rewrote behind our back --
            // there is no transfer, no flush, no command at all -- so this is outside the dirty
            // check, and outside it for a texture bound in an earlier batch and left alone since.
            if let Some(view) = view {
                host.refresh_guest_pixels(view.resource);
            }
            if dirty.contains(i)
                && let Some(view) = view
            {
                gl.active_texture(next_sampler_id);
                if let Some(loc) = sampler_locs[sampler_index] {
                    gl.uniform_1i(loc, next_sampler_id.uniform_value());
                }
                let res = host.bound_resource(view.resource);
                if shadow_mask & (1 << i) != 0 {
                    // A depth texture read through a shadow sampler compares, and the
                    // luminance-style swizzles must not apply: the texture goes back to
                    // identity and the view's swizzle becomes a mask and an add.
                    if let Some(res) = res
                        && let Storage::Texture(t) = &res.storage
                    {
                        // The object the unit will bind, which is the view's own whenever it
                        // has one. Resetting the base texture instead would leave the swizzle
                        // standing on what is actually sampled, and clear it on a texture no
                        // one asked about.
                        let (name, target) = match view.view {
                            Some(name) => (name, view.target),
                            None => (t.name, t.target),
                        };
                        gl.bind_texture(target, Some(name));
                        for (c, sw) in [GL_RED, GL_GREEN, GL_BLUE, GL_ALPHA].iter().enumerate() {
                            gl.tex_parameter_i(
                                target,
                                GL_TEXTURE_SWIZZLE_R + c as GLenum,
                                *sw as GLint,
                            );
                        }
                    }
                    let one_or_zero = |g: GLint| g == GL_ZERO as GLint || g == GL_ONE as GLint;
                    let m = view.gl_swizzle.map(|g| if one_or_zero(g) { 0.0 } else { 1.0 });
                    let a = view.gl_swizzle.map(|g| if g == GL_ONE as GLint { 1.0 } else { 0.0 });
                    if let Some(loc) = mask_locs[sampler_index] {
                        gl.uniform_4f(loc, m);
                    }
                    if let Some(loc) = add_locs[sampler_index] {
                        gl.uniform_4f(loc, a);
                    }
                }
                if let Some(res) = res {
                    let (id, target, is_buffer, multisampled) = match &res.storage {
                        Storage::Buffer { tbo, .. } => (*tbo, GL_TEXTURE_BUFFER, true, false),
                        Storage::Texture(t) => (
                            Some(view.view.unwrap_or(t.name)),
                            view.target,
                            false,
                            res.args.nr_samples > 1,
                        ),
                        _ => (None, view.target, false, false),
                    };
                    if let Some(id) = id {
                        gl.bind_texture(target, Some(id));
                    }
                    // vrend_apply_sampler_state, the sampler-objects leg: a buffer or
                    // multisampled texture takes no sampler.
                    if !is_buffer && !multisampled {
                        let sampler =
                            sub.units[s].sampler(i).and_then(|h| match sub.objects.get(&h) {
                                Some(Object::SamplerState(st)) => Some(st),
                                _ => None,
                            });
                        let alpha_in_red = crate::vrend::formats::is_emulated_alpha(
                            host.features.api(),
                            view.format,
                        );
                        let id = sampler.and_then(|st| match (st.ids, st.alpha_in_red) {
                            (_, Some(a8)) if alpha_in_red => Some(a8),
                            (Some(ids), _) if view.skip_srgb_decode => Some(ids[0]),
                            (Some(ids), _) => Some(ids[1]),
                            _ => None,
                        });
                        if let Some(id) = id {
                            gl.bind_sampler(next_sampler_id, Some(id));
                        }
                    }
                    let levels = match view.span {
                        Span::Levels { first, last } => last.wrapping_sub(first).wrapping_add(1),
                        Span::Elements(_) => 0,
                    };
                    let levels = if levels != 0 { levels } else { res.args.last_level + 1 };
                    levels_used = levels_used.max(sampler_index + 1);
                    levels_out[sampler_index] = levels as GLint;
                }
            }
            sampler_index += 1;
            next_sampler_id = next_sampler_id.next();
        }
        let tl = &mut sub.texture_levels[s];
        if tl.len() < sampler_index {
            tl.resize(sampler_index, 0);
        }
        tl.truncate(sampler_index);
        for (i, l) in levels_out[..levels_used].iter().enumerate() {
            if i < tl.len() {
                tl[i] = *l;
            }
        }
        sub.units[s].bound();
        // A later glBindTexture for another reason must not disturb the units just bound.
        gl.active_texture(TextureUnit::at(max_units.saturating_sub(1)));
        next_sampler_id
    }

    /// `vrend_draw_bind_ssbo_shader`.
    fn draw_bind_ssbo(sub: &SubContext, host: &mut Host<'_>, at: ProgramSlot, stage: ShaderStage) {
        let gl = host.gl;
        if !host.has(Feature::ssbo) {
            return;
        }
        let s = stage.index();
        let prog = sub.program_at(at);
        let offset = prog.ssbo_binding_offset[s];
        let prog_mask = prog.ssbo_used_mask[s];
        for (&i, ssbo) in &sub.ssbos[s] {
            if i as usize >= MAX_SHADER_BUFFERS || prog_mask & (1 << i) == 0 {
                continue;
            }
            if let Some(res) = host.bound_resource(ssbo.resource)
                && let Storage::Buffer { name, .. } = res.storage
            {
                gl.bind_buffer_range(
                    GL_SHADER_STORAGE_BUFFER,
                    BindingPoint::at(i + offset),
                    name,
                    ssbo.offset as usize,
                    ssbo.length as usize,
                );
            }
        }
    }

    /// `vrend_draw_bind_abo_shader`.
    fn draw_bind_abo(sub: &SubContext, host: &mut Host<'_>) {
        let gl = host.gl;
        if !host.has(Feature::atomic_counters) {
            return;
        }
        for (&i, abo) in &sub.abos {
            if let Some(res) = host.bound_resource(abo.resource)
                && let Storage::Buffer { name, .. } = res.storage
            {
                gl.bind_buffer_range(
                    GL_ATOMIC_COUNTER_BUFFER,
                    BindingPoint::at(i),
                    name,
                    abo.offset as usize,
                    abo.length as usize,
                );
            }
        }
    }

    /// `vrend_draw_bind_images_shader`.
    fn draw_bind_images(
        sub: &SubContext,
        host: &mut Host<'_>,
        at: ProgramSlot,
        stage: ShaderStage,
    ) {
        let gl = host.gl;
        let features = host.features;
        let formats = host.formats;
        let s = stage.index();
        let prog = sub.program_at(at);
        if sub.images[s].is_empty() || prog.img_locs[s].is_empty() || !features.has(Feature::images)
        {
            return;
        }
        let prog_mask = prog.images_used_mask[s];
        let offset = prog.image_binding_offset[s];
        for (&i, iview) in &sub.images[s] {
            if i as usize >= MAX_SHADER_IMAGES || prog_mask & (1 << i) == 0 {
                continue;
            }
            let image_unit = ImageUnit::at(i + offset);
            let Some(loc) = prog.img_locs[s].get(i as usize).copied().flatten() else {
                continue;
            };
            // A desktop shader declares its images by `location`, not `binding`, as the C's
            // translator writes them, so the unit is the uniform's to be told.
            if !features.api().is_gles() {
                gl.uniform_1i(loc, (i + offset) as GLint);
            }
            let access = match iview.access {
                ImageAccess::Read => GL_READ_ONLY,
                ImageAccess::Write => GL_WRITE_ONLY,
                ImageAccess::ReadWrite => GL_READ_WRITE,
            };
            let bound = formats.get(iview.format).and_then(|entry| {
                let res = host.bound_resource_mut(iview.resource)?;
                image_binding(gl, features, formats, res, iview.format, iview.span)
                    .map(|b| (b, entry.gl.internalformat))
            });
            // A unit the program reads that cannot be bound is emptied, not left holding what
            // an earlier draw bound there.
            let Some((b, internalformat)) = bound else {
                if matches!(iview.span, ImageSpan::Layers { first, last, .. } if first != last) {
                    host.todo.note("image views of a layer subset that cannot be viewed");
                }
                gl.bind_image_texture(image_unit, None, 0, false, 0, GL_READ_ONLY, GL_R32UI);
                continue;
            };
            gl.bind_image_texture(
                image_unit,
                Some(b.texture),
                b.level,
                b.layered,
                b.layer,
                access,
                internalformat,
            );
        }
    }

    /// `vrend_fill_sysval_uniform_block`.
    fn fill_sysval_uniform_block(&mut self, host: &mut Host<'_>, at: ProgramSlot) {
        let gl = host.gl;
        let sub = self.sub_mut();
        let generation = sub.sysval.generation();
        let prog = sub.program_at(at);
        if prog.virgl_block_bind.is_none() || prog.sysval_uploaded == Some(generation) {
            return;
        }
        let buf = prog.sysval_buffer.expect("a bound block has its buffer");
        let mut bytes = [0; Sysval::SIZE];
        sub.sysval.write_to(&mut bytes);
        gl.bind_buffer(GL_UNIFORM_BUFFER, Some(buf));
        gl.buffer_sub_data(GL_UNIFORM_BUFFER, 0, &bytes);
        gl.bind_buffer(GL_UNIFORM_BUFFER, None);
        sub.program_at_mut(at).sysval_uploaded = Some(generation);
    }

    /// `vrend_draw_bind_objects`.
    fn draw_bind_objects(&mut self, host: &mut Host<'_>, at: ProgramSlot, new_program: bool) {
        let gl = host.gl;
        // The pass's other question -- which sub-context -- is answered once here too. Each binder
        // used to ask both again, per stage.
        let sub = self.sub_mut();
        let last = sub.program_at(at).linkage.last_stage();
        let mut next_ubo_id = BindingPoint::FIRST;
        let mut next_sampler_id = TextureUnit::FIRST;
        for stage in C_STAGE_ORDER {
            if stage.index() > last.index() {
                continue;
            }
            // A pipeline's uniforms are its stages' programs': this stage's is the one written.
            if let Some(program) = sub.stage_program(at, stage) {
                sub.program_at(at).object.activate(gl, program);
            }
            next_ubo_id = Self::draw_bind_ubo(sub, host, at, stage, next_ubo_id);
            Self::draw_bind_const(sub, host, at, stage, new_program);
            next_sampler_id = Self::draw_bind_samplers(sub, host, at, stage, next_sampler_id);
            Self::draw_bind_images(sub, host, at, stage);
            Self::draw_bind_ssbo(sub, host, at, stage);
            if let Some(loc) = sub.program_at(at).tex_levels_uniform_id[stage.index()] {
                gl.uniform_1iv(loc, &sub.texture_levels[stage.index()]);
            }
        }
        let prog = sub.program_at(at);
        if let Some(bind) = prog.virgl_block_bind
            && let Some(buf) = prog.sysval_buffer
        {
            gl.bind_buffer_range(GL_UNIFORM_BUFFER, bind, buf, 0, Sysval::SIZE);
        }
        Self::draw_bind_abo(sub, host);
        let fs =
            sub.stage_program(at, ShaderStage::Fragment).expect("a program has a fragment stage");
        sub.program_at(at).object.activate(gl, fs);
    }

    /// `vrend_draw_bind_vertex_binding`: the bound layout's VAO, and the vertex buffers when
    /// they changed.
    fn draw_bind_vertex_binding(&mut self, host: &mut Host<'_>) {
        let gl = host.gl;
        let sub = self.sub_mut();
        let vao = sub.ve.and_then(|h| match sub.objects.get(&h) {
            Some(Object::VertexElements(v)) => v.vao,
            _ => None,
        });
        let Some(vao) = vao else {
            gl.bind_vertex_array(Some(sub.vao));
            return;
        };
        gl.bind_vertex_array(Some(vao));
        if !sub.vbo_dirty {
            return;
        }
        for (i, vbo) in sub.vbos.iter().enumerate() {
            let name = vbo.resource.and_then(|r| host.bound_resource(r)).and_then(|res| match res
                .storage
            {
                Storage::Buffer { name, .. } => Some(name),
                _ => None,
            });
            match name {
                Some(name) => {
                    gl.bind_vertex_buffer(i as GLuint, Some(name), vbo.offset, vbo.stride)
                }
                None => gl.bind_vertex_buffer(i as GLuint, None, 0, 0),
            }
        }
        for i in stale_vbo_slots(sub.vbos.len(), sub.hw_num_vbos) {
            gl.bind_vertex_buffer(i as GLuint, None, 0, 0);
        }
        sub.hw_num_vbos = sub.vbos.len();
        sub.vbo_dirty = false;
    }

    /// `vrend_draw_vbo`.
    pub(super) fn draw_vbo(&mut self, host: &mut Host<'_>, draw: Draw) -> Result<(), Fault> {
        let cmd = Cmd::DrawVbo;
        let gl = host.gl;
        let features = host.features;
        let need = |feature: Feature| {
            if features.has(feature) { Ok(()) } else { Err(Fault::NoFeature { cmd, feature }) }
        };
        if draw.instance_count > 0 {
            need(Feature::draw_instance)?;
        }
        if draw.start_instance > 0 {
            need(Feature::base_instance)?;
        }
        let indirect = draw.indirect;
        if let Some(ind) = indirect {
            if ind.draw_count > 1 {
                need(Feature::multi_draw_indirect)?;
            }
            need(Feature::indirect_draw)?;
            if ind.draw_count_resource.is_some() {
                // `feat_indirect_params` is desktop-only.
                return Err(Fault::Unimplemented { cmd, what: "an indirect draw count" });
            }
        }
        // GL takes every count as a signed 32-bit value. Past that, GL would raise an error the
        // batch turns into a fault at its end; saying so here names the draw that asked.
        let sized = |v: u32, what: &'static str| -> Result<GLsizei, Fault> {
            GLsizei::try_from(v).map_err(|_| Fault::OutOfRange { cmd, what })
        };
        let start = sized(draw.start, "a draw start")?;
        let count = sized(draw.count, "a draw count")?;
        let instances = sized(draw.instance_count, "an instance count")?;
        let indirect_counts = match indirect {
            Some(ind) => Some((
                sized(ind.draw_count, "an indirect draw count")?,
                sized(ind.stride, "an indirect stride")?,
            )),
            None => None,
        };
        let vertices_per_patch = match draw.tess {
            Some(t) => sized(t.vertices_per_patch, "a patch size")?,
            None => 0,
        };
        let indirect_buffer = match indirect {
            Some(ind) => match host.resource(cmd, ind.resource)?.storage {
                Storage::Buffer { name, .. } => Some(name),
                _ => return Err(Fault::IllegalResource { cmd, handle: ind.resource }),
            },
            None => None,
        };

        self.flush_lazy_state(host);
        if self.sub().blend_dirty {
            self.patch_blend_state(host);
        }

        let sub = self.sub_mut();
        // An injected control stage declares the draw's patch size, so another size is another
        // shader. Asked of the one in use rather than of a remembered size, which could disagree.
        if sub.injects_tcs()
            && sub.passthrough.as_ref().is_some_and(|p| !p.made_for_patch(vertices_per_patch))
        {
            sub.shader_dirty = true;
        }
        if sub.prim_mode != draw.mode {
            // Only a switch in or out of points changes the shader variants.
            if sub.prim_mode == PrimType::Points || draw.mode == PrimType::Points {
                sub.shader_dirty = true;
            }
            sub.prim_mode = draw.mode;
        }

        let mut new_program = false;
        let mut selected = None;
        let mut built = false;
        let sub = self.sub();
        // The C also reselects on every draw while the framebuffer needs a red-blue swizzle or
        // a manual sRGB encode. Both are inputs to the fragment key, but both change only in
        // `set_framebuffer_state`, which marks the shader dirty -- so the C's test is nine key
        // passes per draw on any BGRA target, which is every desktop draw here, for a program
        // that comes back the same. Every other key input marks dirt where it changes, and
        // the flatshade and sampler fixtures are what say so.
        //
        // A dispatch leaves its compute program bound, and nothing about that is a dirty flag:
        // the program itself says it is not one a draw can run, so the draw selects its own.
        let compute_bound = sub.program().is_some_and(|p| matches!(p.linkage, Linkage::Compute(_)));
        if sub.shader_dirty || sub.vbo_dirty || compute_bound {
            selected = host.tally.mark();
            let s = self.select_linked_program(host, cmd, vertices_per_patch as u32)?;
            new_program = s.changed;
            built = s.built;
        }
        host.tally.draw(selected, built);
        // The C drops the draw with a warning; a draw with nothing to run it is a fault here.
        // Resolved once for the whole draw: `select_linked_program` above is the last thing that
        // can move the program list, and everything below is handed the slot rather than asking
        // again.
        let sub = self.sub();
        let Some(at) = sub.program_slot() else {
            return Err(Fault::Shader { cmd, what: "a draw with no program" });
        };
        let prog = sub.program_at(at);
        let object = prog.object;
        let reads_drawid = prog.reads_drawid;
        let fs_blend_advanced = prog.fs_blend_equation_advanced;
        object.use_in(gl, host.current.program());

        if features.has(Feature::draw_parameters) && reads_drawid {
            let drawid = draw.tess.map_or(0, |t| t.drawid) as i32;
            // Read through `Deref` and write only on a change: a write bumps the generation, and
            // bumping it every draw would re-upload the whole block for every drawid-reading
            // program, which is the upload the generation exists to skip.
            let sub = self.sub_mut();
            if sub.sysval.drawid_base != drawid {
                sub.sysval.drawid_base = drawid;
            }
        }

        self.draw_bind_objects(host, at, new_program);
        self.fill_sysval_uniform_block(host, at);
        self.draw_bind_vertex_binding(host);

        let mut index_type = GL_UNSIGNED_INT;
        let mut ib_offset = 0;
        if draw.indexed {
            // The C skips an indexed draw with no index buffer, or one that reads past it,
            // with a warning and success. Both are the guest's claim about its own buffer,
            // and a claim the buffer cannot meet is refused.
            let Some(ib) = self.sub().ib else {
                return Err(Fault::OutOfRange {
                    cmd,
                    what: "an indexed draw with no index buffer",
                });
            };
            let res = host.resource(cmd, ib.resource)?;
            if indirect.is_none() {
                let expected = ib.index_type.bytes() as u64 * draw.count as u64 + ib.offset as u64;
                if expected > res.args.width as u64 {
                    return Err(Fault::OutOfRange { cmd, what: "a draw past its index buffer" });
                }
            }
            let Storage::Buffer { name, .. } = res.storage else {
                return Err(Fault::IllegalResource { cmd, handle: ib.resource });
            };
            gl.bind_buffer(GL_ELEMENT_ARRAY_BUFFER, Some(name));
            index_type = match ib.index_type {
                IndexType::U8 => GL_UNSIGNED_BYTE,
                IndexType::U16 => GL_UNSIGNED_SHORT,
                IndexType::U32 => GL_UNSIGNED_INT,
            };
            ib_offset = ib.offset;
        } else {
            gl.bind_buffer(GL_ELEMENT_ARRAY_BUFFER, None);
        }

        let sub = self.sub_mut();
        if let Some(i) = sub.current_so {
            match sub.streamouts[i].xfb {
                Xfb::NeedBegin => {
                    let mode = if let Some(gs) = sub.bound_program(ShaderStage::Geometry) {
                        gs_xfb_mode(gs.info.gs_out_prim)
                    } else if let Some(tes) = sub.bound_program(ShaderStage::TessEval) {
                        tess_xfb_mode(tes.info.tes_prim, tes.info.tes_point_mode)
                    } else {
                        xfb_mode(draw.mode)
                    };
                    gl.begin_transform_feedback(mode);
                    sub.streamouts[i].xfb = Xfb::Started;
                }
                Xfb::Paused => {
                    gl.resume_transform_feedback();
                    sub.streamouts[i].xfb = Xfb::Started;
                }
                Xfb::Started => {}
            }
        }

        if draw.primitive_restart {
            // GLES restarts at the index type's maximum alone; desktop GL takes the guest's.
            if features.api().is_gles() {
                gl.enable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
            } else {
                gl.enable(GL_PRIMITIVE_RESTART);
                gl.primitive_restart_index(draw.restart_index);
            }
        }
        if features.has(Feature::indirect_draw) {
            gl.bind_buffer(GL_DRAW_INDIRECT_BUFFER, indirect_buffer);
        }
        if vertices_per_patch > 0 && features.has(Feature::tessellation) {
            gl.patch_parameter_i(GL_PATCH_VERTICES, vertices_per_patch);
        }

        // A host with advanced blend equations but no framebuffer fetch takes the equation the
        // guest sent through the blend state. The wire carries it in the alpha factors of a
        // target whose blending is off, which the decoder does not keep; the shape is counted
        // until it does.
        if fs_blend_advanced != 0
            && !features.has(Feature::framebuffer_fetch)
            && features.has(Feature::blend_equation_advanced)
        {
            host.todo.note("advanced blend equations");
        }

        let mode = prim_mode(draw.mode);
        if !draw.indexed {
            // The C draws `cso` vertices from zero when the wire names a stream-out object to
            // count from -- the handle's number, as it is: the count is never read from the
            // object. A handle is not a count, and no corpus has asked; refused until one does.
            if draw.count_from_so.is_some() {
                host.todo.note("a draw counted from a stream-out object");
                return Err(Fault::Unimplemented {
                    cmd,
                    what: "a draw counted from a stream-out object",
                });
            }
            if let Some(ind) = indirect {
                let (draw_count, stride) = indirect_counts.expect("counted with the buffer");
                if draw_count > 1 {
                    gl.multi_draw_arrays_indirect(mode, ind.offset, draw_count, stride);
                } else {
                    gl.draw_arrays_indirect(mode, ind.offset);
                }
            } else if draw.instance_count > 0 {
                if draw.start_instance > 0 {
                    gl.draw_arrays_instanced_base_instance(
                        mode,
                        start,
                        count,
                        instances,
                        draw.start_instance,
                    );
                } else {
                    gl.draw_arrays_instanced(mode, start, count, instances);
                }
            } else {
                gl.draw_arrays(mode, start, count);
            }
        } else {
            let ranged = draw.min_index != 0 || draw.max_index != u32::MAX;
            if let Some(ind) = indirect {
                let (draw_count, stride) = indirect_counts.expect("counted with the buffer");
                if draw_count > 1 {
                    gl.multi_draw_elements_indirect(
                        mode, index_type, ind.offset, draw_count, stride,
                    );
                } else {
                    gl.draw_elements_indirect(mode, index_type, ind.offset);
                }
            } else if draw.index_bias != 0 {
                if draw.instance_count > 0 {
                    if draw.start_instance > 0 {
                        gl.draw_elements_instanced_base_vertex_base_instance(
                            mode,
                            count,
                            index_type,
                            ib_offset,
                            instances,
                            draw.index_bias,
                            draw.start_instance,
                        );
                    } else {
                        gl.draw_elements_instanced_base_vertex(
                            mode,
                            count,
                            index_type,
                            ib_offset,
                            instances,
                            draw.index_bias,
                        );
                    }
                } else if ranged {
                    gl.draw_range_elements_base_vertex(
                        mode,
                        draw.min_index,
                        draw.max_index,
                        count,
                        index_type,
                        ib_offset,
                        draw.index_bias,
                    );
                } else {
                    gl.draw_elements_base_vertex(
                        mode,
                        count,
                        index_type,
                        ib_offset,
                        draw.index_bias,
                    );
                }
            } else if draw.instance_count > 0 {
                if draw.start_instance > 0 {
                    gl.draw_elements_instanced_base_instance(
                        mode,
                        count,
                        index_type,
                        ib_offset,
                        instances,
                        draw.start_instance,
                    );
                } else {
                    gl.draw_elements_instanced(mode, count, index_type, ib_offset, instances);
                }
            } else if ranged {
                gl.draw_range_elements(
                    mode,
                    draw.min_index,
                    draw.max_index,
                    count,
                    index_type,
                    ib_offset,
                );
            } else {
                gl.draw_elements(mode, count, index_type, ib_offset);
            }
        }

        if draw.primitive_restart {
            gl.disable(if features.api().is_gles() {
                GL_PRIMITIVE_RESTART_FIXED_INDEX
            } else {
                GL_PRIMITIVE_RESTART
            });
        }
        let sub = self.sub_mut();
        if let Some(i) = sub.current_so
            && features.has(Feature::transform_feedback2)
            && sub.streamouts[i].xfb == Xfb::Started
        {
            gl.pause_transform_feedback();
            sub.streamouts[i].xfb = Xfb::Paused;
        }
        Ok(())
    }
}

impl Context {
    /// `vrend_launch_grid`'s program: the compute variant selected, then the program that links
    /// it found or made. Answers whether the sub-context's program changed.
    ///
    /// Graphics and compute share the one program a sub-context runs, as they do in the C, so a
    /// change here leaves the draw's program behind. The C marks the shader dirty for the next
    /// draw; here the draw sees a compute program bound and selects its own, so there is no flag
    /// for a later change to forget.
    fn select_compute_program(&mut self, host: &mut Host<'_>, cmd: Cmd) -> Result<bool, Fault> {
        self.select_compute(host, cmd)?;
        let sub = self.sub();
        let Some(cs) = sub.linked_stage(ShaderStage::Compute) else {
            return Err(Fault::Shader { cmd, what: "a stage with no compiled variant" });
        };
        let want = Linkage::Compute(cs.variant.id);
        if sub.program().is_some_and(|p| p.linkage == want) {
            return Ok(false);
        }
        let found = sub
            .programs
            .iter()
            .position(|p| p.linkage == want)
            .map(|at| ProgramSlot { at, serial: sub.programs[at].serial });
        let slot = match found {
            Some(s) => s,
            None => {
                let serial = sub.mint_program_serial();
                let prog = add_cs_shader_program(host, cmd, serial, &cs)?;
                let sub = self.sub_mut();
                sub.programs.push(prog);
                ProgramSlot { at: sub.programs.len() - 1, serial }
            }
        };
        let sub = self.sub_mut();
        sub.prog = Some(slot);
        // A program's sampler uniforms are written only for a dirty unit, so a new one starts
        // with every unit and block of its stage to bind.
        let s = ShaderStage::Compute.index();
        sub.ubos_dirty[s] = Dirty::all();
        sub.units[s].mark_all();
        Ok(true)
    }

    /// `vrend_launch_grid`: the compute program, its stage's bindings, and the dispatch -- its
    /// grid from the wire, or from three words of `indirect` at `indirect_offset`.
    ///
    /// The C returns quietly from a dispatch on a host without compute and from one with no
    /// compute shader bound. Neither is a dispatch a guest told the truth about: the caps
    /// advertise compute only where the host has it, and a guest that dispatches nothing has sent
    /// a command it cannot mean. Both are refused here, as a draw with no program is.
    pub(super) fn launch_grid(
        &mut self,
        host: &mut Host<'_>,
        grid: [u32; 3],
        indirect: Option<ResourceHandle>,
        indirect_offset: u32,
    ) -> Result<(), Fault> {
        let cmd = Cmd::LaunchGrid;
        let gl = host.gl;
        if !host.has(Feature::compute_shader) {
            return Err(Fault::NoFeature { cmd, feature: Feature::compute_shader });
        }
        // Three words, aligned, inside the buffer: GL refuses anything else with an error the
        // batch would report at its end, and saying so here names the dispatch that asked.
        let indirect_buffer = match indirect {
            Some(handle) => {
                let res = host.resource(cmd, handle)?;
                let Storage::Buffer { name, .. } = res.storage else {
                    return Err(Fault::IllegalResource { cmd, handle });
                };
                let end = u64::from(indirect_offset) + 3 * size_of::<u32>() as u64;
                if !indirect_offset.is_multiple_of(4) || end > u64::from(res.args.width) {
                    return Err(Fault::OutOfRange { cmd, what: "an indirect dispatch's grid" });
                }
                Some(name)
            }
            None => None,
        };

        let new_program = self.select_compute_program(host, cmd)?;
        let stage = ShaderStage::Compute;
        let sub = self.sub_mut();
        let at = sub.program_slot().expect("a compute program was selected a moment ago");
        sub.program_at(at).object.use_in(gl, host.current.program());
        Self::draw_bind_ubo(sub, host, at, stage, BindingPoint::FIRST);
        Self::draw_bind_const(sub, host, at, stage, new_program);
        Self::draw_bind_samplers(sub, host, at, stage, TextureUnit::FIRST);
        Self::draw_bind_images(sub, host, at, stage);
        Self::draw_bind_ssbo(sub, host, at, stage);
        // Not in the C's dispatch, which leaves a GLES compute shader's `textureQueryLevels`
        // reading whatever the uniform last held. The draw writes it; so does the dispatch.
        if let Some(loc) = sub.program_at(at).tex_levels_uniform_id[stage.index()] {
            gl.uniform_1iv(loc, &sub.texture_levels[stage.index()]);
        }
        Self::draw_bind_abo(sub, host);

        gl.bind_buffer(GL_DISPATCH_INDIRECT_BUFFER, indirect_buffer);
        match indirect_buffer {
            Some(_) => gl.dispatch_compute_indirect(indirect_offset),
            None => gl.dispatch_compute(grid),
        }
        Ok(())
    }
}

/// What a shader image binds: `glBindImageTexture`'s texture, level and layer arguments.
struct ImageBinding {
    texture: TextureName,
    level: GLint,
    layered: bool,
    layer: GLint,
}

/// Which of a texture's layers an image over `first..=last` reaches.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ImageLayers {
    /// Every layer, through the texture itself.
    Whole,
    /// One layer of an array or 3D texture, bound unlayered.
    One(u32),
    /// Some layers but not all, which only a texture view of them can bind.
    Range { first: u32, layers: u32 },
}

/// The C's reading of an image's layer range (`vrend_draw_bind_images_shader`) over a texture of
/// `array_size` layers and `depth` slices. `None` for a range past the texture's last layer.
fn image_layers(first: u32, last: u32, array_size: u32, depth: u32) -> Option<ImageLayers> {
    let total = array_size.max(depth);
    if last >= total {
        return None;
    }
    let layered = !((array_size > 1 || depth > 1) && first == last);
    Some(match (layered, last - first + 1) {
        (false, _) => ImageLayers::One(first),
        (true, layers) if first == 0 && layers == total => ImageLayers::Whole,
        (true, layers) => ImageLayers::Range { first, layers },
    })
}

/// What an image of `format` over `res` binds, or `None` when nothing can be: a span of the
/// other kind of resource than the one it was set over, or a layer range with no view to serve it.
fn image_binding(
    gl: &Gl,
    features: &Features,
    formats: &Table,
    res: &mut Resource,
    format: Format,
    span: ImageSpan,
) -> Option<ImageBinding> {
    let (args, viewable) = (res.args, res.supports_view());
    match (&mut res.storage, span) {
        (Storage::Buffer { name, tbo, .. }, ImageSpan::Bytes { offset, size }) => {
            let tbo_tex = *tbo.get_or_insert_with(|| gl.gen_texture());
            // `set_shader_images` admits a buffer image only with one of these widths.
            let bs = format.describe().map_or(1, |d| d.block_bytes());
            let internal = match bs {
                16 => GL_RGBA32UI,
                8 => GL_RG32UI,
                4 => GL_R32UI,
                2 => GL_R16UI,
                _ => GL_R8UI,
            };
            gl.bind_buffer(GL_TEXTURE_BUFFER, Some(*name));
            gl.bind_texture(GL_TEXTURE_BUFFER, Some(tbo_tex));
            if features.has(Feature::arb_or_gles_ext_texture_buffer) {
                let range = features.has(Feature::texture_buffer_range).then(|| {
                    let bs = bs as usize;
                    (offset as usize, size as usize / bs * bs)
                });
                gl.tex_buffer(internal, *name, range);
            }
            Some(ImageBinding { texture: tbo_tex, level: 0, layered: true, layer: 0 })
        }
        (Storage::Texture(t), ImageSpan::Layers { level, first, last }) => {
            let level = level as GLint;
            match image_layers(first, last, args.array_size, args.depth)? {
                ImageLayers::Whole => {
                    Some(ImageBinding { texture: t.name, level, layered: true, layer: 0 })
                }
                ImageLayers::One(layer) => Some(ImageBinding {
                    texture: t.name,
                    level,
                    layered: false,
                    layer: layer as GLint,
                }),
                ImageLayers::Range { first, layers } => {
                    // The resource's views span every level, where the C's spans only the one it
                    // binds, so `level` names the same level of the view as of the texture.
                    let src =
                        t.immutable.filter(|_| viewable && features.has(Feature::texture_view))?;
                    let internalformat = formats.get(args.format)?.gl.internalformat;
                    let key = ViewKey { format: args.format, first_layer: first, layers };
                    let view = t.view(gl, src, key, internalformat, args.last_level + 1);
                    Some(ImageBinding { texture: view, level, layered: true, layer: 0 })
                }
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An emulated-alpha target blends its alpha in red: the alpha factors become the colour
    /// ones, a destination alpha reads the red that stores it, only red is written and only if
    /// alpha was, and a constant factor reads the constant's alpha. The C's table, and on GLES,
    /// where nothing is emulated, an A8 target is left as bound.
    #[test]
    fn an_emulated_alpha_target_blends_its_alpha_in_red() {
        use BlendFactor::*;
        let a8 = (0..crate::vrend::proto::FORMAT_MAX)
            .filter_map(Format::from_wire)
            .find(|f| f.name() == "A8_UNORM")
            .expect("A8 is a wire format");
        let eq = |src, dst| BlendEq { func: BlendFunc::Add, src, dst };
        let mut state = ZERO_BLEND;
        state.rt[0] = RtBlend {
            equation: Some(RtBlendEq {
                rgb: eq(SrcColor, InvSrcColor),
                alpha: eq(DstAlpha, InvDstAlpha),
            }),
            colormask: PIPE_MASK_A | 0x7,
        };
        let (patched, swizzle) = patch_blend(Api::Gl(46), &state, &[Some(a8)]);
        assert_eq!(
            patched.rt[0],
            RtBlend {
                equation: Some(RtBlendEq { rgb: eq(DstColor, InvDstColor), alpha: eq(Zero, Zero) }),
                colormask: PIPE_MASK_R,
            }
        );
        assert!(!swizzle, "no constant factor");
        state.rt[0].equation = Some(RtBlendEq { rgb: eq(One, Zero), alpha: eq(ConstAlpha, Zero) });
        state.rt[0].colormask = 0x7;
        let (patched, swizzle) = patch_blend(Api::Gl(46), &state, &[Some(a8)]);
        assert_eq!(patched.rt[0].colormask, 0, "alpha masked off writes nothing");
        assert!(swizzle, "a constant factor reads the constant's alpha in red");
        assert_eq!(patch_blend(Api::Gles(32), &state, &[Some(a8)]), (state, false));
    }

    /// An image over some of an array's layers is a range for a view, not the whole texture and
    /// not nothing: `image.bin` is its pixel gate, and this pins the reading it rests on.
    #[test]
    fn an_image_reads_its_layers_as_the_c_does() {
        use ImageLayers::*;
        // A plain 2D texture has one layer, and it is the whole of it.
        assert_eq!(image_layers(0, 0, 1, 1), Some(Whole));
        assert_eq!(image_layers(1, 1, 1, 1), None, "a layer the texture does not have");
        // Four layers: all of them, some of them, one of them, and one past the end.
        assert_eq!(image_layers(0, 3, 4, 1), Some(Whole));
        assert_eq!(image_layers(1, 2, 4, 1), Some(Range { first: 1, layers: 2 }));
        assert_eq!(image_layers(0, 2, 4, 1), Some(Range { first: 0, layers: 3 }));
        assert_eq!(image_layers(2, 2, 4, 1), Some(One(2)));
        assert_eq!(image_layers(0, 4, 4, 1), None);
        // A 3D texture's slices read the same way.
        assert_eq!(image_layers(0, 7, 1, 8), Some(Whole));
        assert_eq!(image_layers(3, 3, 1, 8), Some(One(3)));
    }

    /// The hazard the slot's serial exists to answer: the program list shifts under a bound slot
    /// whenever a variant is destroyed, and an index that did not move with it would name a
    /// different program -- which draws with the wrong uniform locations rather than failing.
    #[test]
    fn a_slot_follows_its_program_through_a_removal() {
        let bound = ProgramSlot { at: 3, serial: ProgramSerial(70) };

        // Removed before it: the program is now one place earlier, and it is the same program.
        let after = bound.after_removing(1, ProgramSerial(11)).expect("it still names a program");
        assert_eq!(after.at, 2);
        assert_eq!(after.serial, ProgramSerial(70));

        // Removed after it: nothing before the hole moves.
        let after = bound.after_removing(5, ProgramSerial(90)).expect("it still names a program");
        assert_eq!(after.at, 3);

        // Removed *is* it: nothing left to name.
        assert!(bound.after_removing(3, ProgramSerial(70)).is_none());
    }

    #[test]
    fn a_set_that_never_reached_a_draw_still_leaves_its_buffers_unbound() {
        // Four bound, then the guest sets two and one before drawing again: the slots to clear
        // are those the hardware holds, not the ones the last set replaced.
        assert_eq!(stale_vbo_slots(4, 0), 4..4);
        assert_eq!(stale_vbo_slots(1, 4), 1..4);
        assert_eq!(stale_vbo_slots(4, 4), 4..4);
        assert_eq!(stale_vbo_slots(6, 4), 6..6);
    }

    #[test]
    fn only_a_mutable_reach_moves_the_generation() {
        let mut sysval = Tracked::new(Sysval::default());
        let fresh = sysval.generation();

        // Reading the block is not a change, however much of it is read.
        assert_eq!(sysval.clip_planes[0][0], 0.0);
        assert_eq!(sysval.drawid_base, 0);
        assert_eq!(sysval.generation(), fresh);

        sysval.drawid_base = 7;
        let moved = sysval.generation();
        assert_ne!(moved, fresh);

        // Changed back is still changed: the generation only ever advances, so a program that
        // uploaded the old bytes uploads again rather than trusting a value that matches by luck.
        sysval.drawid_base = 0;
        assert_eq!(*sysval, Sysval::default());
        assert_ne!(sysval.generation(), moved);
        assert_ne!(sysval.generation(), fresh);
    }

    #[test]
    fn write_to_lays_the_block_out_as_std140_declares_it() {
        let mut sysval = Sysval::default();
        sysval.clip_planes[1] = [1.0, 2.0, 3.0, 4.0];
        sysval.stipple[2] = 0xdead_beef;
        sysval.drawid_base = -3;

        // A buffer carrying someone else's bytes: every byte of the block is written, so the
        // padding std140 leaves between stipple rows cannot carry them into the shader.
        let mut bytes = [0xabu8; Sysval::SIZE];
        sysval.write_to(&mut bytes);

        let word = |at: usize| u32::from_ne_bytes(bytes[at..at + 4].try_into().unwrap());
        assert_eq!(f32::from_bits(word(16)), 1.0);
        assert_eq!(f32::from_bits(word(28)), 4.0);
        let stipple = shader::NUM_CLIP_PLANES * 16;
        assert_eq!(word(stipple + 2 * 16), 0xdead_beef);
        assert_eq!(word(stipple + 2 * 16 + 4), 0, "std140 pads each row to sixteen bytes");
        let tail = stipple + shader::POLYGON_STIPPLE_SIZE * 16;
        assert_eq!(f32::from_bits(word(tail)), 1.0, "winsys_adjust_y");
        assert_eq!(word(tail + 12) as i32, -3, "drawid_base");
    }
}
