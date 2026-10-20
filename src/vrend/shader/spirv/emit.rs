// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! TGSI to SPIR-V, for the shapes this emitter knows, and a named refusal for every other.
//!
//! What a shader computes is held to what the GLSL emitter's text computes, not to the TGSI
//! specification where the two part: the GLSL is the C's, and the C is what the guest's pixels
//! were drawn by. Two places that matters:
//!
//! - Most opcodes write channel `c` from channel `c` of their swizzled sources. A few the GLSL
//!   spells without a writemask (`RCP`, `POW`, `NOT`, `INEG`) convert a whole `vec4` result into
//!   the destination's width instead, which takes its leading channels: written channel `k` in
//!   order gets result channel `k`. [`Emitter::pack`] is that.
//! - The dot products and `RSQ` write one value to every written channel.
//!
//! Registers are kept as `uvec4` -- TGSI's registers are untyped words -- and each operation
//! bitcasts its operands to the type its opcode reads them as, as the GLSL's `floatBitsToUint`
//! and kin do.
//!
//! The emitter refuses ([`Decline`]) anything outside what it has been taught: a refused shader
//! is translated to GLSL as before, so a gap here costs only the time SPIR-V would have saved.

use std::collections::BTreeMap;

use spirv::{
    BuiltIn, Capability, Decoration, Dim, ExecutionMode, ExecutionModel, ImageOperands, Op,
    StorageClass,
};

use super::writer::{Function, Id, Module};
use crate::vrend::pipe::ShaderStage;
use crate::vrend::proto::StreamOutput;
use crate::vrend::shader::layout::Layout;
use crate::vrend::shader::{Config, Info, Key, NUM_CLIP_PLANES, POLYGON_STIPPLE_SIZE};
use crate::vrend::tgsi::info::OpType;
use crate::vrend::tgsi::{
    self, Declaration, File, ImmType, Instruction, Interpolate, Location, Opcode, Processor,
    Property, ReturnType, Semantic, Src, Texture, Token,
};

/// Why the emitter would not write a shader: the first thing it met that it does not handle.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Decline {
    /// A shape of shader, interface or key, described.
    Shape(&'static str),
    /// An opcode, by its TGSI mnemonic.
    Opcode(&'static str),
}

impl std::fmt::Display for Decline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Decline::Shape(why) => f.write_str(why),
            Decline::Opcode(op) => write!(f, "the opcode {op}"),
        }
    }
}

type Result<T> = std::result::Result<T, Decline>;

fn decline<T>(why: &'static str) -> Result<T> {
    Err(Decline::Shape(why))
}

/// `GLSL.std.450` instruction numbers.
mod std450 {
    pub const ROUND: u32 = 1;
    pub const ROUND_EVEN: u32 = 2;
    pub const TRUNC: u32 = 3;
    pub const F_ABS: u32 = 4;
    pub const S_ABS: u32 = 5;
    pub const F_SIGN: u32 = 6;
    pub const S_SIGN: u32 = 7;
    pub const FLOOR: u32 = 8;
    pub const CEIL: u32 = 9;
    pub const FRACT: u32 = 10;
    pub const SIN: u32 = 13;
    pub const COS: u32 = 14;
    pub const POW: u32 = 26;
    pub const EXP2: u32 = 29;
    pub const LOG2: u32 = 30;
    pub const SQRT: u32 = 31;
    pub const INVERSE_SQRT: u32 = 32;
    pub const F_MIN: u32 = 37;
    pub const U_MIN: u32 = 38;
    pub const S_MIN: u32 = 39;
    pub const F_MAX: u32 = 40;
    pub const U_MAX: u32 = 41;
    pub const S_MAX: u32 = 42;
    pub const F_CLAMP: u32 = 43;
    pub const F_MIX: u32 = 46;
}

/// The type an operation reads its operands as, or writes its result as.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Ty {
    Float,
    Int,
    Uint,
}

impl Ty {
    fn of(t: OpType) -> Result<Ty> {
        match t {
            // MOV and UCMP move bits; their source modifiers are the float ones, as the GLSL's
            // `-` and `abs` on a `vec4` are.
            OpType::Untyped | OpType::Float => Ok(Ty::Float),
            OpType::Signed => Ok(Ty::Int),
            OpType::Unsigned => Ok(Ty::Uint),
            OpType::Void => decline("an opcode with no operand type"),
            OpType::Double | OpType::Unsigned64 | OpType::Signed64 => decline("a 64-bit operation"),
        }
    }
}

/// The module's common types, made once.
struct Types {
    void: Id,
    bool_: Id,
    float: Id,
    int: Id,
    uint: Id,
    vec4: Id,
    ivec4: Id,
    uvec4: Id,
    bvec4: Id,
    vec3: Id,
    vec2: Id,
    fn_uvec4: Id,
    in_vec4: Id,
    out_vec4: Id,
}

impl Types {
    fn new(m: &mut Module) -> Types {
        let void = m.type_void();
        let bool_ = m.type_bool();
        let float = m.type_float(32);
        let int = m.type_int(32, true);
        let uint = m.type_int(32, false);
        let vec4 = m.type_vector(float, 4);
        let ivec4 = m.type_vector(int, 4);
        let uvec4 = m.type_vector(uint, 4);
        let bvec4 = m.type_vector(bool_, 4);
        let vec3 = m.type_vector(float, 3);
        let vec2 = m.type_vector(float, 2);
        let fn_uvec4 = m.type_pointer(StorageClass::Function, uvec4);
        let in_vec4 = m.type_pointer(StorageClass::Input, vec4);
        let out_vec4 = m.type_pointer(StorageClass::Output, vec4);
        Types {
            void,
            bool_,
            float,
            int,
            uint,
            vec4,
            ivec4,
            uvec4,
            bvec4,
            vec3,
            vec2,
            fn_uvec4,
            in_vec4,
            out_vec4,
        }
    }

    fn vec(&self, ty: Ty) -> Id {
        match ty {
            Ty::Float => self.vec4,
            Ty::Int => self.ivec4,
            Ty::Uint => self.uvec4,
        }
    }
}

/// Where a declared output goes when the shader ends.
#[derive(Clone, Copy)]
enum Sink {
    Position,
    PointSize,
    /// A generic varying at its location.
    Varying {
        sid: u32,
        invariant: bool,
    },
    /// A fragment colour output for colour buffer `cbuf`.
    Colour {
        cbuf: u32,
    },
}

/// A sampler as declared: the target the instructions sample it as, its variable, and the type
/// of what it returns.
#[derive(Clone, Copy)]
struct Sampler {
    texture: Texture,
    var: Id,
    sampled_image: Id,
    image: Id,
    ret: Ty,
}

/// The structured construct an open `IF` or `UIF` is in.
struct Selection {
    else_label: Id,
    merge: Id,
    in_else: bool,
}

struct Emitter<'a> {
    m: Module,
    f: Function,
    t: Types,
    key: &'a Key,
    cfg: &'a Config,
    layout: Layout,
    stage: ShaderStage,
    processor: Processor,
    interface: Vec<Id>,
    /// Every register a shader names, by file and index, as a function-local `uvec4`.
    regs: BTreeMap<(u8, i16), Id>,
    /// Inputs, loaded at their use.
    inputs: BTreeMap<i16, Id>,
    outputs: BTreeMap<i16, Sink>,
    immediates: Vec<Id>,
    /// The constant array and how many `uvec4` it holds.
    consts: Option<(Id, u32)>,
    flow: Vec<Selection>,
    early_fragment_tests: bool,
    /// `FS_COLOR0_WRITES_ALL_CBUFS`: colour 0 goes to every draw buffer the host has.
    write_all_cbufs: bool,
    /// The `VirglBlock`, declared on the first read of it.
    sysvals: Option<Id>,
    /// Each sampler view's declared return type, by slot.
    views: BTreeMap<u16, ReturnType>,
    /// The samplers declared, by slot, at the first instruction to sample each.
    samplers: BTreeMap<u16, Sampler>,
    ended: bool,
}

/// The `VirglBlock` members, as `emit_required_sysval_uniforms` declares them.
mod sysval {
    pub const CLIP_PLANES: u32 = 0;
    pub const WINSYS_ADJUST_Y: u32 = 2;
    pub const CLIP_PLANE_ENABLED: u32 = 4;
}

/// The key fields this emitter does not implement, each named when it is off its neutral
/// value. A field missing here is one the shapes accepted below cannot reach.
fn check_key(key: &Key, processor: Processor) -> Result<()> {
    if key.gs_present || key.tcs_present || key.tes_present {
        return decline("a program with a geometry or tessellation stage");
    }
    if key.pstipple_enabled {
        return decline("polygon stipple");
    }
    if key.add_alpha_test {
        return decline("the alpha test");
    }
    if key.color_two_side {
        return decline("two-sided colour");
    }
    if key.fixed_point_size {
        return decline("a fixed point size");
    }
    if key.num_in_clip != 0
        || key.num_in_cull != 0
        || key.num_out_clip != 0
        || key.num_out_cull != 0
    {
        return decline("clip or cull distances");
    }
    if key.require_input_arrays || key.require_output_arrays || key.use_pervertex_in {
        return decline("interface arrays");
    }
    if !key.in_arrays.layout.is_empty() {
        return decline("input arrays");
    }
    if key.force_invariant_inputs != [0; 4] {
        return decline("invariant inputs");
    }
    if key.out_texcoord_expected_mask != 0 || key.in_texcoord_expected_mask != 0 {
        return decline("texture-coordinate varyings");
    }
    if key.in_patch_expected_mask != 0 {
        return decline("patch varyings");
    }
    if processor == Processor::Vertex {
        if key.vs.fog_fixup_mask != 0 {
            return decline("the fog fixup");
        }
        if key.vs.vertex_id_from_input {
            return decline("the vertex id as an input");
        }
    }
    if processor == Processor::Fragment {
        let fs = &key.fs;
        if fs.cbufs_are_a8_bitmask != 0 {
            return decline("alpha-only colour buffers");
        }
        if fs.logicop_func.is_some() {
            return decline("an emulated logic op");
        }
        if fs.swizzle_output_rgb_to_bgr != 0 {
            return decline("a BGR output swizzle");
        }
        if fs.needs_manual_srgb_encode_bitmask != 0 {
            return decline("manual sRGB encoding");
        }
        if fs.cbufs_signed_int_bitmask != 0 || fs.cbufs_unsigned_int_bitmask != 0 {
            return decline("integer colour buffers");
        }
        if fs.coord_replace != 0 {
            return decline("point-sprite coordinate replacement");
        }
    }
    Ok(())
}

impl<'a> Emitter<'a> {
    fn new(cfg: &'a Config, key: &'a Key, stage: ShaderStage, processor: Processor) -> Self {
        let mut m = Module::new();
        m.capability(Capability::Shader);
        let t = Types::new(&mut m);
        let fn_type = m.type_function(t.void, &[]);
        let f = m.begin_function(t.void, fn_type);
        Emitter {
            m,
            f,
            t,
            key,
            cfg,
            layout: Layout::new(key, None),
            stage,
            processor,
            interface: Vec::new(),
            regs: BTreeMap::new(),
            inputs: BTreeMap::new(),
            outputs: BTreeMap::new(),
            immediates: Vec::new(),
            consts: None,
            flow: Vec::new(),
            early_fragment_tests: false,
            write_all_cbufs: false,
            sysvals: None,
            views: BTreeMap::new(),
            samplers: BTreeMap::new(),
            ended: false,
        }
    }

    fn uint(&mut self, v: u32) -> Id {
        let uint = self.t.uint;
        self.m.constant(uint, &[v])
    }

    fn int(&mut self, v: i32) -> Id {
        let int = self.t.int;
        self.m.constant(int, &[v as u32])
    }

    fn float(&mut self, v: f32) -> Id {
        let float = self.t.float;
        self.m.constant(float, &[v.to_bits()])
    }

    fn splat(&mut self, ty: Id, scalar: Id) -> Id {
        self.m.constant_composite(ty, &[scalar; 4])
    }

    fn op(&mut self, op: Op, ty: Id, operands: &[Id]) -> Id {
        let words: Vec<u32> = operands.iter().map(|o| o.word()).collect();
        self.f.op(&mut self.m, op, ty, &words)
    }

    fn ext(&mut self, ty: Id, number: u32, operands: &[Id]) -> Id {
        self.f.glsl(&mut self.m, ty, number, operands)
    }

    fn bitcast(&mut self, ty: Id, v: Id) -> Id {
        self.op(Op::Bitcast, ty, &[v])
    }

    fn shuffle(&mut self, ty: Id, a: Id, b: Id, lanes: &[u32]) -> Id {
        let mut words = vec![a.word(), b.word()];
        words.extend_from_slice(lanes);
        self.f.op(&mut self.m, Op::VectorShuffle, ty, &words)
    }

    // ---- declarations -------------------------------------------------------------------

    fn reg(&mut self, file: File, index: i16) -> Id {
        if let Some(&id) = self.regs.get(&(file as u8, index)) {
            return id;
        }
        let ptr = self.t.fn_uvec4;
        let id = self.f.variable(&mut self.m, ptr);
        self.regs.insert((file as u8, index), id);
        id
    }

    fn declare(&mut self, d: &Declaration) -> Result<()> {
        if d.dimension.is_some() && d.file != File::Constant {
            return decline("a two-dimensional declaration");
        }
        match d.file {
            File::Temporary => {
                for i in d.first..=d.last {
                    self.reg(File::Temporary, i as i16);
                }
                Ok(())
            }
            File::Constant => match d.dimension {
                None | Some(0) => Ok(()),
                Some(_) => decline("a uniform buffer"),
            },
            File::Input => self.declare_input(d),
            File::Output => self.declare_output(d),
            File::Address => decline("address registers"),
            File::Sampler => Ok(()),
            File::SamplerView => {
                for slot in d.first..=d.last {
                    self.views.insert(slot, d.sampler_view.return_type[0]);
                }
                Ok(())
            }
            File::Image => decline("an image"),
            File::Buffer => decline("a shader storage buffer"),
            File::Memory => decline("shared or global memory"),
            File::HwAtomic => decline("an atomic counter"),
            File::SystemValue => decline("a system value"),
            File::Null | File::Immediate | File::Predicate => decline("an unexpected declaration"),
        }
    }

    fn declare_input(&mut self, d: &Declaration) -> Result<()> {
        for (n, i) in (d.first..=d.last).enumerate() {
            let var = self.m.global_variable(self.t.in_vec4, StorageClass::Input);
            match self.processor {
                // A vertex attribute's location is its register index, which is where the GLSL
                // path's `glBindAttribLocation` puts `in_<index>`.
                Processor::Vertex => self.m.decorate(var, Decoration::Location, &[u32::from(i)]),
                Processor::Fragment => {
                    if !d.has_semantic || d.semantic.name != Semantic::Generic {
                        return decline("a fragment input other than a generic varying");
                    }
                    let sid = u32::from(d.semantic.index) + n as u32;
                    let Some(location) = self.layout.varying_location(sid) else {
                        return decline("a generic varying past index 31");
                    };
                    self.m.decorate(var, Decoration::Location, &[location]);
                    self.interpolation(var, d.interp.interpolate, d.interp.location)?;
                }
                _ => return decline("an input of this stage"),
            }
            self.interface.push(var);
            self.inputs.insert(i as i16, var);
        }
        Ok(())
    }

    fn interpolation(
        &mut self,
        var: Id,
        interpolate: Interpolate,
        location: Location,
    ) -> Result<()> {
        match interpolate {
            Interpolate::Perspective => {}
            // The GLSL drops the qualifier on a host without it, and the stages must agree.
            Interpolate::Linear if !self.cfg.has_nopersective => {}
            Interpolate::Linear => self.m.decorate(var, Decoration::NoPerspective, &[]),
            Interpolate::Constant => self.m.decorate(var, Decoration::Flat, &[]),
            Interpolate::Color => return decline("colour interpolation"),
        }
        match location {
            Location::Center => {}
            Location::Centroid => self.m.decorate(var, Decoration::Centroid, &[]),
            Location::Sample => {
                self.m.capability(Capability::SampleRateShading);
                self.m.decorate(var, Decoration::Sample, &[]);
            }
        }
        Ok(())
    }

    fn declare_output(&mut self, d: &Declaration) -> Result<()> {
        if !d.has_semantic {
            return decline("an output with no semantic");
        }
        for (n, i) in (d.first..=d.last).enumerate() {
            let index = u32::from(d.semantic.index) + n as u32;
            let sink = match (self.processor, d.semantic.name) {
                (Processor::Vertex, Semantic::Position) => Sink::Position,
                (Processor::Vertex, Semantic::PSize) => Sink::PointSize,
                (Processor::Vertex, Semantic::Generic) => {
                    Sink::Varying { sid: index, invariant: d.invariant }
                }
                (Processor::Fragment, Semantic::Color) => Sink::Colour { cbuf: index },
                (Processor::Vertex, _) => {
                    return decline("a vertex output other than position, size or generic");
                }
                (Processor::Fragment, _) => return decline("a fragment output other than colour"),
                _ => return decline("an output of this stage"),
            };
            if self.outputs.insert(i as i16, sink).is_some() {
                return decline("an output declared twice");
            }
            self.reg(File::Output, i as i16);
        }
        Ok(())
    }

    fn immediate(&mut self, imm: &tgsi::Immediate) -> Result<()> {
        if !matches!(imm.ty, ImmType::Float32 | ImmType::Uint32 | ImmType::Int32) {
            return decline("a 64-bit immediate");
        }
        let parts = imm.data.map(|w| self.uint(w));
        let uvec4 = self.t.uvec4;
        let id = self.m.constant_composite(uvec4, &parts);
        self.immediates.push(id);
        Ok(())
    }

    fn property(&mut self, p: &tgsi::PropertyToken) -> Result<()> {
        match p.name {
            // Read only through `gl_FragCoord`, which this emitter does not declare.
            Property::FsCoordOrigin | Property::FsCoordPixelCenter => Ok(()),
            Property::NextShader | Property::MulZeroWins => Ok(()),
            Property::FsEarlyDepthStencil => {
                self.early_fragment_tests = p.data != 0;
                Ok(())
            }
            Property::FsColor0WritesAllCbufs => {
                self.write_all_cbufs = p.data == 1;
                Ok(())
            }
            Property::FsDepthLayout if p.data == 0 => Ok(()),
            Property::NumClipdistEnabled | Property::NumCulldistEnabled if p.data == 0 => Ok(()),
            Property::SeparableProgram => decline("a separable program"),
            _ => decline("a shader property this emitter does not handle"),
        }
    }

    fn constants(&mut self, count: u32) -> Id {
        if let Some((var, _)) = self.consts {
            return var;
        }
        let array = self.m.type_array(self.t.uvec4, count);
        let ptr = self.m.type_pointer(StorageClass::UniformConstant, array);
        let var = self.m.global_variable(ptr, StorageClass::UniformConstant);
        let location = Layout::const_location(self.stage);
        self.m.decorate(var, Decoration::Location, &[location]);
        self.consts = Some((var, count));
        var
    }

    /// The `VirglBlock`, at the binding every stage of a SPIR-V program reads it at, with
    /// every member the GLSL's declares and the offsets std140 gives them there -- which is
    /// what the draw path's upload writes (`draw::Sysval::write_to`).
    fn sysvals(&mut self) -> Id {
        if let Some(var) = self.sysvals {
            return var;
        }
        let t = &self.t;
        let (vec4, uint, float, int) = (t.vec4, t.uint, t.float, t.int);
        let planes = self.m.type_array(vec4, NUM_CLIP_PLANES as u32);
        let stipple = self.m.type_array(uint, POLYGON_STIPPLE_SIZE as u32);
        self.m.decorate(planes, Decoration::ArrayStride, &[16]);
        self.m.decorate(stipple, Decoration::ArrayStride, &[16]);
        // `clip_plane_enabled` is a `bool` in the GLSL; a block member cannot be one in SPIR-V,
        // and std140 lays a `bool` out as a 32-bit word, which is what is read here.
        let block = self.m.type_struct(&[planes, stipple, float, float, uint, int]);
        self.m.decorate(block, Decoration::Block, &[]);
        let stipple_end = 16 * NUM_CLIP_PLANES as u32 + 16 * POLYGON_STIPPLE_SIZE as u32;
        let offsets = [
            0,
            16 * NUM_CLIP_PLANES as u32,
            stipple_end,
            stipple_end + 4,
            stipple_end + 8,
            stipple_end + 12,
        ];
        for (member, offset) in offsets.into_iter().enumerate() {
            self.m.member_decorate(block, member as u32, Decoration::Offset, &[offset]);
        }
        let ptr = self.m.type_pointer(StorageClass::Uniform, block);
        let var = self.m.global_variable(ptr, StorageClass::Uniform);
        self.m.decorate(var, Decoration::Binding, &[Layout::VIRGL_BLOCK_BINDING]);
        self.sysvals = Some(var);
        var
    }

    /// The sampler `inst` samples through, declared as the target the instruction names on its
    /// first use. The GLSL declares a sampler by the target its instructions name, too.
    fn sampler(&mut self, inst: &Instruction) -> Result<Sampler> {
        let src = inst.src[1];
        if src.file != File::Sampler || src.indirect || !(0..32).contains(&src.index) {
            return decline("a texture instruction naming no plain sampler");
        }
        let slot = src.index as u16;
        let index = usize::from(slot);
        let mut texture = inst.tex().texture;
        if texture == Texture::Array2d
            && Key::view_mask_get(&self.key.sampler_views_lower_array_mask, index)
        {
            texture = Texture::D2;
        }
        if Key::view_mask_get(&self.key.sampler_views_emulated_rect_mask, index) {
            return decline("a rectangle emulated by a 2D texture");
        }
        if inst.tex().num_offsets != 0 {
            return decline("a texel offset");
        }
        if let Some(s) = self.samplers.get(&slot) {
            if s.texture != texture {
                return decline("a sampler sampled as two targets");
            }
            return Ok(*s);
        }
        let (dim, arrayed) = match texture {
            Texture::D2 => (Dim::Dim2D, false),
            Texture::D3 => (Dim::Dim3D, false),
            Texture::Cube => (Dim::DimCube, false),
            Texture::Array2d => (Dim::Dim2D, true),
            _ => return decline("a texture target this emitter does not sample"),
        };
        let Some(&ret) = self.views.get(&slot) else {
            return decline("a sampler with no view declared");
        };
        let (ret, scalar) = match ret {
            ReturnType::Sint => (Ty::Int, self.t.int),
            ReturnType::Uint => (Ty::Uint, self.t.uint),
            ReturnType::Float | ReturnType::Unorm | ReturnType::Snorm => (Ty::Float, self.t.float),
        };
        let image = self.m.type_image(scalar, dim, arrayed);
        let sampled_image = self.m.type_sampled_image(image);
        let ptr = self.m.type_pointer(StorageClass::UniformConstant, sampled_image);
        let var = self.m.global_variable(ptr, StorageClass::UniformConstant);
        // The draw path writes the unit to the location; the binding is only the initial value
        // GL requires an opaque uniform in a module to carry.
        let location = Layout::sampler_location(self.stage, u32::from(slot));
        self.m.decorate(var, Decoration::Location, &[location]);
        self.m.decorate(var, Decoration::Binding, &[u32::from(slot)]);
        let s = Sampler { texture, var, sampled_image, image, ret };
        self.samplers.insert(slot, s);
        Ok(s)
    }

    /// `TEX`, `TXB`, `TXL`, `TXF` and `TXQ`, as `translate_tex` and `emit_txq` spell them for
    /// the targets [`Emitter::sampler`] takes.
    fn texture(&mut self, inst: &Instruction, num_consts: u32) -> Result<()> {
        let s = self.sampler(inst)?;
        let (ivec4, float, int) = (self.t.ivec4, self.t.float, self.t.int);
        let result_ty = self.t.vec(s.ret);
        // How many coordinate channels the target takes, layer included.
        let n = match s.texture {
            Texture::D2 => 2,
            _ => 3,
        };
        let fragment = self.processor == Processor::Fragment;
        match inst.opcode {
            Opcode::Tex | Opcode::Txb | Opcode::Txl => {
                let c = self.src(inst, 0, Ty::Float, num_consts)?;
                let coord_ty = if n == 2 { self.t.vec2 } else { self.t.vec3 };
                let lanes: Vec<u32> = (0..n).collect();
                let coord = self.shuffle(coord_ty, c, c, &lanes);
                let sampled = self.op(Op::Load, s.sampled_image, &[s.var]);
                let mut words = vec![sampled.word(), coord.word()];
                let op = match inst.opcode {
                    Opcode::Txb if !fragment => {
                        return decline("a biased lookup outside a fragment shader");
                    }
                    Opcode::Tex if fragment => Op::ImageSampleImplicitLod,
                    // Outside a fragment shader there are no derivatives; GLSL's `texture()`
                    // there samples level 0.
                    Opcode::Tex => {
                        let zero = self.float(0.0);
                        words.extend([ImageOperands::LOD.bits(), zero.word()]);
                        Op::ImageSampleExplicitLod
                    }
                    Opcode::Txb => {
                        let bias =
                            self.f.op(&mut self.m, Op::CompositeExtract, float, &[c.word(), 3]);
                        words.extend([ImageOperands::BIAS.bits(), bias.word()]);
                        Op::ImageSampleImplicitLod
                    }
                    _ => {
                        let lod =
                            self.f.op(&mut self.m, Op::CompositeExtract, float, &[c.word(), 3]);
                        words.extend([ImageOperands::LOD.bits(), lod.word()]);
                        Op::ImageSampleExplicitLod
                    }
                };
                let r = self.f.op(&mut self.m, op, result_ty, &words);
                self.store(inst, r, s.ret)
            }
            Opcode::Txf => {
                if s.texture == Texture::Cube {
                    return decline("a texel fetch from a cube");
                }
                let index = inst.src[1].index as usize;
                if Key::view_mask_get(&self.key.sampler_views_lower_swizzle_mask, index) {
                    return decline("a texel fetch through a lowered swizzle");
                }
                // The coordinate and level are integers in the register: `ivec2(src.xy)` of
                // the unsigned words, and `int(src.w)`.
                let c = self.src(inst, 0, Ty::Int, num_consts)?;
                let ivec =
                    if n == 2 { self.m.type_vector(int, 2) } else { self.m.type_vector(int, 3) };
                let lanes: Vec<u32> = (0..n).collect();
                let coord = self.shuffle(ivec, c, c, &lanes);
                let lod = self.f.op(&mut self.m, Op::CompositeExtract, int, &[c.word(), 3]);
                let sampled = self.op(Op::Load, s.sampled_image, &[s.var]);
                let image = self.op(Op::Image, s.image, &[sampled]);
                let words = [image.word(), coord.word(), ImageOperands::LOD.bits(), lod.word()];
                let r = self.f.op(&mut self.m, Op::ImageFetch, result_ty, &words);
                self.store(inst, r, s.ret)
            }
            Opcode::Txq => {
                if s.texture != Texture::D2 {
                    return decline("a size query of a target other than 2D");
                }
                if inst.dst[0].writemask & 0xc != 0 {
                    return decline("a level-count query");
                }
                self.m.capability(Capability::ImageQuery);
                // `textureSize(sampler, int(src.x))`, its two channels in x and y.
                let c = self.src(inst, 0, Ty::Int, num_consts)?;
                let lod = self.f.op(&mut self.m, Op::CompositeExtract, int, &[c.word(), 0]);
                let sampled = self.op(Op::Load, s.sampled_image, &[s.var]);
                let image = self.op(Op::Image, s.image, &[sampled]);
                let ivec2 = self.m.type_vector(int, 2);
                let size = self.op(Op::ImageQuerySizeLod, ivec2, &[image, lod]);
                let zero = self.int(0);
                let zero2 = self.m.constant_composite(ivec2, &[zero, zero]);
                let r = self.shuffle(ivec4, size, zero2, &[0, 1, 2, 3]);
                self.store(inst, r, Ty::Int)
            }
            _ => unreachable!("only the texture opcodes come here"),
        }
    }

    // ---- operands -----------------------------------------------------------------------

    /// The register `src` names, as raw words, before its swizzle.
    fn load_raw(&mut self, src: &Src, num_consts: u32) -> Result<Id> {
        if src.indirect {
            return decline("an indirectly addressed source");
        }
        let uvec4 = self.t.uvec4;
        match src.file {
            File::Temporary | File::Output => {
                let var = self.reg(src.file, src.index);
                Ok(self.op(Op::Load, uvec4, &[var]))
            }
            File::Input => {
                let Some(&var) = self.inputs.get(&src.index) else {
                    return decline("a read of an undeclared input");
                };
                let vec4 = self.t.vec4;
                let mut v = self.op(Op::Load, vec4, &[var]);
                // An attribute whose format the host stores swapped (BGRA) is read back through
                // `.zyxw` before the source's own swizzle, as the GLSL does.
                if self.processor == Processor::Vertex
                    && src.index >= 0
                    && src.index < 32
                    && self.key.vs.attrib_zyxw_bitmask & (1 << src.index) != 0
                {
                    v = self.shuffle(vec4, v, v, &[2, 1, 0, 3]);
                }
                Ok(self.bitcast(uvec4, v))
            }
            File::Immediate => {
                let Some(&imm) = self.immediates.get(src.index as usize) else {
                    return decline("a read of an undeclared immediate");
                };
                Ok(imm)
            }
            File::Constant => {
                if src.dimension && src.dim.index != 0 {
                    return decline("a uniform buffer read");
                }
                if src.dimension && src.dim.indirect {
                    return decline("an indirectly addressed constant buffer");
                }
                let index = u32::try_from(src.index).unwrap_or(u32::MAX);
                if index >= num_consts {
                    return decline("a constant past the declared ones");
                }
                let var = self.constants(num_consts);
                let ptr = self.m.type_pointer(StorageClass::UniformConstant, uvec4);
                let i = self.int(index as i32);
                let p = self.op(Op::AccessChain, ptr, &[var, i]);
                Ok(self.op(Op::Load, uvec4, &[p]))
            }
            _ => decline("a source in a file this emitter does not read"),
        }
    }

    /// Source `n` of `inst`, swizzled, as type `ty`, with its modifiers applied in that type.
    fn src(&mut self, inst: &Instruction, n: usize, ty: Ty, num_consts: u32) -> Result<Id> {
        let src = inst.src[n];
        let raw = self.load_raw(&src, num_consts)?;
        let uvec4 = self.t.uvec4;
        let swz = src.swizzle.map(u32::from);
        let raw = if swz == [0, 1, 2, 3] { raw } else { self.shuffle(uvec4, raw, raw, &swz) };
        let vty = self.t.vec(ty);
        let mut v = if ty == Ty::Uint { raw } else { self.bitcast(vty, raw) };
        if src.absolute {
            v = match ty {
                Ty::Float => self.ext(vty, std450::F_ABS, &[v]),
                Ty::Int => self.ext(vty, std450::S_ABS, &[v]),
                Ty::Uint => return decline("the absolute value of an unsigned source"),
            };
        }
        if src.negate {
            v = match ty {
                Ty::Float => self.op(Op::FNegate, vty, &[v]),
                Ty::Int | Ty::Uint => self.op(Op::SNegate, vty, &[v]),
            };
        }
        Ok(v)
    }

    /// Writes `value`, a `vec4` of type `ty`, to `inst`'s destination under its writemask.
    fn store(&mut self, inst: &Instruction, value: Id, ty: Ty) -> Result<()> {
        let dst = inst.dst[0];
        if dst.indirect || dst.dimension {
            return decline("an indirectly addressed destination");
        }
        if !matches!(dst.file, File::Temporary | File::Output) {
            return decline("a destination in a file this emitter does not write");
        }
        if dst.file == File::Output && !self.outputs.contains_key(&dst.index) {
            return decline("a write to an undeclared output");
        }
        let mut v = value;
        if inst.saturate {
            if ty != Ty::Float {
                return decline("saturation of an integer result");
            }
            let (vec4, zero, one) = (self.t.vec4, self.float(0.0), self.float(1.0));
            let (lo, hi) = (self.splat(vec4, zero), self.splat(vec4, one));
            v = self.ext(vec4, std450::F_CLAMP, &[v, lo, hi]);
        }
        let uvec4 = self.t.uvec4;
        if ty != Ty::Uint {
            v = self.bitcast(uvec4, v);
        }
        let var = self.reg(dst.file, dst.index);
        let mask = u32::from(dst.writemask & 0xf);
        if mask != 0xf {
            let old = self.op(Op::Load, uvec4, &[var]);
            let lanes: Vec<u32> =
                (0..4).map(|c| if mask & (1 << c) != 0 { 4 + c } else { c }).collect();
            v = self.shuffle(uvec4, old, v, &lanes);
        }
        self.f.op_void(Op::Store, &[var.word(), v.word()]);
        Ok(())
    }

    /// A `vec4` result the GLSL converts whole into the destination's width: written channel
    /// `k`, in order, takes the result's channel `k`.
    fn pack(&mut self, inst: &Instruction, ty: Ty, v: Id) -> Id {
        let mask = inst.dst[0].writemask & 0xf;
        let mut k = 0;
        let lanes: Vec<u32> = (0..4u32)
            .map(|c| {
                if mask & (1 << c) != 0 {
                    k += 1;
                    k - 1
                } else {
                    c
                }
            })
            .collect();
        if lanes == [0, 1, 2, 3] {
            return v;
        }
        let vty = self.t.vec(ty);
        self.shuffle(vty, v, v, &lanes)
    }

    fn scalar_splat(&mut self, ty: Id, s: Id) -> Id {
        let words = [s.word(); 4];
        self.f.op(&mut self.m, Op::CompositeConstruct, ty, &words)
    }

    // ---- instructions -------------------------------------------------------------------

    fn instruction(&mut self, inst: &Instruction, num_consts: u32) -> Result<()> {
        use Opcode::*;
        if inst.num_dst > 1 {
            return decline("an instruction with two destinations");
        }
        let src_ty = Ty::of(inst.opcode.src_type());
        let vec4 = self.t.vec4;
        let ivec4 = self.t.ivec4;
        let uvec4 = self.t.uvec4;
        let bvec4 = self.t.bvec4;
        macro_rules! s {
            ($n:expr, $ty:expr) => {
                self.src(inst, $n, $ty, num_consts)?
            };
        }
        // A componentwise operation of the GLSL.std.450 set on one source.
        macro_rules! ext1 {
            ($num:expr) => {{
                let ty = src_ty?;
                let a = s!(0, ty);
                let vty = self.t.vec(ty);
                let r = self.ext(vty, $num, &[a]);
                self.store(inst, r, ty)
            }};
        }
        macro_rules! ext2 {
            ($num:expr) => {{
                let ty = src_ty?;
                let (a, b) = (s!(0, ty), s!(1, ty));
                let vty = self.t.vec(ty);
                let r = self.ext(vty, $num, &[a, b]);
                self.store(inst, r, ty)
            }};
        }
        macro_rules! bin {
            ($op:expr) => {{
                let ty = src_ty?;
                let (a, b) = (s!(0, ty), s!(1, ty));
                let vty = self.t.vec(ty);
                let r = self.op($op, vty, &[a, b]);
                if inst.precise {
                    self.m.decorate(r, Decoration::NoContraction, &[]);
                }
                self.store(inst, r, ty)
            }};
        }
        macro_rules! unary {
            ($op:expr) => {{
                let ty = src_ty?;
                let a = s!(0, ty);
                let vty = self.t.vec(ty);
                let r = self.op($op, vty, &[a]);
                self.store(inst, r, ty)
            }};
        }
        // A comparison whose result is 1.0 or 0.0 per channel (`SLT` and kin).
        macro_rules! fcompare {
            ($op:expr) => {{
                let ty = src_ty?;
                let (a, b) = (s!(0, ty), s!(1, ty));
                let c = self.op($op, bvec4, &[a, b]);
                let (one, zero) = (self.float(1.0), self.float(0.0));
                let (one, zero) = (self.splat(vec4, one), self.splat(vec4, zero));
                let r = self.op(Op::Select, vec4, &[c, one, zero]);
                self.store(inst, r, Ty::Float)
            }};
        }
        // A comparison whose result is every bit or none per channel (`FSLT`, `ISLT` and kin).
        macro_rules! mcompare {
            ($op:expr) => {{
                let ty = src_ty?;
                let (a, b) = (s!(0, ty), s!(1, ty));
                let c = self.op($op, bvec4, &[a, b]);
                let (all, none) = (self.uint(u32::MAX), self.uint(0));
                let (all, none) = (self.splat(uvec4, all), self.splat(uvec4, none));
                let r = self.op(Op::Select, uvec4, &[c, all, none]);
                self.store(inst, r, Ty::Uint)
            }};
        }
        match inst.opcode {
            Mov => {
                let a = s!(0, Ty::Float);
                self.store(inst, a, Ty::Float)
            }
            Add => bin!(Op::FAdd),
            Sub => bin!(Op::FSub),
            Mul => bin!(Op::FMul),
            Div => bin!(Op::FDiv),
            Mad => {
                let (a, b, c) = (s!(0, Ty::Float), s!(1, Ty::Float), s!(2, Ty::Float));
                let ab = self.op(Op::FMul, vec4, &[a, b]);
                let r = self.op(Op::FAdd, vec4, &[ab, c]);
                if inst.precise {
                    self.m.decorate(ab, Decoration::NoContraction, &[]);
                    self.m.decorate(r, Decoration::NoContraction, &[]);
                }
                self.store(inst, r, Ty::Float)
            }
            Min => ext2!(std450::F_MIN),
            Max => ext2!(std450::F_MAX),
            Imin => ext2!(std450::S_MIN),
            Imax => ext2!(std450::S_MAX),
            Umin => ext2!(std450::U_MIN),
            Umax => ext2!(std450::U_MAX),
            Frc => ext1!(std450::FRACT),
            Flr => ext1!(std450::FLOOR),
            Ceil => ext1!(std450::CEIL),
            Trunc => ext1!(std450::TRUNC),
            Round => {
                let even = (self.cfg.is_gles() && self.cfg.glsl_version >= 300)
                    || self.cfg.glsl_version >= 400;
                ext1!(if even { std450::ROUND_EVEN } else { std450::ROUND })
            }
            Ssg => ext1!(std450::F_SIGN),
            Issg => ext1!(std450::S_SIGN),
            Abs => ext1!(std450::F_ABS),
            Iabs => ext1!(std450::S_ABS),
            Sqrt => ext1!(std450::SQRT),
            Ex2 => ext1!(std450::EXP2),
            Lg2 => ext1!(std450::LOG2),
            Sin => ext1!(std450::SIN),
            Cos => ext1!(std450::COS),
            Ddx | Ddy => {
                if self.processor != Processor::Fragment {
                    return decline("a derivative outside a fragment shader");
                }
                unary!(if inst.opcode == Ddx { Op::DPdx } else { Op::DPdy })
            }
            Rcp => {
                let a = s!(0, Ty::Float);
                let one = self.float(1.0);
                let one = self.splat(vec4, one);
                let r = self.op(Op::FDiv, vec4, &[one, a]);
                let r = self.pack(inst, Ty::Float, r);
                self.store(inst, r, Ty::Float)
            }
            Pow => {
                let (a, b) = (s!(0, Ty::Float), s!(1, Ty::Float));
                let r = self.ext(vec4, std450::POW, &[a, b]);
                let r = self.pack(inst, Ty::Float, r);
                self.store(inst, r, Ty::Float)
            }
            Rsq => {
                let a = s!(0, Ty::Float);
                let float = self.t.float;
                let x = self.f.op(&mut self.m, Op::CompositeExtract, float, &[a.word(), 0]);
                let r = self.ext(float, std450::INVERSE_SQRT, &[x]);
                let r = self.scalar_splat(vec4, r);
                self.store(inst, r, Ty::Float)
            }
            Dp2 | Dp3 | Dp4 | Dph => {
                let (mut a, mut b) = (s!(0, Ty::Float), s!(1, Ty::Float));
                match inst.opcode {
                    Dp2 => {
                        let vec2 = self.t.vec2;
                        a = self.shuffle(vec2, a, a, &[0, 1]);
                        b = self.shuffle(vec2, b, b, &[0, 1]);
                    }
                    Dp3 => {
                        let vec3 = self.t.vec3;
                        a = self.shuffle(vec3, a, a, &[0, 1, 2]);
                        b = self.shuffle(vec3, b, b, &[0, 1, 2]);
                    }
                    Dph => {
                        // `dot(vec4(vec3(a), 1.0), b)`.
                        let one = self.float(1.0);
                        let one = self.splat(vec4, one);
                        a = self.shuffle(vec4, a, one, &[0, 1, 2, 4]);
                    }
                    _ => {}
                }
                let float = self.t.float;
                let d = self.op(Op::Dot, float, &[a, b]);
                let r = self.scalar_splat(vec4, d);
                self.store(inst, r, Ty::Float)
            }
            Lrp => {
                // `mix(src2, src1, src0)`.
                let (a, b, c) = (s!(0, Ty::Float), s!(1, Ty::Float), s!(2, Ty::Float));
                let r = self.ext(vec4, std450::F_MIX, &[c, b, a]);
                self.store(inst, r, Ty::Float)
            }
            Cmp => {
                // `mix(src1, src2, greaterThanEqual(src0, vec4(0.0)))`.
                let (a, b, c) = (s!(0, Ty::Float), s!(1, Ty::Float), s!(2, Ty::Float));
                let zero = self.float(0.0);
                let zero = self.splat(vec4, zero);
                let ge = self.op(Op::FOrdGreaterThanEqual, bvec4, &[a, zero]);
                let r = self.op(Op::Select, vec4, &[ge, c, b]);
                self.store(inst, r, Ty::Float)
            }
            Ucmp => {
                // `mix(src2, src1, notEqual(floatBitsToUint(src0), uvec4(0)))`, which moves bits.
                let a = s!(0, Ty::Uint);
                let (b, c) = (s!(1, Ty::Uint), s!(2, Ty::Uint));
                let zero = self.uint(0);
                let zero = self.splat(uvec4, zero);
                let ne = self.op(Op::INotEqual, bvec4, &[a, zero]);
                let r = self.op(Op::Select, uvec4, &[ne, b, c]);
                self.store(inst, r, Ty::Uint)
            }
            Seq => fcompare!(Op::FOrdEqual),
            Sne => fcompare!(Op::FUnordNotEqual),
            Slt => fcompare!(Op::FOrdLessThan),
            Sle => fcompare!(Op::FOrdLessThanEqual),
            Sgt => fcompare!(Op::FOrdGreaterThan),
            Sge => fcompare!(Op::FOrdGreaterThanEqual),
            Fseq => mcompare!(Op::FOrdEqual),
            Fsne => mcompare!(Op::FUnordNotEqual),
            Fslt => mcompare!(Op::FOrdLessThan),
            Fsge => mcompare!(Op::FOrdGreaterThanEqual),
            Useq => mcompare!(Op::IEqual),
            Usne => mcompare!(Op::INotEqual),
            Uslt => mcompare!(Op::ULessThan),
            Usge => mcompare!(Op::UGreaterThanEqual),
            Islt => mcompare!(Op::SLessThan),
            Isge => mcompare!(Op::SGreaterThanEqual),
            I2f => {
                let a = s!(0, Ty::Int);
                let r = self.op(Op::ConvertSToF, vec4, &[a]);
                self.store(inst, r, Ty::Float)
            }
            U2f => {
                let a = s!(0, Ty::Uint);
                let r = self.op(Op::ConvertUToF, vec4, &[a]);
                self.store(inst, r, Ty::Float)
            }
            F2i => {
                let a = s!(0, Ty::Float);
                let r = self.op(Op::ConvertFToS, ivec4, &[a]);
                self.store(inst, r, Ty::Int)
            }
            F2u => {
                let a = s!(0, Ty::Float);
                let r = self.op(Op::ConvertFToU, uvec4, &[a]);
                self.store(inst, r, Ty::Uint)
            }
            Uadd => bin!(Op::IAdd),
            Umul => bin!(Op::IMul),
            And => bin!(Op::BitwiseAnd),
            Or => bin!(Op::BitwiseOr),
            Xor => bin!(Op::BitwiseXor),
            Shl => bin!(Op::ShiftLeftLogical),
            Ishr => bin!(Op::ShiftRightArithmetic),
            Ushr => bin!(Op::ShiftRightLogical),
            Not => {
                let a = s!(0, Ty::Uint);
                let r = self.op(Op::Not, uvec4, &[a]);
                let r = self.pack(inst, Ty::Uint, r);
                self.store(inst, r, Ty::Uint)
            }
            Ineg => {
                let a = s!(0, Ty::Int);
                let r = self.op(Op::SNegate, ivec4, &[a]);
                let r = self.pack(inst, Ty::Int, r);
                self.store(inst, r, Ty::Int)
            }
            If | Uif => {
                let c = if inst.opcode == Uif {
                    let a = s!(0, Ty::Uint);
                    let uint = self.t.uint;
                    let x = self.f.op(&mut self.m, Op::CompositeExtract, uint, &[a.word(), 0]);
                    let zero = self.uint(0);
                    let b = self.t.bool_;
                    self.op(Op::INotEqual, b, &[x, zero])
                } else {
                    // `bool(x)` of a float is `x != 0.0`, which a NaN passes.
                    let a = s!(0, Ty::Float);
                    let float = self.t.float;
                    let x = self.f.op(&mut self.m, Op::CompositeExtract, float, &[a.word(), 0]);
                    let zero = self.float(0.0);
                    let b = self.t.bool_;
                    self.op(Op::FUnordNotEqual, b, &[x, zero])
                };
                let (then, else_label, merge) = (self.m.id(), self.m.id(), self.m.id());
                self.f.op_void(Op::SelectionMerge, &[merge.word(), 0]);
                self.f.op_void(Op::BranchConditional, &[c.word(), then.word(), else_label.word()]);
                self.f.label(then);
                self.flow.push(Selection { else_label, merge, in_else: false });
                Ok(())
            }
            Else => {
                let Some(sel) = self.flow.last_mut() else {
                    return decline("an ELSE outside an IF");
                };
                if sel.in_else {
                    return decline("two ELSEs in one IF");
                }
                sel.in_else = true;
                let (else_label, merge) = (sel.else_label, sel.merge);
                self.f.op_void(Op::Branch, &[merge.word()]);
                self.f.label(else_label);
                Ok(())
            }
            Endif => {
                let Some(sel) = self.flow.pop() else {
                    return decline("an ENDIF outside an IF");
                };
                self.f.op_void(Op::Branch, &[sel.merge.word()]);
                if !sel.in_else {
                    self.f.label(sel.else_label);
                    self.f.op_void(Op::Branch, &[sel.merge.word()]);
                }
                self.f.label(sel.merge);
                Ok(())
            }
            Kill => {
                if self.processor != Processor::Fragment {
                    return decline("a discard outside a fragment shader");
                }
                self.f.op_void(Op::Kill, &[]);
                // Whatever follows in the block is unreachable, and still needs a block to be in.
                let dead = self.m.id();
                self.f.label(dead);
                Ok(())
            }
            KillIf => {
                if self.processor != Processor::Fragment {
                    return decline("a discard outside a fragment shader");
                }
                // `if (any(lessThan(src, vec4(0.0)))) discard;`
                let a = s!(0, Ty::Float);
                let zero = self.float(0.0);
                let zero = self.splat(vec4, zero);
                let lt = self.op(Op::FOrdLessThan, bvec4, &[a, zero]);
                let b = self.t.bool_;
                let any = self.op(Op::Any, b, &[lt]);
                let (kill, merge) = (self.m.id(), self.m.id());
                self.f.op_void(Op::SelectionMerge, &[merge.word(), 0]);
                self.f.op_void(Op::BranchConditional, &[any.word(), kill.word(), merge.word()]);
                self.f.label(kill);
                self.f.op_void(Op::Kill, &[]);
                self.f.label(merge);
                Ok(())
            }
            Tex | Txb | Txl | Txf | Txq => self.texture(inst, num_consts),
            End => {
                if !self.flow.is_empty() {
                    return decline("an END inside an IF");
                }
                self.ended = true;
                Ok(())
            }
            op => Err(Decline::Opcode(op.mnemonic())),
        }
    }

    // ---- the way out --------------------------------------------------------------------

    /// What the GLSL does at the end of a vertex shader that is the last before the rasteriser:
    /// the clip planes, where the host has clip distances and the shader writes none of its own,
    /// then the y prescale.
    fn vertex_epilogue(&mut self, info: &Info) -> Result<()> {
        let Some((&pos_reg, _)) = self.outputs.iter().find(|(_, s)| matches!(s, Sink::Position))
        else {
            // No position: nothing for the clip planes or the prescale to read, as in the GLSL,
            // which writes them into an unwritten gl_Position all the same. Rare enough to leave.
            return decline("a vertex shader that writes no position");
        };
        let pos = self.reg(File::Output, pos_reg);
        let (vec4, uvec4, float, uint) = (self.t.vec4, self.t.uvec4, self.t.float, self.t.uint);
        let block = self.sysvals();
        if self.cfg.has_cull_distance
            && (info.num_outputs as u32) + 2 <= crate::vrend::shader::glsl::MAX_VARYING
        {
            self.m.capability(Capability::ClipDistance);
            let distances = self.m.type_array(float, NUM_CLIP_PLANES as u32);
            let out_ptr = self.m.type_pointer(StorageClass::Output, distances);
            let clip = self.m.global_variable(out_ptr, StorageClass::Output);
            self.m.builtin(clip, BuiltIn::ClipDistance);
            self.interface.push(clip);
            let uniform_uint = self.m.type_pointer(StorageClass::Uniform, uint);
            let member = self.int(sysval::CLIP_PLANE_ENABLED as i32);
            let p = self.op(Op::AccessChain, uniform_uint, &[block, member]);
            let enabled = self.op(Op::Load, uint, &[p]);
            let zero = self.uint(0);
            let b = self.t.bool_;
            let on = self.op(Op::INotEqual, b, &[enabled, zero]);
            let (then, merge) = (self.m.id(), self.m.id());
            self.f.op_void(Op::SelectionMerge, &[merge.word(), 0]);
            self.f.op_void(Op::BranchConditional, &[on.word(), then.word(), merge.word()]);
            self.f.label(then);
            let raw = self.op(Op::Load, uvec4, &[pos]);
            let position = self.bitcast(vec4, raw);
            let uniform_vec4 = self.m.type_pointer(StorageClass::Uniform, vec4);
            let out_float = self.m.type_pointer(StorageClass::Output, float);
            let planes = self.int(sysval::CLIP_PLANES as i32);
            for i in 0..NUM_CLIP_PLANES as i32 {
                let idx = self.int(i);
                let pp = self.op(Op::AccessChain, uniform_vec4, &[block, planes, idx]);
                let plane = self.op(Op::Load, vec4, &[pp]);
                let d = self.op(Op::Dot, float, &[position, plane]);
                let dst = self.op(Op::AccessChain, out_float, &[clip, idx]);
                self.f.op_void(Op::Store, &[dst.word(), d.word()]);
            }
            self.f.op_void(Op::Branch, &[merge.word()]);
            self.f.label(merge);
        }
        // `gl_Position.y = gl_Position.y * winsys_adjust_y;`
        let uniform_float = self.m.type_pointer(StorageClass::Uniform, float);
        let member = self.int(sysval::WINSYS_ADJUST_Y as i32);
        let p = self.op(Op::AccessChain, uniform_float, &[block, member]);
        let adjust = self.op(Op::Load, float, &[p]);
        let raw = self.op(Op::Load, uvec4, &[pos]);
        let position = self.bitcast(vec4, raw);
        let y = self.f.op(&mut self.m, Op::CompositeExtract, float, &[position.word(), 1]);
        let y = self.op(Op::FMul, float, &[y, adjust]);
        let words = [y.word(), position.word(), 1];
        let position = self.f.op(&mut self.m, Op::CompositeInsert, vec4, &words);
        let raw = self.bitcast(uvec4, position);
        self.f.op_void(Op::Store, &[pos.word(), raw.word()]);
        Ok(())
    }

    /// The outputs' registers copied out to the interface, then the padding a vertex shader
    /// declares for the generic inputs the next stage reads and it does not write.
    fn write_outputs(&mut self) -> Result<()> {
        let (vec4, uvec4, float) = (self.t.vec4, self.t.uvec4, self.t.float);
        let mut written = 0u64;
        let outputs: Vec<(i16, Sink)> = self.outputs.iter().map(|(&i, &s)| (i, s)).collect();
        for (index, sink) in outputs {
            let reg = self.reg(File::Output, index);
            let raw = self.op(Op::Load, uvec4, &[reg]);
            let v = self.bitcast(vec4, raw);
            match sink {
                Sink::Position | Sink::Varying { .. } | Sink::Colour { .. } => {
                    let var = self.m.global_variable(self.t.out_vec4, StorageClass::Output);
                    match sink {
                        Sink::Position => self.m.builtin(var, BuiltIn::Position),
                        Sink::Varying { sid, invariant } => {
                            let Some(location) = self.layout.varying_location(sid) else {
                                return decline("a generic varying past index 31");
                            };
                            self.m.decorate(var, Decoration::Location, &[location]);
                            if invariant {
                                self.m.decorate(var, Decoration::Invariant, &[]);
                            }
                            self.output_interpolation(var, sid)?;
                            written |= 1 << sid;
                        }
                        Sink::Colour { cbuf } => {
                            let location = self.layout.colour_output_location(cbuf);
                            self.m.decorate(var, Decoration::Location, &[location]);
                            if self.write_all_cbufs && cbuf == 0 {
                                self.copy_to_every_cbuf(v);
                            }
                        }
                        Sink::PointSize => unreachable!("matched above"),
                    }
                    self.interface.push(var);
                    self.f.op_void(Op::Store, &[var.word(), v.word()]);
                }
                Sink::PointSize => {
                    let ptr = self.m.type_pointer(StorageClass::Output, float);
                    let var = self.m.global_variable(ptr, StorageClass::Output);
                    self.m.builtin(var, BuiltIn::PointSize);
                    self.interface.push(var);
                    let x = self.f.op(&mut self.m, Op::CompositeExtract, float, &[v.word(), 0]);
                    self.f.op_void(Op::Store, &[var.word(), x.word()]);
                }
            }
        }
        if self.processor == Processor::Vertex {
            let mut missing = self.key.out_generic_expected_mask & !written;
            while missing != 0 {
                let sid = missing.trailing_zeros();
                missing &= missing - 1;
                let Some(location) = self.layout.varying_location(sid) else {
                    return decline("a generic varying past index 31");
                };
                let var = self.m.global_variable(self.t.out_vec4, StorageClass::Output);
                self.m.decorate(var, Decoration::Location, &[location]);
                self.output_interpolation(var, sid)?;
                self.interface.push(var);
                let zero = self.float(0.0);
                let zero = self.splat(vec4, zero);
                self.f.op_void(Op::Store, &[var.word(), zero.word()]);
            }
        }
        Ok(())
    }

    /// Colour 0's value to every other draw buffer the host has, as the GLSL's
    /// `fsout_c<i> = fsout_c0` does.
    fn copy_to_every_cbuf(&mut self, v: Id) {
        for cbuf in 1..self.cfg.max_draw_buffers {
            let var = self.m.global_variable(self.t.out_vec4, StorageClass::Output);
            let location = self.layout.colour_output_location(cbuf);
            self.m.decorate(var, Decoration::Location, &[location]);
            self.interface.push(var);
            self.f.op_void(Op::Store, &[var.word(), v.word()]);
        }
    }

    /// A vertex output carries the interpolation the fragment shader declares for it, as the
    /// GLSL's does from the key: GLES requires the two to agree.
    fn output_interpolation(&mut self, var: Id, sid: u32) -> Result<()> {
        let found =
            self.key.fs_info.interps.iter().find(|i| {
                i.semantic_name == Semantic::Generic && u32::from(i.semantic_index) == sid
            });
        match found {
            Some(i) => self.interpolation(var, i.interpolate, i.location),
            None => Ok(()),
        }
    }

    fn finish(mut self, info: &Info) -> Result<Vec<u32>> {
        if !self.ended {
            return decline("a shader with no END");
        }
        if self.processor == Processor::Vertex {
            self.vertex_epilogue(info)?;
        }
        if self.write_all_cbufs
            && self.outputs.values().any(|s| matches!(s, Sink::Colour { cbuf } if *cbuf != 0))
        {
            return decline("colour 0 written to every buffer beside another colour output");
        }
        self.write_outputs()?;
        self.f.op_void(Op::Return, &[]);
        let main = self.f.id;
        let model = match self.processor {
            Processor::Vertex => ExecutionModel::Vertex,
            _ => ExecutionModel::Fragment,
        };
        let Emitter { mut m, f, interface, early_fragment_tests, .. } = self;
        m.end_function(f);
        m.entry_point(model, main, "main", &interface);
        if model == ExecutionModel::Fragment {
            // GL's default; this emitter declares no gl_FragCoord for the choice to show in.
            m.execution_mode(main, ExecutionMode::OriginLowerLeft, &[]);
            if early_fragment_tests {
                m.execution_mode(main, ExecutionMode::EarlyFragmentTests, &[]);
            }
        }
        Ok(m.finish())
    }
}

/// `program` as a SPIR-V module, translated under `key` for `stage`, or why not. `info` is what
/// the GLSL translation of the same program said about it, which the draw path binds by: the
/// two must describe one shader, so this reads the counts from there rather than its own.
pub fn emit(
    cfg: &Config,
    program: &tgsi::Program,
    stage: ShaderStage,
    key: &Key,
    info: &Info,
    so: &StreamOutput,
) -> Result<Vec<u32>> {
    let processor = program.shader.processor;
    if !matches!(processor, Processor::Vertex | Processor::Fragment) {
        return decline("a stage other than vertex or fragment");
    }
    if !so.outputs.is_empty() {
        return decline("stream output");
    }
    if info.separable_program {
        return decline("a separable program");
    }
    // What the draw path binds by name for a GLSL program and by number for this one; each is
    // also refused where it is declared, and this keeps the two from disagreeing.
    if info.ubo_used_mask != 0 {
        return decline("a uniform buffer");
    }
    if !info.sampler_arrays.is_empty() || info.shadow_samp_mask != 0 {
        return decline("a sampler array or a shadow sampler");
    }
    if info.images_used_mask != 0 || info.ssbo_used_mask != 0 || info.reads_drawid {
        return decline("an image, a storage buffer or the draw id");
    }
    check_key(key, processor)?;
    let num_consts = u32::try_from(info.num_consts).unwrap_or(0);
    let mut e = Emitter::new(cfg, key, stage, processor);
    // Declared whenever the GLSL would declare it, read or not: the draw path uploads to it by
    // that count, and an upload to a location nothing declares is an error.
    if num_consts > 0 {
        e.constants(num_consts);
    }
    for token in &program.shader.tokens {
        if e.ended {
            break;
        }
        match token {
            Token::Declaration(d) => e.declare(d)?,
            Token::Immediate(imm) => e.immediate(imm)?,
            Token::Property(p) => e.property(p)?,
            Token::Instruction(inst) => e.instruction(inst, num_consts)?,
        }
    }
    // The draw path writes a unit to every sampler the GLSL translation counted; a module that
    // declared a different set would leave one unwritten or write to nothing.
    let sampled = e.samplers.keys().fold(0u32, |mask, &slot| mask | 1 << slot);
    if sampled != info.samplers_used_mask {
        return decline("a sampler declared and never sampled");
    }
    e.finish(info)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vrend::shader::spirv::writer::tests::validate;
    use crate::vrend::shader::{convert, keytext};
    use crate::vrend::tgsi::fixture;
    use std::collections::BTreeMap;

    /// Every block of the recorded fixtures with its inputs, through the emitter: what it
    /// writes must validate, and what it refuses is counted by reason so a run says how far the
    /// emitter reaches.
    #[test]
    fn every_module_the_emitter_writes_validates() {
        let corpora = [
            ("captures", fixture::capture_blocks()),
            ("piglit", fixture::piglit_blocks()),
            ("desktop", fixture::desktop_rs_blocks()),
        ];
        let mut written = 0;
        let mut invalid = Vec::new();
        for (name, blocks) in corpora {
            let (mut wrote, mut seen) = (0, 0);
            let mut declined: BTreeMap<Decline, u32> = BTreeMap::new();
            for (n, block) in blocks.iter().enumerate() {
                let Some(inputs) = &block.inputs else { continue };
                seen += 1;
                let shader = tgsi::text::parse(block.input().as_bytes(), u32::MAX).expect("parses");
                let program = tgsi::Program::scan(shader).expect("scans");
                let cfg = keytext::parse_config(inputs.config);
                let key = keytext::parse_key(inputs.key);
                let so = keytext::parse_stream_output(inputs.stream_output);
                let (_, info, _) =
                    convert(&cfg, &program, inputs.local_mem, &key, &so).expect("translates");
                let stage = match program.shader.processor {
                    Processor::Vertex => ShaderStage::Vertex,
                    Processor::Fragment => ShaderStage::Fragment,
                    _ => ShaderStage::Geometry,
                };
                match emit(&cfg, &program, stage, &key, &info, &so) {
                    Ok(words) => {
                        wrote += 1;
                        if let Some(Err(e)) = validate(&words) {
                            invalid.push((name, n, e));
                        }
                    }
                    Err(why) => *declined.entry(why).or_default() += 1,
                }
            }
            let mut by_count: Vec<_> =
                declined.into_iter().map(|(d, c)| (d.to_string(), c)).collect();
            by_count.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
            eprintln!("{name}: {wrote} of {seen} written; declined: {by_count:?}");
            written += wrote;
        }
        assert!(written > 0, "the emitter wrote nothing at all");
        assert!(
            invalid.is_empty(),
            "{} modules failed validation, the first ({} block {}):\n{}",
            invalid.len(),
            invalid[0].0,
            invalid[0].1,
            invalid[0].2
        );
    }
}
