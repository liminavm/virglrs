// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! Shader selection: the key the bound state makes for a stage, the variant that key names, and
//! the GL shader the variant compiles to.
//!
//! The C keeps each guest shader as a selector with a chain of variants, one per key it has
//! been translated under, the one last selected at the head. A stage is selected once when its
//! text completes and again whenever the program is assembled -- at `LINK_SHADER`, and before
//! a draw -- and the key it gets comes from everything around it at that moment: the
//! framebuffer, the blend, rasterizer and depth state, the views bound to the stage, and what
//! the neighbouring stages declared. That is `vrend_fill_shader_key` and `vrend_sync_shader_io`,
//! ported line for line, because the key decides which variant -- so which GLSL -- a draw runs,
//! and a key filled differently from the C is a different shader on the screen.
//!
//! `use_core_profile` is set on both flavours. `use_integer` is set on neither, so the signed and
//! unsigned attribute masks stay clear; the colour buffers' integer masks the C also keeps behind
//! it are always filled, since a float output into an integer target is undefined and zink on
//! KosmicKrisp writes zeros for it. Two separable stages meet by location and are not matched
//! against each other; every other pair is.

use super::*;

/// `vrend_shader_selector`'s translated half: the guest's program, and what translating it made.
pub struct Program {
    pub tgsi: tgsi::Program,
    pub translated: Translated,
}

/// What a stage's translations made, whoever wrote the program: what the latest one said about
/// it, and every variant it has been selected as. The neighbours and the link read a stage
/// through this, so they cannot tell a guest's program from one this host injected.
#[derive(Default)]
pub struct Translated {
    /// `sel->sinfo`: written by every translation, so it describes the newest variant. What
    /// it says that the key does not change is what the neighbours read.
    pub info: shader::Info,
    /// `sel->current` and its chain, the variant last selected first. Empty only after a
    /// selection failed, which is the C's `current = NULL`.
    pub variants: Vec<Variant>,
}

impl Translated {
    /// `sel->current->var_sinfo`, or the C's zeroed struct when there is no current variant.
    fn current_var_info(&self) -> shader::VarInfo {
        self.variants.first().map(|v| v.var_info.clone()).unwrap_or_default()
    }
}

/// A variant's name for as long as its sub-context lives: minted once and never reused, so a
/// program keyed on it cannot mean a later variant. The GL shader name is not that: the driver
/// hands a deleted name to the next shader it makes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct VariantId(u64);

/// `vrend_shader`: one translation of a program, under one key.
pub struct Variant {
    pub id: VariantId,
    pub key: shader::Key,
    pub strings: shader::Strings,
    pub var_info: shader::VarInfo,
    /// The compiled shader, once a program has been assembled with it.
    pub gl: Option<ShaderName>,
    /// A separable stage's own program, linked at its compile, which every pipeline the stage
    /// serves names.
    pub separate: Option<Separate>,
}

/// `vrend_shader`'s `program_id` and `last_pipeline_id`: a separable variant's program, and the
/// pipeline whose numbering its block bindings hold.
pub struct Separate {
    pub program: ProgramName,
    /// Which program last numbered this one's uniform blocks. Pipelines sharing the stage number
    /// them from different places, so a pipeline finding another's numbering here binds its own
    /// again. A serial is never reused, so one that died names nothing that can match.
    pub blocks_bound_for: Cell<Option<draw::ProgramSerial>>,
}

/// The control stage this host injects ahead of an evaluation shader the guest bound with none,
/// because GLES will not link one without the other: `vrend_inject_tcs`. The guest never sees
/// it, so it is no object of theirs and no binding; the sub-context holds the one in use.
///
/// The C makes a new one at every selection and leaks the last. This one is made again only when
/// what it is made from changed, and what it is made from is all here, compared whole -- so the
/// TCS a draw links cannot be one made for another vertex shader, patch size or factors.
pub struct Passthrough {
    made_from: PassthroughInputs,
    pub translated: Translated,
}

impl Passthrough {
    /// Whether this was made for a draw of `vertices` vertices a patch.
    pub fn made_for_patch(&self, vertices: GLsizei) -> bool {
        i64::from(self.made_from.vertices_per_patch) == i64::from(vertices)
    }
}

/// Everything `vrend_shader_create_passthrough_tcs` reads, but the translator's config, which a
/// renderer never changes.
#[derive(PartialEq)]
struct PassthroughInputs {
    /// The vertex shader's declarations: the outputs it writes are what the TCS copies.
    vs: Vec<tgsi::Declaration>,
    key: shader::Key,
    vertices_per_patch: u8,
    /// `SET_TESS_STATE`'s levels, by bits: the TCS writes them as constants.
    tess_factors: [u32; 6],
}

/// What a sub-context has bound at a stage, and when. The C holds a reference, so a shader the
/// guest destroys while it is bound stays bound until the next bind; it moves here from the
/// object table, and the slot is the one owner either way.
///
/// The slot is also what a rebuild binds from, as the sampler units are, and not the last
/// `BIND_SHADER` sent: the C ignores a bind naming no shader of the stage, so the last command
/// can name something the slot never held.
pub enum Bound {
    /// Bound from the table, by the command retained at `at`.
    Object {
        handle: ObjectHandle,
        at: Seq,
    },
    Owned(Box<Orphan>),
}

/// A shader destroyed while bound, with what a rebuild needs to reach the same state: its
/// create under the handle it had, the bind, and the destroy, each at the position it was
/// accepted. The handle may name another object by now; the destroy comes before that object's
/// create, so a rebuild frees it for the create exactly as the guest did.
pub struct Orphan {
    pub shader: Shader,
    handle: ObjectHandle,
    created: Retained,
    bound_at: Seq,
    destroyed_at: Seq,
}

impl Bound {
    /// Whether this is `handle`, bound from the table.
    pub fn is(&self, handle: ObjectHandle) -> bool {
        matches!(self, Bound::Object { handle: h, .. } if *h == handle)
    }

    /// The shader bound here leaves the table, destroyed at `destroyed_at`: the slot takes it,
    /// and the create the table retained for it.
    pub fn orphan(&mut self, shader: Shader, created: Retained, destroyed_at: Seq) {
        let Bound::Object { handle, at } = *self else {
            panic!("only a shader bound from the table can leave it");
        };
        *self =
            Bound::Owned(Box::new(Orphan { shader, handle, created, bound_at: at, destroyed_at }));
    }

    /// The commands that put this binding back in a fresh sub-context, each at its position.
    pub fn rebuild(&self, stage: ShaderStage) -> Vec<(Seq, Cow<'_, [Vec<u32>]>)> {
        let bind = |handle| {
            let mut wire = Vec::new();
            encode::encode(&Command::BindShader { stage, handle: Some(handle) }, &mut wire);
            Cow::Owned(vec![wire])
        };
        match self {
            Bound::Object { handle, at } => vec![(*at, bind(*handle))],
            Bound::Owned(o) => {
                let mut destroy = Vec::new();
                let kind = ObjectType::Shader;
                encode::encode(&Command::DestroyObject { kind, handle: o.handle }, &mut destroy);
                vec![
                    (o.created.seq, Cow::Borrowed(o.created.chunks.as_slice())),
                    (o.bound_at, bind(o.handle)),
                    (o.destroyed_at, Cow::Owned(vec![destroy])),
                ]
            }
        }
    }
}

fn gl_shader_kind(stage: ShaderStage) -> GLenum {
    match stage {
        ShaderStage::Vertex => GL_VERTEX_SHADER,
        ShaderStage::Fragment => GL_FRAGMENT_SHADER,
        ShaderStage::Geometry => GL_GEOMETRY_SHADER,
        ShaderStage::TessCtrl => GL_TESS_CONTROL_SHADER,
        ShaderStage::TessEval => GL_TESS_EVALUATION_SHADER,
        ShaderStage::Compute => GL_COMPUTE_SHADER,
    }
}

/// `can_emulate_logicop`: whether the fragment shader can do the blend's logic op, which needs
/// the framebuffer read back unless the op ignores it.
pub(super) fn can_emulate_logicop(features: &Features, op: LogicOp) -> bool {
    if features.has(Feature::framebuffer_fetch_non_coherent)
        || features.has(Feature::framebuffer_fetch)
    {
        return true;
    }
    matches!(op, LogicOp::Clear | LogicOp::Copy | LogicOp::Set | LogicOp::CopyInverted)
}

/// `util_format_get_component_bits(format, UTIL_FORMAT_COLORSPACE_RGB, component)`: the width
/// of the channel the component reads, in the RGB colour space alone -- an sRGB or depth format
/// answers zero, as gallium's does.
fn component_bits(desc: &Description, component: usize) -> u8 {
    if desc.colorspace != super::super::formats::Colorspace::Rgb {
        return 0;
    }
    match desc.swizzle[component] {
        Some(Swizzle::X) => desc.channels[0].bits as u8,
        Some(Swizzle::Y) => desc.channels[1].bits as u8,
        Some(Swizzle::Z) => desc.channels[2].bits as u8,
        Some(Swizzle::W) => desc.channels[3].bits as u8,
        _ => 0,
    }
}

/// `vrend_get_swizzle`, the GLES and core-profile legs: the swizzle a buffer texture's format
/// asks of the shader, since a buffer texture has no swizzle state of its own.
fn buffer_swizzle(formats: &Table, format: Format) -> Option<[Swizzle; 4]> {
    if let Some(sw) = formats.get(format).and_then(|e| e.gl.swizzle) {
        return Some(sw);
    }
    use Swizzle::*;
    const OOOR: [Swizzle; 4] = [Zero, Zero, Zero, X];
    const RRR1: [Swizzle; 4] = [X, X, X, One];
    const RRRG: [Swizzle; 4] = [X, X, X, Y];
    const RRRR: [Swizzle; 4] = [X, X, X, X];
    const RG01: [Swizzle; 4] = [X, Y, Zero, One];
    const R001: [Swizzle; 4] = [X, Zero, Zero, One];
    Some(match format.describe()?.name {
        "A8_UNORM" | "A16_FLOAT" | "A32_FLOAT" => OOOR,
        "L8_UNORM" | "L8_SINT" | "L8_UINT" | "L16_UNORM" | "L16_SINT" | "L16_UINT"
        | "L16_FLOAT" | "L32_SINT" | "L32_UINT" | "L32_FLOAT" => RRR1,
        "L8A8_SINT" | "L8A8_UINT" | "L16A16_SINT" | "L16A16_UINT" | "L16A16_FLOAT"
        | "L32A32_FLOAT" | "L32A32_SINT" | "L32A32_UINT" | "L8A8_UNORM" | "L16A16_UNORM" => RRRG,
        "I8_UNORM" | "I8_SINT" | "I8_UINT" | "I16_UNORM" | "I16_SINT" | "I16_UINT"
        | "I16_FLOAT" | "I32_FLOAT" | "I32_SINT" | "I32_UINT" => RRRR,
        "R32G32_FLOAT" | "R32G32_UINT" | "R32G32_SINT" | "R16G16_FLOAT" | "R16G16_UINT"
        | "R16G16_SINT" | "R16G16_SNORM" | "R16G16_UNORM" | "R8G8_UINT" | "R8G8_SINT"
        | "R8G8_SNORM" | "R8G8_UNORM" | "R8G8_SSCALED" | "R8G8_USCALED" => RG01,
        "R32_FLOAT" | "R32_UINT" | "R32_SINT" | "R16_FLOAT" | "R16_UINT" | "R16_SINT"
        | "R16_SNORM" | "R16_UNORM" | "R8_UINT" | "R8_SINT" | "R8_SNORM" | "R8_UNORM"
        | "R8_SSCALED" | "R8_USCALED" => R001,
        _ => return None,
    })
}

/// `vrend_compile_shader`: a GL shader from the variant's GLSL. The driver's log is printed
/// with the numbered source when it refuses, as the C prints it, so the line it names can be
/// read.
fn compile(gl: &Gl, stage: ShaderStage, variant: &mut Variant) -> bool {
    let Some(id) = gl.create_shader(gl_shader_kind(stage)) else {
        return false;
    };
    let source = variant.strings.source();
    match gl.compile_shader(id, &source) {
        Ok(()) => {
            variant.gl = Some(id);
            true
        }
        Err(log) => {
            let mut numbered = String::new();
            for (n, line) in source.lines().enumerate() {
                numbered.push_str(&format!("{:4}: {line}\n", n + 1));
            }
            eprintln!("[virglrs] vrend: shader failed to compile\n{log}\n{numbered}");
            gl.delete_shader(id);
            false
        }
    }
}

impl SubContext {
    fn mint_variant_id(&mut self) -> VariantId {
        let id = VariantId(self.next_variant_id);
        self.next_variant_id += 1;
        id
    }

    /// Whether the pipeline has a stage at `stage`: the guest bound one there, or it is the
    /// control stage this host injects ahead of an evaluation shader that has none.
    pub(super) fn has_stage(&self, stage: ShaderStage) -> bool {
        self.shaders[stage.index()].is_some()
            || (stage == ShaderStage::TessCtrl && self.injects_tcs())
    }

    /// `vrend_select_program`'s GLES test for `vrend_inject_tcs`: an evaluation shader with no
    /// control shader before it, which GLES will not link.
    pub(super) fn injects_tcs(&self) -> bool {
        self.tcs_required
            && self.shaders[ShaderStage::TessCtrl.index()].is_none()
            && self.shaders[ShaderStage::TessEval.index()].is_some()
    }

    /// What was translated for the stage at `stage`, when the pipeline has one and its text is
    /// whole: the guest's program, or the control stage this host injected.
    pub(super) fn bound_program(&self, stage: ShaderStage) -> Option<&Translated> {
        if stage == ShaderStage::TessCtrl && self.injects_tcs() {
            return self.passthrough.as_ref().map(|p| &p.translated);
        }
        self.bound_tgsi(stage).map(|p| &p.translated)
    }

    /// Whether the shader being selected -- `handle` from the table, or the one bound at
    /// `stage` -- is separable: the C's `sel->sinfo.separable_program`.
    fn selected_separable(&self, handle: Option<ObjectHandle>, stage: ShaderStage) -> bool {
        let program = match handle {
            Some(h) => match self.objects.get(&h) {
                Some(Object::Shader(Shader { text: ShaderText::Whole(p), .. })) => {
                    Some(&p.translated)
                }
                _ => None,
            },
            None => self.bound_program(stage),
        };
        program.is_some_and(|p| p.info.separable_program)
    }

    /// The guest's program bound at `stage`, when its text is whole.
    fn bound_tgsi(&self, stage: ShaderStage) -> Option<&Program> {
        let shader = match self.shaders[stage.index()].as_ref()? {
            Bound::Object { handle, .. } => match self.objects.get(handle) {
                Some(Object::Shader(s)) => s,
                _ => return None,
            },
            Bound::Owned(o) => &o.shader,
        };
        match &shader.text {
            ShaderText::Whole(p) => Some(p),
            ShaderText::Arriving { .. } => None,
        }
    }

    fn bound_shader_mut(&mut self, stage: ShaderStage) -> Option<&mut Shader> {
        match self.shaders[stage.index()].as_mut()? {
            Bound::Object { handle, .. } => match self.objects.get_mut(handle) {
                Some(Object::Shader(s)) => Some(s),
                _ => None,
            },
            Bound::Owned(o) => Some(&mut o.shader),
        }
    }

    /// `vrend_sync_shader_io`: what the stages either side ask of this one.
    fn sync_shader_io(
        &self,
        features: &Features,
        handle: Option<ObjectHandle>,
        stage: ShaderStage,
        key: &mut shader::Key,
    ) {
        use ShaderStage::*;
        let mut prev_type = (stage != Vertex).then_some(Vertex);
        // Gallium sends and binds the shaders in reverse order, so an old shader still bound
        // at this stage says nothing about the one before it -- unless the bound one is this.
        let is_bound = match handle {
            Some(h) => self.shaders[stage.index()].as_ref().is_some_and(|b| b.is(h)),
            None => true,
        };
        if is_bound {
            match stage {
                Geometry if key.tcs_present || key.tes_present => prev_type = Some(TessEval),
                Fragment if key.gs_present => prev_type = Some(Geometry),
                Fragment if key.tcs_present || key.tes_present => prev_type = Some(TessEval),
                TessEval if key.tcs_present => prev_type = Some(TessCtrl),
                _ => {}
            }
        }
        let separable = self.selected_separable(handle, stage);
        let prev = prev_type.and_then(|t| self.bound_program(t).map(|p| (t, p)));
        if let Some((_, prev)) = prev {
            // Two separable stages meet by location: neither is translated to the other.
            if !prev.info.separable_program || !separable {
                key.require_input_arrays = prev.info.has_output_arrays;
                key.in_generic_expected_mask = prev.info.out_generic_emitted_mask;
                key.in_texcoord_expected_mask = u64::from(prev.info.out_texcoord_emitted_mask);
                key.in_patch_expected_mask = prev.info.out_patch_emitted_mask;
                key.in_arrays = prev.info.output_arrays.clone();
                key.force_invariant_inputs = prev.info.invariant_outputs;
            }
            key.ssbo_binding_offset = (prev.info.ssbo_last_binding + 1) as u8;
            key.image_binding_offset = (prev.info.image_last_binding + 1) as u8;
            let var = prev.current_var_info();
            key.num_in_clip = var.num_out_clip;
            key.num_in_cull = var.num_out_cull;
            if stage == Fragment && features.api().is_gles() {
                key.fs.available_color_in_bits = var.legacy_color_bits as u8;
            }
        }

        let mut next_type = None;
        if stage == Fragment {
            key.fs.lower_left_origin = !self.fbo_origin_upper_left;
            key.fs.swizzle_output_rgb_to_bgr = self.swizzle_output_rgb_to_bgr;
            key.fs.needs_manual_srgb_encode_bitmask = self.needs_manual_srgb_encode;
            // GLES has no logic op, so the shader does it; desktop GL does it in the blender.
            if let Some(blend) = self.blend
                && features.api().is_gles()
                && blend.logicop_enable
                && can_emulate_logicop(features, blend.logicop_func)
            {
                key.fs.logicop_func = Some(blend.logicop_func);
            }
            // The draw's mode, unless the stage before decides what reaches the rasterizer.
            let mut fs_prim_mode = self.prim_mode;
            if let Some((prev_stage, prev)) = prev {
                match prev_stage {
                    TessEval if prev.info.tes_point_mode => fs_prim_mode = PrimType::Points,
                    Geometry => fs_prim_mode = prev.info.gs_out_prim.unwrap_or(PrimType::Points),
                    _ => {}
                }
            }
            key.fs.prim_is_points = fs_prim_mode == PrimType::Points;
            let rs = self.rs_state();
            key.fs.coord_replace = if rs.point_quad_rasterization && key.fs.prim_is_points {
                rs.sprite_coord_enable
            } else {
                0
            };
        } else if self.shaders[Fragment.index()].is_some() {
            next_type = Some(Fragment);
        }
        match stage {
            Vertex => {
                if key.tcs_present {
                    next_type = Some(TessCtrl);
                } else if key.gs_present {
                    next_type = Some(Geometry);
                } else if key.tes_present {
                    // GLES: a TCS is injected before the TES, and it is the vertex stage's next.
                    next_type = Some(if self.tcs_required { TessCtrl } else { TessEval });
                }
            }
            TessCtrl => next_type = Some(TessEval),
            TessEval if key.gs_present => next_type = Some(Geometry),
            _ => {}
        }
        if let Some(next_type) = next_type
            && let Some(next) = self.bound_program(next_type)
            && (!next.info.separable_program || !separable)
        {
            key.use_pervertex_in = next.info.use_pervertex_in;
            key.require_output_arrays = next.info.has_input_arrays;
            key.out_generic_expected_mask = next.info.in_generic_emitted_mask;
            key.out_texcoord_expected_mask = next.info.in_texcoord_emitted_mask;
            if next_type == Fragment {
                // The fragment stage takes the clip and cull counts from the key instead, so
                // this stage need not be re-translated for them.
                key.fs_info = next.current_var_info().fs_info;
                if stage == Vertex
                    && let Some(vs) = self.bound_program(Vertex)
                {
                    // Only the fog inputs the stage before does not feed are fixed up.
                    let fog_input = next.info.fog_input_mask;
                    let fog_output = vs.info.fog_output_mask;
                    key.vs.fog_fixup_mask = (fog_input ^ fog_output) & fog_input;
                }
            } else {
                let var = next.current_var_info();
                key.num_out_clip = var.num_in_clip;
                key.num_out_cull = var.num_in_cull;
            }
        }
    }

    /// `vrend_fill_shader_key`: the key for `stage` under the state bound now. `handle` names
    /// the shader when it is selected from the table; a selection of the bound one passes
    /// `None`.
    fn fill_shader_key(
        &self,
        host: &Host<'_>,
        handle: Option<ObjectHandle>,
        stage: ShaderStage,
    ) -> shader::Key {
        use ShaderStage::*;
        let mut key = shader::Key::default();
        if stage == Fragment {
            let mut add_alpha_test = true;
            let logicop = self.blend.is_some_and(|b| b.logicop_enable);
            for (i, surf) in self.cbufs.iter().enumerate() {
                let Some(surf) = surf else {
                    continue;
                };
                if crate::vrend::formats::is_emulated_alpha(host.formats, surf.format) {
                    key.fs.cbufs_are_a8_bitmask |= 1 << i;
                }
                let Some(desc) = surf.format.describe() else {
                    continue;
                };
                if desc.is_pure_integer() {
                    add_alpha_test = false;
                    // The C's `UPDATE_INT_SIGN_MASK`, without its `use_integer` gate.
                    if desc.is_pure_uint() {
                        key.fs.cbufs_unsigned_int_bitmask |= 1 << i;
                    } else {
                        key.fs.cbufs_signed_int_bitmask |= 1 << i;
                    }
                }
                // Read only under a logic op, as the C reads it.
                if logicop {
                    key.fs.surface_component_bits[i] = component_bits(desc, 0);
                }
            }
            if add_alpha_test {
                let dsa = self.dsa_state();
                key.add_alpha_test = dsa.alpha.enabled;
                key.alpha_test = dsa.alpha.func;
            }
        }
        let rs = self.rs_state();
        key.pstipple_enabled = rs.poly_stipple_enable;
        key.color_two_side = rs.light_twoside;
        key.flatshade = rs.flatshade;
        if stage == Vertex
            && let Some(Object::VertexElements(ve)) = self.ve.and_then(|h| self.objects.get(&h))
        {
            key.vs.attrib_zyxw_bitmask = ve.zyxw_bitmask;
        }
        key.gs_present = self.shaders[Geometry.index()].is_some() || stage == Geometry;
        key.tcs_present = self.has_stage(TessCtrl) || stage == TessCtrl;
        key.tes_present = self.shaders[TessEval.index()].is_some() || stage == TessEval;
        if stage != Compute {
            self.sync_shader_io(host.features, handle, stage, &mut key);
        }
        if stage == Geometry {
            key.gs.emit_clip_distance = rs.clip_plane_enable != 0;
        }
        for (slot, h) in self.units[stage.index()].views() {
            let Some(Object::SamplerView(view)) = self.objects.get(&h) else {
                continue;
            };
            let i = slot as usize;
            if view.emulated_rect {
                shader::Key::view_mask_set(&mut key.sampler_views_emulated_rect_mask, i);
            }
            // A 2D_ARRAY sampler over a plain 2D view: the shader is compiled to the view.
            if view.target == GL_TEXTURE_2D {
                shader::Key::view_mask_set(&mut key.sampler_views_lower_array_mask, i);
            }
            if view.target == GL_TEXTURE_BUFFER
                && let Some(sw) = buffer_swizzle(host.formats, view.format)
            {
                shader::Key::view_mask_set(&mut key.sampler_views_lower_swizzle_mask, i);
                key.tex_swizzle[i] =
                    sw[0] as u16 | (sw[1] as u16) << 3 | (sw[2] as u16) << 6 | (sw[3] as u16) << 9;
            }
        }
        key
    }
}

/// `vrend_shader_create`: the variant for `key`, translated and printed when asked, in the C's
/// format so the two logs diff. A refused translation empties the chain, as the C's
/// `current = NULL` does, and hands back what it emptied: those variants own GL names, which only
/// [`draw::release_variants`] may delete, so [`Context::release_refused`] is where they go.
fn translate(
    host: &Host<'_>,
    cmd: Cmd,
    shader: &mut Shader,
    key: shader::Key,
    id: VariantId,
) -> Result<(), Refused> {
    let refuse = |fault| Refused { fault, emptied: Vec::new() };
    let stage = shader.stage;
    let ShaderText::Whole(program) = &mut shader.text else {
        return Err(refuse(Fault::Shader {
            cmd,
            what: "a shader selected before its text was whole",
        }));
    };
    let none = StreamOutput::default();
    let (req_local_mem, so_info) = match &shader.kind {
        ShaderKind::Compute { req_local_mem } => (*req_local_mem, &none),
        ShaderKind::Graphics { stream_output } => (0, stream_output),
    };
    let log = host.debug.enabled(debug::Switch::Shader);
    if log {
        eprint!("TGSI received:\n{}\n", tgsi::dump::dump(&program.tgsi.shader));
    }
    match shader::convert(host.shader_cfg, &program.tgsi, req_local_mem, &key, so_info) {
        Ok((strings, info, var_info)) => {
            if log {
                eprint!("GLSL:\n{}\n", strings.source());
            }
            program.translated.info = info;
            program
                .translated
                .variants
                .insert(0, Variant { id, key, strings, var_info, gl: None, separate: None });
            Ok(())
        }
        Err(error) => Err(Refused {
            fault: Fault::Glsl { cmd, stage, error },
            emptied: std::mem::take(&mut program.translated.variants),
        }),
    }
}

/// A translation refused: the fault, and the variants the refusal emptied out of the chain.
#[must_use]
struct Refused {
    fault: Fault,
    emptied: Vec<Variant>,
}

/// `vrend_shader_select`'s cache: the variant for `key` moved to the head when the chain has
/// it, or `None` when it must be made.
fn select_variant(shader: &mut Shader, key: &shader::Key) -> bool {
    let ShaderText::Whole(program) = &mut shader.text else {
        return false;
    };
    let variants = &mut program.translated.variants;
    let Some(i) = variants.iter().position(|v| v.key == *key) else {
        return false;
    };
    if i != 0 {
        let v = variants.remove(i);
        variants.insert(0, v);
    }
    true
}

impl Context {
    /// `vrend_shader_select` on a shader in the table: the moment its text is whole, under
    /// whatever is bound then.
    pub(super) fn select_object(
        &mut self,
        host: &mut Host<'_>,
        cmd: Cmd,
        handle: ObjectHandle,
    ) -> Result<(), Fault> {
        let sub = self.sub();
        let Some(Object::Shader(shader)) = sub.objects.get(&handle) else {
            return Err(Fault::IllegalHandle { cmd, handle });
        };
        let key = sub.fill_shader_key(host, Some(handle), shader.stage);
        let sub = self.sub_mut();
        let id = sub.mint_variant_id();
        let Some(Object::Shader(shader)) = sub.objects.get_mut(&handle) else {
            unreachable!("looked up a moment ago");
        };
        if select_variant(shader, &key) {
            return Ok(());
        }
        let refused = translate(host, cmd, shader, key, id);
        self.release_refused(host, refused)
    }

    /// A refused translation's fault, once the variants it emptied are released -- their GL
    /// shaders, their own programs, and every program linking them.
    fn release_refused(
        &mut self,
        host: &mut Host<'_>,
        r: Result<(), Refused>,
    ) -> Result<(), Fault> {
        let Err(Refused { fault, emptied }) = r else {
            return Ok(());
        };
        draw::release_variants(self.sub_mut(), host.gl, host.current.program(), emptied);
        Err(fault)
    }

    /// `vrend_shader_select` on the shader bound at `stage`. Answers whether a variant had to be
    /// translated, which is the expensive case; a key the chain already holds costs a fill and a
    /// compare.
    fn select_bound(
        &mut self,
        host: &mut Host<'_>,
        cmd: Cmd,
        stage: ShaderStage,
    ) -> Result<bool, Fault> {
        let sub = self.sub();
        let handle = match sub.shaders[stage.index()].as_ref() {
            Some(Bound::Object { handle, .. }) => Some(*handle),
            Some(Bound::Owned(_)) => None,
            None => return Ok(false),
        };
        let key = sub.fill_shader_key(host, handle, stage);
        let sub = self.sub_mut();
        let id = sub.mint_variant_id();
        let Some(shader) = sub.bound_shader_mut(stage) else {
            let handle = handle.expect("a bound shader that is not owned is in the table");
            return Err(Fault::IllegalHandle { cmd, handle });
        };
        if select_variant(shader, &key) {
            return Ok(false);
        }
        let refused = translate(host, cmd, shader, key, id);
        self.release_refused(host, refused)?;
        Ok(true)
    }

    /// `vrend_select_program`, as far as the variants: every bound stage selected under the
    /// state of the moment, in the C's order -- the fragment stage last, then each again, since
    /// a stage's key reads its neighbours' newest variants -- and compiled. The program that
    /// links them is `Context::select_linked_program`'s. Answers whether any stage was
    /// translated or compiled.
    ///
    /// `vertices_per_patch` is the draw's, which an injected control stage declares as its
    /// output patch size.
    pub(super) fn select_program(
        &mut self,
        host: &mut Host<'_>,
        cmd: Cmd,
        vertices_per_patch: u32,
    ) -> Result<bool, Fault> {
        use ShaderStage::*;
        let bound = |s: &Context, stage: ShaderStage| s.sub().shaders[stage.index()].is_some();
        if !bound(self, Vertex) || !bound(self, Fragment) {
            return Err(Fault::Shader {
                cmd,
                what: "a program without a vertex and a fragment shader",
            });
        }
        let mut built = self.select_bound(host, cmd, Vertex)?;
        built |= self.select_tcs(host, cmd, vertices_per_patch)?;
        built |= self.select_bound(host, cmd, TessEval)?;
        built |= self.select_bound(host, cmd, Geometry)?;
        built |= self.select_bound(host, cmd, Fragment)?;
        // The C's second round, its workaround for duplicated compilation (#180).
        built |= self.select_bound(host, cmd, Geometry)?;
        built |= self.select_bound(host, cmd, TessEval)?;
        built |= self.select_tcs(host, cmd, vertices_per_patch)?;
        built |= self.select_bound(host, cmd, Vertex)?;

        for stage in [Vertex, Fragment, Geometry, TessCtrl, TessEval] {
            built |= self.compile_bound(host, cmd, stage)?;
        }
        Ok(built)
    }

    /// The control stage: the guest's, or the one injected ahead of an evaluation shader that
    /// has none. Answers whether anything was translated or compiled.
    fn select_tcs(
        &mut self,
        host: &mut Host<'_>,
        cmd: Cmd,
        vertices_per_patch: u32,
    ) -> Result<bool, Fault> {
        if self.sub().injects_tcs() {
            self.select_passthrough(host, cmd, vertices_per_patch)
        } else {
            self.select_bound(host, cmd, ShaderStage::TessCtrl)
        }
    }

    /// `vrend_inject_tcs`: the control stage for the bound vertex shader under the key and the
    /// tessellation state of the moment, translated and compiled as the C does at once, and kept
    /// until what it was made from changes. Answers whether it was made.
    fn select_passthrough(
        &mut self,
        host: &mut Host<'_>,
        cmd: Cmd,
        vertices_per_patch: u32,
    ) -> Result<bool, Fault> {
        let stage = ShaderStage::TessCtrl;
        let vertices_per_patch = u8::try_from(vertices_per_patch)
            .map_err(|_| Fault::OutOfRange { cmd, what: "a patch size" })?;
        let sub = self.sub();
        let Some(vs) = sub.bound_tgsi(ShaderStage::Vertex) else {
            return Err(Fault::Shader {
                cmd,
                what: "a control stage for a vertex shader not whole",
            });
        };
        let made_from = PassthroughInputs {
            vs: vs
                .tgsi
                .shader
                .tokens
                .iter()
                .filter_map(|t| match t {
                    tgsi::Token::Declaration(d) => Some(*d),
                    _ => None,
                })
                .collect(),
            key: sub.fill_shader_key(host, None, stage),
            vertices_per_patch,
            tess_factors: sub.tess_factors.map(f32::to_bits),
        };
        if sub.passthrough.as_ref().is_some_and(|p| p.made_from == made_from) {
            return Ok(false);
        }
        let (strings, info) = shader::create_passthrough_tcs(
            host.shader_cfg,
            &vs.tgsi.shader,
            &made_from.key,
            &sub.tess_factors,
            vertices_per_patch,
        )
        .map_err(|error| Fault::Glsl { cmd, stage, error })?;
        let sub = self.sub_mut();
        let id = sub.mint_variant_id();
        let key = made_from.key.clone();
        let mut variant = Variant {
            id,
            key,
            strings,
            var_info: shader::VarInfo::default(),
            gl: None,
            separate: None,
        };
        if !compile(host.gl, stage, &mut variant) {
            return Err(Fault::Shader {
                cmd,
                what: "an injected control shader the driver refused",
            });
        }
        let translated = Translated { info, variants: vec![variant] };
        if let Some(old) = sub.passthrough.replace(Passthrough { made_from, translated }) {
            draw::release_variants(sub, host.gl, host.current.program(), old.translated.variants);
        }
        Ok(true)
    }

    /// `vrend_compile_shader` on the current variant of the shader bound at `stage`, when it has
    /// no GL shader yet -- and, for a separable stage on a host with pipelines, its own program
    /// linked. Answers whether it compiled one.
    fn compile_bound(
        &mut self,
        host: &mut Host<'_>,
        cmd: Cmd,
        stage: ShaderStage,
    ) -> Result<bool, Fault> {
        let gl = host.gl;
        let separable_host = host.shader_cfg.serves_separable();
        let dual_src = self.sub().blend.as_ref().is_some_and(|b| draw::blend_is_dual(b, 0));
        let Some(shader) = self.sub_mut().bound_shader_mut(stage) else {
            return Ok(false);
        };
        let ShaderText::Whole(program) = &mut shader.text else {
            return Ok(false);
        };
        let Some(current) = program.translated.variants.first_mut() else {
            return Err(Fault::Shader { cmd, what: "a stage with no variant to compile" });
        };
        if current.gl.is_some() {
            return Ok(false);
        }
        if !compile(gl, stage, current) {
            return Err(Fault::Shader { cmd, what: "a shader the driver refused to compile" });
        }
        if separable_host && program.translated.info.separable_program {
            let compiled = current.gl.expect("compiled a moment ago");
            let info = &program.translated.info;
            let made = draw::link_separable(host, cmd, stage, info, compiled, dual_src)?;
            let current = program.translated.variants.first_mut().expect("compiled a moment ago");
            current.separate = Some(Separate { program: made, blocks_bound_for: Cell::new(None) });
        }
        Ok(true)
    }

    /// `vrend_launch_grid`'s selection, as far as the variant: the compute shader selected under
    /// the state of the moment and compiled. Answers whether it was translated or compiled.
    ///
    /// The C selects only when a bind marked the stage dirty (`cs_shader_dirty`), so a dispatch
    /// after a change to what the compute key reads -- the views bound to the stage -- runs the
    /// variant selected before it. A dispatch is rare beside a draw, so this one fills the key
    /// every time and keeps no flag that could be left standing or forgotten.
    pub(super) fn select_compute(&mut self, host: &mut Host<'_>, cmd: Cmd) -> Result<bool, Fault> {
        if self.sub().shaders[ShaderStage::Compute.index()].is_none() {
            return Err(Fault::Shader { cmd, what: "a dispatch with no compute shader" });
        }
        let translated = self.select_bound(host, cmd, ShaderStage::Compute)?;
        let compiled = self.compile_bound(host, cmd, ShaderStage::Compute)?;
        Ok(translated || compiled)
    }

    /// `vrend_link_program_hook`: the program the handles name, assembled now rather than at
    /// its first draw. The handles are bound for the duration and what was bound before is put
    /// back after.
    pub(super) fn link_shader(
        &mut self,
        host: &mut Host<'_>,
        handles: [Option<ObjectHandle>; ShaderStage::COUNT],
    ) -> Result<(), Fault> {
        use ShaderStage::*;
        let cmd = Cmd::LinkShader;
        // Pre-compiling a compute shader needs more than this does.
        if handles[Compute.index()].is_some() {
            return Ok(());
        }
        // Nothing to link without both ends; and a control shader cannot be linked without
        // its evaluation shader.
        if handles[Vertex.index()].is_none() || handles[Fragment.index()].is_none() {
            return Ok(());
        }
        if handles[TessCtrl.index()].is_some() && handles[TessEval.index()].is_none() {
            return Ok(());
        }
        let prev = std::mem::take(&mut self.sub_mut().shaders);
        for (i, h) in handles.iter().enumerate() {
            let stage = ShaderStage::from_wire(i as u32).expect("six stages");
            self.bind_shader(host, *h, stage);
        }
        // The C links with a patch size of one; a draw selects again with its own.
        let r = self.select_linked_program(host, cmd, 1).map(|_| ());
        let sub = self.sub_mut();
        sub.shaders = prev;
        sub.shader_dirty = true;
        r
    }
}
