// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! Whether the host takes SPIR-V in the context vrend runs, asked once at init by linking a
//! program written the way the emitter writes them: a varying at location 31, the top of the
//! separable numbering, and a uniform block at binding 80, the shared `VirglBlock` slot.
//!
//! A desktop GL 4.6 host has the path in core. A GLES host has it only with our host Mesa's
//! patch (limina's `docs/design/vrend-spirv.md` §4); without it the entry point resolves to a
//! stub that raises an error. Neither is visible in the extension string, so only a link
//! answers.

use spirv::{BuiltIn, Capability, Decoration, ExecutionMode, ExecutionModel, Op, StorageClass};

use super::writer::Module;
use crate::vrend::gl::gles::{
    GL_FRAGMENT_SHADER, GL_SHADER_BINARY_FORMAT_SPIR_V, GL_VERTEX_SHADER,
};
use crate::vrend::gl::{BoundProgram, Gl};

const VARYING: u32 = 31;
const BLOCK_BINDING: u32 = 80;

/// The probe's vertex shader: the position at the origin and the varying at one.
fn vertex() -> Vec<u32> {
    let mut m = Module::new();
    m.capability(Capability::Shader);
    let void = m.type_void();
    let float = m.type_float(32);
    let vec4 = m.type_vector(float, 4);
    let out_ptr = m.type_pointer(StorageClass::Output, vec4);
    let position = m.global_variable(out_ptr, StorageClass::Output);
    m.builtin(position, BuiltIn::Position);
    let varying = m.global_variable(out_ptr, StorageClass::Output);
    m.decorate(varying, Decoration::Location, &[VARYING]);
    let zero = m.constant(float, &[0.0f32.to_bits()]);
    let one = m.constant(float, &[1.0f32.to_bits()]);
    let origin = m.constant_composite(vec4, &[zero, zero, zero, one]);
    let ones = m.constant_composite(vec4, &[one, one, one, one]);
    let fn_type = m.type_function(void, &[]);
    let mut f = m.begin_function(void, fn_type);
    f.op_void(Op::Store, &[position.word(), origin.word()]);
    f.op_void(Op::Store, &[varying.word(), ones.word()]);
    f.op_void(Op::Return, &[]);
    let main = f.id;
    m.end_function(f);
    m.entry_point(ExecutionModel::Vertex, main, "main", &[position, varying]);
    m.finish()
}

/// The probe's fragment shader: the varying times the block's colour.
fn fragment() -> Vec<u32> {
    let mut m = Module::new();
    m.capability(Capability::Shader);
    let void = m.type_void();
    let float = m.type_float(32);
    let int = m.type_int(32, true);
    let vec4 = m.type_vector(float, 4);
    let in_ptr = m.type_pointer(StorageClass::Input, vec4);
    let out_ptr = m.type_pointer(StorageClass::Output, vec4);
    let varying = m.global_variable(in_ptr, StorageClass::Input);
    m.decorate(varying, Decoration::Location, &[VARYING]);
    let colour = m.global_variable(out_ptr, StorageClass::Output);
    m.decorate(colour, Decoration::Location, &[0]);
    let block = m.type_struct(&[vec4]);
    m.decorate(block, Decoration::Block, &[]);
    m.member_decorate(block, 0, Decoration::Offset, &[0]);
    let block_ptr = m.type_pointer(StorageClass::Uniform, block);
    let ubo = m.global_variable(block_ptr, StorageClass::Uniform);
    m.decorate(ubo, Decoration::Binding, &[BLOCK_BINDING]);
    let member_ptr = m.type_pointer(StorageClass::Uniform, vec4);
    let first = m.constant(int, &[0]);
    let fn_type = m.type_function(void, &[]);
    let mut f = m.begin_function(void, fn_type);
    let p = f.op(&mut m, Op::AccessChain, member_ptr, &[ubo.word(), first.word()]);
    let c = f.op(&mut m, Op::Load, vec4, &[p.word()]);
    let v = f.op(&mut m, Op::Load, vec4, &[varying.word()]);
    let o = f.op(&mut m, Op::FMul, vec4, &[c.word(), v.word()]);
    f.op_void(Op::Store, &[colour.word(), o.word()]);
    f.op_void(Op::Return, &[]);
    let main = f.id;
    m.end_function(f);
    m.entry_point(ExecutionModel::Fragment, main, "main", &[varying, colour]);
    m.execution_mode(main, ExecutionMode::OriginUpperLeft, &[]);
    m.finish()
}

/// Whether `gl`'s context takes SPIR-V programs, and why not when it does not.
pub fn probe(gl: &Gl) -> Result<(), String> {
    if !gl.shader_binary_formats().contains(&GL_SHADER_BINARY_FORMAT_SPIR_V) {
        return Err("GL_SHADER_BINARY_FORMATS does not name SPIR-V".to_string());
    }
    let program = gl.create_program().ok_or("glCreateProgram failed")?;
    let mut shaders = Vec::new();
    let mut linked = Ok(());
    for (kind, words) in [(GL_VERTEX_SHADER, vertex()), (GL_FRAGMENT_SHADER, fragment())] {
        let Some(shader) = gl.create_shader(kind) else {
            linked = Err("glCreateShader failed".to_string());
            break;
        };
        shaders.push(shader);
        if let Err(e) = gl.load_spirv(shader, &words) {
            linked = Err(e);
            break;
        }
        gl.attach_shader(program, shader);
    }
    if linked.is_ok() {
        linked = gl.link_program(program).map_err(|log| format!("the link failed: {log}"));
    }
    for shader in shaders {
        gl.delete_shader(shader);
    }
    // The program was never bound, so nothing records it as in use.
    gl.delete_program(&mut BoundProgram::default(), program);
    linked
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vrend::egl::{Flavour, Version, Winsys};
    use crate::vrend::features::Api;
    use crate::vrend::shader::spirv::writer::tests::validate;

    #[test]
    fn the_probe_modules_validate() {
        for (name, words) in [("vertex", vertex()), ("fragment", fragment())] {
            if let Some(verdict) = validate(&words) {
                verdict.unwrap_or_else(|e| panic!("spirv-val refused the {name} module:\n{e}"));
            }
        }
    }

    /// A desktop GL 4.6 host has SPIR-V in core, so the probe must pass there: a failure is the
    /// probe's or the writer's, not the host's.
    #[test]
    fn a_desktop_host_takes_the_probe_program() {
        let _display = crate::vrend::one_display_at_a_time();
        let winsys = Winsys::open(Flavour::Gl).expect("the surfaceless display opens");
        let version = Version { major: 4, minor: 6 };
        let ctx = winsys.create_context(version, None).expect("a 4.6 core context");
        winsys.make_current(&ctx).expect("current");
        let gl = Gl::new(winsys.procs(), Api::Gl(46));
        probe(&gl).unwrap_or_else(|why| panic!("desktop GL refused SPIR-V: {why}"));
    }

    /// A GLES host answers either way, by whether its Mesa carries the patch; what matters is
    /// that it answers rather than crashes, and that a refusal says why.
    #[test]
    fn a_gles_host_answers_the_probe() {
        let _display = crate::vrend::one_display_at_a_time();
        let winsys = Winsys::open(Flavour::Gles).expect("the surfaceless display opens");
        let version = Version { major: 3, minor: 1 };
        let ctx = winsys.create_context(version, None).expect("a 3.1 context");
        winsys.make_current(&ctx).expect("current");
        let gl = Gl::new(winsys.procs(), Api::Gles(31));
        match probe(&gl) {
            Ok(()) => eprintln!("this GLES host takes SPIR-V"),
            Err(why) => {
                assert!(!why.is_empty());
                eprintln!("this GLES host refuses SPIR-V: {why}");
            }
        }
    }
}
