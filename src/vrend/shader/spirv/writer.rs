// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! A SPIR-V module, written word by word: the binary encoding, the logical layout's sections in
//! their required order, types and constants made once each, and function bodies whose variables
//! the spec requires at the top of the entry block. What the words mean -- opcodes, enumerants --
//! comes from the `spirv` crate, which is generated from the Khronos grammar.
//!
//! It checks nothing about the module's meaning: an emitter that writes an ill-typed operation
//! gets an ill-typed module. `spirv-val` is the check (the tests run it), and a module the host
//! refuses falls back to GLSL.

use std::collections::HashMap;
use std::num::NonZeroU32;

use spirv::{
    AddressingModel, BuiltIn, Capability, Decoration, Dim, ExecutionMode, ExecutionModel,
    FunctionControl, MemoryModel, Op, StorageClass,
};

/// A result id: what one instruction defines and others refer to. Zero is not an id.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Id(NonZeroU32);

impl Id {
    /// The id as an operand word.
    pub fn word(self) -> u32 {
        self.0.get()
    }
}

/// A type as the module dedups it. Structs are left out: two of one shape can carry different
/// decorations, so each is made fresh.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum TypeKey {
    Void,
    Bool,
    Int {
        width: u32,
        signed: bool,
    },
    Float {
        width: u32,
    },
    Vector {
        component: Id,
        count: u32,
    },
    Array {
        element: Id,
        length: Id,
    },
    Pointer {
        class: StorageClass,
        pointee: Id,
    },
    Function {
        ret: Id,
        params: Vec<Id>,
    },
    /// A sampled (not storage) image of unknown format, neither depth nor multisampled.
    Image {
        sampled_type: Id,
        dim: Dim,
        arrayed: bool,
    },
    SampledImage {
        image: Id,
    },
}

/// One instruction's words appended to `out`: the word count and opcode, then the operands.
fn instruction(out: &mut Vec<u32>, op: Op, operands: &[u32]) {
    let count = u32::try_from(operands.len() + 1).expect("an instruction's operands fit");
    assert!(count <= 0xffff, "an instruction is at most 65535 words");
    out.push(count << 16 | op as u32);
    out.extend_from_slice(operands);
}

/// A literal string's words: UTF-8, nul-terminated, zero-padded to a whole word, little end
/// first in each word.
fn string_words(s: &str) -> Vec<u32> {
    assert!(!s.contains('\0'), "a SPIR-V string holds no nul");
    let mut bytes = s.as_bytes().to_vec();
    bytes.push(0);
    while !bytes.len().is_multiple_of(4) {
        bytes.push(0);
    }
    bytes.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect()
}

/// A module under construction. Each field is one section of the logical layout, written in
/// this order by [`Module::finish`].
#[derive(Default)]
pub struct Module {
    next_id: u32,
    capabilities: Vec<u32>,
    capability_set: Vec<Capability>,
    ext_inst_imports: Vec<u32>,
    entry_points: Vec<u32>,
    execution_modes: Vec<u32>,
    annotations: Vec<u32>,
    /// Types, constants and global variables, which may refer to one another in definition
    /// order and so share a section.
    globals: Vec<u32>,
    functions: Vec<u32>,
    types: HashMap<TypeKey, Id>,
    constants: HashMap<(Id, Vec<u32>), Id>,
    glsl_std_450: Option<Id>,
}

impl Module {
    pub fn new() -> Module {
        Module { next_id: 1, ..Module::default() }
    }

    /// A fresh id, defined by whatever instruction the caller writes next with it.
    pub fn id(&mut self) -> Id {
        let id = Id(NonZeroU32::new(self.next_id).expect("ids count up from one"));
        self.next_id += 1;
        id
    }

    /// Declares `cap`, once however often it is asked for.
    pub fn capability(&mut self, cap: Capability) {
        if !self.capability_set.contains(&cap) {
            self.capability_set.push(cap);
            instruction(&mut self.capabilities, Op::Capability, &[cap as u32]);
        }
    }

    /// The `GLSL.std.450` extended instruction set, imported on first use.
    pub fn glsl_std_450(&mut self) -> Id {
        if let Some(id) = self.glsl_std_450 {
            return id;
        }
        let id = self.id();
        let mut operands = vec![id.word()];
        operands.extend(string_words("GLSL.std.450"));
        instruction(&mut self.ext_inst_imports, Op::ExtInstImport, &operands);
        self.glsl_std_450 = Some(id);
        id
    }

    /// An entry point `function` for `model`, named `name`, with the global variables it
    /// reads and writes (`interface`).
    pub fn entry_point(
        &mut self,
        model: ExecutionModel,
        function: Id,
        name: &str,
        interface: &[Id],
    ) {
        let mut operands = vec![model as u32, function.word()];
        operands.extend(string_words(name));
        operands.extend(interface.iter().map(|id| id.word()));
        instruction(&mut self.entry_points, Op::EntryPoint, &operands);
    }

    /// An execution mode of `function`, with its literal operands.
    pub fn execution_mode(&mut self, function: Id, mode: ExecutionMode, literals: &[u32]) {
        let mut operands = vec![function.word(), mode as u32];
        operands.extend_from_slice(literals);
        instruction(&mut self.execution_modes, Op::ExecutionMode, &operands);
    }

    /// Decorates `target` with `decoration` and its literal operands.
    pub fn decorate(&mut self, target: Id, decoration: Decoration, literals: &[u32]) {
        let mut operands = vec![target.word(), decoration as u32];
        operands.extend_from_slice(literals);
        instruction(&mut self.annotations, Op::Decorate, &operands);
    }

    /// Decorates member `member` of the struct type `target`.
    pub fn member_decorate(
        &mut self,
        target: Id,
        member: u32,
        decoration: Decoration,
        literals: &[u32],
    ) {
        let mut operands = vec![target.word(), member, decoration as u32];
        operands.extend_from_slice(literals);
        instruction(&mut self.annotations, Op::MemberDecorate, &operands);
    }

    /// Decorates `target` as the built-in `builtin`.
    pub fn builtin(&mut self, target: Id, builtin: BuiltIn) {
        self.decorate(target, Decoration::BuiltIn, &[builtin as u32]);
    }

    fn ty(&mut self, key: TypeKey) -> Id {
        if let Some(&id) = self.types.get(&key) {
            return id;
        }
        let id = self.id();
        let r = id.word();
        match &key {
            TypeKey::Void => instruction(&mut self.globals, Op::TypeVoid, &[r]),
            TypeKey::Bool => instruction(&mut self.globals, Op::TypeBool, &[r]),
            TypeKey::Int { width, signed } => {
                instruction(&mut self.globals, Op::TypeInt, &[r, *width, u32::from(*signed)]);
            }
            TypeKey::Float { width } => instruction(&mut self.globals, Op::TypeFloat, &[r, *width]),
            TypeKey::Vector { component, count } => {
                instruction(&mut self.globals, Op::TypeVector, &[r, component.word(), *count]);
            }
            TypeKey::Array { element, length } => {
                instruction(&mut self.globals, Op::TypeArray, &[r, element.word(), length.word()]);
            }
            TypeKey::Pointer { class, pointee } => {
                instruction(
                    &mut self.globals,
                    Op::TypePointer,
                    &[r, *class as u32, pointee.word()],
                );
            }
            TypeKey::Function { ret, params } => {
                let mut operands = vec![r, ret.word()];
                operands.extend(params.iter().map(|p| p.word()));
                instruction(&mut self.globals, Op::TypeFunction, &operands);
            }
            TypeKey::Image { sampled_type, dim, arrayed } => {
                // Depth 0, not multisampled, sampled 1, format unknown.
                let operands =
                    [r, sampled_type.word(), *dim as u32, 0, u32::from(*arrayed), 0, 1, 0];
                instruction(&mut self.globals, Op::TypeImage, &operands);
            }
            TypeKey::SampledImage { image } => {
                instruction(&mut self.globals, Op::TypeSampledImage, &[r, image.word()]);
            }
        }
        self.types.insert(key, id);
        id
    }

    pub fn type_void(&mut self) -> Id {
        self.ty(TypeKey::Void)
    }

    pub fn type_bool(&mut self) -> Id {
        self.ty(TypeKey::Bool)
    }

    pub fn type_int(&mut self, width: u32, signed: bool) -> Id {
        self.ty(TypeKey::Int { width, signed })
    }

    pub fn type_float(&mut self, width: u32) -> Id {
        self.ty(TypeKey::Float { width })
    }

    pub fn type_vector(&mut self, component: Id, count: u32) -> Id {
        self.ty(TypeKey::Vector { component, count })
    }

    /// An array of `length` elements; the length is made as a 32-bit unsigned constant.
    pub fn type_array(&mut self, element: Id, length: u32) -> Id {
        let uint = self.type_int(32, false);
        let length = self.constant(uint, &[length]);
        self.ty(TypeKey::Array { element, length })
    }

    /// An image sampled through a sampler, of `sampled_type` texels.
    pub fn type_image(&mut self, sampled_type: Id, dim: Dim, arrayed: bool) -> Id {
        self.ty(TypeKey::Image { sampled_type, dim, arrayed })
    }

    pub fn type_sampled_image(&mut self, image: Id) -> Id {
        self.ty(TypeKey::SampledImage { image })
    }

    pub fn type_pointer(&mut self, class: StorageClass, pointee: Id) -> Id {
        self.ty(TypeKey::Pointer { class, pointee })
    }

    pub fn type_function(&mut self, ret: Id, params: &[Id]) -> Id {
        self.ty(TypeKey::Function { ret, params: params.to_vec() })
    }

    /// A struct of `members`, always a new type (see [`TypeKey`]).
    pub fn type_struct(&mut self, members: &[Id]) -> Id {
        let id = self.id();
        let mut operands = vec![id.word()];
        operands.extend(members.iter().map(|m| m.word()));
        instruction(&mut self.globals, Op::TypeStruct, &operands);
        id
    }

    /// A scalar constant of type `ty` with its value's words (one for 32 bits, two for 64, low
    /// word first), made once per type and value.
    pub fn constant(&mut self, ty: Id, words: &[u32]) -> Id {
        let key = (ty, words.to_vec());
        if let Some(&id) = self.constants.get(&key) {
            return id;
        }
        let id = self.id();
        let mut operands = vec![ty.word(), id.word()];
        operands.extend_from_slice(words);
        instruction(&mut self.globals, Op::Constant, &operands);
        self.constants.insert(key, id);
        id
    }

    /// A `true` or `false` of type `ty`, made once each.
    pub fn constant_bool(&mut self, ty: Id, value: bool) -> Id {
        // Kept apart from the scalar constants by a word no 32-bit value has beside it.
        let key = (ty, vec![u32::from(value), u32::MAX, u32::MAX]);
        if let Some(&id) = self.constants.get(&key) {
            return id;
        }
        let id = self.id();
        let op = if value { Op::ConstantTrue } else { Op::ConstantFalse };
        instruction(&mut self.globals, op, &[ty.word(), id.word()]);
        self.constants.insert(key, id);
        id
    }

    /// A composite constant of type `ty` from `parts`, made once per type and parts.
    pub fn constant_composite(&mut self, ty: Id, parts: &[Id]) -> Id {
        // Ids are never zero, so a leading zero keeps these apart from scalar constants.
        let mut words = vec![0];
        words.extend(parts.iter().map(|p| p.word()));
        let key = (ty, words);
        if let Some(&id) = self.constants.get(&key) {
            return id;
        }
        let id = self.id();
        let mut operands = vec![ty.word(), id.word()];
        operands.extend(parts.iter().map(|p| p.word()));
        instruction(&mut self.globals, Op::ConstantComposite, &operands);
        self.constants.insert(key, id);
        id
    }

    /// A global variable of pointer type `pointer_type` in `class`.
    pub fn global_variable(&mut self, pointer_type: Id, class: StorageClass) -> Id {
        let id = self.id();
        instruction(
            &mut self.globals,
            Op::Variable,
            &[pointer_type.word(), id.word(), class as u32],
        );
        id
    }

    /// Starts a function returning `ret` of type `function_type`, with no parameters. The
    /// function's body is written through the [`Function`] returned and appended to the
    /// module by [`Module::end_function`].
    pub fn begin_function(&mut self, ret: Id, function_type: Id) -> Function {
        let id = self.id();
        let mut head = Vec::new();
        instruction(
            &mut head,
            Op::Function,
            &[ret.word(), id.word(), FunctionControl::NONE.bits(), function_type.word()],
        );
        let entry = self.id();
        instruction(&mut head, Op::Label, &[entry.word()]);
        Function { id, head, variables: Vec::new(), body: Vec::new() }
    }

    /// Appends a finished function: its head and entry label, its variables, then its body.
    pub fn end_function(&mut self, f: Function) {
        self.functions.extend(f.head);
        self.functions.extend(f.variables);
        self.functions.extend(f.body);
        instruction(&mut self.functions, Op::FunctionEnd, &[]);
    }

    /// The module's words: the header, then every section in the logical layout's order.
    pub fn finish(self) -> Vec<u32> {
        let mut words = vec![
            spirv::MAGIC_NUMBER,
            // SPIR-V 1.0: what GL's SPIR-V environment (ARB_gl_spirv, OpenGL 4.6) takes.
            0x0001_0000,
            // No registered generator.
            0,
            self.next_id,
            0,
        ];
        words.extend(self.capabilities);
        words.extend(self.ext_inst_imports);
        instruction(
            &mut words,
            Op::MemoryModel,
            &[AddressingModel::Logical as u32, MemoryModel::GLSL450 as u32],
        );
        words.extend(self.entry_points);
        words.extend(self.execution_modes);
        words.extend(self.annotations);
        words.extend(self.globals);
        words.extend(self.functions);
        words
    }
}

/// A function's body under construction. Its variables are kept apart, since the spec wants
/// every one of them at the top of the entry block whatever order the emitter makes them in.
pub struct Function {
    pub id: Id,
    head: Vec<u32>,
    variables: Vec<u32>,
    body: Vec<u32>,
}

impl Function {
    /// A function-local variable of pointer type `pointer_type`.
    pub fn variable(&mut self, module: &mut Module, pointer_type: Id) -> Id {
        let id = module.id();
        instruction(
            &mut self.variables,
            Op::Variable,
            &[pointer_type.word(), id.word(), StorageClass::Function as u32],
        );
        id
    }

    /// An instruction with a result: `result_type`, a fresh id, then `operands`.
    pub fn op(&mut self, module: &mut Module, op: Op, result_type: Id, operands: &[u32]) -> Id {
        let id = module.id();
        let mut words = vec![result_type.word(), id.word()];
        words.extend_from_slice(operands);
        instruction(&mut self.body, op, &words);
        id
    }

    /// An instruction with no result.
    pub fn op_void(&mut self, op: Op, operands: &[u32]) {
        instruction(&mut self.body, op, operands);
    }

    /// A `GLSL.std.450` instruction `number` on `operands`.
    pub fn glsl(
        &mut self,
        module: &mut Module,
        result_type: Id,
        number: u32,
        operands: &[Id],
    ) -> Id {
        let set = module.glsl_std_450();
        let mut words = vec![set.word(), number];
        words.extend(operands.iter().map(|o| o.word()));
        self.op(module, Op::ExtInst, result_type, &words)
    }

    /// Starts the block `label`, which a branch made earlier names.
    pub fn label(&mut self, label: Id) {
        instruction(&mut self.body, Op::Label, &[label.word()]);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::Write;
    use std::process::{Command, Stdio};

    /// `spirv-val`'s verdict on `words` in GL's SPIR-V environment: `None` where the tool is not
    /// installed, so a host without it skips rather than fails.
    pub(crate) fn validate(words: &[u32]) -> Option<Result<(), String>> {
        let mut child = Command::new("spirv-val")
            .args(["--target-env", "opengl4.5", "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .ok()?;
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        child.stdin.take().expect("piped").write_all(&bytes).expect("spirv-val reads its input");
        let out = child.wait_with_output().expect("spirv-val runs");
        Some(if out.status.success() {
            Ok(())
        } else {
            Err(format!(
                "{}{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            ))
        })
    }

    #[test]
    fn a_string_is_nul_terminated_and_padded_to_a_word() {
        assert_eq!(string_words("main"), [u32::from_le_bytes(*b"main"), 0]);
        assert_eq!(string_words("abc"), [u32::from_le_bytes(*b"abc\0")]);
        assert_eq!(string_words(""), [0]);
    }

    #[test]
    fn types_and_constants_are_made_once() {
        let mut m = Module::new();
        let f = m.type_float(32);
        assert_eq!(m.type_float(32), f);
        let v = m.type_vector(f, 4);
        assert_eq!(m.type_vector(f, 4), v);
        assert_ne!(m.type_vector(f, 3), v);
        let one = m.constant(f, &[1.0f32.to_bits()]);
        assert_eq!(m.constant(f, &[1.0f32.to_bits()]), one);
        let b = m.type_bool();
        assert_ne!(m.constant_bool(b, true), m.constant_bool(b, false));
        assert_eq!(
            m.constant_composite(v, &[one, one, one, one]),
            m.constant_composite(v, &[one; 4])
        );
    }

    /// A vertex shader copying its one input to the position: the smallest module the
    /// emitter's output shares a shape with, and valid as written.
    #[test]
    fn a_pass_through_vertex_shader_validates() {
        let mut m = Module::new();
        m.capability(Capability::Shader);
        let void = m.type_void();
        let float = m.type_float(32);
        let vec4 = m.type_vector(float, 4);
        let in_ptr = m.type_pointer(StorageClass::Input, vec4);
        let out_ptr = m.type_pointer(StorageClass::Output, vec4);
        let input = m.global_variable(in_ptr, StorageClass::Input);
        m.decorate(input, Decoration::Location, &[0]);
        let position = m.global_variable(out_ptr, StorageClass::Output);
        m.builtin(position, BuiltIn::Position);
        let fn_type = m.type_function(void, &[]);
        let mut f = m.begin_function(void, fn_type);
        let v = f.op(&mut m, Op::Load, vec4, &[input.word()]);
        f.op_void(Op::Store, &[position.word(), v.word()]);
        f.op_void(Op::Return, &[]);
        let main = f.id;
        m.end_function(f);
        m.entry_point(ExecutionModel::Vertex, main, "main", &[input, position]);
        let words = m.finish();
        assert_eq!(words[0], spirv::MAGIC_NUMBER);
        match validate(&words) {
            Some(verdict) => verdict.unwrap_or_else(|e| panic!("spirv-val refused it:\n{e}")),
            None => eprintln!("spirv-val is not installed; the module was not validated"),
        }
    }
}
