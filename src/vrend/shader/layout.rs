// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! The numbers a shader's interfaces are declared at: the binding of each image, SSBO and
//! atomic counter, and the location of each interface a stage declares by number. One place
//! for both the GLSL emitter, which prints them, and an emitter that decorates with them, so the
//! two cannot number one interface differently.
//!
//! Which interfaces carry a number at all stays the emitter's decision -- GLSL leaves most to
//! the linker's name matching, and a SPIR-V module cannot -- so this answers "at what number",
//! never "whether".
//!
//! The numbering is the C's, collisions included. A separable stage's declared generics count
//! down from 31 while the outputs [`Layout::padding_location`] numbers count up from 0, so a
//! stage with both can put two outputs at one location. That is left as it is here: changing it
//! changes the GLSL, which is a fix in its own right once a case reproduces it.

use super::Key;
use crate::vrend::pipe::ShaderStage;

/// The uniform locations one stage's default-block uniforms take in a SPIR-V program. Every
/// stage of a program shares one location space, and a SPIR-V module cannot know what the
/// other stages declare, so each stage takes its own range. Six ranges fit the 81,820
/// locations zink-KosmicKrisp offers (limina's `docs/design/vrend-spirv.md` §2).
const STAGE_LOCATIONS: u32 = 8192;

/// The numbers of one translation, from its key and what its declarations bound.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    image_binding_offset: u32,
    /// The guest's SSBO binding offset less the first SSBO the shader declares, so slot `n`
    /// binds at `n + ssbo_base`. In 64 bits as the C's arithmetic is: the difference can be
    /// negative before the slot is added.
    ssbo_base: i64,
}

impl Layout {
    /// `ssbo_first_binding` is the lowest SSBO slot the declarations name, or `None` with no
    /// SSBO declared.
    pub fn new(key: &Key, ssbo_first_binding: Option<u32>) -> Layout {
        // With no SSBO declared nothing asks for an SSBO binding; the C's sentinel is the
        // unsigned maximum, and the base is kept as that arithmetic gives it.
        let first = ssbo_first_binding.unwrap_or(u32::MAX);
        Layout {
            image_binding_offset: u32::from(key.image_binding_offset),
            ssbo_base: i64::from(key.ssbo_binding_offset) - i64::from(first),
        }
    }

    /// The binding of image slot `slot`, or of the first image of an array starting there. On
    /// desktop GL the C declares this number as the image's uniform location, and the renderer
    /// finds it again by name.
    pub fn image_binding(&self, slot: u32) -> u32 {
        slot + self.image_binding_offset
    }

    /// The binding of SSBO slot `slot`, or of the first of an indirectly addressed run.
    pub fn ssbo_binding(&self, slot: u32) -> i32 {
        (i64::from(slot) + self.ssbo_base) as i32
    }

    /// The binding and byte offset of an atomic counter: the buffer the guest declared it in
    /// and the counter's first index there, four bytes a counter.
    pub fn atomic_counter(&self, buffer: i32, first: i32) -> (i32, i32) {
        (buffer, first * 4)
    }

    /// The location of a separable stage's generic varying with semantic index `sid`: down from
    /// 31, clear of the patches.
    pub fn separable_generic_location(&self, sid: u32) -> i32 {
        31 - sid as i32
    }

    /// The location of a separable stage's patch varying with semantic index `sid`: up from 0.
    pub fn separable_patch_location(&self, sid: u32) -> u32 {
        sid
    }

    /// The location of the generic output a separable stage declares only because the next
    /// stage expects it, at its semantic index -- not the 31-down its declared generics take.
    pub fn padding_location(&self, sid: u32) -> u32 {
        sid
    }

    /// The location of the fragment colour output for colour buffer `cbuf`.
    pub fn colour_output_location(&self, cbuf: u32) -> u32 {
        cbuf
    }

    /// The binding every stage of a SPIR-V program reads the `VirglBlock` at: past the five
    /// graphics stages' fixed sixteen uniform-block slots each.
    pub const VIRGL_BLOCK_BINDING: u32 = 80;

    /// The location of a SPIR-V stage's constant array (the GLSL's `<stage>const0`), at the
    /// start of the stage's range.
    pub fn const_location(stage: ShaderStage) -> u32 {
        stage.index() as u32 * STAGE_LOCATIONS
    }

    /// The binding of a SPIR-V stage's uniform buffer `slot`: sixteen a stage, slot 0 (the
    /// constants the stage reads as plain uniforms) never taking one. `None` past fifteen.
    pub fn ubo_binding(stage: ShaderStage, slot: u32) -> Option<u32> {
        (1..16).contains(&slot).then(|| stage.index() as u32 * 16 + slot)
    }

    /// The location of a SPIR-V stage's sampler `slot`, past the room its constant array can
    /// take (4096 `uvec4`, `MAX_CONSTS`).
    pub fn sampler_location(stage: ShaderStage, slot: u32) -> u32 {
        stage.index() as u32 * STAGE_LOCATIONS + 4096 + slot
    }

    /// The location of a generic varying with semantic index `sid` in a SPIR-V program, which
    /// takes the separable numbering whether it is separable or not: a SPIR-V stage meets its
    /// neighbours only by location. `None` past 31, which the numbering has no room for.
    pub fn varying_location(&self, sid: u32) -> Option<u32> {
        31u32.checked_sub(sid)
    }
}
