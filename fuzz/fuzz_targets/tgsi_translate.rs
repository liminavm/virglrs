// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! A shader as a guest sends it -- TGSI text -- parsed, scanned and translated to GLSL, which is
//! the path `create_shader` takes. The guest writes every byte of it, so any panic is a guest
//! aborting the renderer.
#![no_main]

use libfuzzer_sys::fuzz_target;
use virglrenderer::vrend::proto::StreamOutput;
use virglrenderer::vrend::shader::{Config, Dialect, Key, convert};
use virglrenderer::vrend::tgsi::Program;

/// A capable GLES host, as the translator's own corpus tests configure it.
const GLES: Config = Config {
    dialect: Dialect::Es,
    glsl_version: 310,
    max_draw_buffers: 8,
    max_shader_patch_varyings: 30,
    has_arrays_of_arrays: true,
    has_gpu_shader5: true,
    has_es31_compat: true,
    has_conservative_depth: false,
    has_dual_src_blend: true,
    has_fbfetch_coherent: false,
    has_cull_distance: true,
    has_nopersective: false,
    has_texture_shadow_lod: false,
    has_vs_layer: false,
    has_vs_viewport_index: false,
    has_separate_shader_objects: false,
};

/// A desktop GL 4.6 core host, as the translator's desktop corpus tests configure it.
const DESKTOP: Config = Config {
    dialect: Dialect::Core,
    glsl_version: 460,
    max_draw_buffers: 8,
    max_shader_patch_varyings: 32,
    has_arrays_of_arrays: true,
    has_gpu_shader5: true,
    has_es31_compat: true,
    has_conservative_depth: true,
    has_dual_src_blend: true,
    has_fbfetch_coherent: true,
    has_cull_distance: true,
    has_nopersective: true,
    has_texture_shadow_lod: true,
    has_vs_layer: true,
    has_vs_viewport_index: true,
    has_separate_shader_objects: true,
};

fuzz_target!(|text: &[u8]| {
    // The guest states the token count; the C parses into ten more. A bound here keeps a runaway
    // input from being a timeout rather than a finding.
    let Ok(shader) = Program::parse(text, 4096) else { return };
    let Ok(program) = Program::scan(shader) else { return };
    // Each input through both dialects: the corpus is shader text, so a selector byte in front
    // of it would shift every seed.
    for cfg in [&GLES, &DESKTOP] {
        let _ = convert(cfg, &program, 0, &Key::default(), &StreamOutput::default());
    }
});
