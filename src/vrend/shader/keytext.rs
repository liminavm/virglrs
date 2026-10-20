// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! A translation's inputs beyond its program, as text: the host's [`Config`], the [`Key`], the
//! [`StreamOutput`] and a compute shader's local memory, printed by `VIRGLRS_DEBUG=shader-exact`
//! beside its TGSI so a fixture can translate a block under exactly what produced its GLSL.
//!
//! Every field is printed, default or not, as `name=value` separated by spaces: nested fields
//! dotted, arrays comma-separated, lists `;`-separated with `-` for an empty one. The printer
//! destructures each struct with no `..`, and the parser builds each with a literal, so a field
//! added to the key fails the build here until the text carries it. A format that left defaults
//! out would go on parsing after a field was added, and translate under its default in silence.

use std::fmt::Write;

use super::{
    Config, Dialect, FragmentInfo, FsKey, GsKey, InterpInfo, IoArray, IoArrayInfo, Key, VsKey,
};
use crate::vrend::proto::{SoOutput, StreamOutput};

fn list<T: std::fmt::Display>(out: &mut String, name: &str, values: &[T]) {
    let joined: Vec<String> = values.iter().map(ToString::to_string).collect();
    write!(out, " {name}={}", joined.join(",")).expect("String");
}

fn records(out: &mut String, name: &str, items: Vec<String>) {
    let joined = if items.is_empty() { "-".to_string() } else { items.join(";") };
    write!(out, " {name}={joined}").expect("String");
}

fn dialect_name(d: Dialect) -> &'static str {
    match d {
        Dialect::Es => "es",
        Dialect::Core => "core",
    }
}

/// The host's translator configuration as one line of text.
pub fn print_config(cfg: &Config) -> String {
    let Config {
        dialect,
        glsl_version,
        max_draw_buffers,
        max_shader_patch_varyings,
        has_arrays_of_arrays,
        has_gpu_shader5,
        has_es31_compat,
        has_conservative_depth,
        has_dual_src_blend,
        has_fbfetch_coherent,
        has_cull_distance,
        has_nopersective,
        has_texture_shadow_lod,
        has_vs_layer,
        has_vs_viewport_index,
        has_separate_shader_objects,
    } = cfg;
    let mut o = String::new();
    write!(o, "dialect={}", dialect_name(*dialect)).expect("String");
    write!(o, " glsl_version={glsl_version}").expect("String");
    write!(o, " max_draw_buffers={max_draw_buffers}").expect("String");
    write!(o, " max_shader_patch_varyings={max_shader_patch_varyings}").expect("String");
    write!(o, " has_arrays_of_arrays={has_arrays_of_arrays}").expect("String");
    write!(o, " has_gpu_shader5={has_gpu_shader5}").expect("String");
    write!(o, " has_es31_compat={has_es31_compat}").expect("String");
    write!(o, " has_conservative_depth={has_conservative_depth}").expect("String");
    write!(o, " has_dual_src_blend={has_dual_src_blend}").expect("String");
    write!(o, " has_fbfetch_coherent={has_fbfetch_coherent}").expect("String");
    write!(o, " has_cull_distance={has_cull_distance}").expect("String");
    write!(o, " has_nopersective={has_nopersective}").expect("String");
    write!(o, " has_texture_shadow_lod={has_texture_shadow_lod}").expect("String");
    write!(o, " has_vs_layer={has_vs_layer}").expect("String");
    write!(o, " has_vs_viewport_index={has_vs_viewport_index}").expect("String");
    write!(o, " has_separate_shader_objects={has_separate_shader_objects}").expect("String");
    o
}

/// The key as one line of text.
pub fn print_key(key: &Key) -> String {
    let Key {
        out_generic_expected_mask,
        out_texcoord_expected_mask,
        in_generic_expected_mask,
        in_texcoord_expected_mask,
        in_patch_expected_mask,
        force_invariant_inputs,
        fs_info,
        in_arrays,
        fs,
        vs,
        gs,
        sampler_views_lower_swizzle_mask,
        sampler_views_emulated_rect_mask,
        sampler_views_lower_array_mask,
        tex_swizzle,
        ssbo_binding_offset,
        image_binding_offset,
        alpha_test,
        num_in_cull,
        num_in_clip,
        num_out_cull,
        num_out_clip,
        pstipple_enabled,
        add_alpha_test,
        color_two_side,
        gs_present,
        tcs_present,
        tes_present,
        flatshade,
        require_input_arrays,
        require_output_arrays,
        use_pervertex_in,
        fixed_point_size,
    } = key;
    let mut o = String::new();
    write!(o, "out_generic_expected_mask={out_generic_expected_mask}").expect("String");
    write!(o, " out_texcoord_expected_mask={out_texcoord_expected_mask}").expect("String");
    write!(o, " in_generic_expected_mask={in_generic_expected_mask}").expect("String");
    write!(o, " in_texcoord_expected_mask={in_texcoord_expected_mask}").expect("String");
    write!(o, " in_patch_expected_mask={in_patch_expected_mask}").expect("String");
    list(&mut o, "force_invariant_inputs", force_invariant_inputs);

    let FragmentInfo { glsl_ver, has_sample_input, has_noperspective, interps } = fs_info;
    write!(o, " fs_info.glsl_ver={glsl_ver}").expect("String");
    write!(o, " fs_info.has_sample_input={has_sample_input}").expect("String");
    write!(o, " fs_info.has_noperspective={has_noperspective}").expect("String");
    let interps = interps
        .iter()
        .map(|i| {
            let InterpInfo { semantic_name, semantic_index, interpolate, location } = i;
            format!(
                "{}:{semantic_index}:{}:{}",
                semantic_name.name(),
                interpolate.name(),
                location.name()
            )
        })
        .collect();
    records(&mut o, "fs_info.interps", interps);

    let IoArrayInfo { layout } = in_arrays;
    let layout = layout
        .iter()
        .map(|a| {
            let IoArray { name, sid, size, array_id } = a;
            format!("{}:{sid}:{size}:{array_id}", name.name())
        })
        .collect();
    records(&mut o, "in_arrays.layout", layout);

    let FsKey {
        surface_component_bits,
        coord_replace,
        swizzle_output_rgb_to_bgr,
        needs_manual_srgb_encode_bitmask,
        cbufs_are_a8_bitmask,
        cbufs_signed_int_bitmask,
        cbufs_unsigned_int_bitmask,
        logicop_func,
        prim_is_points,
        lower_left_origin,
        available_color_in_bits,
    } = fs;
    list(&mut o, "fs.surface_component_bits", surface_component_bits);
    write!(o, " fs.coord_replace={coord_replace}").expect("String");
    write!(o, " fs.swizzle_output_rgb_to_bgr={swizzle_output_rgb_to_bgr}").expect("String");
    write!(o, " fs.needs_manual_srgb_encode_bitmask={needs_manual_srgb_encode_bitmask}")
        .expect("String");
    write!(o, " fs.cbufs_are_a8_bitmask={cbufs_are_a8_bitmask}").expect("String");
    write!(o, " fs.cbufs_signed_int_bitmask={cbufs_signed_int_bitmask}").expect("String");
    write!(o, " fs.cbufs_unsigned_int_bitmask={cbufs_unsigned_int_bitmask}").expect("String");
    write!(o, " fs.logicop_func={}", logicop_func.map_or("-", |l| l.name())).expect("String");
    write!(o, " fs.prim_is_points={prim_is_points}").expect("String");
    write!(o, " fs.lower_left_origin={lower_left_origin}").expect("String");
    write!(o, " fs.available_color_in_bits={available_color_in_bits}").expect("String");

    let VsKey { attrib_zyxw_bitmask, fog_fixup_mask, vertex_id_from_input } = vs;
    write!(o, " vs.attrib_zyxw_bitmask={attrib_zyxw_bitmask}").expect("String");
    write!(o, " vs.fog_fixup_mask={fog_fixup_mask}").expect("String");
    write!(o, " vs.vertex_id_from_input={vertex_id_from_input}").expect("String");

    let GsKey { emit_clip_distance } = gs;
    write!(o, " gs.emit_clip_distance={emit_clip_distance}").expect("String");

    list(&mut o, "sampler_views_lower_swizzle_mask", sampler_views_lower_swizzle_mask);
    list(&mut o, "sampler_views_emulated_rect_mask", sampler_views_emulated_rect_mask);
    list(&mut o, "sampler_views_lower_array_mask", sampler_views_lower_array_mask);
    list(&mut o, "tex_swizzle", tex_swizzle);
    write!(o, " ssbo_binding_offset={ssbo_binding_offset}").expect("String");
    write!(o, " image_binding_offset={image_binding_offset}").expect("String");
    write!(o, " alpha_test={}", alpha_test.name()).expect("String");
    write!(o, " num_in_cull={num_in_cull}").expect("String");
    write!(o, " num_in_clip={num_in_clip}").expect("String");
    write!(o, " num_out_cull={num_out_cull}").expect("String");
    write!(o, " num_out_clip={num_out_clip}").expect("String");
    write!(o, " pstipple_enabled={pstipple_enabled}").expect("String");
    write!(o, " add_alpha_test={add_alpha_test}").expect("String");
    write!(o, " color_two_side={color_two_side}").expect("String");
    write!(o, " gs_present={gs_present}").expect("String");
    write!(o, " tcs_present={tcs_present}").expect("String");
    write!(o, " tes_present={tes_present}").expect("String");
    write!(o, " flatshade={flatshade}").expect("String");
    write!(o, " require_input_arrays={require_input_arrays}").expect("String");
    write!(o, " require_output_arrays={require_output_arrays}").expect("String");
    write!(o, " use_pervertex_in={use_pervertex_in}").expect("String");
    write!(o, " fixed_point_size={fixed_point_size}").expect("String");
    o
}

/// The stream output as one line of text.
pub fn print_stream_output(so: &StreamOutput) -> String {
    let StreamOutput { stride, outputs } = so;
    let mut o = String::new();
    let strides: Vec<String> = stride.iter().map(ToString::to_string).collect();
    write!(o, "stride={}", strides.join(",")).expect("String");
    let outputs = outputs
        .iter()
        .map(|s| {
            let SoOutput {
                register_index,
                start_component,
                num_components,
                output_buffer,
                dst_offset,
                stream,
            } = s;
            format!(
                "{register_index}:{start_component}:{num_components}:{}:{dst_offset}:{stream}",
                output_buffer.index()
            )
        })
        .collect();
    records(&mut o, "outputs", outputs);
    o
}

#[cfg(test)]
pub use parse::{parse_config, parse_key, parse_stream_output};

#[cfg(test)]
mod parse {
    use std::collections::HashMap;
    use std::str::FromStr;

    use super::*;
    use crate::vrend::pipe::{CompareFunc, LogicOp};
    use crate::vrend::proto::SoBuffer;
    use crate::vrend::tgsi::{Interpolate, Location, Semantic};

    /// The line's fields by name. Each is taken once; whatever is left at the end was printed and
    /// not read, which is as much a mismatch as a field read and not printed.
    struct Fields<'a>(HashMap<&'a str, &'a str>);

    impl<'a> Fields<'a> {
        fn new(line: &'a str) -> Fields<'a> {
            let mut map = HashMap::new();
            for word in line.split(' ').filter(|w| !w.is_empty()) {
                let (k, v) = word.split_once('=').unwrap_or_else(|| panic!("no `=` in {word:?}"));
                assert!(map.insert(k, v).is_none(), "{k} printed twice");
            }
            Fields(map)
        }

        fn raw(&mut self, name: &str) -> &'a str {
            self.0.remove(name).unwrap_or_else(|| panic!("no {name} in the key text"))
        }

        fn num<T: FromStr>(&mut self, name: &str) -> T {
            let v = self.raw(name);
            v.parse().unwrap_or_else(|_| panic!("{name}={v} is not a number"))
        }

        fn flag(&mut self, name: &str) -> bool {
            self.num(name)
        }

        fn array<T: FromStr + Copy + Default, const N: usize>(&mut self, name: &str) -> [T; N] {
            let v = self.raw(name);
            let mut out = [T::default(); N];
            let parts: Vec<&str> = v.split(',').collect();
            assert_eq!(parts.len(), N, "{name} has {} values, not {N}", parts.len());
            for (slot, p) in out.iter_mut().zip(parts) {
                *slot = p.parse().unwrap_or_else(|_| panic!("{name}: {p} is not a number"));
            }
            out
        }

        fn records(&mut self, name: &str) -> Vec<Vec<&'a str>> {
            match self.raw(name) {
                "-" => Vec::new(),
                v => v.split(';').map(|r| r.split(':').collect()).collect(),
            }
        }

        fn done(self) {
            assert!(self.0.is_empty(), "fields printed but not read: {:?}", self.0.keys());
        }
    }

    fn named<T: Copy>(all: &[T], name: fn(T) -> &'static str, text: &str) -> T {
        *all.iter().find(|v| name(**v) == text).unwrap_or_else(|| panic!("no value named {text}"))
    }

    fn n<T: FromStr>(text: &str) -> T {
        text.parse().unwrap_or_else(|_| panic!("{text} is not a number"))
    }

    /// The configuration [`print_config`] printed.
    pub fn parse_config(line: &str) -> Config {
        let mut f = Fields::new(line);
        let cfg = Config {
            dialect: match f.raw("dialect") {
                "es" => Dialect::Es,
                "core" => Dialect::Core,
                d => panic!("no dialect named {d}"),
            },
            glsl_version: f.num("glsl_version"),
            max_draw_buffers: f.num("max_draw_buffers"),
            max_shader_patch_varyings: f.num("max_shader_patch_varyings"),
            has_arrays_of_arrays: f.flag("has_arrays_of_arrays"),
            has_gpu_shader5: f.flag("has_gpu_shader5"),
            has_es31_compat: f.flag("has_es31_compat"),
            has_conservative_depth: f.flag("has_conservative_depth"),
            has_dual_src_blend: f.flag("has_dual_src_blend"),
            has_fbfetch_coherent: f.flag("has_fbfetch_coherent"),
            has_cull_distance: f.flag("has_cull_distance"),
            has_nopersective: f.flag("has_nopersective"),
            has_texture_shadow_lod: f.flag("has_texture_shadow_lod"),
            has_vs_layer: f.flag("has_vs_layer"),
            has_vs_viewport_index: f.flag("has_vs_viewport_index"),
            has_separate_shader_objects: f.flag("has_separate_shader_objects"),
        };
        f.done();
        cfg
    }

    /// The key [`print_key`] printed.
    pub fn parse_key(line: &str) -> Key {
        let mut f = Fields::new(line);
        let key = Key {
            out_generic_expected_mask: f.num("out_generic_expected_mask"),
            out_texcoord_expected_mask: f.num("out_texcoord_expected_mask"),
            in_generic_expected_mask: f.num("in_generic_expected_mask"),
            in_texcoord_expected_mask: f.num("in_texcoord_expected_mask"),
            in_patch_expected_mask: f.num("in_patch_expected_mask"),
            force_invariant_inputs: f.array("force_invariant_inputs"),
            fs_info: FragmentInfo {
                glsl_ver: f.num("fs_info.glsl_ver"),
                has_sample_input: f.flag("fs_info.has_sample_input"),
                has_noperspective: f.flag("fs_info.has_noperspective"),
                interps: f
                    .records("fs_info.interps")
                    .into_iter()
                    .map(|r| InterpInfo {
                        semantic_name: named(Semantic::ALL, Semantic::name, r[0]),
                        semantic_index: n(r[1]),
                        interpolate: named(Interpolate::ALL, Interpolate::name, r[2]),
                        location: named(Location::ALL, Location::name, r[3]),
                    })
                    .collect(),
            },
            in_arrays: IoArrayInfo {
                layout: f
                    .records("in_arrays.layout")
                    .into_iter()
                    .map(|r| IoArray {
                        name: named(Semantic::ALL, Semantic::name, r[0]),
                        sid: n(r[1]),
                        size: n(r[2]),
                        array_id: n(r[3]),
                    })
                    .collect(),
            },
            fs: FsKey {
                surface_component_bits: f.array("fs.surface_component_bits"),
                coord_replace: f.num("fs.coord_replace"),
                swizzle_output_rgb_to_bgr: f.num("fs.swizzle_output_rgb_to_bgr"),
                needs_manual_srgb_encode_bitmask: f.num("fs.needs_manual_srgb_encode_bitmask"),
                cbufs_are_a8_bitmask: f.num("fs.cbufs_are_a8_bitmask"),
                cbufs_signed_int_bitmask: f.num("fs.cbufs_signed_int_bitmask"),
                cbufs_unsigned_int_bitmask: f.num("fs.cbufs_unsigned_int_bitmask"),
                logicop_func: match f.raw("fs.logicop_func") {
                    "-" => None,
                    l => Some(named(LogicOp::ALL, LogicOp::name, l)),
                },
                prim_is_points: f.flag("fs.prim_is_points"),
                lower_left_origin: f.flag("fs.lower_left_origin"),
                available_color_in_bits: f.num("fs.available_color_in_bits"),
            },
            vs: VsKey {
                attrib_zyxw_bitmask: f.num("vs.attrib_zyxw_bitmask"),
                fog_fixup_mask: f.num("vs.fog_fixup_mask"),
                vertex_id_from_input: f.flag("vs.vertex_id_from_input"),
            },
            gs: GsKey { emit_clip_distance: f.flag("gs.emit_clip_distance") },
            sampler_views_lower_swizzle_mask: f.array("sampler_views_lower_swizzle_mask"),
            sampler_views_emulated_rect_mask: f.array("sampler_views_emulated_rect_mask"),
            sampler_views_lower_array_mask: f.array("sampler_views_lower_array_mask"),
            tex_swizzle: f.array("tex_swizzle"),
            ssbo_binding_offset: f.num("ssbo_binding_offset"),
            image_binding_offset: f.num("image_binding_offset"),
            alpha_test: named(CompareFunc::ALL, CompareFunc::name, f.raw("alpha_test")),
            num_in_cull: f.num("num_in_cull"),
            num_in_clip: f.num("num_in_clip"),
            num_out_cull: f.num("num_out_cull"),
            num_out_clip: f.num("num_out_clip"),
            pstipple_enabled: f.flag("pstipple_enabled"),
            add_alpha_test: f.flag("add_alpha_test"),
            color_two_side: f.flag("color_two_side"),
            gs_present: f.flag("gs_present"),
            tcs_present: f.flag("tcs_present"),
            tes_present: f.flag("tes_present"),
            flatshade: f.flag("flatshade"),
            require_input_arrays: f.flag("require_input_arrays"),
            require_output_arrays: f.flag("require_output_arrays"),
            use_pervertex_in: f.flag("use_pervertex_in"),
            fixed_point_size: f.flag("fixed_point_size"),
        };
        f.done();
        key
    }

    /// The stream output [`print_stream_output`] printed.
    pub fn parse_stream_output(line: &str) -> StreamOutput {
        let mut f = Fields::new(line);
        let so = StreamOutput {
            stride: f.array("stride"),
            outputs: f
                .records("outputs")
                .into_iter()
                .map(|r| SoOutput {
                    register_index: n(r[0]),
                    start_component: n(r[1]),
                    num_components: n(r[2]),
                    output_buffer: SoBuffer::from_wire(n(r[3])).expect("a stream-output buffer"),
                    dst_offset: n(r[4]),
                    stream: n(r[5]),
                })
                .collect(),
        };
        f.done();
        so
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vrend::pipe::{CompareFunc, LogicOp};
    use crate::vrend::proto::SoBuffer;
    use crate::vrend::tgsi::{Interpolate, Location, Semantic};

    /// A key with no field at its default, so a field the text dropped or misread shows.
    fn busy_key() -> Key {
        let mut tex_swizzle = [0u16; super::super::MAX_SHADER_SAMPLER_VIEWS];
        tex_swizzle.iter_mut().enumerate().for_each(|(i, s)| *s = (i as u16 * 37) & 0xfff);
        Key {
            out_generic_expected_mask: 1 << 40 | 3,
            out_texcoord_expected_mask: 5,
            in_generic_expected_mask: 1 << 63,
            in_texcoord_expected_mask: 9,
            in_patch_expected_mask: 17,
            force_invariant_inputs: [1, 2, 3, 4],
            fs_info: FragmentInfo {
                glsl_ver: 320,
                has_sample_input: true,
                has_noperspective: true,
                interps: vec![
                    InterpInfo {
                        semantic_name: Semantic::Generic,
                        semantic_index: 3,
                        interpolate: Interpolate::Linear,
                        location: Location::Centroid,
                    },
                    InterpInfo {
                        semantic_name: Semantic::Color,
                        semantic_index: 1,
                        interpolate: Interpolate::Constant,
                        location: Location::Sample,
                    },
                ],
            },
            in_arrays: IoArrayInfo {
                layout: vec![IoArray { name: Semantic::Patch, sid: 2, size: 4, array_id: 7 }],
            },
            fs: FsKey {
                surface_component_bits: [1, 2, 3, 4, 5, 6, 7, 8],
                coord_replace: 0x55,
                swizzle_output_rgb_to_bgr: 3,
                needs_manual_srgb_encode_bitmask: 6,
                cbufs_are_a8_bitmask: 9,
                cbufs_signed_int_bitmask: 10,
                cbufs_unsigned_int_bitmask: 12,
                logicop_func: Some(LogicOp::Xor),
                prim_is_points: true,
                lower_left_origin: true,
                available_color_in_bits: 5,
            },
            vs: VsKey { attrib_zyxw_bitmask: 33, fog_fixup_mask: 66, vertex_id_from_input: true },
            gs: GsKey { emit_clip_distance: true },
            sampler_views_lower_swizzle_mask: [1, 1 << 63],
            sampler_views_emulated_rect_mask: [2, 3],
            sampler_views_lower_array_mask: [4, 5],
            tex_swizzle,
            ssbo_binding_offset: 3,
            image_binding_offset: 4,
            alpha_test: CompareFunc::GreaterEqual,
            num_in_cull: 1,
            num_in_clip: 2,
            num_out_cull: 3,
            num_out_clip: 4,
            pstipple_enabled: true,
            add_alpha_test: true,
            color_two_side: true,
            gs_present: true,
            tcs_present: true,
            tes_present: true,
            flatshade: true,
            require_input_arrays: true,
            require_output_arrays: true,
            use_pervertex_in: true,
            fixed_point_size: true,
        }
    }

    #[test]
    fn a_config_round_trips_through_its_text() {
        let quiet = Config {
            dialect: Dialect::Es,
            glsl_version: 310,
            max_draw_buffers: 4,
            max_shader_patch_varyings: 0,
            has_arrays_of_arrays: false,
            has_gpu_shader5: false,
            has_es31_compat: false,
            has_conservative_depth: false,
            has_dual_src_blend: false,
            has_fbfetch_coherent: false,
            has_cull_distance: false,
            has_nopersective: false,
            has_texture_shadow_lod: false,
            has_vs_layer: false,
            has_vs_viewport_index: false,
            has_separate_shader_objects: false,
        };
        let busy = Config {
            dialect: Dialect::Core,
            glsl_version: 460,
            max_draw_buffers: 8,
            max_shader_patch_varyings: 30,
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
        for cfg in [quiet, busy] {
            let text = print_config(&cfg);
            assert_eq!(parse_config(&text), cfg, "{text}");
        }
    }

    #[test]
    fn a_key_round_trips_through_its_text() {
        for key in [Key::default(), busy_key()] {
            let text = print_key(&key);
            assert_eq!(parse_key(&text), key, "{text}");
        }
    }

    #[test]
    fn a_stream_output_round_trips_through_its_text() {
        let busy = StreamOutput {
            stride: [4, 8, 12, 16],
            outputs: vec![
                SoOutput {
                    register_index: 2,
                    start_component: 1,
                    num_components: 3,
                    output_buffer: SoBuffer::from_wire(3).unwrap(),
                    dst_offset: 9,
                    stream: 1,
                },
                SoOutput {
                    register_index: 5,
                    start_component: 0,
                    num_components: 4,
                    output_buffer: SoBuffer::from_wire(0).unwrap(),
                    dst_offset: 0,
                    stream: 0,
                },
            ],
        };
        for so in [StreamOutput::default(), busy] {
            let text = print_stream_output(&so);
            assert_eq!(parse_stream_output(&text), so, "{text}");
        }
    }
}
