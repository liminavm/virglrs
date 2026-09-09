// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Gustavo Noronha Silva

/* The classic capset dump, shared by the replayer and the standalone `caps-dump`.
 *
 * One copy of the format on purpose. The dump is a fixture: a golden recorded from one tool and
 * compared against the other is only a comparison while both spell the fields the same way, and
 * two copies of a field list are two lists free to drift.
 *
 * Nothing here touches the limina fork's entry points -- get_cap_set and fill_caps are stock
 * virglrenderer -- so this builds against any virglrenderer, which is what lets the reference leg
 * be the C the VMM under study actually loaded.
 */

#ifndef VREND_REPLAY_CAPS_H
#define VREND_REPLAY_CAPS_H

#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* The classic capsets, field by field. A word each for the masks and arrays, in hex, so a diff
 * names the field that moved and the bit that moved in it. */
static void caps_words(FILE *f, const char *name, const void *p, size_t bytes)
{
   const uint32_t *w = p;
   fprintf(f, "%s =", name);
   for (size_t i = 0; i < bytes / 4; i++) fprintf(f, " %08x", w[i]);
   fprintf(f, "\n");
}

/* virtgpu's VIRTGPU_DRM_CAPSET_VIRGL and _VIRGL2: the ids a guest names the sets by. */
enum { CAPSET_VIRGL = 1, CAPSET_VIRGL2 = 2 };

static int dump_caps(const char *path)
{
   FILE *f = fopen(path, "w");
   if (!f) { perror(path); return 2; }
   for (uint32_t set = CAPSET_VIRGL; set <= CAPSET_VIRGL2; set++) {
      uint32_t max_ver = 0, max_size = 0;
      virgl_renderer_get_cap_set(set, &max_ver, &max_size);
      fprintf(f, "set %u: max_ver=%u max_size=%u\n", set, max_ver, max_size);
      if (!max_size) continue;
      union virgl_caps caps;
      memset(&caps, 0xa5, sizeof(caps));   /* so an unfilled byte shows */
      virgl_renderer_fill_caps(set, max_ver, &caps);
      const struct virgl_caps_v1 *v1 = &caps.v1;
#define U(field) fprintf(f, "  %s = %u\n", #field, (unsigned)v1->field)
#define W(field) caps_words(f, "  " #field, &v1->field, sizeof(v1->field))
      U(max_version); W(sampler); W(render); W(depthstencil); W(vertexbuffer); W(bset);
      U(glsl_level); U(max_texture_array_layers); U(max_streamout_buffers);
      U(max_dual_source_render_targets); U(max_render_targets); U(max_samples); U(prim_mask);
      U(max_tbo_size); U(max_uniform_blocks); U(max_viewports); U(max_texture_gather_components);
#undef U
#undef W
      if (set != CAPSET_VIRGL2) continue;
      const struct virgl_caps_v2 *v2 = &caps.v2;
#define U(field) fprintf(f, "  %s = %u\n", #field, (unsigned)v2->field)
#define I(field) fprintf(f, "  %s = %d\n", #field, (int)v2->field)
#define F(field) fprintf(f, "  %s = %g\n", #field, (double)v2->field)
#define W(field) caps_words(f, "  " #field, &v2->field, sizeof(v2->field))
      F(min_aliased_point_size); F(max_aliased_point_size); F(min_smooth_point_size);
      F(max_smooth_point_size); F(min_aliased_line_width); F(max_aliased_line_width);
      F(min_smooth_line_width); F(max_smooth_line_width); F(max_texture_lod_bias);
      U(max_geom_output_vertices); U(max_geom_total_output_components); U(max_vertex_outputs);
      U(max_vertex_attribs); U(max_shader_patch_varyings); I(min_texel_offset); I(max_texel_offset);
      I(min_texture_gather_offset); I(max_texture_gather_offset); U(texture_buffer_offset_alignment);
      U(uniform_buffer_offset_alignment); U(shader_buffer_offset_alignment); W(capability_bits);
      W(sample_locations); U(max_vertex_attrib_stride); U(max_shader_buffer_frag_compute);
      U(max_shader_buffer_other_stages); U(max_shader_image_frag_compute);
      U(max_shader_image_other_stages); U(max_image_samples); U(max_compute_work_group_invocations);
      U(max_compute_shared_memory_size); W(max_compute_grid_size); W(max_compute_block_size);
      U(max_texture_2d_size); U(max_texture_3d_size); U(max_texture_cube_size);
      U(max_combined_shader_buffers); W(max_atomic_counters); W(max_atomic_counter_buffers);
      U(max_combined_atomic_counters); U(max_combined_atomic_counter_buffers);
      U(host_feature_check_version); W(supported_readback_formats); W(scanout); W(capability_bits_v2);
      U(max_video_memory); fprintf(f, "  renderer = %.64s\n", v2->renderer); F(max_anisotropy);
      U(max_texture_samplers); W(supported_multisample_formats); W(max_const_buffer_size);
      U(num_video_caps); W(video_caps); U(max_uniform_block_size); U(max_tcs_outputs);
      U(max_tes_outputs); W(max_shader_storage_blocks);
#undef U
#undef I
#undef F
#undef W
   }
   fclose(f);
   return 0;
}

#endif /* VREND_REPLAY_CAPS_H */
