#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright © 2026 Gustavo Noronha Silva
#
# Write a synthetic classic corpus for a shader image over a subset of an array texture's layers.
#
#   make-image-corpus.py [OUT]      default ../vm/captures/image.bin
#
# A four-layer array texture, and one draw whose fragment shader stores into it through an image
# view of layers 1 and 2 only: 1.0 into the view's first layer, 0.5 into its second. Then four
# draws, each sampling one layer of the whole texture into a 2D offscreen, which the sweep scores
# at its unref. The C binds a texture view of the two layers (`vrend_draw_bind_images_shader`), so
# layers 1 and 2 read back red at two strengths and layers 0 and 3 stay clear. A renderer that
# does not make that view writes nothing, or writes through whatever the image unit held before.
#
# No recorded session reaches this: a desktop's images, where it has any, span whole textures.
# The layers are read back through a draw because the sweep reads only plain 2D colour targets.
import struct, sys, zlib

from corpus import (Corpus, BIND_RENDER_TARGET, BIND_SAMPLER_VIEW, BIND_VERTEX_BUFFER,
                    B8G8R8A8_UNORM, R32_FLOAT, R32G32B32A32_FLOAT, IMAGE_ACCESS_WRITE,
                    OBJ_BLEND, OBJ_DSA, OBJ_RASTERIZER, OBJ_SAMPLER_VIEW, OBJ_SURFACE,
                    OBJ_VERTEX_ELEMENTS,
                    STAGE_VERTEX, STAGE_FRAGMENT, TARGET_2D_ARRAY)

SIDE = 16
LAYERS = 4
R8_UNORM = 64
TOKENS = 300

QUAD = [(-1.0, -1.0, 0.0, 0.0), (1.0, -1.0, 1.0, 0.0),
        (-1.0, 1.0, 0.0, 1.0), (1.0, 1.0, 1.0, 1.0)]

VS = """VERT
DCL IN[0]
DCL IN[1]
DCL OUT[0], POSITION
DCL OUT[1], GENERIC[0]
0: MOV OUT[0], IN[0]
1: MOV OUT[1], IN[1]
2: END"""

# Layer 0 of the view is layer 1 of the texture. The colour output only gives the draw a target.
STORE_FS = """FRAG
DCL IN[0], POSITION, LINEAR
DCL OUT[0], COLOR
DCL IMAGE[0], 2D_ARRAY, PIPE_FORMAT_R32_FLOAT, WR
DCL TEMP[0]
IMM[0] FLT32 { 1.0, 0.5, 0.0, 1.0 }
IMM[1] UINT32 { 0, 1, 0, 0 }
0: F2U TEMP[0].xy, IN[0].xyyy
1: MOV TEMP[0].z, IMM[1].xxxx
2: STORE IMAGE[0], TEMP[0], IMM[0].xxxx, 2D_ARRAY, PIPE_FORMAT_R32_FLOAT
3: MOV TEMP[0].z, IMM[1].yyyy
4: STORE IMAGE[0], TEMP[0], IMM[0].yyyy, 2D_ARRAY, PIPE_FORMAT_R32_FLOAT
5: MOV OUT[0], IMM[0].zzzw
6: END"""


def read_fs(layer):
    """Sample one layer of the whole texture: an R32 texel reads as (v, 0, 0, 1)."""
    return """FRAG
DCL IN[0], GENERIC[0], PERSPECTIVE
DCL OUT[0], COLOR
DCL SAMP[0]
DCL SVIEW[0], 2D_ARRAY, FLOAT
DCL TEMP[0]
IMM[0] FLT32 { %d.0, 0.0, 0.0, 0.0 }
0: MOV TEMP[0].xy, IN[0].xyyy
1: MOV TEMP[0].z, IMM[0].xxxx
2: TEX OUT[0], TEMP[0], SAMP[0], 2D_ARRAY
3: END""" % layer


def build():
    c = Corpus()
    rt = BIND_RENDER_TARGET | BIND_SAMPLER_VIEW

    ARRAY = 30
    STORE_TARGET = 31
    READ = [40 + l for l in range(LAYERS)]
    VBO = 33

    VS_H, STORE_FS_H = 100, 101
    READ_FS_H = [102 + l for l in range(LAYERS)]
    RAST_H, BLEND_H, DSA_H, VE_H, SAMP_H = 110, 111, 112, 113, 114
    VIEW_H = 200
    STORE_SURF_H = 201
    READ_SURF_H = [210 + l for l in range(LAYERS)]

    c.submit()
    c.create(ARRAY, R32_FLOAT, BIND_SAMPLER_VIEW, SIDE, array=LAYERS, target=TARGET_2D_ARRAY)
    c.create(STORE_TARGET, B8G8R8A8_UNORM, rt, SIDE)
    for h in READ:
        c.create(h, B8G8R8A8_UNORM, rt, SIDE)
    c.create(VBO, R8_UNORM, BIND_VERTEX_BUFFER, 128, 1, target=0)

    # A token budget, not a count: generous, as the parser only needs room.
    c.shader(VS_H, STAGE_VERTEX, VS, tokens=TOKENS)
    c.shader(STORE_FS_H, STAGE_FRAGMENT, STORE_FS, tokens=TOKENS)
    for l in range(LAYERS):
        c.shader(READ_FS_H[l], STAGE_FRAGMENT, read_fs(l), tokens=TOKENS)
    c.rasterizer(RAST_H)
    c.blend(BLEND_H)
    c.dsa(DSA_H)
    c.vertex_elements(VE_H, [(0, 0, R32G32B32A32_FLOAT), (16, 0, R32G32B32A32_FLOAT)])
    c.sampler_state(SAMP_H)
    c.bind_object(OBJ_RASTERIZER, RAST_H)
    c.bind_object(OBJ_BLEND, BLEND_H)
    c.bind_object(OBJ_DSA, DSA_H)
    c.bind_object(OBJ_VERTEX_ELEMENTS, VE_H)
    c.bind_shader(VS_H, STAGE_VERTEX)

    data = b"".join(struct.pack("<8f", x, y, 0.0, 1.0, u, v, 0.0, 0.0) for x, y, u, v in QUAD)
    c.inline_write(VBO, data, len(data), 1, 0)
    c.set_viewport(SIDE, SIDE)
    c.set_vertex_buffers([(32, 0, VBO)])

    # The store: layers 1 to 2 of the four, at level 0.
    c.surface(STORE_SURF_H, STORE_TARGET, B8G8R8A8_UNORM)
    c.set_framebuffer([STORE_SURF_H])
    c.bind_shader(STORE_FS_H, STAGE_FRAGMENT)
    c.set_shader_images(STAGE_FRAGMENT, [(ARRAY, R32_FLOAT, IMAGE_ACCESS_WRITE, 1, 2, 0)])
    c.draw(4)
    c.set_shader_images(STAGE_FRAGMENT, [None])
    c.memory_barrier()

    # The reads: every layer of the whole texture, one offscreen each.
    c.sampler_view(VIEW_H, ARRAY, R32_FLOAT, TARGET_2D_ARRAY, first_layer=0, last_layer=LAYERS - 1)
    c.set_sampler_views(STAGE_FRAGMENT, [VIEW_H])
    c.bind_sampler_states(STAGE_FRAGMENT, [SAMP_H])
    for l in range(LAYERS):
        c.surface(READ_SURF_H[l], READ[l], B8G8R8A8_UNORM)
        c.set_framebuffer([READ_SURF_H[l]])
        c.bind_shader(READ_FS_H[l], STAGE_FRAGMENT)
        c.draw(4)

    c.set_framebuffer([])
    c.set_sampler_views(STAGE_FRAGMENT, [])
    c.destroy_object(OBJ_SURFACE, STORE_SURF_H)
    for h in READ_SURF_H:
        c.destroy_object(OBJ_SURFACE, h)
    c.destroy_object(OBJ_SAMPLER_VIEW, VIEW_H)
    c.submit()

    # The sweep reads each 2D offscreen at its unref. The store's own target is scored as the
    # control that its draw ran at all.
    for h in [STORE_TARGET] + READ + [ARRAY]:
        c.unref(h)
    c.submit()
    return c


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "../vm/captures/image.bin"
    blob = build().dump()
    open(out, "wb").write(blob)
    print("wrote %s: %d bytes, crc32 %08x" % (out, len(blob), zlib.crc32(blob)))


if __name__ == "__main__":
    main()
