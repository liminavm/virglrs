#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright © 2026 Gustavo Noronha Silva
#
# Write a synthetic classic corpus for compute dispatch.
#
#   make-compute-corpus.py [OUT]      default ../vm/captures/compute.bin
#
# Three R32 textures, each written by a compute shader through a shader image and then sampled
# by a draw into a 2D offscreen, which the sweep scores at its unref:
#
# - A, a direct dispatch of 2x2 blocks of 8x8 threads: the whole texture, a gradient that numbers
#   every texel, so a grid or block read wrongly lands somewhere the score sees.
# - B, the same shader dispatched indirectly, from a buffer that asks for 2x1x1 blocks: the top
#   half only. A dispatch that took its grid from anywhere but the buffer writes the bottom half
#   too, or nothing.
# - C, a second compute shader, the gradient reversed, dispatched after the reads of A and B have
#   drawn. Graphics and compute share the program the sub-context runs, so this is the switch in
#   both directions: the dispatch must leave the draw's program, and the draw after it must come
#   back to its own.
#
# No recorded session dispatches compute: a desktop does not. The textures are read back through
# a draw because the sweep reads only plain 2D colour targets.
import struct, sys, zlib

from corpus import (Corpus, BIND_RENDER_TARGET, BIND_SAMPLER_VIEW, BIND_VERTEX_BUFFER,
                    B8G8R8A8_UNORM, R32_FLOAT, R32G32B32A32_FLOAT, IMAGE_ACCESS_WRITE,
                    OBJ_BLEND, OBJ_DSA, OBJ_RASTERIZER, OBJ_SAMPLER_VIEW, OBJ_SURFACE,
                    OBJ_VERTEX_ELEMENTS,
                    STAGE_VERTEX, STAGE_FRAGMENT, TARGET_2D)

SIDE = 16
BLOCK = 8
R8_UNORM = 64
TOKENS = 300
STAGE_COMPUTE = 5
BIND_COMMAND_ARGS = 1 << 8
CCMD_LAUNCH_GRID = 37

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

READ_FS = """FRAG
DCL IN[0], GENERIC[0], PERSPECTIVE
DCL OUT[0], COLOR
DCL SAMP[0]
DCL SVIEW[0], 2D, FLOAT
0: TEX OUT[0], IN[0], SAMP[0], 2D
1: END"""


def gradient_cs(scale, bias):
    """Store `(x + 16 * y + 1) * scale + bias` at every texel (x, y) the dispatch reaches."""
    return """COMP
PROPERTY CS_FIXED_BLOCK_WIDTH %d
PROPERTY CS_FIXED_BLOCK_HEIGHT %d
PROPERTY CS_FIXED_BLOCK_DEPTH 1
DCL SV[0], THREAD_ID
DCL SV[1], BLOCK_ID
DCL IMAGE[0], 2D, PIPE_FORMAT_R32_FLOAT, WR
DCL TEMP[0..1]
IMM[0] UINT32 { %d, %d, 1, 0 }
IMM[1] FLT32 { %r, %r, 0.0, 0.0 }
0: UMAD TEMP[0].xy, SV[1].xyyy, IMM[0].xxxx, SV[0].xyyy
1: UMAD TEMP[1].x, TEMP[0].yyyy, IMM[0].yyyy, TEMP[0].xxxx
2: UADD TEMP[1].x, TEMP[1].xxxx, IMM[0].zzzz
3: U2F TEMP[1].x, TEMP[1].xxxx
4: MAD TEMP[1].x, TEMP[1].xxxx, IMM[1].xxxx, IMM[1].yyyy
5: STORE IMAGE[0], TEMP[0], TEMP[1].xxxx, 2D, PIPE_FORMAT_R32_FLOAT
6: END""" % (BLOCK, BLOCK, BLOCK, SIDE, scale, bias)


def launch_grid(c, grid, indirect=0, offset=0):
    """LAUNCH_GRID: the block size the shader fixed, the grid, and an optional indirect buffer
    whose three words at `offset` replace the grid."""
    c.emit(CCMD_LAUNCH_GRID, 0, [BLOCK, BLOCK, 1] + list(grid) + [indirect, offset])


def build():
    c = Corpus()
    rt = BIND_RENDER_TARGET | BIND_SAMPLER_VIEW

    TEX = {"a": 30, "b": 31, "c": 32}
    READ = {"a": 40, "b": 41, "c": 42}
    VBO, ARGS = 33, 34

    VS_H, READ_FS_H, CS_UP_H, CS_DOWN_H = 100, 101, 102, 103
    RAST_H, BLEND_H, DSA_H, VE_H, SAMP_H = 110, 111, 112, 113, 114
    VIEW_H = {"a": 200, "b": 201, "c": 202}
    SURF_H = {"a": 210, "b": 211, "c": 212}

    c.submit()
    for k in "abc":
        c.create(TEX[k], R32_FLOAT, BIND_SAMPLER_VIEW, SIDE, target=TARGET_2D)
        c.create(READ[k], B8G8R8A8_UNORM, rt, SIDE)
    c.create(VBO, R8_UNORM, BIND_VERTEX_BUFFER, 128, 1, target=0)
    # Leading padding, so the dispatch reads its words from an offset and not from the start.
    c.create(ARGS, R8_UNORM, BIND_COMMAND_ARGS, 32, 1, target=0)

    c.shader(VS_H, STAGE_VERTEX, VS, tokens=TOKENS)
    c.shader(READ_FS_H, STAGE_FRAGMENT, READ_FS, tokens=TOKENS)
    c.shader(CS_UP_H, STAGE_COMPUTE, gradient_cs(1.0 / 256, 0.0), tokens=TOKENS)
    c.shader(CS_DOWN_H, STAGE_COMPUTE, gradient_cs(-1.0 / 256, 1.0), tokens=TOKENS)
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
    c.bind_shader(READ_FS_H, STAGE_FRAGMENT)
    c.bind_sampler_states(STAGE_FRAGMENT, [SAMP_H])

    data = b"".join(struct.pack("<8f", x, y, 0.0, 1.0, u, v, 0.0, 0.0) for x, y, u, v in QUAD)
    c.inline_write(VBO, data, len(data), 1, 0)
    args = struct.pack("<4I", 0xdead, 2, 1, 1)
    c.inline_write(ARGS, args, len(args), 1, 0)
    c.set_viewport(SIDE, SIDE)
    c.set_vertex_buffers([(32, 0, VBO)])

    def image(k):
        c.set_shader_images(STAGE_COMPUTE, [(TEX[k], R32_FLOAT, IMAGE_ACCESS_WRITE, 0, 0, 0)])

    def read(k):
        c.sampler_view(VIEW_H[k], TEX[k], R32_FLOAT, TARGET_2D)
        c.set_sampler_views(STAGE_FRAGMENT, [VIEW_H[k]])
        c.surface(SURF_H[k], READ[k], B8G8R8A8_UNORM)
        c.set_framebuffer([SURF_H[k]])
        c.draw(4)

    c.bind_shader(CS_UP_H, STAGE_COMPUTE)
    image("a")
    launch_grid(c, [SIDE // BLOCK, SIDE // BLOCK, 1])
    image("b")
    launch_grid(c, [0, 0, 0], indirect=ARGS, offset=4)
    c.memory_barrier()
    read("a")
    read("b")

    c.bind_shader(CS_DOWN_H, STAGE_COMPUTE)
    image("c")
    launch_grid(c, [SIDE // BLOCK, SIDE // BLOCK, 1])
    c.memory_barrier()
    read("c")

    c.set_shader_images(STAGE_COMPUTE, [None])
    c.set_framebuffer([])
    c.set_sampler_views(STAGE_FRAGMENT, [])
    for k in "abc":
        c.destroy_object(OBJ_SURFACE, SURF_H[k])
        c.destroy_object(OBJ_SAMPLER_VIEW, VIEW_H[k])
    c.submit()

    for k in "abc":
        c.unref(READ[k])
    for k in "abc":
        c.unref(TEX[k])
    c.submit()
    return c


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "../vm/captures/compute.bin"
    blob = build().dump()
    open(out, "wb").write(blob)
    print("wrote %s: %d bytes, crc32 %08x" % (out, len(blob), zlib.crc32(blob)))


if __name__ == "__main__":
    main()
