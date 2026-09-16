#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright © 2026 Gustavo Noronha Silva
#
# Write a synthetic classic corpus for a sampler-only change between two draws.
#
#   make-sampler-corpus.py [OUT]      default ../vm/captures/sampler.bin
#
# Two draws of one textured quad through one sampler view, one program and one vertex buffer.
# Between them the guest changes exactly one thing about the unit: the sampler state bound to it,
# from clamp-to-edge to repeat. The texture coordinates run to 2.0, so the wrap mode decides
# three quarters of the pixels and the two reads cannot hash equal by accident.
#
# A renderer that re-binds a unit only when its VIEW changed draws the second quad through the
# first sampler, and the two reads come back identical. The C marks the unit dirty in
# `vrend_bind_sampler_states`, so on that leg they differ. No recorded session isolates this: a
# desktop changes a sampler beside a view, a shader or a framebuffer, any of which re-binds the
# unit for its own reason and covers for a sampler bind that did not.
#
# So the stream between the draws carries only what the second destination needs -- its
# framebuffer -- and the sampler bind. A second view bind, a shader bind or a fresh vertex buffer
# would dirty the unit on its own, and the corpus would then measure that instead.
import struct, sys, zlib

from corpus import (Corpus, BIND_RENDER_TARGET, BIND_SAMPLER_VIEW, BIND_VERTEX_BUFFER,
                    B8G8R8A8_UNORM, R32G32B32A32_FLOAT,
                    OBJ_BLEND, OBJ_DSA, OBJ_RASTERIZER, OBJ_SAMPLER_VIEW, OBJ_SURFACE,
                    OBJ_VERTEX_ELEMENTS,
                    STAGE_VERTEX, STAGE_FRAGMENT, TARGET_2D)

SIDE = 64
R8_UNORM = 64

# pipe_tex_wrap. Not CLAMP (1): on a GLES host both legs remap it, and the corpus should not
# depend on that remap agreeing.
WRAP_REPEAT, WRAP_CLAMP_TO_EDGE = 0, 2

# A quad covering the whole target, as a triangle strip. Positions are clip space; the texture
# coordinate runs 0..2 so the right and top halves are where the wrap mode shows.
QUAD = [(-1.0, -1.0, 0.0, 0.0), (1.0, -1.0, 2.0, 0.0),
        (-1.0, 1.0, 0.0, 2.0), (1.0, 1.0, 2.0, 2.0)]

VS = """VERT
DCL IN[0]
DCL IN[1]
DCL OUT[0], POSITION
DCL OUT[1], GENERIC[0]
0: MOV OUT[0], IN[0]
1: MOV OUT[1], IN[1]
2: END"""

FS = """FRAG
DCL IN[0], GENERIC[0], PERSPECTIVE
DCL OUT[0], COLOR
DCL SAMP[0]
0: TEX OUT[0], IN[0], SAMP[0], 2D
1: END"""


def pattern(tag, w, h):
    """Pixels with no symmetry in x or y and no two channels alike, so a clamped edge and a
    repeated tile cannot hash equal, and neither can hash like the source."""
    px = bytearray()
    for y in range(h):
        for x in range(w):
            px += bytes(((x * 3 + tag * 37) & 0xFF, (y * 5 + tag * 11) & 0xFF,
                         ((x ^ y) * 7 + tag * 53) & 0xFF, (0x30 + x + y + tag) & 0xFF))
    return bytes(px)


def build():
    c = Corpus()
    tex = BIND_RENDER_TARGET | BIND_SAMPLER_VIEW

    SRC = 30
    READ_CLAMP, READ_REPEAT = 31, 32
    VBO = 33

    # Handle numbering is by kind so a refused object names itself in a log.
    VS_H, FS_H = 100, 101
    RAST_H, BLEND_H, DSA_H, VE_H = 110, 111, 112, 113
    SAMP_CLAMP_H, SAMP_REPEAT_H = 114, 115
    VIEW_H = 200
    SURF_CLAMP_H, SURF_REPEAT_H = 201, 202

    c.submit()
    c.create(SRC, B8G8R8A8_UNORM, tex, SIDE)
    c.create(READ_CLAMP, B8G8R8A8_UNORM, tex, SIDE)
    c.create(READ_REPEAT, B8G8R8A8_UNORM, tex, SIDE)
    c.create(VBO, R8_UNORM, BIND_VERTEX_BUFFER, 128, 1, target=0)

    c.shader(VS_H, STAGE_VERTEX, VS)
    c.shader(FS_H, STAGE_FRAGMENT, FS)
    c.rasterizer(RAST_H)
    c.blend(BLEND_H)
    c.dsa(DSA_H)
    # Two attributes out of one buffer: position at 0, texture coordinate at 16.
    c.vertex_elements(VE_H, [(0, 0, R32G32B32A32_FLOAT), (16, 0, R32G32B32A32_FLOAT)])
    c.sampler_state(SAMP_CLAMP_H, wrap=WRAP_CLAMP_TO_EDGE)
    c.sampler_state(SAMP_REPEAT_H, wrap=WRAP_REPEAT)
    c.bind_object(OBJ_RASTERIZER, RAST_H)
    c.bind_object(OBJ_BLEND, BLEND_H)
    c.bind_object(OBJ_DSA, DSA_H)
    c.bind_object(OBJ_VERTEX_ELEMENTS, VE_H)
    c.bind_shader(VS_H, STAGE_VERTEX)
    c.bind_shader(FS_H, STAGE_FRAGMENT)

    c.inline_write(SRC, pattern(1, SIDE, SIDE), SIDE, SIDE, SIDE * 4)
    data = b"".join(struct.pack("<8f", x, y, 0.0, 1.0, u, v, 0.0, 0.0) for x, y, u, v in QUAD)
    c.inline_write(VBO, data, len(data), 1, 0)

    c.sampler_view(VIEW_H, SRC, B8G8R8A8_UNORM, TARGET_2D)
    c.surface(SURF_CLAMP_H, READ_CLAMP, B8G8R8A8_UNORM)
    c.surface(SURF_REPEAT_H, READ_REPEAT, B8G8R8A8_UNORM)

    # Everything both draws share, bound once.
    c.set_viewport(SIDE, SIDE)
    c.set_vertex_buffers([(32, 0, VBO)])
    c.set_sampler_views(STAGE_FRAGMENT, [VIEW_H])

    c.set_framebuffer([SURF_CLAMP_H])
    c.bind_sampler_states(STAGE_FRAGMENT, [SAMP_CLAMP_H])
    c.draw(4)

    # The change under test, and the destination to see it in. Nothing else.
    c.set_framebuffer([SURF_REPEAT_H])
    c.bind_sampler_states(STAGE_FRAGMENT, [SAMP_REPEAT_H])
    c.draw(4)

    # Unbind and destroy the per-draw objects, the way a guest releases them: a view or a surface
    # holds its resource, and the unref sweep below has to be a state a guest can reach.
    c.set_framebuffer([])
    c.set_sampler_views(STAGE_FRAGMENT, [])
    c.destroy_object(OBJ_SURFACE, SURF_CLAMP_H)
    c.destroy_object(OBJ_SURFACE, SURF_REPEAT_H)
    c.destroy_object(OBJ_SAMPLER_VIEW, VIEW_H)
    c.submit()

    # The sweep reads each scored offscreen AT its unref. The source is scored too, as the
    # control that the pattern landed as written.
    for h in (SRC, READ_CLAMP, READ_REPEAT):
        c.unref(h)
    c.submit()
    return c


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "../vm/captures/sampler.bin"
    blob = build().dump()
    open(out, "wb").write(blob)
    print("wrote %s: %d bytes, crc32 %08x" % (out, len(blob), zlib.crc32(blob)))


if __name__ == "__main__":
    main()
