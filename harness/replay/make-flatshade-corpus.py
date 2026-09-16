#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright © 2026 Gustavo Noronha Silva
#
# Write a synthetic classic corpus for a rasterizer-only change between two draws.
#
#   make-flatshade-corpus.py [OUT]      default ../vm/captures/flatshade.bin
#
# Two draws of one triangle with a different colour at each vertex, into one RGBA target,
# through one program. Between them the guest changes exactly one thing: the rasterizer bound,
# from smooth to flat shading. Flat shading is not a GL state on this host; it is the `flat`
# qualifier the shader translator puts on the colour varying when the rasterizer's bit is in
# the fragment shader's key. So the second draw needs a different fragment program from the
# first, and a renderer only gets one if the rasterizer bind marked the shader dirty.
#
# The C marks it in `vrend_bind_object` for a rasterizer. A renderer that does not draws the
# second triangle through the first draw's smooth program, and the target reads back as the
# gradient of the first draw instead of the one flat colour the C leaves.
#
# The target is RGBA, deliberately. A BGRA target needs a red-blue swizzle in the fragment
# shader, and the draw path reselects the program on every draw while one is bound -- which
# would cover for the missing mark and the corpus would test nothing. Nothing else changes
# between the draws: a framebuffer or shader bind marks the shader dirty on its own, and the
# corpus would then measure that instead.
import struct, sys, zlib

from corpus import (Corpus, BIND_RENDER_TARGET, BIND_SAMPLER_VIEW, BIND_VERTEX_BUFFER,
                    R8G8B8A8_UNORM, R32G32B32A32_FLOAT,
                    OBJ_BLEND, OBJ_DSA, OBJ_RASTERIZER, OBJ_SURFACE, OBJ_VERTEX_ELEMENTS,
                    PRIM_TRIANGLES, STAGE_VERTEX, STAGE_FRAGMENT)

SIDE = 64
R8_UNORM = 64

# One triangle covering most of the target, red, green and blue at its corners. Flat shading
# paints the whole of it with the provoking vertex's colour; smooth shading paints a gradient.
# Neither hashes like the other, and neither hashes like the clear.
TRIANGLE = [(-0.9, -0.9, 1.0, 0.0, 0.0), (0.9, -0.9, 0.0, 1.0, 0.0), (0.0, 0.9, 0.0, 0.0, 1.0)]

VS = """VERT
DCL IN[0]
DCL IN[1]
DCL OUT[0], POSITION
DCL OUT[1], COLOR
0: MOV OUT[0], IN[0]
1: MOV OUT[1], IN[1]
2: END"""

# The input is declared with the COLOR interpolation mode: that is the one the translator turns
# into `flat` under a flat-shading rasterizer. A GENERIC varying would be smooth under both.
FS = """FRAG
DCL IN[0], COLOR, COLOR
DCL OUT[0], COLOR
0: MOV OUT[0], IN[0]
1: END"""


def build():
    c = Corpus()

    TARGET = 30
    VBO = 33

    VS_H, FS_H = 100, 101
    RAST_SMOOTH_H, RAST_FLAT_H, BLEND_H, DSA_H, VE_H = 110, 111, 112, 113, 114
    SURF_H = 201

    c.submit()
    c.create(TARGET, R8G8B8A8_UNORM, BIND_RENDER_TARGET | BIND_SAMPLER_VIEW, SIDE)
    c.create(VBO, R8_UNORM, BIND_VERTEX_BUFFER, 128, 1, target=0)

    c.shader(VS_H, STAGE_VERTEX, VS)
    c.shader(FS_H, STAGE_FRAGMENT, FS)
    c.rasterizer(RAST_SMOOTH_H)
    c.rasterizer(RAST_FLAT_H, flatshade=True)
    c.blend(BLEND_H)
    c.dsa(DSA_H)
    # Two attributes out of one buffer: position at 0, colour at 16.
    c.vertex_elements(VE_H, [(0, 0, R32G32B32A32_FLOAT), (16, 0, R32G32B32A32_FLOAT)])
    c.bind_object(OBJ_RASTERIZER, RAST_SMOOTH_H)
    c.bind_object(OBJ_BLEND, BLEND_H)
    c.bind_object(OBJ_DSA, DSA_H)
    c.bind_object(OBJ_VERTEX_ELEMENTS, VE_H)
    c.bind_shader(VS_H, STAGE_VERTEX)
    c.bind_shader(FS_H, STAGE_FRAGMENT)

    data = b"".join(struct.pack("<8f", x, y, 0.0, 1.0, r, g, b, 1.0) for x, y, r, g, b in TRIANGLE)
    c.inline_write(VBO, data, len(data), 1, 0)

    c.surface(SURF_H, TARGET, R8G8B8A8_UNORM)
    c.set_viewport(SIDE, SIDE)
    c.set_vertex_buffers([(32, 0, VBO)])
    c.set_framebuffer([SURF_H])

    c.draw(3, PRIM_TRIANGLES)

    # The change under test. Nothing else: the same target, so that no framebuffer bind marks
    # the shader dirty on the corpus's behalf.
    c.bind_object(OBJ_RASTERIZER, RAST_FLAT_H)
    c.draw(3, PRIM_TRIANGLES)

    c.set_framebuffer([])
    c.destroy_object(OBJ_SURFACE, SURF_H)
    c.submit()

    # The sweep reads the target at its unref: the flat triangle over the gradient, or the
    # gradient alone.
    c.unref(TARGET)
    c.submit()
    return c


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "../vm/captures/flatshade.bin"
    blob = build().dump()
    open(out, "wb").write(blob)
    print("wrote %s: %d bytes, crc32 %08x" % (out, len(blob), zlib.crc32(blob)))


if __name__ == "__main__":
    main()
