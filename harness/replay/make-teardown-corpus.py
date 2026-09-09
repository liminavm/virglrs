#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright © 2026 Gustavo Noronha Silva
#
# Write a synthetic classic corpus that destroys a program and a sub-context out from under a
# renderer that is still holding them.
#
#   make-teardown-corpus.py [OUT]      default ../vm/captures/teardown.bin
#
# Both paths run constantly on a real guest -- mesa gives every `pipe_context` its own
# sub-context and destroys it on teardown, and every shader state deletion encodes
# DESTROY_OBJECT/SHADER -- so a recorded corpus reaches them. Reaching is not exercising. What
# neither a recording nor the boot suite selects for is the ORDERING that makes a wrong answer
# visible, and without that ordering a renderer that mishandles either one draws something
# plausible and scores green.
#
#   program-outlives-its-neighbour  Three programs, the MIDDLE one bound, and the FIRST one's
#                                   shader destroyed. Destroying a program shifts every program
#                                   after it down one, so a renderer holding a bare index into
#                                   that list now names the program that was AFTER the bound
#                                   one. Bound at index 1 of three, destroy index 0: a stale
#                                   index 1 names the third program and draws its colour. Two
#                                   programs would not do -- the stale index would be off the
#                                   end, which crashes rather than lies, and a crash is the easy
#                                   half to get right.
#
#   state-survives-a-sub-destroy    A second sub-context, made current, given its own shaders,
#                                   then destroyed while it is the current one. Sub-context 0
#                                   becomes current again and must still hold its own bindings:
#                                   the draw after the destroy names no state at all, so what it
#                                   draws is whatever the restored sub-context had bound.
#
# Each case is scored by drawing a flat colour into its own offscreen, which the sweep reads at
# unref. The colours are distinct per program, so a wrong program is a wrong hash and not a
# missing one -- and every target is pre-filled with a pattern, so a draw that lands nowhere is
# distinguishable from a draw that lands wrong.
#
# ARMING THE CONTROL, and what the control is NOT. A colour assertion means nothing until the
# colours are shown to differ, so `--no-destroy` writes the same corpus with both destroys
# removed.
#
# It is not a differential: a CORRECT renderer scores the two arms identically, because the draw
# after the destroy draws COLOUR_B whether or not the neighbour was destroyed. Expecting the
# arms to differ would condemn a renderer for being right.
#
# What the control establishes is that the three programs' colours land as three DISTINCT hashes
# on RES_A, RES_B and RES_C. That is the property the real corpus leans on: it is what makes
# "drew the wrong program" a different hash on RES_AFTER_DESTROY rather than an invisible one.
# If those three hash alike, this corpus cannot see a wrong program at all and the fixture is
# worth nothing, however green it reads.
import struct
import sys
import zlib

from corpus import (Corpus, BIND_RENDER_TARGET, BIND_SAMPLER_VIEW, BIND_VERTEX_BUFFER,
                    B8G8R8A8_UNORM, R32G32B32A32_FLOAT,
                    OBJ_BLEND, OBJ_DSA, OBJ_RASTERIZER, OBJ_SHADER, OBJ_SURFACE,
                    OBJ_VERTEX_ELEMENTS, STAGE_VERTEX, STAGE_FRAGMENT)

SIDE = 64   # the sweep scores 2D colour targets wider than 8

R8_UNORM = 64

QUAD = [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]

VS = """VERT
DCL IN[0]
DCL OUT[0], POSITION
0: MOV OUT[0], IN[0]
1: END"""


def fs(rgba):
    """A fragment shader writing one constant colour, as an immediate.

    The immediate is spelled UINT32 over the float bit patterns, which is the form the guest's
    own shaders arrive in (`fixtures/vrend-shaders.txt`) -- so the translator sees nothing here
    it does not see from mesa.
    """
    bits = [struct.unpack("<I", struct.pack("<f", v))[0] for v in rgba]
    return """FRAG
DCL OUT[0], COLOR
IMM[0] UINT32 {%d, %d, %d, %d}
0: MOV OUT[0], IMM[0]
1: END""" % tuple(bits)


# Distinct in every channel, so no pair can hash equal through a swizzle or a lost channel.
COLOUR_A = (0.85, 0.20, 0.10, 1.0)
COLOUR_B = (0.10, 0.80, 0.25, 1.0)
COLOUR_C = (0.15, 0.25, 0.90, 1.0)
COLOUR_SUB = (0.95, 0.85, 0.05, 1.0)

VS_H = 100
FS_A_H, FS_B_H, FS_C_H = 101, 102, 103
RAST_H, BLEND_H, DSA_H, VE_H = 110, 111, 112, 113
SUB_VS_H, SUB_FS_H = 120, 121
SUB_RAST_H, SUB_BLEND_H, SUB_DSA_H, SUB_VE_H = 130, 131, 132, 133

RES_A, RES_B, RES_C = 10, 11, 12          # one per program, linking the three
RES_AFTER_DESTROY = 13                     # the bound program, drawn after its neighbour dies
RES_SUB = 14                               # drawn inside the second sub-context
RES_AFTER_SUB = 15                         # drawn in sub 0 after the second is destroyed

SUB_ID = 1   # mesa's first, since its counter is pre-incremented and never yields 0


def fill_pixels(tag):
    """A pattern with no flat channel: a draw that never lands cannot hash equal to one that
    lands in the wrong colour."""
    px = bytearray()
    for y in range(SIDE):
        for x in range(SIDE):
            px += bytes(((x * 3 + tag * 37) & 0xFF, (y * 5 + tag * 11) & 0xFF,
                         ((x ^ y) * 7 + tag * 53) & 0xFF, (0x30 + x + y + tag) & 0xFF))
    return bytes(px)


class Rig:
    """One sub-context's draw state. Each sub-context needs its own: objects are per-sub, and a
    handle created in one names nothing in another."""

    def __init__(self, c, vs_h, rast_h, blend_h, dsa_h, ve_h, vbo_h):
        self.c = c
        self.next_surface = 200 + vbo_h
        c.shader(vs_h, STAGE_VERTEX, VS)
        c.rasterizer(rast_h)
        c.blend(blend_h)
        c.dsa(dsa_h)
        c.vertex_elements(ve_h, [(0, 0, R32G32B32A32_FLOAT)])
        c.bind_object(OBJ_RASTERIZER, rast_h)
        c.bind_object(OBJ_BLEND, blend_h)
        c.bind_object(OBJ_DSA, dsa_h)
        c.bind_object(OBJ_VERTEX_ELEMENTS, ve_h)
        c.bind_shader(vs_h, STAGE_VERTEX)
        # The vertex buffer is a resource, which is context-global, but binding it is per-sub.
        c.create(vbo_h, R8_UNORM, BIND_VERTEX_BUFFER, 128, 1, target=0)
        data = b"".join(struct.pack("<4f", x, y, 0.0, 1.0) for x, y in QUAD)
        c.inline_write(vbo_h, data, len(data), 1, 0)
        c.set_vertex_buffers([(16, 0, vbo_h)])
        self.surfaces = []

    def draw_into(self, dst):
        """Draw the quad into `dst` with whatever fragment shader is bound."""
        c = self.c
        self.next_surface += 1
        surf = self.next_surface
        c.surface(surf, dst, B8G8R8A8_UNORM)
        self.surfaces.append(surf)
        c.set_framebuffer([surf])
        c.set_viewport(SIDE, SIDE)
        c.draw(4)

    def retire(self):
        self.c.set_framebuffer([])
        for surf in self.surfaces:
            self.c.destroy_object(OBJ_SURFACE, surf)
        self.surfaces = []


def build(destroy=True):
    c = Corpus()
    tex = BIND_RENDER_TARGET | BIND_SAMPLER_VIEW

    c.submit()
    for h in (RES_A, RES_B, RES_C, RES_AFTER_DESTROY, RES_SUB, RES_AFTER_SUB):
        c.create(h, B8G8R8A8_UNORM, tex, SIDE)
        c.inline_write(h, fill_pixels(h), SIDE, SIDE, SIDE * 4)

    # ---- program-outlives-its-neighbour ----
    rig = Rig(c, VS_H, RAST_H, BLEND_H, DSA_H, VE_H, 140)
    c.shader(FS_A_H, STAGE_FRAGMENT, fs(COLOUR_A))
    c.shader(FS_B_H, STAGE_FRAGMENT, fs(COLOUR_B))
    c.shader(FS_C_H, STAGE_FRAGMENT, fs(COLOUR_C))

    # One draw per shader, in order, so the program list is [A, B, C] by construction: a program
    # is linked the first time a draw needs it, and appended.
    c.bind_shader(FS_A_H, STAGE_FRAGMENT)
    rig.draw_into(RES_A)
    c.bind_shader(FS_B_H, STAGE_FRAGMENT)
    rig.draw_into(RES_B)
    c.bind_shader(FS_C_H, STAGE_FRAGMENT)
    rig.draw_into(RES_C)

    # Bind the middle one and draw, so that IT is what the renderer holds.
    c.bind_shader(FS_B_H, STAGE_FRAGMENT)
    rig.draw_into(RES_B)

    if destroy:
        # Destroy the FIRST program's shader. It is not the bound one -- a bound shader is kept
        # alive by its slot and its programs are not released, so destroying B here would
        # exercise nothing. Destroying A releases the program at index 0 and shifts B and C down.
        c.destroy_object(OBJ_SHADER, FS_A_H)

    # No bind between the destroy and this draw: the renderer draws with the program it is
    # already holding. Correct is COLOUR_B. A renderer that did not follow B down to its new
    # index draws COLOUR_C, which is the whole point of the middle slot.
    rig.draw_into(RES_AFTER_DESTROY)
    rig.retire()
    c.submit()

    # ---- state-survives-a-sub-destroy ----
    c.create_sub_ctx(SUB_ID)
    c.set_sub_ctx(SUB_ID)
    # A fresh sub-context shares no objects with sub 0, so it builds its own rig from scratch.
    sub_rig = Rig(c, SUB_VS_H, SUB_RAST_H, SUB_BLEND_H, SUB_DSA_H, SUB_VE_H, 141)
    c.shader(SUB_FS_H, STAGE_FRAGMENT, fs(COLOUR_SUB))
    c.bind_shader(SUB_FS_H, STAGE_FRAGMENT)
    sub_rig.draw_into(RES_SUB)
    sub_rig.retire()
    c.submit()

    if destroy:
        # Destroyed while current, which is what a guest does: mesa's context teardown encodes
        # DESTROY_SUB_CTX for its own sub-context. Sub-context 0 has to become current again.
        c.destroy_sub_ctx(SUB_ID)
    else:
        c.set_sub_ctx(0)

    # Nothing is bound here: no shader, no rasterizer, no vertex buffer, no framebuffer beyond
    # the surface. Sub 0's own bindings are what draw this, so COLOUR_B says its state came back
    # intact -- and that the sub-context restored was 0 and not a hole where it used to be.
    rig.draw_into(RES_AFTER_SUB)
    rig.retire()
    c.submit()

    for h in (RES_A, RES_B, RES_C, RES_AFTER_DESTROY, RES_SUB, RES_AFTER_SUB):
        c.unref(h)
    c.submit()
    return c


def main():
    args = [a for a in sys.argv[1:] if a != "--no-destroy"]
    destroy = "--no-destroy" not in sys.argv[1:]
    out = args[0] if args else "../vm/captures/teardown.bin"
    blob = build(destroy).dump()
    open(out, "wb").write(blob)
    print("wrote %s: %d bytes, crc32 %08x%s"
          % (out, len(blob), zlib.crc32(blob), "" if destroy else "  (control: no destroys)"))


if __name__ == "__main__":
    main()
