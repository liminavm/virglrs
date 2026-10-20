#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright © 2026 Gustavo Noronha Silva
#
# Write a synthetic classic corpus in which two contexts live one after the other on the same id.
#
#   make-ctx-reuse-corpus.py [OUT]      default ../vm/captures/ctx-reuse.bin
#
# A guest hands a destroyed context's id to the next context it creates, so a session of
# short-lived processes -- a test suite, one process per test -- is many contexts on a few ids.
# A context that faults stays faulted: both renderers refuse every later batch on it. A negative
# test faults its context on purpose and exits, and replayed as ONE context per id, every process
# after it on that id is refused with it -- most of a piglit session's batches, measured. The
# trace's version-3 context records are what say where one context ended and the next began.
#
#   first   The first context draws COLOUR_FIRST into RES_FIRST, then sends a surface over a
#           resource that does not exist, which faults it, and is destroyed.
#   second  A new context on the same id builds the same handles again, with a different
#           fragment shader, and draws COLOUR_SECOND into RES_SECOND.
#
# Both targets are pre-filled with a pattern, so a draw that never lands scores that pattern, and
# the colours differ in every channel, so the second context drawing with the first one's shader
# is a wrong hash rather than an invisible one.
#
# ARMING THE CONTROL: replay with `--no-ctx-lifecycle`, which ignores the context records and
# keeps one context for the id. The second context's batch is then refused as part of the first
# one's fault (`submit-errors` counts it) and RES_SECOND scores its fill pattern. The pinned score
# is the C leg's WITH the records followed.
import struct
import sys
import zlib

from corpus import (Corpus, BIND_RENDER_TARGET, BIND_SAMPLER_VIEW, BIND_VERTEX_BUFFER,
                    B8G8R8A8_UNORM, R32G32B32A32_FLOAT,
                    OBJ_BLEND, OBJ_DSA, OBJ_RASTERIZER, OBJ_SURFACE, OBJ_VERTEX_ELEMENTS,
                    STAGE_VERTEX, STAGE_FRAGMENT)

SIDE = 64   # the sweep scores 2D colour targets wider than 8

R8_UNORM = 64

QUAD = [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]

VS = """VERT
DCL IN[0]
DCL OUT[0], POSITION
0: MOV OUT[0], IN[0]
1: END"""


def fs(rgba):
    """A fragment shader writing one constant colour, its immediate spelled as the guest's are."""
    bits = [struct.unpack("<I", struct.pack("<f", v))[0] for v in rgba]
    return """FRAG
DCL OUT[0], COLOR
IMM[0] UINT32 {%d, %d, %d, %d}
0: MOV OUT[0], IMM[0]
1: END""" % tuple(bits)


COLOUR_FIRST = (0.85, 0.20, 0.10, 1.0)
COLOUR_SECOND = (0.10, 0.80, 0.25, 1.0)

# The same handles in both contexts: that is the collision.
VS_H, FS_H = 100, 101
RAST_H, BLEND_H, DSA_H, VE_H = 110, 111, 112, 113
SURF_H = 120
SURF_BAD_H = 121
MISSING_RES = 999   # no resource of that handle is ever created

RES_FIRST, RES_SECOND = 10, 11
VBO_FIRST, VBO_SECOND = 20, 21


def fill_pixels(tag):
    """A pattern with no flat channel, so a draw that never lands is its own hash."""
    px = bytearray()
    for y in range(SIDE):
        for x in range(SIDE):
            px += bytes(((x * 3 + tag * 37) & 0xFF, (y * 5 + tag * 11) & 0xFF,
                         ((x ^ y) * 7 + tag * 53) & 0xFF, (0x30 + x + y + tag) & 0xFF))
    return bytes(px)


def process(c, colour, vbo, dst):
    """One guest process's whole life on the context: build, draw into `dst`, tear down."""
    c.submit()
    c.shader(VS_H, STAGE_VERTEX, VS)
    c.shader(FS_H, STAGE_FRAGMENT, fs(colour))
    c.rasterizer(RAST_H)
    c.blend(BLEND_H)
    c.dsa(DSA_H)
    c.vertex_elements(VE_H, [(0, 0, R32G32B32A32_FLOAT)])
    c.bind_object(OBJ_RASTERIZER, RAST_H)
    c.bind_object(OBJ_BLEND, BLEND_H)
    c.bind_object(OBJ_DSA, DSA_H)
    c.bind_object(OBJ_VERTEX_ELEMENTS, VE_H)
    c.bind_shader(VS_H, STAGE_VERTEX)
    c.bind_shader(FS_H, STAGE_FRAGMENT)
    c.create(vbo, R8_UNORM, BIND_VERTEX_BUFFER, 128, 1, target=0)
    data = b"".join(struct.pack("<4f", x, y, 0.0, 1.0) for x, y in QUAD)
    c.inline_write(vbo, data, len(data), 1, 0)
    c.set_vertex_buffers([(16, 0, vbo)])
    c.surface(SURF_H, dst, B8G8R8A8_UNORM)
    c.set_framebuffer([SURF_H])
    c.set_viewport(SIDE, SIDE)
    c.draw(4)
    c.set_framebuffer([])
    c.destroy_object(OBJ_SURFACE, SURF_H)
    c.submit()


def build():
    c = Corpus()
    tex = BIND_RENDER_TARGET | BIND_SAMPLER_VIEW
    c.ctx_create()
    c.submit()
    for h in (RES_FIRST, RES_SECOND):
        c.create(h, B8G8R8A8_UNORM, tex, SIDE)
        c.inline_write(h, fill_pixels(h), SIDE, SIDE, SIDE * 4)

    process(c, COLOUR_FIRST, VBO_FIRST, RES_FIRST)
    # A negative test's last act: a surface over a resource nobody created faults the context.
    c.submit()
    c.surface(SURF_BAD_H, MISSING_RES, B8G8R8A8_UNORM)
    c.submit()
    # The process exits without deleting its objects one by one, as a real one does: the
    # context's destruction is what takes them.
    c.ctx_destroy()
    c.ctx_create()
    process(c, COLOUR_SECOND, VBO_SECOND, RES_SECOND)

    c.submit()
    # Only the scored targets are unreffed: they are scored at their unref. The vertex buffers stay,
    # as teardown.bin's do, so the rebuild gate is not asked to rebuild a binding to a resource
    # the stream has already let go.
    for h in (RES_FIRST, RES_SECOND):
        c.unref(h)
    c.submit()
    # The second process exits too, as every process in a test session does: a stream can end
    # with no context alive, and the rebuild gate has to say it skipped them rather than fail.
    c.ctx_destroy()
    return c


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "../vm/captures/ctx-reuse.bin"
    blob = build().dump()
    open(out, "wb").write(blob)
    print("wrote %s: %d bytes, crc32 %08x" % (out, len(blob), zlib.crc32(blob)))


if __name__ == "__main__":
    main()
