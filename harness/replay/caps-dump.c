// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Gustavo Noronha Silva

/* caps-dump -- init a virglrenderer and write the classic capsets it fills.
 *
 * The capset is the whole of what a guest's driver configures itself from, so a difference
 * between two renderers' capsets is a difference in every guest that runs on them. This writes
 * one in the fixture's format (`vrend-replay-caps.h`, shared with the replayer so the two cannot
 * drift) and nothing else.
 *
 * Separate from the replayer because it has to build against a renderer that is NOT the limina
 * fork: the replayer calls the fork's journal, content-export and IOSurface entry points, and a
 * stock libvirglrenderer exports none of them. The reference leg worth measuring is the C a VMM
 * actually loaded, which on a Linux host is the distribution's stock build.
 *
 * The flag word is an argument because the GL-vs-GLES choice lives in it and a capset is a
 * different capset on either side of that choice. A VMM that passes no USE_GLES gets a desktop
 * GL context and a capset built from it; passing 0x1b here reproduces the replayer's GLES
 * default. What a given VMM passed is a fact to look up, not to assume.
 *
 *   caps-dump <out> [flags]     flags default to 0, which is what QEMU passes.
 */

#define VIRGL_RENDERER_UNSTABLE_APIS
#include "virglrenderer.h"
#include "virgl_hw.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vrend-replay-caps.h"

/* The renderer requires callbacks and a non-NULL cookie; a caps dump drives nothing that would
 * call them back, so they are present and empty rather than absent.
 *
 * Version 3 is a floor, not a formality: the table is size-versioned, a renderer reads only the
 * prefix the version names, and virglrs refuses anything below 3 outright because a v2 caller has
 * nowhere for a context fence to retire to. The whole struct is allocated either way, so naming a
 * version here promises a prefix that is really there. */
static void cb_write_fence(void *cookie, uint32_t fence) { (void)cookie; (void)fence; }
static void cb_write_context_fence(void *cookie, uint32_t ctx, uint32_t ring, uint64_t fence)
{ (void)cookie; (void)ctx; (void)ring; (void)fence; }

static struct virgl_renderer_callbacks cbs = {
   .version = 3,
   .write_fence = cb_write_fence,
   .write_context_fence = cb_write_context_fence,
};

int main(int argc, char **argv)
{
   if (argc < 2) {
      fprintf(stderr, "usage: caps-dump <out> [flags]\n");
      return 2;
   }
   const char *out = argv[1];
   int flags = argc > 2 ? (int)strtoul(argv[2], NULL, 0) : 0;

   /* Before init, deliberately. A VMM decides how many capsets to expose to its guest from
    * this query, and QEMU makes it while realizing the device -- which is before it calls
    * virgl_renderer_init. A renderer that answers 0 here tells the VMM it has no VIRGL2, and the
    * guest is then offered v1 for the life of the machine however rich the capset later becomes.
    * So the answer before init is a fact about the capset, not a fact about the renderer's
    * state, and this prints both to keep the two honest. */
   for (uint32_t set = 1; set <= 2; set++) {
      uint32_t v = 0xdeadbeef, sz = 0xdeadbeef;
      virgl_renderer_get_cap_set(set, &v, &sz);
      fprintf(stderr, "caps-dump: pre-init  get_cap_set(%u) -> max_ver=%u max_size=%u\n", set, v, sz);
   }

   static int cookie;
   int ret = virgl_renderer_init(&cookie, flags, &cbs);
   if (ret) {
      fprintf(stderr, "virgl_renderer_init(flags 0x%x) failed: %d\n", flags, ret);
      return 2;
   }
   fprintf(stderr, "caps-dump: initialised (flags 0x%x)\n", flags);
   for (uint32_t set = 1; set <= 2; set++) {
      uint32_t v = 0xdeadbeef, sz = 0xdeadbeef;
      virgl_renderer_get_cap_set(set, &v, &sz);
      fprintf(stderr, "caps-dump: post-init get_cap_set(%u) -> max_ver=%u max_size=%u\n", set, v, sz);
   }

   int rc = dump_caps(out);
   if (!rc)
      fprintf(stderr, "caps-dump: wrote %s\n", out);
   return rc;
}
