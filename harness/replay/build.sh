#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright © 2026 Gustavo Noronha Silva
#
# Build vrend-replay against a chosen virglrenderer prefix.
#
# The prefix is the whole point: this tool exists to run the SAME corpus against two
# implementations, so which library it links is a parameter, not a constant. Linking a stock
# system virglrenderer would produce a replayer that runs and reports nothing meaningful —
# `otool -L` afterwards is the check, exactly as for the limina worker.
#
#   VIRGL_PREFIX=/path/to/prefix ./build.sh
#
# Defaults to this tree's own build/ if it has been configured, else the limina C prefix.
set -eu
cd "$(dirname "$0")"
SRC="$(cd ../.. && pwd)"
# The C headers this replayer speaks -- `virgl_hw.h` and the generated `config.h` -- come from the
# vendored C tree and the build the harness makes of it, not from `$SRC/src`, which in this
# repository is Rust.
CSRC="$SRC/third_party/virglrenderer"
. "$SRC/scripts/platform.sh"
CBUILD_INC="$(virgl_generated_include "$SRC")" || {
  echo "no built C tree to take generated headers from: run scripts/build-reference.sh" >&2
  exit 1
}

PREFIX="${VIRGL_PREFIX:-$SRC/third_party/virgl-prefix}"
LIB="$(virgl_find_lib "$PREFIX")" || {
  echo "no libvirglrenderer under $PREFIX — build it first" >&2; exit 1; }
LIBDIR="$(dirname "$LIB")"

# The IOSurface scanout leg is Darwin's; nothing links those frameworks elsewhere.
FRAMEWORKS=""
if [ "$(uname -s)" = Darwin ]; then
  FRAMEWORKS="-framework IOSurface -framework CoreFoundation"
fi

# Whether this leg carries the limina extensions -- the snapshot journal and the IOSurface reads.
# Asked of the library being linked rather than assumed from the platform, because it is a
# property of the leg: a stock upstream virglrenderer exports none of them, and calling one is an
# undefined symbol at link time. The replayer compiles those sections out and refuses the options
# that drive them, rather than failing to build or, worse, reporting a gate it never ran.
NEEDED="virgl_renderer_limina_journal_export virgl_renderer_limina_journal_restore
virgl_renderer_limina_journal_replay_upto virgl_renderer_limina_replay_begin
virgl_renderer_limina_replay_end virgl_renderer_limina_classic_content_export
virgl_renderer_limina_classic_content_restore virgl_renderer_limina_dump_state"
HAVE_LIMINA=1
EXPORTS="$(virgl_exported_symbols "$LIB")"
for sym in $NEEDED; do
  printf '%s\n' "$EXPORTS" | grep -qx "$sym" || { HAVE_LIMINA=0; break; }
done

# shellcheck disable=SC2086
cc -O2 -Wall -Wextra -o vrend-replay vrend-replay.c \
   -DHAVE_LIMINA_EXT=$HAVE_LIMINA \
   -I"$CSRC/src" -I"$CBUILD_INC" -I"$PREFIX/include/virgl" \
   -L"$LIBDIR" -lvirglrenderer -Wl,-rpath,"$LIBDIR" \
   $FRAMEWORKS

if [ "$HAVE_LIMINA" = 0 ]; then
  echo "note: this leg exports no limina extensions -- the snapshot-journal gate (--rebuild)"
  echo "      is compiled out and will be refused rather than silently skipped."
fi
echo "built: $PWD/vrend-replay"
# Which leg it linked, said out loud: the whole point of the prefix being a parameter is that the
# answer is not obvious from the command that produced it.
virgl_link_report vrend-replay | grep virgl
