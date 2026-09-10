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

# shellcheck disable=SC2086
cc -O2 -Wall -Wextra -o vrend-replay vrend-replay.c \
   -I"$CSRC/src" -I"$CBUILD_INC" -I"$PREFIX/include/virgl" \
   -L"$LIBDIR" -lvirglrenderer -Wl,-rpath,"$LIBDIR" \
   $FRAMEWORKS

echo "built: $PWD/vrend-replay"
# Which leg it linked, said out loud: the whole point of the prefix being a parameter is that the
# answer is not obvious from the command that produced it.
virgl_link_report vrend-replay | grep virgl
