#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright © 2026 Gustavo Noronha Silva
#
# Build THIS tree's virglrenderer into harness/vm/prefix, with the same options the limina
# worker is built against: venus + video, surfaceless EGL, in-process render server.
#
# Two host prefixes still come from the limina checkout — an epoxy built WITH EGL (Homebrew's is
# CGL-only) and the zink-on-KosmicKrisp Mesa on its case-sensitive volume. Reproducing those here
# would mean vendoring two more forks; borrowing them is the honest short path, and it is one of
# the things the P6 reconcile has to settle.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"

LIMINA="${LIMINA_ROOT:-$HOME/Projects/limina}"
EPOXY="$LIMINA/third_party/epoxy-egl-prefix"
MESA="${MESA_PREFIX:-/Volumes/mesa-cs/zink-kk-prefix}"

grep -qi epoxy_has_egl=1 "$EPOXY/lib/pkgconfig/epoxy.pc" 2>/dev/null || {
  echo "epoxy-with-EGL missing at $EPOXY" >&2; exit 1; }
[ -f "$MESA/lib/pkgconfig/egl.pc" ] || {
  echo "zink-on-KK Mesa missing at $MESA (mount the case-sensitive volume)" >&2; exit 1; }

export PKG_CONFIG_PATH="$EPOXY/lib/pkgconfig:$MESA/lib/pkgconfig:$(brew --prefix)/opt/molten-vk/lib/pkgconfig:$(brew --prefix)/lib/pkgconfig:$(brew --prefix)/share/pkgconfig"

BUILD="$ROOT/harness/vm/build"
PREFIX="$ROOT/harness/vm/prefix"
ARGS=(-Dvenus=true -Dvideo=true -Dvulkan-dload=false -Drender-server-mode=thread
      -Drender-server-worker=thread -Dplatforms=egl --prefix "$PREFIX" --buildtype release)

# The C tree is a pinned build input under this repository -- `meson.build` is there, not at the
# root, which is where it was when virglrs lived inside it.
CTREE="$ROOT/third_party/virglrenderer"
if [ -d "$BUILD" ]; then meson setup --reconfigure "$BUILD" "$CTREE" "${ARGS[@]}"
else meson setup "$BUILD" "$CTREE" "${ARGS[@]}"; fi
ninja -C "$BUILD"
meson install -C "$BUILD" >/dev/null

echo "==> $PREFIX/lib/libvirglrenderer.1.dylib"
