#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright © 2026 Gustavo Noronha Silva

# Build the pinned C renderer into a prefix, for the harness's other leg.
#
# The harness scores a corpus through both implementations and compares; `--renderer c` needs a
# libvirglrenderer to load, and this builds it. Nothing else in this repository needs it — the
# crate takes only headers and generator inputs from the C tree, never a built library.
#
# Output: third_party/virgl-prefix. Point the harness at it with VIRGL_PREFIX.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/third_party/virglrenderer"
BUILD="$SRC/build"
PREFIX="${VIRGL_PREFIX:-$ROOT/third_party/virgl-prefix}"

[ -f "$SRC/meson.build" ] || { echo "the C tree is not vendored: run scripts/vendor.sh" >&2; exit 1; }

# Two host prefixes this build needs and does not produce. Both come from the limina tree's
# spikes (build-epoxy-egl.sh, build-mesa-zink-kk.sh); override either if yours live elsewhere.
#
# Homebrew's epoxy is CGL-only, and virglrenderer's EGL platform silently builds without EGL
# against it — which is a renderer that cannot bring up host GL at all, discovered at run time.
EPOXY_PREFIX="${EPOXY_PREFIX:-$ROOT/third_party/epoxy-egl-prefix}"
grep -qi epoxy_has_egl=1 "$EPOXY_PREFIX/lib/pkgconfig/epoxy.pc" 2>/dev/null || {
    echo "epoxy-with-EGL missing at $EPOXY_PREFIX (set EPOXY_PREFIX)" >&2; exit 1; }

# epoxy.pc's `Requires.private: egl` means pkg-config must see the zink-on-KK Mesa too, or the
# configure fails with "Could not generate cflags for epoxy".
MESA_PREFIX="${MESA_PREFIX:-/Volumes/mesa-cs/zink-kk-prefix}"
[ -f "$MESA_PREFIX/lib/pkgconfig/egl.pc" ] || {
    echo "zink-on-KK Mesa egl.pc missing at $MESA_PREFIX (set MESA_PREFIX)" >&2; exit 1; }

export PKG_CONFIG_PATH="$EPOXY_PREFIX/lib/pkgconfig:$MESA_PREFIX/lib/pkgconfig:$(brew --prefix)/opt/molten-vk/lib/pkgconfig:$(brew --prefix)/lib/pkgconfig:$(brew --prefix)/share/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"

# The same configuration limina ran the C renderer under, so the goldens are recorded from the
# renderer that shipped rather than from a differently-built one: venus and vrend in one process
# (render-server-mode=thread), EGL platform, the VideoToolbox-backed video path, and the Vulkan
# library linked rather than dlopened by bare soname.
# Unquoted on purpose: empty must expand to no argument at all. Quoted, a first build passes
# meson an empty string, which it takes for the build directory and then rejects the source
# tree as an extra -- so this only ever worked where a build directory already existed.
RECONFIGURE=
if [ -d "$BUILD" ]; then
    RECONFIGURE=--reconfigure
fi
# shellcheck disable=SC2086
meson setup $RECONFIGURE "$BUILD" "$SRC" \
    -Dvenus=true \
    -Dvideo=true \
    -Dvulkan-dload=false \
    -Drender-server-mode=thread \
    -Drender-server-worker=thread \
    -Dplatforms=egl \
    --prefix "$PREFIX" \
    --buildtype release
ninja -C "$BUILD"
meson install -C "$BUILD"

echo "==> reference renderer installed to $PREFIX"
echo "    use it: VIRGL_PREFIX=$PREFIX harness/replay/vrend-replay.sh <corpus> --renderer c"
