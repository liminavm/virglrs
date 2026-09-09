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
PREFIX="${VIRGL_PREFIX:-$ROOT/third_party/virgl-prefix}"

# Which C tree is the leg differs by host, and only here.
#
# The limina fork is the *build input* on every host and is never swapped: `build.rs` generates
# the wire numbering from its `virgl_hw.h`, so a per-host build input would make virglrs a
# different renderer on each. What differs is the tree the harness *scores against*.
#
# On Darwin that is the fork, which is what shipped. Off Darwin it is upstream: the fork does not
# compile there -- its IOSurface fields and five of its functions are declared under
# `#ifdef __APPLE__` while a dozen uses are not guarded at all -- and upstream is the better
# oracle on this host anyway, being what a distribution ships and what a Linux VMM will meet.
# See third_party/manifest.toml for the two cap bits this costs a comparison.
if [ "$(uname -s)" = Darwin ]; then
    SRC="$ROOT/third_party/virglrenderer"
    [ -f "$SRC/meson.build" ] || { echo "the C tree is not vendored: run scripts/vendor.sh" >&2; exit 1; }
else
    SRC="$ROOT/third_party/virglrenderer-upstream"
    # Materialized here rather than by vendor.sh, which brings in what the crate is built from.
    # Idempotent the same way: an existing clone is fetched and re-checked-out at the pinned rev.
    read -r UPSTREAM_REPO UPSTREAM_REV <<EOF
$(python3 - "$ROOT/third_party/manifest.toml" <<'PY'
import sys, tomllib
m = tomllib.load(open(sys.argv[1], "rb"))["virglrenderer-upstream"]
print(m["repo"], m["rev"])
PY
)
EOF
    [ -n "$UPSTREAM_REV" ] || { echo "no upstream rev pinned in third_party/manifest.toml" >&2; exit 1; }
    [ -d "$SRC/.git" ] || git clone --quiet "$UPSTREAM_REPO" "$SRC"
    git -C "$SRC" fetch --quiet origin
    git -C "$SRC" checkout --quiet --detach "$UPSTREAM_REV" || {
        echo "the pinned rev $UPSTREAM_REV is not in $UPSTREAM_REPO" >&2; exit 1; }
fi
BUILD="$SRC/build"

# This build needs an epoxy carrying EGL, and a Mesa whose egl.pc pkg-config can see -- epoxy.pc's
# `Requires.private: egl` means a missing one fails the configure with "Could not generate cflags
# for epoxy".
#
# Where they come from is the only part that differs by host. A Linux distribution ships both, so
# nothing is named and no prefix is invented. On macOS neither is a system package and both come
# from the limina tree's spikes (build-epoxy-egl.sh, build-mesa-zink-kk.sh) -- Homebrew's epoxy is
# CGL-only, and virglrenderer's EGL platform silently builds *without* EGL against it, which is a
# renderer that cannot bring up host GL at all, discovered at run time. Set EPOXY_PREFIX or
# MESA_PREFIX to override on either host.
if [ "$(uname -s)" = Darwin ]; then
    EPOXY_PREFIX="${EPOXY_PREFIX:-$ROOT/third_party/epoxy-egl-prefix}"
    MESA_PREFIX="${MESA_PREFIX:-/Volumes/mesa-cs/zink-kk-prefix}"
    BREW="$(brew --prefix)"
    PKG_CONFIG_PATH="$BREW/opt/molten-vk/lib/pkgconfig:$BREW/lib/pkgconfig:$BREW/share/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
fi
for prefix in "${MESA_PREFIX:-}" "${EPOXY_PREFIX:-}"; do
    [ -n "$prefix" ] && PKG_CONFIG_PATH="$prefix/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
done
export PKG_CONFIG_PATH

# Asked of pkg-config rather than grepped out of a .pc at a path we reconstructed: this is the
# same pkg-config meson is about to run, so its answer is the one the build will get.
[ "$(pkg-config --variable=epoxy_has_egl epoxy 2>/dev/null)" = 1 ] || {
    echo "epoxy is missing, or was built without EGL (set EPOXY_PREFIX)" >&2; exit 1; }
pkg-config --exists egl || {
    echo "no egl.pc on the pkg-config path (set MESA_PREFIX)" >&2; exit 1; }

# The same configuration limina ran the C renderer under, so the goldens are recorded from the
# renderer that shipped rather than from a differently-built one: venus and vrend in one process
# (render-server-mode=thread), EGL platform, the VideoToolbox-backed video path, and the Vulkan
# library linked rather than dlopened by bare soname.
# Unquoted on purpose: empty must expand to no argument at all. Quoted, a first build passes
# meson an empty string, which it takes for the build directory and then rejects the source tree
# as an extra -- so the script only ever worked where a build directory already existed.
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
echo "    leg: $(basename "$SRC") at $(git -C "$SRC" rev-parse --short HEAD)"
echo "    use it: VIRGL_PREFIX=$PREFIX harness/replay/vrend-replay.sh <corpus> --renderer c"
