#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright © 2026 Gustavo Noronha Silva

# Build the C tree's own tests/, and a virglrs-linked copy of them.
#
# These are the one gate here that reaches the ABI's ERROR contract. Every corpus replays a
# well-behaved guest, so no fixture in this tree passes a null pointer, an out-of-range version
# or a second init -- and the C's tests do nothing else. Run them with ctests.sh.
#
# Output: harness/vm/build-tests, with tests/ (the C leg) and tests-rs/ (virglrs).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$ROOT/third_party/virglrenderer"
BUILD="$ROOT/harness/vm/build-tests"
RS="$ROOT/prefix/lib/libvirglrenderer.1.dylib"

[ -f "$SRC/meson.build" ] || { echo "the C tree is not vendored: run scripts/vendor.sh" >&2; exit 1; }

# NOT harness/vm/build. That is the C leg every golden was recorded from, and -Dtests=true sets
# ENABLE_TESTS, which reaches the library -- a COPY_TRANSFER3D debug flag bit and a field in
# vrend_iov.h's transfer info. The effect looks inert; "looks inert" is not a reason to rebuild
# the reference underneath the fixtures that were pinned from it.
EPOXY_PREFIX="${EPOXY_PREFIX:-$ROOT/third_party/epoxy-egl-prefix}"
[ -f "$EPOXY_PREFIX/lib/pkgconfig/epoxy.pc" ] ||
    EPOXY_PREFIX=/Users/kov/Projects/limina/third_party/epoxy-egl-prefix
grep -qi epoxy_has_egl=1 "$EPOXY_PREFIX/lib/pkgconfig/epoxy.pc" 2>/dev/null || {
    echo "epoxy-with-EGL missing at $EPOXY_PREFIX (set EPOXY_PREFIX)" >&2; exit 1; }

MESA_PREFIX="${MESA_PREFIX:-/Volumes/mesa-cs/zink-kk-prefix}"
[ -f "$MESA_PREFIX/lib/pkgconfig/egl.pc" ] || {
    echo "zink-on-KK Mesa egl.pc missing at $MESA_PREFIX (set MESA_PREFIX)" >&2; exit 1; }

export PKG_CONFIG_PATH="$EPOXY_PREFIX/lib/pkgconfig:$MESA_PREFIX/lib/pkgconfig:$(brew --prefix)/opt/molten-vk/lib/pkgconfig:$(brew --prefix)/lib/pkgconfig:$(brew --prefix)/share/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"

# tests/meson.build gives test_virgl_gbm_resources `test_depends`, which has no epoxy -- while
# fuzzytest_depends beside it does. On Linux that is masked by epoxy living in /usr/include. The
# -I is the workaround; the C tree is a pinned build input and is not patched here.
meson setup "$([ -d "$BUILD" ] && echo --reconfigure)" "$BUILD" "$SRC" \
    -Dvenus=true \
    -Dvideo=true \
    -Dvulkan-dload=false \
    -Drender-server-mode=thread \
    -Drender-server-worker=thread \
    -Dplatforms=egl \
    -Dtests=true \
    -Dc_args="-I$EPOXY_PREFIX/include -I$MESA_PREFIX/include" \
    --prefix "$ROOT/harness/vm/prefix-tests" \
    --buildtype release

# test_virgl_gbm_resources is named nowhere below and cannot be: it references `gbm`, the minigbm
# allocation path, which does not exist on macOS. It fails to LINK, not to pass, so the targets
# are listed rather than building everything.
#
# test_virgl_strbuf and test_virgl_journal are built and run, but only on the C leg: they load no
# libvirglrenderer at all (0 dylib loads, 0 imported virgl_renderer_* symbols) and exercise static
# code from the C tree. Copying them to the rs leg would report the C twice and inflate the score
# with two tests that cannot disagree.
ABI_TESTS=(test_virgl_init test_virgl_fence test_virgl_resource test_virgl_transfer test_virgl_cmd)
STATIC_TESTS=(test_virgl_strbuf test_virgl_journal)

ninja -C "$BUILD" $(printf 'tests/%s ' "${ABI_TESTS[@]}" "${STATIC_TESTS[@]}")

[ -f "$RS" ] || { echo "no virglrs dylib at $RS -- run ./install.sh" >&2; exit 1; }

# The rs leg is the same binaries with their load command rewritten. DYLD_LIBRARY_PATH does NOT
# work for this: the binaries load @rpath/libvirglrenderer.1.dylib and dyld resolves it from their
# own LC_RPATH whatever the environment says. Measured -- a deliberately corrupt dylib placed on
# DYLD_LIBRARY_PATH is ignored and the test still passes. A differential built on that env var
# compares the C leg with itself and agrees perfectly while measuring nothing.
mkdir -p "$BUILD/tests-rs"
for t in "${ABI_TESTS[@]}"; do
    cp "$BUILD/tests/$t" "$BUILD/tests-rs/$t"
    install_name_tool -change @rpath/libvirglrenderer.1.dylib "$RS" "$BUILD/tests-rs/$t"
    # Rewriting a load command invalidates the ad-hoc signature macOS applies to arm64 binaries.
    codesign --force --sign - "$BUILD/tests-rs/$t"
    # Assert the rewrite landed: a silent no-op would put the C back on both legs, which is the
    # whole failure this indirection exists to avoid.
    otool -L "$BUILD/tests-rs/$t" | grep -q "$RS" ||
        { echo "FAIL: $t does not name virglrs after the rewrite" >&2; exit 1; }
    if otool -L "$BUILD/tests-rs/$t" | grep -q '@rpath/libvirglrenderer'; then
        echo "FAIL: $t still names @rpath after the rewrite" >&2; exit 1
    fi
done

echo "==> C leg:       $BUILD/tests        (${#ABI_TESTS[@]} ABI + ${#STATIC_TESTS[@]} static)"
echo "==> virglrs leg: $BUILD/tests-rs     (${#ABI_TESTS[@]} ABI)"
echo "    run them: harness/ctests/ctests.sh [c|rs|diff]"
