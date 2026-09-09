#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright © 2026 Gustavo Noronha Silva

# Run the C tree's own tests against one implementation, or diff the two.
#
#   ctests.sh c      the reference C renderer
#   ctests.sh rs     virglrs, through the same binaries with a rewritten load command
#   ctests.sh diff   both, and diff what they assert  <- the gate
#
# `diff` is the one that means something. Neither leg passes this suite on this host and a green
# is not the goal: the C fails 45 assertions here for reasons that are the HOST's, not the
# renderer's -- zink-on-KosmicKrisp advertises no cube map arrays, and multisample targets are
# refused. So the oracle is agreement with the C, exactly as it is for every corpus in this tree.
#
# DYLD_* must be exported INSIDE this script: /bin/bash is SIP-restricted and strips them at
# launch, so setting them on the calling command line never reaches the test binaries.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD="$ROOT/harness/vm/build-tests"
OUT="${TMPDIR:-/tmp}/virglrs-ctests"
mkdir -p "$OUT"

[ -d "$BUILD/tests" ] || { echo "not built: run harness/ctests/build.sh" >&2; exit 1; }

MESA_PREFIX="${MESA_PREFIX:-/Volumes/mesa-cs/zink-kk-prefix}"
EPOXY_PREFIX="${EPOXY_PREFIX:-$ROOT/third_party/epoxy-egl-prefix}"
[ -d "$EPOXY_PREFIX" ] || EPOXY_PREFIX=/Users/kov/Projects/limina/third_party/epoxy-egl-prefix

export VK_DRIVER_FILES="${VK_DRIVER_FILES:-$MESA_PREFIX/share/vulkan/icd.d/kosmickrisp_mesa_icd.aarch64.json}"
export DYLD_LIBRARY_PATH="$MESA_PREFIX/vulkan-rpath"
export DYLD_FALLBACK_LIBRARY_PATH="$MESA_PREFIX/lib:$EPOXY_PREFIX/lib:$(brew --prefix)/lib"
export MESA_LOADER_DRIVER_OVERRIDE=zink
export GALLIUM_DRIVER=zink
export LIBGL_DRIVERS_PATH="$MESA_PREFIX/lib"
export EGL_PLATFORM=surfaceless

# The tests default to a bare VIRGL_RENDERER_USE_EGL, which on macOS means a native display and
# epoxy routing desktop GL into Apple's OpenGL framework. Both of these are the C's own escape
# hatch: surfaceless for the display, GLES so epoxy resolves through libGLESv2 into our Mesa.
export VRENDTEST_USE_EGL_SURFACELESS=1
export VRENDTEST_USE_EGL_GLES=1

# check(1) forks a child per test case. MTLCompilerService is an XPC service and an XPC connection
# does not survive fork(), so every GPU test dies at MTLLibrary creation with "Unable to reach
# MTLCompilerService" -- which reads like a driver fault and is not one.
#
# The cost is that state leaks between cases in one process, so a case that leaves the renderer
# initialized poisons every case after it. That is a property of the run, not a bug in either
# renderer, and it is why a divergence should be read from the FIRST differing case.
export CK_FORK=no

ABI_TESTS=(test_virgl_init test_virgl_fence test_virgl_resource test_virgl_transfer test_virgl_cmd)
STATIC_TESTS=(test_virgl_strbuf test_virgl_journal)

# Keep only check's own verdict lines. Everything else a leg prints is renderer chatter, driver
# warnings and timings -- none of it comparable between two implementations.
run_leg() {
    local leg="$1" dir="$2" out="$OUT/$1.txt"
    shift 2
    : > "$out"
    for t in "$@"; do
        [ -x "$dir/$t" ] || continue
        "$dir/$t" 2>&1 \
            | grep -E "^[0-9]+%: Checks:|:[FES]:" \
            | sed -e "s|^.*/tests/|$t |" -e "s|^\([0-9]\)|$t SUMMARY \1|" >> "$out"
    done
    printf '%s: %s failing assertions\n' "$leg" "$(grep -c ':[FES]:' "$out")"
}

case "${1:-diff}" in
  c)
    run_leg c "$BUILD/tests" "${ABI_TESTS[@]}" "${STATIC_TESTS[@]}"
    grep SUMMARY "$OUT/c.txt" | sed 's/SUMMARY //'
    ;;
  rs)
    run_leg rs "$BUILD/tests-rs" "${ABI_TESTS[@]}"
    grep SUMMARY "$OUT/rs.txt" | sed 's/SUMMARY //'
    ;;
  diff)
    # Only the ABI tests on both legs; the static two cannot disagree and would pad the diff
    # with lines that are the C's on either side.
    run_leg c  "$BUILD/tests"    "${ABI_TESTS[@]}"
    run_leg rs "$BUILD/tests-rs" "${ABI_TESTS[@]}"
    echo "=== per-test totals (c | rs) ==="
    paste <(grep SUMMARY "$OUT/c.txt") <(grep SUMMARY "$OUT/rs.txt") | sed 's/SUMMARY //g'
    echo "=== assertions the legs disagree about ==="
    if diff <(grep ':[FES]:' "$OUT/c.txt") <(grep ':[FES]:' "$OUT/rs.txt"); then
        echo "(identical -- virglrs answers this suite exactly as the C does)"
    else
        # A divergence cascades under CK_FORK=no, so the first line is the one to read.
        echo
        echo "read the FIRST differing case: a leg that leaves the renderer initialized"
        echo "fails every case after it, and those are consequences, not findings."
        exit 1
    fi
    ;;
  *)
    echo "usage: ctests.sh [c|rs|diff]" >&2; exit 1 ;;
esac
