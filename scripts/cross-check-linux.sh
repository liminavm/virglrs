#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright © 2026 Gustavo Noronha Silva

# Type-check the Linux build from macOS: clippy over every `cfg(target_os = "linux")` arm, the
# VA-API backend included, without a Linux machine.
#
#   scripts/cross-check-linux.sh [extra clippy args]
#
# It never links, so two inputs only need to exist, not work:
#
# - libva's headers, which cros-libva's build script turns into bindings. They come from the libva
#   rev `third_party/manifest.toml` pins, laid out the way an install lays them out: `va/drm/` flat
#   into `va/`, and `va_version.h` generated from its template, as libva's own build does. The pin
#   matches what goiaba builds against, because cros-libva gates API on the version it reads.
# - a libEGL, which `build.rs` only checks for. An empty file is enough.
#
# bindgen is told the target is glibc and freestanding. There is no Linux sysroot here, and clang's
# own headers send a musl target to the system's `stddef.h`; libva needs nothing beyond what a
# freestanding compiler ships.
#
# What it does not do is run anything. goiaba runs the Linux tests before a push.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/third_party/libva"
OUT="$ROOT/target/cross-linux"
TARGET=aarch64-unknown-linux-musl

read -r REPO REV < <(python3 - "$ROOT/third_party/manifest.toml" <<'PY'
import sys, tomllib
m = tomllib.load(open(sys.argv[1], 'rb'))['libva']
print(m['repo'], m['rev'])
PY
)

# Idempotent like vendor.sh: an existing clone is fetched and re-checked-out, never reset.
[ -d "$SRC/.git" ] || git clone --quiet "$REPO" "$SRC"
git -C "$SRC" cat-file -e "$REV^{commit}" 2>/dev/null || git -C "$SRC" fetch --quiet origin
git -C "$SRC" checkout --quiet --detach "$REV"

mkdir -p "$OUT/include/va" "$OUT/egl"
cp -f "$SRC"/va/*.h "$SRC"/va/drm/va_drm.h "$OUT/include/va/"
read -r MAJOR MINOR MICRO < <(python3 - "$SRC/meson.build" <<'PY'
import re, sys
src = open(sys.argv[1]).read()
print(*(re.search(r'va_api_%s_version = (\d+)' % p, src).group(1) for p in ('major', 'minor', 'micro')))
PY
)
sed -e "s/@VA_API_MAJOR_VERSION@/$MAJOR/" -e "s/@VA_API_MINOR_VERSION@/$MINOR/" \
    -e "s/@VA_API_MICRO_VERSION@/$MICRO/" -e "s/@VA_API_VERSION@/$MAJOR.$MINOR.$MICRO/" \
    "$SRC/va/va_version.h.in" > "$OUT/include/va/va_version.h"
: > "$OUT/egl/libEGL.so"

rustup target list --installed | grep -qx "$TARGET" || {
    echo "the $TARGET target is not installed: rustup target add $TARGET" >&2; exit 1; }

cd "$ROOT"
exec env \
    CROS_LIBVA_H_PATH="$OUT/include" \
    BINDGEN_EXTRA_CLANG_ARGS_aarch64_unknown_linux_musl="--target=aarch64-unknown-linux-gnu -ffreestanding" \
    EGL_LIB_DIR="$OUT/egl" \
    cargo clippy --lib --tests --target "$TARGET" "$@" -- -D warnings
