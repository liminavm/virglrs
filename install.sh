#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright © 2026 Gustavo Noronha Silva
#
# Build virglrs and lay it out as a virglrenderer prefix -- the same shape meson installs, so the
# two implementations are swapped by pointing VIRGL_PREFIX at one or the other and nothing that
# consumes the prefix can tell which it got.
#
#   install.sh [prefix]     default: ./prefix
set -euo pipefail
cd "$(dirname "$0")"
HERE="$(pwd)"
# The C tree is a pinned build input under this repository, not the parent directory it was when
# virglrs lived inside it. The headers come from there because they define the ABI both
# implementations serve -- see the copy below.
CTREE="$HERE/third_party/virglrenderer"

PREFIX="${1:-$HERE/prefix}"
mkdir -p "$PREFIX/lib/pkgconfig" "$PREFIX/include/virgl"
PREFIX="$(cd "$PREFIX" && pwd)"

VERSION=1.3.0

cargo build --release
cp "$HERE/target/release/libvirglrenderer.dylib" "$PREFIX/lib/libvirglrenderer.1.dylib"

# The install name is the ABSOLUTE path of the installed library, matching what meson records.
# A consumer links against the path it finds here, and dyld resolves it from the recorded id --
# get this wrong and the app loads whichever libvirglrenderer is on the default path instead.
install_name_tool -id "$PREFIX/lib/libvirglrenderer.1.dylib" \
  "$PREFIX/lib/libvirglrenderer.1.dylib"
ln -sf libvirglrenderer.1.dylib "$PREFIX/lib/libvirglrenderer.dylib"

# The headers are the C tree's. They define the ABI both implementations serve, so there is no
# Rust-side copy to drift -- virglrs is checked against them by harness/abi.
cp "$CTREE/src/virglrenderer.h" "$PREFIX/include/virgl/"
if [ -f "$HERE/harness/vm/build/src/virgl-version.h" ]; then
  cp "$HERE/harness/vm/build/src/virgl-version.h" "$PREFIX/include/virgl/"
fi

# No Requires.private: virglrs links no epoxy, no vulkan and no dav1d yet. A .pc that claimed them
# would make a consumer's build fail on packages this implementation does not need.
cat > "$PREFIX/lib/pkgconfig/virglrenderer.pc" <<PC
prefix=$PREFIX
includedir=\${prefix}/include
libdir=\${prefix}/lib

Name: virglrenderer
Description: virgl GL renderer (virglrs)
Version: $VERSION
Libs: -L\${libdir} -lvirglrenderer
Cflags: -I\${includedir}/virgl
PC

echo "installed virglrs to $PREFIX"
