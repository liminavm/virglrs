#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright © 2026 Gustavo Noronha Silva
#
# Build a self-contained VM rig under harness/vm: a limina app bundle carrying THIS tree's
# renderer, plus APFS clones of a couple of guest disks.
#
# WHY A COPY AND NOT limina's OWN BUNDLE. Capturing a corpus means booting a real guest against a
# renderer we are actively changing. Doing that in limina's tree would make every capture a
# mutation of their working set — and the whole point of this repository right now is that the
# rewrite is contained. The rig is ours to break.
#
# The disks are APFS clones (cp -c): they cost no space until the guest writes, and the originals
# are never touched. Copying 15 GB per image the ordinary way would not fit on this host.
#
# Two renderers are under test in this tree, so the rig is two bundles. `--renderer` picks one,
# and it picks the build and the bundle together: a bundle named for one tree holding the other
# tree's renderer is a rig that lies about what it booted, and nothing downstream could tell.
#
# THE TWO LEGS ARE BUILT DIFFERENTLY, and not by choice. limina compiles virglrs in, as a cargo
# path dependency: for the rust leg there is no dylib to swap, so the bundle has to be BUILT here,
# from a limina worktree whose third_party/virglrs is this tree. The C renderer is still a dylib,
# and its leg is still limina's own bundle with that dylib swapped in -- but only a bundle from
# before limina's cutover has a `libvirglrenderer` load command to swap into, which is why the C
# path asserts on one.
#
# Usage: make-rig.sh [--disks-only | --bundle-only] [--renderer c|rust]
set -euo pipefail
cd "$(dirname "$0")"
RIG="$(pwd)"
cd ../..
ROOT="$(pwd)"

LIMINA="${LIMINA_ROOT:-$HOME/Projects/limina}"
DISKS="$RIG/disks"
# The rig's own limina checkout, built against this tree. See build_rust_app.
WT="$RIG/limina-src"

# Three guests, because the corpus needs three different renderers exercised:
#   synoik    a Vulkan compositor — the desktop workload that is venus end to end
#   enhanced  GNOME, whose shell runs on classic virgl (GALLIUM_DRIVER=virgl); venus here comes
#             from Vulkan clients, so it is the mixed vrend+venus case
#   stock     an unmodified guest: classic vrend and the VA-API video path
SRC_DISKS=(
  "$LIMINA/Fedora-Workstation-44.enhanced.synoik.raw"
  "$LIMINA/Fedora-Workstation-44.enhanced.test.raw"
  "$LIMINA/Fedora-Workstation-44.stock.test.raw"
)

want_bundle=1 want_disks=1 renderer=c
while [ $# -gt 0 ]; do
  case "$1" in
    --disks-only) want_bundle=0; shift ;;
    --bundle-only) want_disks=0; shift ;;
    --renderer) renderer="$2"; shift 2 ;;
    *) echo "usage: make-rig.sh [--disks-only|--bundle-only] [--renderer c|rust]" >&2; exit 2 ;;
  esac
done

case "$renderer" in
  c)    PREFIX="$ROOT/harness/vm/prefix"; APP="$RIG/Limina.app" ;;
  rust) PREFIX=""; APP="$RIG/Limina-rust.app" ;;
  *) echo "unknown renderer: $renderer (c|rust)" >&2; exit 2 ;;
esac

# Build limina against THIS tree, and hand back its bundle.
#
# WHY THIS EXISTS. rutabaga names virglrs as a cargo path dependency, so the renderer is compiled
# into limina-vmm — `otool -L limina-vmm` has no libvirglrenderer at all. The bundle limina builds
# therefore carries whatever `limina/third_party/virglrs` held, which is a SECOND clone of this
# repository. The swap that used to stand here installed a dylib nothing loads and re-signed the
# bundle afterwards, so a stale renderer reported as a fresh one; only breaking the fix on purpose
# and watching the score not move could have found it, and that is how it was found.
#
# A WORKTREE, and not limina's own tree, for the reason at the top of this file: the rig is ours to
# break, and re-pointing limina's third_party would change what THEIR builds compile. third_party
# is untracked there (and 14 GB), so the worktree's copy is a directory of symlinks to theirs —
# with virglrs pointed here, which is the whole point.
build_rust_app() {
  [ -d "$LIMINA/.git" ] || { echo "no limina checkout at $LIMINA (set LIMINA_ROOT)" >&2; exit 1; }
  local rev resolved
  # limina's HEAD, not its working tree: the rig must be reproducible from a commit, and the
  # in-flight edits in that tree are theirs. LIMINA_REV pins an older one deliberately.
  rev="${LIMINA_REV:-$(git -C "$LIMINA" rev-parse HEAD)}"

  if [ -d "$WT" ]; then
    git -C "$WT" checkout --quiet --detach "$rev"
  else
    echo "==> creating the rig's limina worktree at $WT"
    git -C "$LIMINA" worktree add --detach "$WT" "$rev"
  fi

  # third_party is gitignored in limina, so the worktree has none; it is theirs by symlink.
  mkdir -p "$WT/third_party"
  for e in "$LIMINA"/third_party/*; do
    ln -sfn "$e" "$WT/third_party/$(basename "$e")"
  done
  ln -sfn "$ROOT" "$WT/third_party/virglrs"

  # The GOP firmware is an edk2 build and not a cargo one; share limina's rather than spend an
  # hour rebuilding a file that is an input to both legs anyway.
  mkdir -p "$WT/target"
  ln -sfn "$LIMINA/target/krun-efi" "$WT/target/krun-efi"

  # THE check that this leg is this tree, and the only one that cannot be satisfied by a build
  # that ran. rutabaga spells the dependency `../../../virglrs` from inside third_party/libkrun,
  # which is a symlink: cargo normalizes that lexically and lands on the worktree's own virglrs,
  # but a resolver that walked the symlink physically would land in limina's clone instead and
  # every score after it would be of the wrong source, silently.
  resolved="$(cd "$WT" && cargo metadata --format-version 1 2>/dev/null | python3 -c '
import json, sys
for pkg in json.load(sys.stdin)["packages"]:
    if pkg["name"] == "virglrs":
        print(pkg["manifest_path"])
')"
  [ "$resolved" = "$WT/third_party/virglrs/Cargo.toml" ] || {
    echo "cargo resolves virglrs to: ${resolved:-<nothing>}" >&2
    echo "expected: $WT/third_party/virglrs/Cargo.toml (a symlink to $ROOT)" >&2
    echo "refusing to build a rig that would score a different checkout." >&2
    exit 1
  }

  echo "==> building limina ${rev:0:12} against this tree (a cold build takes a while)"
  (cd "$WT" && ./scripts/build-app.sh release) || exit 1
  # Beside the bundle and not inside it: build-app.sh seals the app, and a file added afterwards
  # breaks the signature.
  printf '%s\n' "$rev" > "$RIG/Limina-rust.rev"
}

if [ "$want_disks" = 1 ]; then
  mkdir -p "$DISKS"
  for src in "${SRC_DISKS[@]}"; do
    [ -f "$src" ] || {
      echo "missing source disk: $src" >&2
      echo "The rig only clones images; limina/docs/images.md is how they are built." >&2
      exit 1
    }
    dst="$DISKS/$(basename "$src")"
    if [ -f "$dst" ]; then
      echo "==> keeping existing $(basename "$dst") (delete it to re-clone)"
      continue
    fi
    # -c is the whole point: an APFS clone, not a copy. Without it this is 15 GB per image.
    echo "==> cloning $(basename "$src")"
    cp -c "$src" "$dst"
  done
fi

[ "$want_bundle" = 1 ] || exit 0

if [ "$renderer" = rust ]; then
  build_rust_app
  SRC_APP="$WT/target/Limina.app"
else
  DYLIB="$PREFIX/lib/libvirglrenderer.1.dylib"
  [ -f "$DYLIB" ] || { echo "build it first: harness/vm/build-renderer.sh" >&2; exit 1; }
  SRC_APP="$LIMINA/target/Limina.app"
fi
[ -d "$SRC_APP" ] || { echo "missing source bundle: $SRC_APP (cargo xtask app in limina)" >&2; exit 1; }

echo "==> cloning $SRC_APP"
rm -rf "$APP"
cp -Rc "$SRC_APP" "$APP"

FW="$APP/Contents/Frameworks"
MACOS="$APP/Contents/MacOS"

# The rust bundle IS the build: it was compiled against this tree, checked to have been, and
# signed by build-app.sh. There is nothing to swap and nothing to re-sign.
if [ "$renderer" = rust ]; then
  codesign --verify --deep --strict "$APP" && echo "==> bundle verifies"
  echo "==> rig ready: $APP"
  echo "    limina $(cut -c1-12 < "$RIG/Limina-rust.rev"), renderer compiled from $ROOT"
  exit 0
fi

# A bundle with no libvirglrenderer load command loads no dylib, so swapping one in changes
# nothing and re-signing hides that it changed nothing. limina past its cutover builds exactly
# such a bundle -- it compiles virglrs in -- so the C leg needs a bundle from BEFORE it, and this
# is where that stops being silent.
otool -L "$MACOS/limina-vmm" | grep -q libvirglrenderer || {
  echo "$SRC_APP does not link libvirglrenderer: it compiles the renderer in, so this bundle" >&2
  echo "would boot virglrs while claiming to be the C leg. The C leg needs a PRE-CUTOVER limina" >&2
  echo "bundle -- keep the existing $APP, or build one from a limina revision before it." >&2
  exit 1
}

echo "==> swapping in $(basename "$DYLIB")"
cp "$DYLIB" "$FW/libvirglrenderer.1.dylib"

# Our build links its dependencies by absolute path; the bundle reaches them through @rpath. Left
# as built, the bundled worker would load epoxy/vulkan/dav1d from OUTSIDE the bundle — which
# happens to work on this host and would silently stop being the thing under test the moment one
# of those prefixes moved.
install_name_tool -id "@rpath/libvirglrenderer.1.dylib" "$FW/libvirglrenderer.1.dylib"
otool -L "$FW/libvirglrenderer.1.dylib" | awk 'NR>1 {print $1}' | while read -r dep; do
  case "$dep" in
    /Users/*|/Volumes/*|/opt/homebrew/*)
      name="$(basename "$dep")"
      # limina bundles Mesa's libraries under their unversioned names (libEGL.dylib for
      # libEGL.1.dylib); the Rust build links the versioned one. Name what the bundle has.
      if [ ! -f "$FW/$name" ]; then
        bare="$(echo "$name" | sed -E 's/\.[0-9]+\.dylib$/.dylib/')"
        [ -f "$FW/$bare" ] && name="$bare"
      fi
      install_name_tool -change "$dep" "@rpath/$name" "$FW/libvirglrenderer.1.dylib"
      echo "    $dep -> @rpath/$name" ;;
  esac
done

# Anything we rewrote to @rpath must actually BE in Frameworks, or the worker fails to launch with
# a dyld error that reads like a signing problem. Check now, while the cause is obvious.
missing=0
otool -L "$FW/libvirglrenderer.1.dylib" | awk 'NR>1 && $1 ~ /^@rpath\// {print $1}' | while read -r dep; do
  [ -f "$FW/${dep#@rpath/}" ] || { echo "    MISSING in bundle: $dep" >&2; missing=1; }
  [ "$missing" = 0 ] || exit 1
done

# Re-sign inside-out, exactly as build-app.sh does: the bundle seal covers Frameworks, so
# replacing a file there invalidates the app whether or not the worker validates libraries.
SIGN_ID="${LIMINA_SIGN_IDENTITY:-$(security find-identity -v -p codesigning 2>/dev/null \
          | sed -n 's/^ *1) [0-9A-F]* "\(.*\)"$/\1/p' | head -1)}"
[ -n "$SIGN_ID" ] || SIGN_ID="-"
echo "==> re-signing as '$SIGN_ID'"

ENT="$(mktemp -t limina-ent).plist"
sign_like() {  # preserve each binary's own entitlements rather than inventing them
  local bin="$1"
  if codesign -d --entitlements "$ENT" --xml "$bin" >/dev/null 2>&1 && [ -s "$ENT" ]; then
    codesign -s "$SIGN_ID" --options runtime --entitlements "$ENT" --force "$bin"
  else
    codesign -s "$SIGN_ID" --options runtime --force "$bin"
  fi
}

codesign -s "$SIGN_ID" --force "$FW/libvirglrenderer.1.dylib"
sign_like "$MACOS/limina-vmm"
sign_like "$MACOS/limina"
[ -f "$MACOS/gvproxy" ] && sign_like "$MACOS/gvproxy"
sign_like "$APP"
rm -f "$ENT"

codesign --verify --deep --strict "$APP" && echo "==> bundle verifies"
echo "==> rig ready: $APP"
otool -L "$MACOS/limina-vmm" | grep -i virgl || true
