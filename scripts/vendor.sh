#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright © 2026 Gustavo Noronha Silva

# Materialize third_party/ from the revs third_party/manifest.toml pins, then materialize the
# meson subprojects the C tree's own wraps pin. Both are build inputs: see build.rs's `c_tree`.
#
# Idempotent — an existing clone is fetched and re-checked-out at the pinned rev, never reset,
# so local work in it is never silently discarded.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="$ROOT/third_party/manifest.toml"
DEST="$ROOT/third_party/virglrenderer"

read -r REPO REV < <(python3 - "$MANIFEST" <<'PY'
import sys, tomllib
m = tomllib.load(open(sys.argv[1], 'rb'))['virglrenderer']
print(m['repo'], m['rev'])
PY
)

# A local clone to source from instead of the network. The pinned rev has to be reachable in it;
# this only changes where it is fetched from, never which rev is used.
SRC="${VIRGLRENDERER_SRC:-$REPO}"

if [ ! -d "$DEST/.git" ]; then
    echo "==> cloning $SRC"
    git clone "$SRC" "$DEST"
    git -C "$DEST" remote set-url origin "$REPO"
    [ "$SRC" = "$REPO" ] || git -C "$DEST" remote add local "$SRC"
fi

echo "==> checking out $REV"
git -C "$DEST" fetch --quiet "$([ "$SRC" = "$REPO" ] && echo origin || echo local)"
git -C "$DEST" checkout --quiet --detach "$REV" || {
    echo "the pinned rev $REV is not in $SRC." >&2
    echo "If it has not been pushed to $REPO yet, point VIRGLRENDERER_SRC at a clone that has it." >&2
    exit 1
}

# venus-protocol: the wire generator's model. Pinned by the C tree's own wrap, so it rides
# inside the clone rather than being a second record of one revision.
echo "==> materializing meson subprojects"
( cd "$DEST" && meson subprojects download venus-protocol )

echo "==> vendored: $DEST @ $REV"
