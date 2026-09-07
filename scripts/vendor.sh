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

if [ ! -d "$DEST/.git" ]; then
    echo "==> cloning $REPO"
    git clone "$REPO" "$DEST"
fi

echo "==> checking out $REV"
git -C "$DEST" fetch --quiet origin
git -C "$DEST" checkout --quiet --detach "$REV"

# venus-protocol: the wire generator's model. Pinned by the C tree's own wrap, so it rides
# inside the clone rather than being a second record of one revision.
echo "==> materializing meson subprojects"
( cd "$DEST" && meson subprojects download venus-protocol )

echo "==> vendored: $DEST @ $REV"
