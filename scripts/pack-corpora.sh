#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright © 2026 Gustavo Noronha Silva
#
# Pack the recorded corpora for a release, and write the manifest that pins them.
#
#   pack-corpora.sh [--prune] [OUT]      OUT default: dist/corpora/ under the repository root
#
# The manifest is regenerated from what is in harness/vm/captures/, so packing on a host that has
# only some of the recordings would drop the pins for the rest. That is refused by name; --prune
# is how a corpus is actually retired.
#
# The corpora are recordings, not source: they run to gigabytes raw and compress by about fifty
# times, because a command stream over mostly-repetitive pixel data is what they are. So a release
# carries the compressed form and the manifest carries the hash of the ORIGINAL, which is what the
# pinned scores were recorded against and what `fetch-corpora.sh` verifies after it decompresses.
# Hashing the compressed form as well would be a second value free to disagree with the first, and
# zstd's own frame checksum already refuses a damaged download.
#
# The corpora a `make-*-corpus.py` writes are not packed. They are deterministic — measured
# 2026-09-08, two runs of each reproduce the stored file byte for byte — so the manifest records
# how to generate them and the fetch runs the script instead of the network.
#
# Uploading is deliberately not part of this: it is outward-facing, and the tag rule below makes
# it a decision rather than a step. See harness/replay/corpora.toml.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
. "$ROOT/scripts/platform.sh"
CAPTURES="$ROOT/harness/vm/captures"
MANIFEST="$ROOT/harness/replay/corpora.toml"

PRUNE=0
if [ "${1:-}" = --prune ]; then PRUNE=1; shift; fi
OUT="${1:-$ROOT/dist/corpora}"

# One corpus generation, one tag. A release asset replaced in place under an existing tag would
# leave every checkout that already fetched it holding different bytes under the same name, and
# the scores would go on passing against whichever copy a machine happened to have.
TAG="${CORPORA_TAG:-corpora-$(date +%Y-%m-%d)}"
BASE="${CORPORA_BASE:-https://github.com/liminavm/virglrs/releases/download}"

# The generated corpora, and the script that writes each. Kept here rather than discovered so that
# a new synthetic corpus is a deliberate line in a diff.
generator_for() {
    case "$1" in
        blit.bin)    echo "harness/replay/make-blit-corpus.py" ;;
        sampled.bin) echo "harness/replay/make-sampled-corpus.py" ;;
        surface.bin) echo "harness/replay/make-surface-corpus.py" ;;
        *)           echo "" ;;
    esac
}

command -v zstd >/dev/null || { echo "zstd is not installed" >&2; exit 1; }
[ -d "$CAPTURES" ] || { echo "no corpora at $CAPTURES — see harness/README.md" >&2; exit 1; }

# The manifest is rewritten from whatever is in CAPTURES, so a host holding a subset of the
# recordings silently drops the rest -- and the dropped ones are exactly the hosted corpora a
# second machine has never fetched, which is the normal state of a second machine. The result is
# a plausible manifest that pins less than the tree needs, and nothing downstream can tell it from
# a deliberate prune. So say which entries would go and refuse; --prune is how a real removal is
# spelled, and it has to be typed.
missing=""
if [ -f "$MANIFEST" ]; then
    while read -r name; do
        [ -n "$name" ] || continue
        [ -f "$CAPTURES/$name" ] || missing="$missing $name"
    done <<EOF
$(sed -n 's/^name = "\(.*\)"$/\1/p' "$MANIFEST")
EOF
fi
if [ -n "$missing" ] && [ "$PRUNE" = 0 ]; then
    echo "refusing to rewrite $MANIFEST: it pins corpora this host does not have," >&2
    echo "and rewriting it here would drop them:" >&2
    for name in $missing; do echo "    $name" >&2; done
    echo >&2
    echo "Fetch them first (scripts/fetch-corpora.sh), or pass --prune to drop them on purpose." >&2
    exit 1
fi

# Assets go under the tag, which is the layout a release download URL already has
# (`$base/$tag/$asset`). So `CORPORA_BASE=file://$OUT` is not an approximation of the real fetch
# -- it is the same path arithmetic against a local directory, which is what makes it worth
# rehearsing with before anything is uploaded.
ASSETS="$OUT/$TAG"
mkdir -p "$ASSETS"

{
    cat <<HEADER
# The recorded corpora, pinned.
#
# A corpus is a recording of a real guest, and the scores in fixtures/ only mean anything against
# the corpus they were recorded from — so this file pins both halves of that pair. It is written
# by scripts/pack-corpora.sh; do not edit it by hand.
#
# Bumping = pack, create a NEW tag, upload, and commit the regenerated manifest together with any
# score the new corpora move. Never replace an asset under an existing tag: a checkout that
# already fetched it would keep different bytes under the same name and go on passing.
#
# \`sha256\` and \`size\` are of the UNCOMPRESSED corpus, which is what a score was recorded
# against. An entry with \`generated\` is not hosted at all — the script reproduces it byte for
# byte, so fetching it is running that script.

tag = "$TAG"
base = "$BASE"
HEADER

    for path in "$CAPTURES"/*.bin "$CAPTURES"/*.vkrc; do
        [ -f "$path" ] || continue
        name="$(basename "$path")"
        sha="$(virgl_sha256 "$path")"
        size="$(virgl_file_size "$path")"
        gen="$(generator_for "$name")"

        printf '\n[[corpus]]\nname = "%s"\nsha256 = "%s"\nsize = %s\n' "$name" "$sha" "$size"
        if [ -n "$gen" ]; then
            printf 'generated = "%s"\n' "$gen"
            echo "==> $name: generated by $gen, not packed" >&2
        else
            printf 'asset = "%s.zst"\n' "$name"
            if [ -f "$ASSETS/$name.zst" ] && [ "$ASSETS/$name.zst" -nt "$path" ]; then
                echo "==> $name: already packed" >&2
            else
                echo "==> packing $name" >&2
                nice zstd -q -19 --long -T0 -f -o "$ASSETS/$name.zst" "$path"
            fi
        fi
    done
} > "$MANIFEST.tmp"
# Only now, with every hash computed and every asset packed, does the pinned file change: the
# redirection above truncates its target before the loop runs, and a zstd that runs out of disk
# halfway would otherwise leave a manifest pinning the corpora it got to.
mv "$MANIFEST.tmp" "$MANIFEST"

echo "==> manifest: $MANIFEST (tag $TAG)"
echo "==> assets:   $ASSETS"
du -sh "$ASSETS"
