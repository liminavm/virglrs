#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright © 2026 Gustavo Noronha Silva
#
# Materialize harness/vm/captures/ from the release harness/replay/corpora.toml pins.
#
#   fetch-corpora.sh [NAME...]     fetch the named corpora, or every missing one
#   fetch-corpora.sh --verify      hash what is already here against the manifest
#   fetch-corpora.sh --list        what the manifest pins, and what is present
#
# The corpora are what the pinned scores were recorded against, so this is the same kind of thing
# `vendor.sh` does for the C tree and it reads its manifest the same way. It never overwrites a
# corpus that is already present: a recording in place is either the pinned one or one you are
# deliberately holding, and silently replacing the second would throw away a capture.
#
# CORPORA_BASE overrides where assets come from, exactly as VIRGLRENDERER_SRC does for the clone —
# `CORPORA_BASE=file:///path/to/dist/corpora` fetches from a local pack without a network. It
# changes where bytes come from, never which bytes are required: the hash is checked either way.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="$ROOT/harness/replay/corpora.toml"
DEST="${CORPORA_DEST:-$ROOT/harness/vm/captures}"

[ -f "$MANIFEST" ] || { echo "no manifest at $MANIFEST" >&2; exit 1; }

# name<TAB>sha256<TAB>size<TAB>url<TAB>generator, one corpus per line, "-" where a field does
# not apply. Not an empty field: tab is IFS whitespace, so `read` would collapse two of them and
# shift every later field left -- which reads as a corpus whose URL is a script path.
entries() {
    CORPORA_BASE="${CORPORA_BASE:-}" python3 - "$MANIFEST" <<'PY'
import os, sys, tomllib
m = tomllib.load(open(sys.argv[1], 'rb'))
base = os.environ.get('CORPORA_BASE') or m['base']
tag = m['tag']
for c in m.get('corpus', []):
    url = f"{base}/{tag}/{c['asset']}" if 'asset' in c else '-'
    print('\t'.join([c['name'], c['sha256'], str(c['size']), url, c.get('generated', '-')]))
PY
}

. "$ROOT/scripts/platform.sh"

verify_one() { # $1 = path, $2 = expected sha256
    [ "$(virgl_sha256 "$1")" = "$2" ]
}

case "${1:-}" in
--list)
    printf '%-28s %12s  %s\n' CORPUS SIZE STATE
    while IFS=$'\t' read -r name sha size url gen; do
        if [ ! -f "$DEST/$name" ]; then state="missing"
        elif verify_one "$DEST/$name" "$sha"; then state="ok"
        else state="DIFFERS from the pin"; fi
        [ "$gen" = "-" ] || state="$state (generated)"
        printf '%-28s %12s  %s\n' "$name" "$size" "$state"
    done < <(entries)
    exit 0 ;;
--verify)
    bad=0
    while IFS=$'\t' read -r name sha size url gen; do
        if [ ! -f "$DEST/$name" ]; then
            echo "missing: $name"; bad=1
        elif verify_one "$DEST/$name" "$sha"; then
            echo "ok:      $name"
        else
            echo "DIFFERS: $name — not the corpus the scores were recorded against"; bad=1
        fi
    done < <(entries)
    exit "$bad" ;;
esac

WANT=("$@")
wanted() { # no names given means every missing one
    [ "${#WANT[@]}" -eq 0 ] && return 0
    for w in "${WANT[@]}"; do [ "$w" = "$1" ] && return 0; done
    return 1
}

mkdir -p "$DEST"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fetched=0
while IFS=$'\t' read -r name sha size url gen; do
    wanted "$name" || continue
    if [ -f "$DEST/$name" ]; then
        echo "==> $name: present, keeping it"
        continue
    fi

    if [ "$gen" != "-" ]; then
        echo "==> $name: generating with $gen"
        python3 "$ROOT/$gen" "$TMP/$name" >/dev/null
    else
        echo "==> $name: fetching $url"
        curl -fL --progress-bar -o "$TMP/$name.zst" "$url" || {
            echo "could not fetch $name from $url" >&2
            echo "If the release is not published yet, pack locally and point CORPORA_BASE at it:" >&2
            echo "  scripts/pack-corpora.sh && CORPORA_BASE=file://$ROOT/dist/corpora $0 $name" >&2
            exit 1
        }
        zstd -q -d -o "$TMP/$name" "$TMP/$name.zst"
    fi

    verify_one "$TMP/$name" "$sha" || {
        echo "$name does not hash to the pin — refusing to install it" >&2
        exit 1
    }
    mv "$TMP/$name" "$DEST/$name"
    fetched=$((fetched + 1))
done < <(entries)

echo "==> $fetched corpus/corpora materialized into $DEST"
