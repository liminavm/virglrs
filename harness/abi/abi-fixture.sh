#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright © 2026 Gustavo Noronha Silva
#
# Pin, and check, the C ABI: the symbols the dylib exports and the layout of every struct that
# crosses it. Both are things a Rust port declares by hand on its own side, and both fail in a way
# nothing else in the harness sees -- a missing symbol only shows at dlopen, a wrong offset never
# shows at all, it just corrupts.
#
# The layout is exact; the symbols are a floor. See the checks at the bottom for why.
#
#   abi-fixture.sh            check the current build against the pinned fixtures
#   abi-fixture.sh --pin      re-record them (do this only when the ABI is meant to change)
#
# VIRGL_PREFIX selects the build under test; it defaults to this tree's rig prefix, so pointing it
# at a Rust build is how the port gets checked.
set -euo pipefail
cd "$(dirname "$0")"
HERE="$(pwd)"
ROOT="$(cd ../.. && pwd)"

PIN=0
[ "${1:-}" = "--pin" ] && PIN=1

. "$ROOT/scripts/platform.sh"

PREFIX="${VIRGL_PREFIX:-$ROOT/third_party/virgl-prefix}"

# Own the freshness of the prefix we are about to score, the way vkr-replay.sh does. This gate
# reads a built dylib and nothing else, so without this it happily pins or checks a dylib from
# whatever the tree looked like when someone last installed -- and reports "symbols matches" about
# a build that no longer exists. The replay ladder already learned this once; there is no reason
# for the ABI gate to learn it again.
if [ "$PREFIX" = "$ROOT/prefix" ]; then
  "$ROOT/install.sh" >/dev/null
fi

LIB="$(virgl_find_lib "$PREFIX")" || { echo "no libvirglrenderer under $PREFIX" >&2; exit 1; }

# Any other prefix is built by someone else, so all we can do is say when it looks stale.
if [ "$PREFIX" != "$ROOT/prefix" ] \
   && find "$ROOT/third_party" -name '*.c' -path '*/virglrenderer*/src/*' -newer "$LIB" 2>/dev/null | read -r _; then
  echo "warning: $LIB is older than the C sources it was built from -- stale build" >&2
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Every exported symbol, not an intersection with any one consumer: what the dylib exports is
# defined by this tree, while who calls what is defined outside it and moves without warning.
# A port that exports the whole list satisfies every consumer of it.
virgl_exported_symbols "$LIB" > "$TMP/symbols.txt"

# Against the FORK's headers on either host, because they are the ABI virglrs implements -- not
# whichever C tree this host happens to score against. Only `virgl-version.h` is taken from a
# build directory, and it contributes nothing to the dump.
GEN_INC="$(virgl_generated_include "$ROOT")" || {
  echo "no built C tree to take generated headers from: run scripts/build-reference.sh" >&2
  exit 1
}
cc -O0 -o "$TMP/abi-dump" "$HERE/abi-dump.c" -I"$ROOT/third_party/virglrenderer/src" -I"$GEN_INC"
"$TMP/abi-dump" > "$TMP/layout.txt"

fail=0

# The layout is an exact match, on either implementation. A struct that crosses the ABI has one
# shape; a field at the wrong offset compiles clean on both sides and corrupts at run time, so
# there is no such thing as an acceptable difference here.
if [ "$PIN" = 1 ]; then
  cp "$TMP/layout.txt" "$HERE/layout.txt"
  echo "pinned layout.txt ($(wc -l < "$TMP/layout.txt" | tr -d ' ') lines)"
elif diff -u "$HERE/layout.txt" "$TMP/layout.txt"; then
  echo "layout matches"
else
  echo "ABI FIXTURE MISMATCH: layout" >&2
  fail=1
fi

# The symbols are a floor, not an exact list, because the two implementations legitimately differ:
# virglrs serves `journal_held`, which the C header has no equivalent of. The failure this gate
# exists for is a *missing* symbol -- it shows at dlopen and nowhere earlier -- and an extra one
# satisfies every consumer of the list just as well. Extras are printed rather than ignored, so an
# unintended export is still visible; it is just not a failure.
if [ "$PIN" = 1 ]; then
  # Pinning from the Rust build would raise the floor to include its extensions and fail the C
  # leg on the next run -- and the diff would name the C as the regression.
  if [ "$PREFIX" = "$ROOT/prefix" ]; then
    echo "refusing to pin the symbol floor from the Rust build: it exports extensions the C" >&2
    echo "does not, and a floor both must meet can only be recorded from the C." >&2
    exit 2
  fi
  # And pinning from a stock upstream build would LOWER it: upstream carries none of the limina
  # extensions, so the new floor would drop them silently and every later run would pass while
  # virglrs quietly stopped exporting the symbols limina actually calls. The floor is the fork's
  # export set; only a fork build may record it.
  if [ -s "$HERE/symbols-limina.txt" ] \
     && [ -n "$(LC_ALL=C comm -23 "$HERE/symbols-limina.txt" "$TMP/symbols.txt")" ]; then
    echo "refusing to pin the symbol floor from a build that carries no limina extensions:" >&2
    echo "it would lower the floor rather than record it. Pin from a fork build." >&2
    exit 2
  fi
  cp "$TMP/symbols.txt" "$HERE/symbols.txt"
  echo "pinned symbols.txt ($(wc -l < "$TMP/symbols.txt" | tr -d ' ') lines)"
else
  missing="$(LC_ALL=C comm -23 "$HERE/symbols.txt" "$TMP/symbols.txt")"
  extra="$(LC_ALL=C comm -13 "$HERE/symbols.txt" "$TMP/symbols.txt")"

  # The floor is the fork's export set, and it has two halves that fail differently.
  #
  # The limina extensions (symbols-limina.txt) exist only in our fork. virglrs must export them,
  # because limina calls them; a stock upstream virglrenderer has no reason to and legitimately
  # does not. Scoring an upstream leg against the whole floor reports twenty-two failures for a
  # tree that is behaving correctly -- a red that means nothing, which is worse than no gate,
  # because the next person learns to expect it red.
  #
  # So: everything must meet the core floor. Only the Rust build must also meet the extensions.
  if [ -s "$HERE/symbols-limina.txt" ] && [ "$PREFIX" != "$ROOT/prefix" ]; then
    missing="$(LC_ALL=C comm -23 <(printf '%s\n' "$missing" | sed '/^$/d') "$HERE/symbols-limina.txt")"
    absent_ext="$(LC_ALL=C comm -12 \
      <(LC_ALL=C comm -23 "$HERE/symbols.txt" "$TMP/symbols.txt") "$HERE/symbols-limina.txt")"
    if [ -n "$absent_ext" ]; then
      echo "note: this leg carries none of the limina extensions ($(printf '%s\n' "$absent_ext" | wc -l | tr -d ' ') symbols)."
      echo "      expected of a stock upstream build; the Rust build is held to them."
    fi
  fi

  if [ -n "$missing" ]; then
    echo "symbols missing from $LIB:" >&2
    echo "$missing" | sed 's/^/  - /' >&2
    echo "ABI FIXTURE MISMATCH: symbols" >&2
    fail=1
  elif [ -n "$extra" ]; then
    echo "symbols matches, plus extensions beyond the pinned floor:"
    echo "$extra" | sed 's/^/  + /'
  else
    echo "symbols matches"
  fi
fi

exit $fail
