#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright © 2026 Gustavo Noronha Silva

# Score the conformance suites against one renderer, or diff the two.
#
#   fluster.sh c              the reference C renderer
#   fluster.sh rs             virglrs
#   fluster.sh diff           both, and diff what they decoded  <- the gate
#   fluster.sh diff --record  record the pin
#
# WHAT THIS ADDS OVER THE THREE VIDEO CORPORA. `vrend-h264.score` and its siblings are one clip
# per codec, scored by agreement with the C. These are the official conformance vectors, scored
# by agreement with the md5 the standards body published -- the only ABSOLUTE oracle in this
# tree. A stream both legs decode wrong is invisible to a differential and caught here.
#
# It does not replace the differential, because the absolute oracle is not the whole answer on
# this host: VideoToolbox refuses profiles the vectors cover (`vainfo` here advertises HEVC Main
# and no Main10, VP9 Profile 0 and nothing else), so streams fail for reasons that are the
# host's. Those are what the pin absorbs -- the gate is that the two legs decode the SAME set,
# and the absolute pass count is reported beside it because that is the number a reader can
# compare with anyone else's.
#
# Each vector is its own process, so unlike ctests.sh there is no cascade and no reason to pin
# only the first divergence: the whole result list is the fixture.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RIG="$HERE/../vm"
UPSTREAM="$HERE/upstream"
RESOURCES="$HERE/resources"
OUT="${TMPDIR:-/tmp}/virglrs-fluster"
PIN="$HERE/fixtures/results.txt"
PORT="${FLUSTER_SSH_PORT:-2299}"
# scp spells the port -P; two lists rather than rewriting one, so neither can drift.
SSH_COMMON=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR)
SSH_OPTS=("${SSH_COMMON[@]}" -p "$PORT")
SCP_OPTS=("${SSH_COMMON[@]}" -P "$PORT")
mkdir -p "$OUT"

[ -d "$UPSTREAM" ] && [ -d "$RESOURCES" ] || { echo "not set up: run harness/fluster/setup.sh" >&2; exit 1; }

# Only what is downloaded, and only if it is downloaded WHOLE. A suite missing vectors still
# runs -- fluster reports the absent ones as failures -- and pinning that would record the state
# of a download as if it were the renderer's answer.
# A command substitution and not `while read < <(...)`: /usr/bin/env bash is 3.2 on macOS,
# which has no mapfile and cannot take a heredoc inside a process substitution.
COMPLETE=$(python3 - "$UPSTREAM" "$RESOURCES" <<'COMPLETE_PY'
import json, os, sys

upstream, resources = sys.argv[1:]
index = {f[: -len(".json")]: os.path.join(root, f)
         for root, _, files in os.walk(os.path.join(upstream, "test_suites"))
         for f in files if f.endswith(".json")}

for suite in sorted(os.listdir(resources)):
    if suite not in index:
        continue
    vectors = json.load(open(index[suite]))["test_vectors"]
    missing = [tv["name"] for tv in vectors
               if not os.path.exists(os.path.join(resources, suite, tv["name"], tv["input_file"]))]
    if missing:
        print(f"{suite}: {len(missing)} of {len(vectors)} vectors not downloaded, skipping"
              f" (first: {missing[0]})", file=sys.stderr)
        continue
    print(suite)
COMPLETE_PY
)
SUITES=($COMPLETE)
[ -n "$COMPLETE" ] || { echo "no complete suite in $RESOURCES -- run setup.sh" >&2; exit 1; }

# -t 120 and not fluster's default 30: a Timeout is a verdict about the clock, and a verdict about
# the clock cannot be pinned -- it flaps with host load and would make the gate red for reasons
# that are nobody's. Typical decodes here are well under a second (305 VP9 vectors in 56 s), so
# 120 is ample headroom; what still times out at 120 is reliably too slow rather than borderline,
# and THAT is a stable thing to pin.

# Every VA decoder we have an element for. fluster matches a decoder to a suite by codec, so
# naming all of them runs each suite with the one that fits it and nothing else.
#
# GStreamer-AV1-VA is absent because `vaav1dec` is not an element on a host without AV1 silicon.
DECODERS=(GStreamer-H.264-VA GStreamer-H.265-VA GStreamer-VP9-VA)

# FLUSTER_VECTORS narrows the run to named vectors, for reading a single stream's renderer log
# rather than a whole suite's. It makes the run unpinnable on purpose -- a score over a subset is
# not the fixture -- so `diff` refuses to --record while it is set.
VECTORS=${FLUSTER_VECTORS:-}
TV=
[ -n "$VECTORS" ] && TV="-tv $VECTORS"

# Boot the stock guest -- the tier the video path lives on, and the one the three video corpora
# were recorded from -- run every complete suite in it, and bring it down.
#
# The vectors are shared read-only rather than copied in: several GB against a 13 GB image, and
# both legs must read the same bytes for the diff to mean anything.
run_leg() {
    local leg="$1" app disk log
    case "$leg" in
        c)     app="$RIG/Limina.app" ;;
        rs)    app="$RIG/Limina-rust.app" ;;
        *) echo "unknown leg: $leg (c|rs)" >&2; return 2 ;;
    esac
    [ -x "$app/Contents/MacOS/limina" ] || {
        echo "no rig -- run harness/vm/make-rig.sh --renderer $([ "$leg" = c ] && echo c || echo rust)" >&2
        return 1
    }
    disk="$RIG/disks/Fedora-Workstation-44.stock.test.raw"
    [ -f "$disk" ] || { echo "missing disk: $disk -- run harness/vm/make-rig.sh" >&2; return 1; }
    log="$OUT/$leg-boot.log"

    # --display-capture and not --window: it attaches the virtio-gpu a headless boot otherwise
    # gets none of, and the guest with no GPU has no render node and so no VA driver at all.
    "$app/Contents/MacOS/limina" \
        --disk "$disk" --net --ssh-port "$PORT" \
        --share "fluster=$HERE:ro" \
        --display-capture "$OUT/$leg-frame.png" --display-size 1280x800 \
        --firmware "$app/Contents/Resources/KRUN_EFI.gop.fd" > "$log" 2>&1 &
    local vm=$!
    # Bring the VM down however this returns -- including a failed run. A limina left holding the
    # disk makes the NEXT leg fail to boot, which reads like a renderer that cannot start.
    trap 'kill "$vm" 2>/dev/null; wait "$vm" 2>/dev/null' RETURN

    echo "==> $leg: booting (log: $log)"
    local waited=0
    until ssh "${SSH_OPTS[@]}" -o ConnectTimeout=2 -o BatchMode=yes claude@127.0.0.1 true 2>/dev/null; do
        sleep 3
        waited=$((waited + 3))
        [ "$waited" -lt 180 ] || { echo "$leg: no ssh after ${waited}s, see $log" >&2; return 1; }
        kill -0 "$vm" 2>/dev/null || { echo "$leg: the VM exited, see $log" >&2; return 1; }
    done

    echo "==> $leg: running ${SUITES[*]}"
    # The guest has no limina agent, so the share is mounted by hand rather than at /media.
    ssh "${SSH_OPTS[@]}" claude@127.0.0.1 "
        set -e
        sudo mkdir -p /media/fluster
        mountpoint -q /media/fluster || sudo mount -t virtiofs limina-fluster /media/fluster
        python3 /media/fluster/upstream/fluster.py \
            -r /media/fluster/resources -o /tmp/fluster-out -ne \
            run -ts ${SUITES[*]} -d ${DECODERS[*]} $TV -j 1 -q -t 120 \
                -so /tmp/summary.json -f json
    " > "$OUT/$leg-run.log" 2>&1
    local rc=$?

    scp "${SCP_OPTS[@]}" claude@127.0.0.1:/tmp/summary.json "$OUT/$leg.json" 2>/dev/null || {
        echo "$leg: no summary came back (fluster rc=$rc), see $OUT/$leg-run.log" >&2
        return 1
    }
    ssh "${SSH_OPTS[@]}" claude@127.0.0.1 'sudo systemctl poweroff' 2>/dev/null || true

    # The fixture is the verdicts alone. Times, the machine's own description and the per-profile
    # tallies all vary run to run and would make a pin that never matches twice.
    python3 - "$OUT/$leg.json" <<'REDUCE_PY' > "$OUT/$leg.txt"
import json, sys

report = json.load(open(sys.argv[1]))
for suite, data in sorted(report["test_suites"].items()):
    for decoder, d in sorted(data["decoders"].items()):
        for vector, v in sorted(d["vectors"].items()):
            print(f"{suite} {decoder} {vector} {v['result']}")
REDUCE_PY

    local total pass
    total=$(wc -l < "$OUT/$leg.txt" | tr -d ' ')
    pass=$(grep -c ' Success$' "$OUT/$leg.txt")
    printf '%s: %s/%s vectors match the published md5\n' "$leg" "$pass" "$total"
}

case "${1:-diff}" in
  c|rs)
    run_leg "$1" ;;
  diff)
    run_leg c  || exit 1
    run_leg rs || exit 1

    echo "=== per-suite (c | rs) ==="
    for s in "${SUITES[@]}"; do
        printf '%-20s %4s | %4s of %s\n' "$s" \
            "$(grep -c "^$s .* Success$" "$OUT/c.txt")" \
            "$(grep -c "^$s .* Success$" "$OUT/rs.txt")" \
            "$(grep -c "^$s " "$OUT/c.txt")"
    done

    echo "=== divergence ==="
    if diff -u "$OUT/c.txt" "$OUT/rs.txt" > "$OUT/legs.diff"; then
        echo "the two legs decoded every vector the same way"
    else
        echo "THE LEGS DISAGREE -- every line here is a finding, not a host limitation:"
        /bin/cat "$OUT/legs.diff"
    fi

    if [ "${2:-}" = "--record" ]; then
        [ -z "$VECTORS" ] || { echo "refusing to record a pin from a FLUSTER_VECTORS subset" >&2; exit 1; }
        mkdir -p "$(dirname "$PIN")"
        cp "$OUT/c.txt" "$PIN"
        echo "recorded $PIN"
        exit 0
    fi
    [ -f "$PIN" ] || { echo "no pin at $PIN -- record one with: fluster.sh diff --record" >&2; exit 1; }

    echo "=== against the pin ==="
    # The pin is the C leg's verdicts. It moves when a vector starts or stops decoding to the
    # published md5 on the REFERENCE -- a host, driver or VideoToolbox change, never ours -- and
    # it is separate from the legs disagreeing, which is ours.
    if diff -u "$PIN" "$OUT/c.txt"; then
        echo "matches $PIN"
    else
        echo
        echo "the C leg's own verdicts moved. That is the host underneath both legs changing," \
             "not a renderer difference; re-record only once you know which."
        echo "A whole suite appearing here is the benign case: the pin was recorded before that" \
             "suite had finished downloading. Re-record."
        exit 1
    fi
    diff -q "$OUT/c.txt" "$OUT/rs.txt" > /dev/null || exit 1
    ;;
  *)
    echo "usage: fluster.sh [c|rs|diff] [--record]" >&2; exit 1 ;;
esac
