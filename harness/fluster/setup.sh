#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright © 2026 Gustavo Noronha Silva

# Materialize fluster and the conformance vectors it scores against, HOST-side.
#
#   setup.sh [SUITE ...]      default: the three suites this host can score
#
# The vectors stay on the host and are shared read-only into the guest, never copied into a
# disk image: they are several GB against a 13 GB stock image, they are identical for both
# legs, and a byte the two legs did not both read is not a differential.
#
# Downloading is host-side for the same reason -- fluster's `download` only fetches and
# unpacks, so there is nothing about it that wants a guest.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
UPSTREAM="$HERE/upstream"
RESOURCES="$HERE/resources"

# Pinned, like every other input here. A suite's expected md5s live in the checkout's
# test_suites/*.json, so a bump can change what a stream is scored against -- which is a
# fixture change, and wants to be one.
REPO=https://github.com/fluendo/fluster.git
REV=f3ad284a9e6cac70dc01b02e0de71c2994181d34

# AV1-TEST-VECTORS is deliberately absent: AV1 decode needs M3-or-later silicon, so on a
# machine without it every stream fails for a reason that is the host's -- `vaav1dec` is not
# even an element there. It is scored on the AV1 machine, the way vrend-av1.score is.
SUITES=("$@")
[ ${#SUITES[@]} -gt 0 ] || SUITES=(JVT-AVC_V1 JCT-VC-HEVC_V1 VP9-TEST-VECTORS)

if [ ! -d "$UPSTREAM/.git" ]; then
    echo "==> cloning fluster"
    git clone "$REPO" "$UPSTREAM"
fi
git -C "$UPSTREAM" fetch --quiet origin "$REV" 2>/dev/null || git -C "$UPSTREAM" fetch --quiet origin
git -C "$UPSTREAM" checkout --quiet "$REV"

mkdir -p "$RESOURCES"
echo "==> downloading ${SUITES[*]} into $RESOURCES"

# JVT-AVC_V1 and JCT-VC-HEVC_V1 come from www.itu.int, which sits behind a WAF that answers
# HTTP 200 with a 245-byte "Request Rejected" page instead of the file. fluster stores that and
# reports a checksum mismatch -- which is also what a changed upstream vector looks like, so the
# message does not tell them apart. Two attempts giving two DIFFERENT checksums for one URL does,
# and `file` on the .zip says HTML outright.
#
# The block is by IP and covers the whole host -- www.itu.int/ itself is refused the same way --
# so no header, User-Agent or browser gets past it, and only going slow avoids earning it. Hence
# -j 1 and a minute between attempts. VP9-TEST-VECTORS is on storage.googleapis.com and has none
# of this: download it on its own when the ITU host is sulking.
#
# -r 2 and not more: fluster raises the flag to itself (`ctx.retries ** ctx.retries`), so -r 5 is
# 3125 inner attempts -- inert while the WAF returns 200, a multi-hour hang the day it returns 403.
#
# -k keeps the verified archives, which is what lets an attempt resume rather than re-fetch.
for attempt in $(seq 1 5); do
    if python3 "$UPSTREAM/fluster.py" -r "$RESOURCES" download -j 1 -r 2 -k "${SUITES[@]}"; then
        break
    fi
    [ "$attempt" = 5 ] && { echo "download still rejected after 5 attempts" >&2; exit 1; }
    echo "==> attempt $attempt rejected, waiting out the block"
    sleep 60
done

# An attempt that died mid-suite leaves archives that verify but were never unpacked: fluster's
# skip-if-already-verified path returns before extracting, so no later run extracts them and the
# vector is silently absent from every score. Close it here rather than trusting the resume.
python3 - "$UPSTREAM" "$RESOURCES" "${SUITES[@]}" <<'REPAIR_PY'
import json, os, sys, zipfile

upstream, resources, *suites = sys.argv[1:]
index = {f[:-len(".json")]: os.path.join(root, f)
         for root, _, files in os.walk(os.path.join(upstream, "test_suites"))
         for f in files if f.endswith(".json")}

for suite in suites:
    for tv in json.load(open(index[suite]))["test_vectors"]:
        d = os.path.join(resources, suite, tv["name"])
        archive = os.path.join(d, os.path.basename(tv["source"]))
        if os.path.exists(os.path.join(d, tv["input_file"])):
            continue
        if not os.path.exists(archive) or not zipfile.is_zipfile(archive):
            continue
        print(f"\textracting {tv['name']}, which a resumed download had skipped")
        zipfile.ZipFile(archive).extract(tv["input_file"], d)
REPAIR_PY

echo
du -sh "$RESOURCES"/* 2>/dev/null || true
echo "==> run them: harness/fluster/fluster.sh c|rs|diff"
