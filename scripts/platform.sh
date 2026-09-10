# SPDX-License-Identifier: MIT
# Copyright © 2026 Gustavo Noronha Silva
#
# What differs between the hosts this tree builds on. Sourced, never run.
#
# One copy, because four scripts each carrying their own is four chances to disagree about which
# library they are scoring -- and the failure mode of that disagreement is not an error. It is a
# full, plausible score for the wrong leg, which reads exactly like a regression in the right one.

# Where a built virglrenderer lands, in the order the builds here lay it down. meson on Fedora
# installs into lib64, meson on Darwin and this tree's own install.sh into lib, and a meson build
# directory that was never installed keeps it under src/.
virgl_lib_candidates() {
    case "$(uname -s)" in
    Darwin)
        printf '%s\n' \
            lib/libvirglrenderer.1.dylib \
            lib/libvirglrenderer.dylib \
            src/libvirglrenderer.dylib
        ;;
    *)
        printf '%s\n' \
            lib64/libvirglrenderer.so.1 \
            lib/libvirglrenderer.so.1 \
            lib64/libvirglrenderer.so \
            lib/libvirglrenderer.so \
            src/libvirglrenderer.so
        ;;
    esac
}

# The built library under a prefix. Prints its path; non-zero if the prefix holds none.
virgl_find_lib() {
    prefix="$1"
    for rel in $(virgl_lib_candidates); do
        if [ -f "$prefix/$rel" ]; then
            printf '%s\n' "$prefix/$rel"
            return 0
        fi
    done
    return 1
}

# A build directory carrying the generated headers -- `virgl-version.h`, and `config.h` for
# anything that includes the C's internals. Which C tree was built differs by host
# (scripts/build-reference.sh decides); this finds whichever one exists rather than encoding
# that decision a second time.
virgl_generated_include() {
    root="$1"
    for cand in \
        "$root/harness/vm/build/src" \
        "$root/third_party/virglrenderer/build/src" \
        "$root/third_party/virglrenderer-upstream/build/src"
    do
        if [ -f "$cand/virgl-version.h" ]; then
            printf '%s\n' "$cand"
            return 0
        fi
    done
    return 1
}

# The exported symbols of a shared library, one per line, without the leading underscore Mach-O
# prefixes them with. Normalised so ONE pinned floor serves both hosts: the underscore is a
# calling-convention detail of the object format, not a difference in what the library exports,
# and a second fixture would be a second thing to keep in step.
virgl_exported_symbols() {
    case "$(uname -s)" in
    Darwin)
        nm -gU "$1" | awk '$2 == "T" || $2 == "S" { print $3 }' | sed 's/^_//'
        ;;
    *)
        # Dynamic symbols only: that is the exported set. Version suffixes are stripped because
        # they belong to the build's linker script, not to the ABI's name.
        nm -D --defined-only "$1" | awk 'NF >= 3 { print $NF }' | sed 's/@@.*//'
        ;;
    esac | LC_ALL=C sort -u
}

# What a binary actually linked, for saying out loud which leg it got.
virgl_link_report() {
    case "$(uname -s)" in
    Darwin) otool -L "$1" ;;
    *)      ldd "$1" ;;
    esac
}

# The SHA-256 of a file, as a bare hex digest.
#
# `shasum` is Perl's and ships with macOS; `sha256sum` is coreutils' and ships with Linux. Neither
# host has both. This matters more than a missing tool usually does, because the callers use the
# answer to decide whether a download is intact: with `shasum` absent the command substitution is
# empty, the comparison fails, and the corpus is reported as not matching its pin -- which blames
# the bytes for the absence of a program. Fail loudly instead.
virgl_sha256() {
    if command -v sha256sum > /dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    elif command -v shasum > /dev/null 2>&1; then
        shasum -a 256 "$1" | cut -d' ' -f1
    else
        echo "no sha256 tool: install coreutils (sha256sum) or perl (shasum)" >&2
        exit 1
    fi
}

# The size of a file in bytes.
#
# `stat` is not one program: BSD's takes -f%z and GNU's takes -c%s, and each rejects the other's
# flag. Same shape of trap as virgl_sha256 above -- the caller writes the answer into a manifest,
# so a silent empty string becomes a pinned size of nothing.
virgl_file_size() {
    case "$(uname -s)" in
    Darwin) stat -f%z "$1" ;;
    *)      stat -c%s "$1" ;;
    esac
}
