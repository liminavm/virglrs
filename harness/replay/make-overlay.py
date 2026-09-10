#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright © 2026 Gustavo Noronha Silva
#
# Write the overlay that carries a fixture onto another GL driver.
#
#   make-overlay.py <fixture.score> <run.score> <run.log> <out.score>
#
# An overlay names the lines a second driver reads differently, and `vrend-replay.sh
# --expect-overlay` applies it over the fixture before comparing. It exists because most of a
# score must agree on every host: a whole second golden would put those lines under two writers,
# and the pair drifts the first time only one of them is re-recorded.
#
# Derived from a run rather than transcribed from a diff, because a transcribed hash is a hash
# nobody can check. What this refuses is the point: a line that is not a pixel hash has no
# business differing between drivers, and a line whose INK moved is not obviously the driver's
# arithmetic -- the same pixels are no longer lit, and something drew differently. Ink that moves
# by a few counts on an antialiased edge is the known exception, so it is reported rather than
# refused, and the number is there to be looked at.
import re, sys

LINE = re.compile(r'^res=(\d+) (\S+) hash=(\S+) ink=(\d+)/(\d+)$')

def scored(path):
    out = {}
    for line in open(path):
        m = LINE.match(line.rstrip('\n'))
        if m:
            out[m.group(1)] = (m.group(2), m.group(3), int(m.group(4)), int(m.group(5)), line.rstrip('\n'))
    return out

def main():
    if len(sys.argv) != 5:
        sys.exit(__doc__ or 'usage: make-overlay.py <fixture> <run> <log> <out>')
    fixture, run, log, out = sys.argv[1:]

    # The GL the renderer actually got, from the renderer's own mouth: it prints the host
    # context's version string at init. Asking eglinfo would report eglinfo's own context.
    driver = 'unknown'
    for line in open(log, errors='replace'):
        m = re.search(r'\[virglrs\] vrend: (OpenGL[^,]*)', line)
        if m:
            driver = m.group(1)
            break

    want, have = scored(fixture), scored(run)
    lines, moved = [], []
    for handle, w in want.items():
        h = have.get(handle)
        if h is None:
            sys.exit(f'{run}: res={handle} is in the fixture and not in the run -- this run does '
                     f'not score the same corpus, and an overlay from it would pin the wrong thing')
        if h[:4] == w[:4]:
            continue
        if h[0] != w[0]:
            sys.exit(f'res={handle}: the fixture says {w[0]} and the run says {h[0]}. An extent is '
                     f'not the driver\'s arithmetic; look at it before pinning anything')
        lines.append(h[4])
        if h[2] != w[2]:
            moved.append(f'res={handle} {w[0]}: ink {w[2]} -> {h[2]} ({h[2] - w[2]:+d})')

    with open(out, 'w') as f:
        f.write(f'# The lines of {fixture.rsplit("/", 1)[-1]} that this GL driver reads\n'
                f'# differently, and only those: everything else must agree with it byte for byte.\n'
                f'#\n'
                f'# Written by make-overlay.py from a scored run. Pixel hashes only -- an extent that\n'
                f'# moved is refused here, because that is not arithmetic.\n'
                f'#\n'
                f'# Recorded against: {driver}\n')
        if moved:
            f.write('#\n# Ink moved on these, which is an antialiased edge landing on different\n'
                    '# pixels rather than a different drawing. Small, and worth re-reading if it grows:\n')
            for m in moved:
                f.write(f'#   {m}\n')
        for l in lines:
            f.write(l + '\n')

    print(f'{out}: {len(lines)} line(s), {len(moved)} with ink moved, driver {driver!r}')

main()
