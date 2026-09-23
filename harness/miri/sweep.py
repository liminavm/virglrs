#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright © 2026 Gustavo Noronha Silva
"""Run every unit test Miri can, one test module at a time, skipping the ones that reach C.

    harness/miri/sweep.py [module ...]

A module is a test path prefix such as `venus::cs::tests::`. With none, every module the library's
test list names is swept. Prints one line per module: `CLEAN`, `UB`, `FAIL` or `TIMEOUT`, how many
tests ran, and which were skipped.

Miri cannot follow a foreign call, and the first one ends the whole run, hiding every test after
it. So when a run stops on one, that test is skipped by name and the module is run again, until it
finishes. A module with many such tests takes many runs; `venus::context` takes over a hundred.
Skipping is the only thing done automatically: undefined behaviour or any other failure ends the
module and is reported as it is.

`Driver::abandon_planted` leaks the tables a test planted on purpose -- dropping them would call
entry points the test never planted -- so leak checking is off for the modules that use it.
"""

import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Miri gets its own target directory: sharing the normal one would rebuild it for every run.
ENV = dict(os.environ, CARGO_TARGET_DIR=str(ROOT / 'target' / 'miri'))
LOGS = ROOT / 'target' / 'miri' / 'sweep-logs'
# One run of one module. `venus::context`'s run through the whole command set is the long pole.
BUDGET = 1800
LEAKY = ('venus::driver::tests::', 'venus::context::tests::')

STARTED = re.compile(r'^test (\S+)(?: - should panic)? \.\.\. ', re.M)
FINISHED = re.compile(r'^test (\S+)(?: - should panic)? \.\.\. (?:ok|FAILED|ignored)', re.M)


def modules():
    """Every test module the library has, as the prefix its tests share."""
    out = subprocess.run(
        ['cargo', 'test', '--lib', '--', '--list', '--format=terse'],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    found = set()
    for line in out.splitlines():
        name = line.removesuffix(': test')
        if name == line:
            continue
        # The module is everything up to the innermost `tests::`-like segment the test sits in:
        # its own path minus the test's name.
        found.add(name.rsplit('::', 1)[0] + '::')
    # A nested module is swept as part of its parent.
    return sorted(m for m in found if not any(m != p and m.startswith(p) for p in found))


def run(module, skips, log):
    argv = ['cargo', '+nightly', 'miri', 'test', '--lib', module, '--', '--test-threads=1']
    for s in skips:
        argv += ['--skip', s]
    env = dict(ENV)
    if module in LEAKY:
        env['MIRIFLAGS'] = (env.get('MIRIFLAGS', '') + ' -Zmiri-ignore-leaks').strip()
    with open(log, 'w') as out:
        p = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT,
                             start_new_session=True)
        try:
            return p.wait(timeout=BUDGET), False
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            return None, True


def stuck_on(module, text):
    """The test a foreign call stopped, which the log does not always finish printing."""
    finished = set(FINISHED.findall(text))
    stuck = [t for t in STARTED.findall(text) if t not in finished]
    if stuck:
        return stuck[-1]
    # Its own line may not be flushed before Miri stops; its frame is in the backtrace, the
    # outermost one under this module's prefix.
    frames = re.findall(r'^\s*\d+: (%s\w+)\s*$' % re.escape(module), text, re.M)
    return frames[-1] if frames else None


def sweep(module):
    skips, began, n = [], time.monotonic(), 0
    while True:
        n += 1
        log = LOGS / ('%s.%d.log' % (module.replace(':', '_'), n))
        code, timed_out = run(module, skips, log)
        text = log.read_text()
        result = re.findall(r'test result: .*', text)
        stuck = stuck_on(module, text) if 'unsupported operation' in text else None
        if timed_out:
            verdict = 'TIMEOUT'
        elif 'Undefined Behavior' in text:
            verdict = 'UB'
        elif stuck and stuck not in skips:
            skips.append(stuck)
            continue
        elif code == 0:
            verdict = 'CLEAN'
        else:
            verdict = 'FAIL(%s)' % code
        print('%-8s %6.0fs %-40s %s | skipped %d: %s | %s' % (
            verdict, time.monotonic() - began, module, result[-1] if result else '-', len(skips),
            ' '.join(s.rsplit('::', 1)[-1] for s in skips), log), flush=True)
        return verdict == 'CLEAN'


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    wanted = sys.argv[1:] or modules()
    clean = [sweep(m) for m in wanted]
    print('\n%d of %d modules clean' % (sum(clean), len(clean)))
    sys.exit(0 if all(clean) else 1)


if __name__ == '__main__':
    main()
