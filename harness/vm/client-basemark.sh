#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright © 2026 Gustavo Noronha Silva
# Basemark Web 3.0 graphics suite on the stock guest, driven rather than clicked, because the
# sample has to be aimed: the suite is seven short tests in sequence, and a profile that smears
# across several of them cannot be attributed to any one.
#
# What this prints to stdout, interleaved and timestamped, is the whole point -- the host samples
# the vmm in parallel and the two logs are correlated afterwards. Nothing here renders a verdict.
#
# The trap this shares with client-gl-synoik.sh, and the reason that script exists: an SSH shell
# does not have the session's GL environment. Without it Firefox falls back to llvmpipe -- it
# draws, the benchmark completes, it reports a score, and the renderer under test never sees a
# command. That profile reads as "no hotspot", which is the most misleading result available, so
# the renderer is asserted below and a fallback refuses the run rather than scoring it.
set -u

PROBE_HTML=/tmp/basemark-probe.html
URL_PROBE="file://$PROBE_HTML"
# suite=2 is the Graphics suite alone: WebGL 1.0.2, WebGL 2.0, Shader Pipeline, Draw-call Stress,
# Geometry Stress, Canvas, SVG. Per-test selection is a Corporate-version feature, so the suite is
# the finest cut available -- which is exactly the seven tests we care about.
# Community mode is ACTIVATED at the root and persists in the profile; it is not a query parameter
# that /run/ understands on its own. A fresh profile each run means activating each run, so this is
# loaded first and its only job is to leave that state behind. Skipping it leaves the suite sitting
# behind a Start button while the profile records an idle desktop.
#
# Every option is set HERE, on the root, and none of them are understood by /run/. Community mode
# prints its own configuration to the console, and that block -- not the URL we asked for -- is
# what says which suite is about to run. It is asserted below, because a suite=2 that silently
# stayed "All suites" scores the JS tests too and costs eight minutes to discover.
URL_MODE='https://web.gpuscore.com/?mode=community&suite=2&conformance=0&battery=0'
URL_RUN='https://web.gpuscore.com/run/'

PROFILE=/tmp/basemark-profile
LOG=/tmp/basemark.log

stamp() { while IFS= read -r l; do printf '%s %s\n' "$(date +%s.%N)" "$l"; done; }

# The session's GL environment, taken from the compositor itself -- see the note above.
COMP=$(pgrep -u "$(id -u)" -x gnome-shell || pgrep -u "$(id -u)" -x synoik || true)
COMP=${COMP%%$'\n'*}
if [ -n "$COMP" ]; then
  echo "compositor pid=$COMP ($(tr -d '\0' < /proc/$COMP/comm))"
  while IFS= read -r -d '' kv; do
    case "$kv" in
      LD_LIBRARY_PATH=*|MESA_*|GALLIUM_*|ZINK_*|LIBGL_*|EGL_*|VK_*|XDG_RUNTIME_DIR=*|DISPLAY=*)
        export "$kv"; echo "  inherited: $kv" ;;
    esac
  done < /proc/$COMP/environ
else
  echo "no compositor found; Firefox would fall back to llvmpipe and profile nothing" >&2
  exit 1
fi
: "${XDG_RUNTIME_DIR:=/run/user/$(id -u)}"; export XDG_RUNTIME_DIR
for s in "$XDG_RUNTIME_DIR"/wayland-*; do
  case "$s" in *.lock) continue ;; esac
  [ -S "$s" ] && { export WAYLAND_DISPLAY="$(basename "$s")"; break; }
done
echo "WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-UNSET}"
export MOZ_ENABLE_WAYLAND=1

rm -rf /tmp/basemark-profile
mkdir -p "$PROFILE"
# Content-process console.log to stdout: this is what carries community mode's machine-readable
# per-test lines, and so what the sample windows are aimed by. Verify on the first run that these
# actually appear -- a silent pref rename would leave the run looking fine and unattributable.
cat > "$PROFILE/user.js" <<'PREFS'
user_pref("devtools.console.stdout.content", true);
user_pref("webgl.sanitize-unmasked-renderer", false);
user_pref("browser.shell.checkDefaultBrowser", false);
user_pref("browser.sessionstore.resume_from_crash", false);
user_pref("browser.startup.homepage_override.mstone", "ignore");
user_pref("datareporting.policy.dataSubmissionEnabled", false);
user_pref("app.update.auto", false);
user_pref("toolkit.telemetry.enabled", false);
PREFS

# A real file, not a data: URL: Firefox has blocked top-level data: navigation since 59, so a
# data: probe never loads at all -- and the failure is indistinguishable from the console pref not
# taking, which is a diagnostic pointing at the wrong thing.
cat > "$PROBE_HTML" <<'HTML'
<canvas id=c></canvas><script>
var gl = document.getElementById("c").getContext("webgl");
var e = gl && gl.getExtension("WEBGL_debug_renderer_info");
console.log("BASEMARK_PROBE renderer=" +
  (e ? gl.getParameter(e.UNMASKED_RENDERER_WEBGL) : "unknown") +
  " version=" + (gl ? gl.getParameter(gl.VERSION) : "no-webgl"));
</script>
HTML

echo "=== one Firefox for the whole run"
# ONE process, start to finish. The configuration community mode records is session state, and a
# probe/configure/run sequence of separate processes relies on it surviving a SIGTERM and a
# round trip through the profile on disk. It does not, reliably: the run comes up with defaults
# and the suite sits behind its Start button while every log line still looks right. Later URLs
# are handed to the instance that is already running, which is what a person does.
timeout 600 firefox --profile "$PROFILE" --new-instance "$URL_PROBE" 2>&1 | stamp | tee -a "$LOG" &
sleep 22

probe=$(grep -o 'BASEMARK_PROBE renderer=.*' "$LOG" | tail -1 || true)
echo "probe: ${probe:-NONE}"
case "$probe" in
  *llvmpipe*|*softpipe*|*swrast*)
    echo "REFUSING: Firefox is on a software rasteriser -- this run would profile nothing" >&2
    exit 1 ;;
  "")
    echo "REFUSING: no probe line reached the log; the console pref did not take, so nothing" >&2
    echo "          here can be correlated or trusted" >&2
    exit 1 ;;
esac

echo "=== configuring: community mode and the graphics suite, in the same session"
firefox --profile "$PROFILE" "$URL_MODE" > /dev/null 2>&1
sleep 20

echo "--- reported configuration:"
grep -o '"[ ]*[A-Za-z][A-Za-z ]*: [^"]*"' "$LOG" | sort -u
suite_line=$(grep -o '"    Suite: [^"]*"' "$LOG" | tail -1)
mode_line=$(grep -o '"    Mode: [^"]*"' "$LOG" | tail -1)
case "$suite_line" in
  *Graphics*) ;;
  *) echo "REFUSING: the suite is ${suite_line:-unreported}, not Graphics" >&2; exit 1 ;;
esac
case "$mode_line" in
  *community*) ;;
  *) echo "REFUSING: mode is ${mode_line:-unreported}" >&2; exit 1 ;;
esac

echo "=== launching the suite in that same session"
firefox --profile "$PROFILE" "$URL_RUN" > /dev/null 2>&1
sleep 10

# Framing, not motion: see tap-keys.py. The overview keeps presenting, but it composites the
# window as a thumbnail inside the shell's UI rather than showing it at its own size.
sudo python3 /tmp/tap-keys.py esc || echo "WARNING: could not tap esc" >&2

echo "=== suite launched; sample the host vmm now"
echo "=== host side: sample \$(pgrep -f '[l]imina-vmm --cpus') 10 -f prof.txt"
wait
