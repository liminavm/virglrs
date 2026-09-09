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

URL_PROBE='data:text/html,<canvas id=c></canvas><script>
var gl=document.getElementById("c").getContext("webgl");
var e=gl.getExtension("WEBGL_debug_renderer_info");
console.log("BASEMARK_PROBE renderer="+(e?gl.getParameter(e.UNMASKED_RENDERER_WEBGL):"unknown")
  +" version="+gl.getParameter(gl.VERSION));
</script>'
# suite=2 is the Graphics suite alone: WebGL 1.0.2, WebGL 2.0, Shader Pipeline, Draw-call Stress,
# Geometry Stress, Canvas, SVG. Per-test selection is a Corporate-version feature, so the suite is
# the finest cut available -- which is exactly the seven tests we care about.
URL_RUN='https://web.gpuscore.com/run/?mode=community&suite=2&conformance=0&battery=0'

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
user_pref("browser.shell.checkDefaultBrowser", false);
user_pref("browser.sessionstore.resume_from_crash", false);
user_pref("browser.startup.homepage_override.mstone", "ignore");
user_pref("datareporting.policy.dataSubmissionEnabled", false);
user_pref("app.update.auto", false);
user_pref("toolkit.telemetry.enabled", false);
PREFS

echo "=== probing the renderer Firefox actually got"
timeout 60 firefox --profile "$PROFILE" --new-instance "$URL_PROBE" 2>&1 | stamp | tee "$LOG" &
sleep 25
probe=$(grep -o 'BASEMARK_PROBE renderer=[^ ]*' "$LOG" | tail -1 || true)
echo "probe: ${probe:-NONE}"
case "$probe" in
  *llvmpipe*|*softpipe*|*swrast*)
    echo "REFUSING: Firefox is on a software rasteriser -- this run would profile nothing" >&2
    pkill -f "$PROFILE" 2>/dev/null
    exit 1 ;;
  "")
    echo "REFUSING: no probe line reached the log; the console pref did not take, so the sample" >&2
    echo "          windows would have nothing to be aimed by" >&2
    pkill -f "$PROFILE" 2>/dev/null
    exit 1 ;;
esac
pkill -f "$PROFILE" 2>/dev/null
sleep 3

echo "=== running the graphics suite; sample the host vmm now"
echo "=== host side: sample \$(pgrep -f '[l]imina-vmm --cpus') 20 -f /tmp/prof.txt"
exec firefox --profile "$PROFILE" --new-instance --kiosk "$URL_RUN" 2>&1 | stamp | tee -a "$LOG"
