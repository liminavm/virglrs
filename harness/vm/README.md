# The capture rig

A self-contained way to boot a real Linux guest against **this tree's** renderer, so a real
workload becomes a replay corpus for `harness/replay/`. Nothing here is tracked by git except the
scripts and this file — the bundle, the disks, the build and the captures are all local artifacts.

```sh
./build-renderer.sh          # this tree -> harness/vm/prefix
./make-rig.sh                # clone limina's bundle + two disks, swap our renderer in
./capture.sh venus  # boot the enhanced guest with the venus recorder armed
./dump.sh venus              # ask for the dump, then check it
```

## What the rig is

`make-rig.sh` clones limina's signed `Limina.app`, replaces
`Contents/Frameworks/libvirglrenderer.1.dylib` with ours, rewrites its dependencies from absolute
paths back to `@rpath` so the bundle stays self-contained, and re-signs inside-out preserving each
binary's own entitlements (the worker needs `com.apple.security.hypervisor`).

Working against a copy rather than limina's own bundle is the point: capturing a corpus means
booting a guest against a renderer we are actively changing, and doing that in limina's tree would
make every capture a mutation of their working set.

The guest disks are **APFS clones** (`cp -c`). They cost no space until the guest writes, and the
source images are never touched — an ordinary copy of two ~15 GB images would not fit on this
host.

## What the three images are

The tiers are **not** graphics capability tiers, and reading them as such is how a bug gets
explained away as "the stock guest would never do that". Every foundational graphics capability is
in all three.

- **stock** — a stock Fedora Workstation. Renders through virgl/vrend, has venus available, and
  hardware acceleration works.
- **enhanced** — stock, plus a 16k-page-size kernel and fixes to guest-mesa issues we found. It
  adds no foundational graphics capability over stock.
- **synoik** — enhanced, with a Vulkan compositor in place of GNOME.

How the three are built, refreshed and brought forward is limina's to say, not this tree's: see
**`limina/docs/images.md`**, whose "The virglrs harness rig" section names which canonical image
each rig disk is a clone of. `make-rig.sh --disks-only` clones them; nothing here builds one.
Only recording a corpus needs them — every pinned score replays with no guest at all.

So a difference in what a workload does between stock and enhanced is a guest-mesa behaviour
difference or a workload difference, never "that tier can't". When a capture on one tier shows a
command the other's corpus lacks, the question to ask is what the two sessions *did* differently.

## The three guests, and what each one needs

All three images boot to **multi-user**, not to a seated desktop, so a capture that just boots
records nothing. What starts the workload differs per image, and so does which renderer it drives:

| `capture.sh` | guest | workload | drives |
|---|---|---|---|
| `synoik` | enhanced + synoik | starts on its own at boot | venus, end to end |
| `venus` | enhanced GNOME | `sudo systemctl isolate graphical.target`, then a Vulkan client (`vkcube`, `vulkaninfo`) over SSH | venus from clients only |
| `vrend` | stock | `sudo systemctl isolate graphical.target` | classic vrend, plus the video path |

The surprise worth keeping: on the **enhanced** image gnome-shell renders through **classic virgl**
(`GALLIUM_DRIVER=virgl` in its environ), not zink→venus. Its venus traffic comes from Vulkan
clients, which makes it the mixed case rather than the venus one. Synoik is the desktop workload
that is venus throughout.

## Pixels are the oracle

Everything else in this rig is a proxy: a systemd target, a `vulkaninfo` line, a renderer log,
a corpus score. Each answers a real question and none of them answers "is there a desktop on the
screen" -- a compositor that exits at startup leaves `graphical-session.target` active, a clean
renderer log and a corpus that replays. The frame is the only thing that settles it.

`frame.py [PNG...]` reports on a captured frame: dimensions, distinct colours, dominant colour and
its share, and a coarse luminance sketch, then the path. It prints no verdict on purpose -- a
"looks seated" line would be a new proxy, and a more dangerous one for sounding like it looked.
Use the numbers to know where to look and then open the file.

Two properties of the capture decide how to use it. It exists only for a **headless** boot, since
limina refuses `--display-capture` together with a window -- so a windowed boot's oracle is the
human in front of it, and a headless boot's is the file. And the file holds the **last presented
frame**, rewritten as frames arrive: read it while the workload runs and it is the current screen,
read it after shutdown and it is the teardown console. The `*-frame.png` files a capture leaves
behind are therefore poweroff screens and evidence of nothing.

Which makes one frame insufficient on its own: a compositor whose context died leaves the last
frame it presented sitting in the surface, and that frame looks exactly like a seated desktop.
**Two samples, and something in them that has to move.** The clock is the obvious candidate and
a poor one -- it changes once a minute, so two samples inside the same minute prove nothing.
Anything animating is better: a rotating `vkcube` in the frame settles both liveness and its own
import at once, since a cube at a different angle in the second sample is a client still drawing
and a compositor still sampling it.

`--renderer c|rust` picks which build to boot, and picks the bundle with it: `Limina.app` holds
the C, `Limina-rust.app` holds virglrs. Only the C records — the recorder is a C-tree feature, so a
virglrs boot yields no corpus and the two legs divide accordingly: the C leg captures, the Rust leg
is what the capture is then replayed against. A boot also reaches what replay cannot -- the ring
transport commands, for the reasons `../README.md` gives. Booting both on the same guest with the same client is
how a refusal is attributed: `vkcube` running its full 25 seconds on one and segfaulting on the
other is a gap in this tree, not in vkcube.

A Vulkan client over SSH needs only `VK_DRIVER_FILES=/usr/share/vulkan/icd.d/virtio_icd.aarch64.json`
plus `XDG_RUNTIME_DIR` and `WAYLAND_DISPLAY` for a windowed one. A **GL** client needs the session's
zink environment too, which an SSH shell does not inherit — without it the stack silently falls back
to llvmpipe and the capture records nothing while looking healthy.

**For a video workload, drive Showtime or `waylandsink` — not `glimagesink`.** A `gst-launch-1.0`
pipeline ending in `glimagesink` decodes correctly and runs to EOS with rc=0, and never puts a
window on the screen. It does this on **both** renderer legs, so it is the sink and not the port;
the C leg's captured frame shows the same empty desktop after the same playback. It is worth
finding out why, because a decoder whose only visible failure is an absent window is a workload
that scores green while measuring nothing on screen — but until it is, a corpus recorded through
it carries the decode and not the presentation. Showtime drives the whole path, composite planar
decode target included, with the picture up.

Two parser notes that cost nothing to know: `matroskademux ! vavp9dec` fails to negotiate without
a `vp9parse` between them, and `ffplay` is not a route here at all — no `libopenh264`, and its
vaapi-from-vulkan derivation fails.

**A capture can tap keys.** `tap-keys.py` is `type-into-overview.py`'s other half: it taps named
keys through `/dev/uinput` rather than typing words (`sudo python3 /tmp/tap-keys.py esc`). A guest
boots into the overview, which composites the session's windows as scaled thumbnails inside the
shell's own UI. They are live and the compositor keeps presenting, so what this costs is not
motion but framing: what gets captured, or scored, is the overview rather than the client at its
own size. Escaping puts the focused window up, which is the shape the measurement is meant to be
taken in.

## A WebGL client asking for MSAA takes the VM down

`webgl.html` is the browser workload, run as `firefox --kiosk file:///tmp/webgl.html`. Requesting
the WebGL context with `antialias:true` reliably kills the VM, and the sequence is worth knowing
because only its first step names the cause:

    MESA: error: ZINK: vkQueueSubmit failed (VK_ERROR_DEVICE_LOST)
    [LIMINA-ALLOC-POOL] class 0 grew to 65 allocators -- in-flight depth is outrunning completion
    ... to 7291 ...
    VM stopped -- worker terminated by signal 6

The device is lost first; after that nothing completes, so the allocator pool grows without bound
until the abort about a minute later. The runaway pool is the loud part and is only the symptom.

It is **not** ours, and the corpus must not be read as if it were: the same page kills the VM on
the C leg and on virglrs alike, and a desktop booted without the page runs indefinitely. It is
below virglrenderer, in the host's zink-on-KosmicKrisp path. Measured 2026-09-05 on alface: four
runs with MSAA, all dead inside ~50 s of the page loading; the identical page with
`antialias:false` runs on with `device-lost=0`.

So the recorded corpus asks for `antialias:false`. That is a deliberate retreat from the crashing
case, not a belief that MSAA works -- and the crash keeps a reproducer here rather than a fixture,
because a corpus of it would only ever record the seconds before the host gave up.

**A capture can type.** `type-into-overview.py` creates a keyboard on `/dev/uinput` and taps
keys through it, so mutter sees a real device and a headless capture can drive the overview's
search entry -- no window, no human. Copy it into the guest and run it as root
(`sudo python3 /tmp/type.py firefox settings`). It is how `vrend-overview.bin` was recorded, and
that corpus exists because typing allocates a resource nothing else in the tree did.

## Cross-path corpora: a client and a compositor on opposite stacks

Four workloads live here as scripts, because quoting a workload through `ssh` is where these
captures go wrong silently -- the client starts, a window appears, the capture completes, and it
recorded the wrong stack.

`client-shm.sh` runs a GTK4 terminal with `GSK_RENDERER=cairo`, `GDK_DEBUG=gl-disable` and
`LIBGL_ALWAYS_SOFTWARE=1`, so its surface arrives as a `wl_shm` buffer and the client issues no
GL at all. Captured with `capture.sh vrend --out shm`.

`client-vulkan.sh` runs `vkcube` on the **stock** guest. Its Vulkan goes out through venus while
the classic tracer records the other half -- the GL shell importing and compositing a buffer a
Vulkan client produced. Neither recorder sees both halves, which is the point: this is the
import side, in classic commands.

`client-basemark.sh` drives Basemark Web 3.0's graphics suite on the **stock** guest, for
profiling rather than for a corpus. It is the worked example for *Profiling a workload* below, and
it refuses rather than scores when the renderer is software or the reported configuration is not
the one asked for.

`client-gl-synoik.sh` is the mirror, and the harder one to get right. A GL client on the Vulkan
compositor needs the session's environment, which an SSH shell does not inherit -- without it the
stack falls back to llvmpipe and records nothing while looking healthy -- so the script reads it
out of the compositor's own `/proc/<pid>/environ` rather than assuming. Two traps found the hard
way: `WAYLAND_DISPLAY` is *absent* from that environ, because the compositor is what serves it,
and the socket here is `wayland-1`, not `wayland-0`; and `glxgears` is not a route at all, being
GLX on a guest with no X server. `glmark2-wayland` is, and it prints the renderer it actually
got -- check for `GL_RENDERER: virgl` before trusting a capture.

**This leg is the gate on classic-into-venus import, and nothing else gates it.** The venus
replayer skips classic contexts, so `synoik-glclient.vkrc` replays green whether or not the import
works -- it did, for as long as it did not. Booting is the only thing that scores it: run this
script, then read the frame while glmark2 is still running. `glmark2 pids:` in its output is the
first signal (a rejected buffer kills the client after one benchmark), and the frame is the
verdict -- the client's window carries the scene, and the scene changes between two samples.

**Channel order needs a coloured scene, and most of glmark2's are not.** `shading` is a grey horse
and `texture` a near-neutral wood crate: measured over the client's window they come out
144/144/144 and 55/55/55, so a swapped red and blue is invisible on them and a frame that shows
them says nothing about it. `build` is the one that carries it -- a saturated blue cat, 0/0/182 on
the C reference -- so a sample that lands on `build` and reads blue is the observation, and any
other scene is a frame that gates compositing and geometry only.

Unlike the Vulkan client corpora below, the C **can** replay this one: the GL client's contexts
are classic and are skipped, so nothing is recorded out of execution order.

## Two synoik corpora, and why one cannot do both jobs

The synoik guest yields two corpora, and they measure different things because a capture cannot
carry both properties at once:

| | `synoik.vkrc` | `synoik-lifecycle.vkrc` |
|---|---|---|
| dumped | mid-workload, compositor still running | after the session is stopped |
| teardown command kinds | 9 | 19 — `vkDestroyDevice`, `vkDestroyInstance`, `vkDestroyRingMESA`, the pipeline, render-pass, pool and layout destroys |
| `vkFreeMemory` | 4 | 41 |
| device allocations left to census | 22 | 0 |

The census hashes device memory that is still live, so it can only score what the workload has
not freed. A workload that exits cleanly frees everything — which is exactly what makes the
lifecycle corpus prove teardown, and exactly why it has no content hashes to offer. Keep both:
`synoik.score` is the content oracle, `synoik-lifecycle.score` is the teardown one, and a renderer
that leaks a `VkDeviceMemory` fails the second while passing the first.

### Capturing the lifecycle one

synoik runs as the user unit `org.gnome.Shell@user.service`, which sets `RefuseManualStop=yes`, as
do `gnome-session@gnome.target` and `gnome-session.target`. The one stoppable handle is
`graphical-session.target`; synoik's SIGTERM handler quits its loop outright rather than waiting
for clients, so stopping the target runs its drops and puts real `vkDestroy*` traffic on the wire
instead of a bare fd close at exit.

```sh
./capture.sh synoik --out synoik-lifecycle       # boot; synoik starts itself
# let it render for a couple of minutes, then, in the guest:
#   ssh -p <port from the boot log> claude@127.0.0.1 \
#     'XDG_RUNTIME_DIR=/run/user/1000 systemctl --user stop graphical-session.target'
./dump.sh synoik-lifecycle
```

`--out` is what keeps the two apart. Without it every synoik capture writes `synoik.vkrc` and
replaces the corpus a pinned score was recorded from.

## Profiling a workload: the sampling cycle

The loop is: run the workload, sample the host renderer, read where the **gpu worker** thread's
samples land, attack one area, re-run the same measurement unchanged. It found the classic fence
`glFinish` under the WebGL aquarium and again under Basemark's graphics suite. Nothing below is
about a particular benchmark; the traps are the ones any driven graphics workload hits here.

**Boot with `--window`, never `--display-capture`.** A capture boot re-encodes every presented
frame to PNG and reads the scanout back out of its IOSurface to do it. Under a workload that
presents constantly, that puts `fdeflate::compress`, `write_png` and
`Renderer::resource_read_iosurface` near the top of the profile -- the vehicle measuring itself.
Video decode presents rarely and does not notice, which is why `capture.sh` is right to use it and
a graphics profile is not.

**`--display-size` is a request, not a setting.** It is clamped to what the host window can be, so
a boot asking for 3840x2160 comes up at 2560x1440. Take the resolution from the boot log's
`iosurface scanout:` line or from the workload's own report, never from what was passed.

**Sample the worker, not the supervisor.** `pgrep -f '[l]imina-vmm'` matches the supervisor, whose
argv carries `--vmm-bin ...limina-vmm`. Use `pgrep -f '[l]imina-vmm --cpus'`, then
`sample <pid> 10 -f prof.txt` in short back-to-back windows. Read the `gpu worker` thread's own
subtree from the call graph; the whole-process leaf list is dominated by idle threads waiting and
says almost nothing.

**The gpu worker looking busy is not the workload running.** It is busy for the compositor, for
the browser's own UI, for a page that has merely loaded. This is the proxy that will cost a cycle:
a profile taken while a benchmark sat behind its Start button was full of plausible draw work.
**Look at the screen** (*Pixels are the oracle*, above), or assert something the workload
itself prints. Nothing in a renderer log can settle it.

### Driving a browser workload

`client-basemark.sh` is the worked example, and every rule in it was paid for:

**One browser process from start to finish.** Configuration a benchmark records is session state.
A probe/configure/run sequence of separate processes -- each killed, each relying on the state
reaching disk -- silently loses it, and the run comes up with defaults. Hand later URLs to the
instance already running (`firefox --profile P <url>` with no `--new-instance`) rather than
starting another.

**Options go where the site says, and are asserted afterwards.** Basemark's take on its root URL
and are ignored on `/run/`. Community mode (`?mode=community`) prints its own configuration to the
console -- that block, not the URL that was asked for, is what says what will run. Assert it:
a `suite=2` that quietly stayed "All suites" scores the wrong tests and costs a whole run to find.

**Probe the renderer the browser actually got, and refuse a software one.** An SSH shell does not
carry the session's GL environment, and Firefox will fall back to llvmpipe: the workload draws,
completes, reports a score, and the renderer under test never sees a command -- a profile that
reads as "no hotspot". Take the environment from the compositor's `/proc/<pid>/environ`, the way
`client-gl-synoik.sh` does, then check `WEBGL_debug_renderer_info` from a page.

Two things make that probe lie if you let them. Firefox returns `"Generic Renderer"` to content
unless `webgl.sanitize-unmasked-renderer` is `false`, so the check passes on llvmpipe. And Firefox
has blocked top-level `data:` navigation since 59, so a `data:` probe page never loads at all and
the failure is indistinguishable from the console pref not taking. Write the probe to a real file.

**Console to stdout is how a run is correlated.** `devtools.console.stdout.content` routes content
`console.log` to the process's stdout; timestamp each line and the host's sample windows can be
aimed by them.

**Reading the scores back, and aiming the windows: Marionette.** Start Firefox with
`--marionette` and drive it over TCP 2828; `marionette.py` is a ~100-line client with three verbs
(`js`, `click`, `wait`). It answers both problems this rig had:

  * **Aiming.** While a test runs, the page's location is
    `/run/tests/<n>/graphics_suite/<test_name>/`. Poll `document.location.pathname` before each
    sample window and the window is attributable to one named test rather than to the suite.
  * **Scores.** They are in the DOM of the page that ran them; `document.body.innerText` on the
    result page gives every per-test number. Do not go looking for them on the server: the UID a
    community-mode run prints is not a stored result (`/api/results/details/<uid>/` answers 404 on
    both hosts, and the configuration block says `Database: Unavailable`), and `/result/json/`
    returns the SPA shell for any unknown path. `firefox --headless --screenshot` renders but fires
    on the load event, so on a client-rendered page it captures "Loading, please wait...".

**A run gets its own negative control for free.** The suite ends on a result page that draws
nothing, so a window sampled there should show the gpu worker at ~0% renderer work -- measured
0.1%, against 43.6% on Geometry Stress in the same run. Take one. Without it, "the worker was busy"
is not evidence the aiming worked, and this rig has already produced a full, plausible profile of a
benchmark that was not running.

**Aiming means the WHOLE window sat inside one test.** The tests run about 11 s each, so a 10 s
window labelled only at its start straddles a boundary — and a straddled window attributes one
test's work to another: a "Geometry Stress" window carried 21.7% `transfer::write`, which is
Canvas's signature, and the same window put `resource_sync_iosurface` at 26% where a clean one puts
it at 55%. Read the pathname **before and after** each window and discard the window when the two
disagree; 5 s windows keep about half. Start on the first `graphics_suite` path rather than
whenever the host is ready, or the first three tests are never sampled at all.

**Symbols in a `sample` call graph are mangled, and the count comes after the tree prefix.** A
needle like `transfer::write` matches nothing (the frame reads `..5vrend8transfer5write`), and a
count regex anchored at the start of the line matches nothing either, because the line begins
`+ ! : | 1427 `. Both failures read as a clean zero for every needle at once, which is why the
aggregation asserts a **positive control** — a frame that must always be there, `Worker::service`.
A sweep where the control is also zero is a broken sweep, not a fixed renderer.

**What the aiming is worth**: the tests are four different targets, and a whole-suite profile
averages them into one misleading number. Percentages are of the `gpu worker` thread's own subtree,
from clean windows, with the classic fence's `finish_all` removed (it is 0.0% in every window):

| window | top of the worker's subtree |
|---|---|
| WebGL 1.0.2 | 36.1% `take_fence`, and **all** of it `tc_flush`; no `glFinish` at all |
| Draw-call Stress | 33-40% `take_fence` (31-38% `tc_flush`), 5-25% `resource_sync_iosurface` |
| Geometry Stress | 55% `resource_sync_iosurface`, 10% `take_fence` |
| Canvas | 42% `transfer::write`, 52% `Context::submit`, 1.3% `take_fence` |
| SVG | barely reaches the renderer at all (2-5%) |
| result page | 0.1% — the negative control |

So there is no single hotspot: WebGL 1.0.2 and Draw-call Stress are the fence's `glFenceSync`
draining mesa's threaded-context queue, Geometry Stress is the present-path `glFinish`, and Canvas
is texture upload with no fence cost worth naming.

**Two runs, and read them as two.** Per-test run-to-run spread on this rig reaches 11% (WebGL 2.0
measured 4409 then 3914), so no single-run single-digit difference is a result. What a pair does
settle is agreement: Canvas and Draw-call Stress came back within 0.2% of each other across two
runs, which makes a difference against a third run worth believing.

## Client corpora, and why the C cannot score them

Two more corpora come from the synoik guest with a Vulkan client in front of the compositor:

| | `synoik-vkcube.vkrc` | `synoik-ptyxis.vkrc` |
|---|---|---|
| client | `vkcube --wsi wayland` | Ptyxis (GTK4, `GSK_RENDERER=vulkan`), rendering a long directory listing |
| adds to the other corpora | the cross-context import: a compositor reaching a client's buffer; `vkCmdCopyImageToBuffer` | `vkGetPipelineCacheData`, both calls -- an out-blob, the guest saving its pipeline cache; `vkFreeDescriptorSets` |
| dumped | mid-workload, client still running | mid-workload, after the guest wrote `~/.cache/gtk-4.0/vulkan-pipeline-cache` |

Neither has a fixture under `../replay/fixtures`, because the C build cannot replay them: the
client's export and the compositor's import are recorded out of execution order, the C's import
of the not-yet-created resource tombstones the image, and KosmicKrisp asserts on the next
descriptor that samples it. So they gate virglrs against its own previous build -- the score at
HEAD against the score with the change -- rather than against the C. What a score sees of the
commands they add is that each decodes and is served to the end of the stream: a build that
refuses or poisons on one shows as `cmds` short of the total. A reply's *shape* it cannot see,
because a recording holds requests only -- an out-blob decoded as absent replays to a
byte-identical score, and that half is the unit tests' to pin (`../sabotage/sweep.py` lists
which).

To capture the Ptyxis one:

```sh
./capture.sh synoik --renderer c --out synoik-ptyxis
# in the guest, once /run/user/1000/wayland-1 exists:
#   XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-1 GSK_RENDERER=vulkan \
#     ptyxis -- sh -c 'ls -lR /usr/share/icons | head -n 4000; sleep 300'
# wait for a new file under ~/.cache/gtk-4.0/vulkan-pipeline-cache, then
./dump.sh synoik-ptyxis
```

GTK's gpu renderer creates no query pools, so no corpus carries the query-pool commands; they
are pinned by unit tests alone.

## Capturing

Both recorders are armed by capacity and write only when asked, through a FIFO — so arming one
costs the render path a predictable branch and nothing is written until `dump.sh` asks. A capture
defaults to **headless**, which still drives the whole renderer (the display sink reads the
scanout IOSurface back instead of presenting it) without taking over the screen; `--window` shows
the guest.

Venus captures stop at the cap and say so in the header. That is deliberate: a truncated corpus is
a valid **prefix** and still replays from its own beginning, where a FIFO ring would keep the most
recent commands and be replayable from nowhere. A small cap costs coverage, never validity.

`dump.sh` runs the decoder's structural check on what came back. Run it before pinning anything as
a fixture — a lost head record, an orphan context or a miscounted header all read as a working
capture right up until something tries to replay it.

## What the rig still borrows from limina

Two host prefixes, because reproducing them here would mean vendoring two more forks:

- an epoxy built **with** EGL (Homebrew's is CGL-only), from `third_party/epoxy-egl-prefix`;
- the zink-on-KosmicKrisp Mesa on its case-sensitive volume, `/Volumes/mesa-cs/zink-kk-prefix`.

It also seeds itself from limina's built `Limina.app` and its disk images. `LIMINA_ROOT` overrides
where it looks. Settling these is part of the P6 reconcile in `docs/design.md`.
