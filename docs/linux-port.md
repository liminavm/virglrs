# virglrs on Linux

A plan to run the renderer on Linux/KVM: the crate building, testing and rendering against a
Linux guest through the C ABI and through the Rust API rutabaga consumes.

## The claim this rests on

virglrs is not macOS-shaped; it is **KosmicKrisp-shaped**, and the two are not the same thing.

The Apple-only code is 2,430 of 82,903 lines (`metal.rs` 1,374, `videotoolbox.rs` 1,056; measured
2026-09-08) and there are exactly two `cfg` gates in the tree (`guest_mem.rs:449/461`). Much of
what would be expensive to port is already portable: the venus wire decoder, every handler in
`venus/context.rs`, the TGSI→GLSL compiler, `vrend/` decode/transfer/blit, `fence.rs` and
`venus/sync.rs` (pure Rust — no eventfd, no dispatch queues, no Mach ports), `guest_mem.rs`,
`abi.rs`. The Vulkan path is the linked Khronos loader with tables generated from vk.xml, the
same loader on Linux. EGL is surfaceless-Mesa with everything resolved through
`eglGetProcAddress` — the Linux-native shape.

`design.md`'s "What we delete rather than port" already removed what would otherwise be a large
item: Rust accepts `RENDER_SERVER` at the ABI and calls vkr directly. An in-process Linux consumer
hits no missing architecture.

**What costs is a direction, not a module.** On KK the host mints the storage and the driver
imports it; Linux is the inverse — the driver allocates, the host exports an fd, the compositor
imports a dma-buf. The inversion is not confined to IOSurface. It is written into the allocate
path: `driver.rs:4044-4057` prepends `VkImportMemoryHostPointerInfoEXT` to **every**
`Backing::Owned` allocation, pages and surfaces alike, and `scanout_surface`
(`driver.rs:4204-4262`) exists to pitch-match host-minted storage against
`vkGetImageSubresourceLayout`. On Linux that becomes `VkExportMemoryAllocateInfo` at allocate plus
`vkGetMemoryFdKHR` after. **That path is replaced, not generalised.**

Of the three `Storage` arms (`driver.rs:4804`), only `Heap` survives unchanged. `Linear` is
defined as pages "handed the driver by host-pointer import" (`driver.rs:4815-4820`) and is a
KK-ism, not a portable arm.

## Three KosmicKrisp-isms that dissolve rather than port

None may be carried across.

1. **Host-pointer import as the universal backing** — the above. It also makes
   `VK_EXT_external_memory_host` a real driver dependency today (radv has it, others vary), and
   after the inversion we should not need it at all.
2. **force-LINEAR** (`external_images_are_linear`, `driver.rs:5395`) — load-bearing only because
   we import over host-minted pages an OPTIMAL image's texels are not in.
3. **The census hole** — the 41 all-zero pinned entries in `synoik`/`synoik-glclient` are the same
   fact. Phase 5's gate says why "non-zero" does not prove it closed.

## Where the ABI stands

- `virgl_renderer_resource_export_blob` (`ffi.rs:801`) is not a stub. It is an unconditional
  `EINVAL` whose doc comment calls the refusal the finished answer, because rutabaga calls it
  unconditionally and reads failure as "no handle". On Linux this is a **design reversal**, and
  `BlobStorage` (`renderer.rs:283-296`) has no variant that can hold an fd — a type change, not a
  fill-in.
- The `todo_phase!("P3: dmabuf export")` sites are `virgl_renderer_get_fd_for_texture` and
  `get_fd_for_texture2` (`ffi.rs:1200-1211`) — the classic GL-texture export.
- `export_query` (`ffi.rs:364`) answers the header's "not exportable" for everything;
  `export_fence`/`export_signalled_fence` (`ffi.rs:1283/1288`) are stubs.
- `driver.rs:96` `EMULATED_ON_THE_HOST` advertises `VK_KHR_external_memory_fd` and
  `VK_EXT_external_memory_dma_buf` to the guest and strips them from the device, because the
  driver has neither. The switch at `driver.rs:775` is already conditioned on
  `VK_EXT_external_memory_metal && !VK_KHR_external_memory_fd`, so on Linux it turns itself off.
  Pin both branches in a test rather than assuming it.

## Out of scope

**limina on Linux** — the Mach-port id transport is a VMM port. The boundary is not where it first
appears, though: `resource_iosurface_id`, `resource_read_iosurface` and `resource_sync_iosurface`
are `Renderer` methods (`renderer.rs:1501-1574`), i.e. the **Rust API**, not only `ffi.rs`. What
those three become on Linux is in scope, and phase 5 decides it.

**A VA-API backend.** Phase 1 says why cfg'ing the video directory out is nonetheless the wrong
move.

## The decision phase 1 forces

**`Held` becomes the abstraction, rather than `Storage` growing a cfg'd arm.**

This cannot be deferred to the export work. `Held` lives in `metal.rs:342`, `lib.rs:33` declares
`pub mod metal` unconditionally, and six platform-neutral files are built on it: `budget.rs:348`,
`renderer.rs:350`, `vrend/resource.rs:22`, `vrend/egl.rs:26`, `venus/driver.rs:60`,
`vrend/vrend.rs:530`. Putting `metal.rs` behind a cfg deletes the trait those files need, along
with `SurfaceId` (`ids.rs:62`) and `NoSurface` (`driver.rs:4931-4960`, whose `Display` text is
IOSurface-specific). The first compile forces the choice.

The alternative — cfg `Storage::Texture` and add a `DmaBuf` arm beside it — puts two containers
under one fact, and makes every future arm a cfg: `driver.rs:5029-5089` has exhaustive matches for
`PartialEq`, `Debug`, `span`, `pixels`, `surface` and `first_refusal`. Rehoming `Held` to a
platform-neutral module and having it yield host-generic storage keeps one owner, with `metal.rs`
and a `dmabuf.rs` as two implementations.

**It is not a rename.** `surface() -> &Surface` is the *entire* trait, and the semantics behind it
are IOSurface-shaped: `features.rs:175 adopts_iosurfaces`, the `external_images_are_linear` pins
at `driver.rs:6398-6407`, and the six exhaustive `Storage` matches. About 60 non-test lines across
seven files name the concrete type, but the line count is not the cost. Budget 2–3 weeks.

## The reference leg is our fork, built on Linux

**A reference leg is only as strong as the harness's ability to drive it.** Upstream
virglrenderer against mature Mesa would be the better oracle and cannot be used:

- `harness/replay/rs/src/abi.rs:99-128` dlopens sixteen `virgl_renderer_limina_*` symbols
  (`replay_begin/submit/ring_cmd/end`, `journal_*`, `memory_census/read/write`, `sync_*`) plus
  `resource_get_iosurface_id` and `resource_read_iosurface`. Upstream exports none of them; the
  venus replayer cannot `dlopen` an upstream build at all.
- `harness/replay/vrend-replay.c:70` includes `<IOSurface/IOSurface.h>` and scores every scanout
  through `IOSurfaceLookup`/`IOSurfaceLock` (lines 408-425); `build.sh:41` links
  `-framework IOSurface -framework CoreFoundation`. The classic replayer does not compile on
  Linux.
- The corpora are the fork recorder's format (`LIMINA_VKR_RECORD`/`LIMINA_VREND_TRACE`), and
  upstream has no recorder.

So the leg is the fork on Linux, and that is code nobody has run. Its `limina_*` entry points sit
outside the `__APPLE__` gates so it can serve the replay ABI, but its non-Apple venus memory path
(`vkr_device_memory.c`) and its libva video backend are unverified since the fork; every tag on it
is macOS work. And `read_iosurface`/`sync_iosurface` answer `-EINVAL` off Apple
(`virglrenderer.c:1392-1445`), so **every scanout line of every `.score` is unscoreable on the C
leg** until a scanout read both legs serve exists. That read is a dma-buf export in miniature,
which is why phase 2 and phase 5 partly reach into each other.

**Goldens do not travel** — every fixture was recorded from the C on KK. Corpora are data and do
travel. `harness/abi/layout.txt` is pinned from macOS aarch64 and is answered by re-running the
fixture, not by assumption.

## Phases

Each gate is an artifact, not a status field.

### Phase 0 — the host, and an inventory of the leg (2–3 days)

- `scripts/vendor.sh`; build the **fork** on Linux via `scripts/build-reference.sh`. The C tree is
  a build input before it is a reference — `build.rs` generates the format tables from its
  `virgl_hw.h`.
- Record distro, arch, Mesa version, Vulkan driver. Every later score is relative to it.
- **Inventory which of the fork's non-Apple branches have ever run.** `vkr_device_memory.c` has
  thirteen `#ifdef __APPLE__` branches; the meson build swaps in the libva `virgl_video.c`. This
  is the phase's real output.

**Gate:** the fork builds on Linux, `harness/abi/abi-fixture.sh` runs against it producing a Linux
`layout.txt`/`symbols.txt`, and the inventory exists as a file.

### Phase 1 — rehome `Held`, then compile (2–3 weeks)

Order within the phase matters: **rehome before any cfg lands.**

1. Move `Held`, `SurfaceId` and `NoSurface` into a platform-neutral module and make `Held` yield
   host-generic storage. The six neutral files stop naming `metal::Surface`.
2. Then `metal.rs` and the VideoToolbox *backend* go behind `cfg(target_os = "macos")`.
3. `build.rs:125` `link_egl` asserts `libEGL.dylib` — needs the `.so` name and Linux pkg-config.
   `link_vulkan_loader` already uses pkg-config; check the rpath spelling for GNU ld.
4. **Video is a trait, not a directory cfg.** `vrend/video/` is 7,469 lines of which
   `videotoolbox.rs` is the only Apple part — the bitstream builders and the H.264/H.265/AV1
   parameter-set synthesis (`video/mod.rs:7`: safe throughout) are pure, portable, and carry their
   own differential tests. Splitting the backend behind a trait keeps that coverage running on
   Linux for nearly free; cfg'ing the directory out throws it away. The `video-oracle` feature
   (`build.rs:255-283`) compiles Darwin-only C and stays macOS-gated.
5. **The scripts are macOS-shaped and belong to this phase:** `install.sh` (`install_name_tool`,
   `.dylib`), `harness/replay/build.sh`, `vkr-replay.sh` (a hardcoded
   `kosmickrisp_mesa_icd.aarch64.json`), `abi-fixture.sh` (a macOS meson dir).

**Gate:** `cargo test` on Linux, reported in three buckets — passed, skipped for want of a GPU,
not compiled. A green run whose needles went dead is the failure this floor exists to avoid, so
the count of *executed* tests is part of the gate, not the pass rate.

### Phase 2 — port the replayers (1–2 weeks)

The replayers are what make any scoring possible.

- `vrend-replay.c`: what it may call is a property of the leg it linked, not of the platform, so
  `build.sh` asks the library what it exports and compiles the rest out. `--rebuild` refuses on a
  leg without the journal rather than silently scoring nothing.
- **The venus replayer has one leg on Linux.** `vkr-replay` feeds the ring without a VM through
  `virgl_renderer_limina_replay_begin`/`_end`, which are ours; upstream exports neither, and there
  is no route to the ring that does not go through them. It refuses such a library by name rather
  than producing a score for a renderer it never drove. The two ways back are to make the fork
  build off Darwin, or to drive a real ring through the public ABI — which is a VMM's job, and
  would not score the transport anyway. Neither is worth its cost yet.
- **The replayer exports and mmaps the dma-buf itself.** The alternative — teaching the C leg a
  Linux scanout read — means carrying a patch on upstream, which is the thing choosing upstream as
  the Linux leg was meant to avoid. Doing it in the replayer also keeps the read on the side that
  is allowed to be host-specific: the leg stays a renderer both hosts can obtain unmodified, and
  the harness owns the platform knowledge, which is already true of every other host difference
  here. It costs the replayer an `EGL_MESA_image_dma_buf_export` call and an mmap of what comes
  back, against a fork that would have to be rebased forever.

**Gate:** the replayers run a corpus end to end on every leg they have and produce a score file.
Not a matching score — a score. Matching is phase 3.

### Phase 3 — score, and triage the divergences (1 week)

Re-record every classic fixture from the Linux C leg, then score the Rust leg against it.
Venus has no second leg here, so its scores are pinned against themselves and only move
deliberately.

Each divergence goes into one of three buckets: (a) a real virglrs gap the macOS host could not
reach; (b) a KK-ism encoded as a general truth; (c) a corpus that does not mean the same thing on
this driver. **Bucket (b) is the valuable output** and the reason this precedes the export work.
The trap runs both ways: a virglrs refusal that is correct on KK can read as a regression against
a more capable leg.

The snapshot-journal fixed-point gate runs here — and a fixed-point gate needs a live world, so
confirm it compares a populated one rather than nothing to nothing.

**Gate:** a committed Linux fixture set with every divergence resolved or categorised. Met for
the nine corpora this host can score; `harness/README.md` carries which those are, and why the
video corpora and `vrend-overview` are not among them. Bucket (a) came back **empty** — on
`vrend.bin`, a 71 MB recording of a real session, the two legs agree across all 341 lines.

### Phase 4 — enumerate the command gap (3–4 days)

The largest variable in the estimate, measured before the work it prices.

The capset advertises far more than the build serves: ~326 generated commands against ~170
handlers in `venus/context.rs`, and what a guest is told (`driver.rs:752-780`) is the driver's
extensions intersected with the serializable table. **On radv or anv that intersection is much
larger than on KK**, and every reachable unserved command poisons the context.

A throwaway boot of a Linux guest against the Rust renderer, purely to enumerate what it reaches
that we do not serve. One run, one list — never chased command by command.

**Gate:** the list. Met, and it is short.

**What a boot reaches.** With `unsupported`'s poisoning taken out so a boot could count more than
one command, vkmark's whole scene set — twice, with its non-default options — reaches nothing
unserved. zink reaches exactly one, `vkCmdSetColorWriteEnableEXT`, and reaches it from glmark2 on
wayland and from kmscube on KMS alike. Nothing gets further, and not because there is nothing
further to find: the guest's next host-visible `CREATE_BLOB` reads `not addressable by the host`,
which is phase 6's gap, and gnome-shell segfaults on the error. **A Linux boot census is bounded
by the export work, not by the command gap** — so the list below is cut from what the guest's
venus device advertises rather than from what one boot survived to send.

**What the host makes reachable.** The doc's premise was right and its size was not. The Linux
guest's venus device advertises 169 extensions on anv, and re-cutting `src/venus/unserved.txt`
against them moves **five commands** in four groups from `out-of-reach` to `wanted`:
`vkCmdSetColorWriteEnableEXT`, `vkCmdSetDepthBias2EXT`, `vkCmdSetFragmentShadingRateKHR`,
`vkGetPhysicalDeviceFragmentShadingRatesKHR`, `vkCmdSetVertexInputEXT`. Nothing moves the other
way: every `wanted` group's extension is advertised here too. The 34 commands still out of reach
are acceleration structures, ray tracing, mesh shaders, the descriptor heap, cooperative matrix,
`maintenance10` and shader objects — none of which anv exposes through venus either.

So `out-of-reach` could not go on meaning "this host cannot send it", and the ledger now reads it
as *no* supported host can. That is the phase's real output: the gap did not grow by a category,
it grew by five lines, and the estimate does not move.

### Phase 5 — the export direction (3–5 weeks, the bulk)

- Replace the allocate path's backing arm: `VkExportMemoryAllocateInfo` + `vkGetMemoryFdKHR`
  where `driver.rs:4044` prepends the host-pointer import.
- `EMULATED_ON_THE_HOST` stops emulating; pin both branches of `driver.rs:775`.
- `export_query` (`ffi.rs:364`) answers fourcc, fds, strides, offsets, modifier for real.
- `export_blob` (`ffi.rs:801`) reverses, so `BlobStorage` grows a variant holding an fd.
- `get_fd_for_texture`/`2` (`ffi.rs:1200-1211`) lose their `todo_phase!`.
- Fence export via `VK_KHR_external_fence_fd`, already in `HOST_EXTENSIONS` and never used.
- Decide what `resource_read_iosurface` and `resource_sync_iosurface` become on the Rust API.
- **Drop force-LINEAR** and let the driver tile.

**Gate.** The census turning non-zero is *not* a control. The census hashes an allocation's
*pages*; with force-LINEAR dropped, an OPTIMAL image on radv is tiled, `vkMapMemory` hands back
tiled bytes, and the hash is non-zero driver-specific noise. Non-zero is not texels — that is a
diagnostic read as an oracle. The control is **a known-content image read back through a copy**
(`vkCopyImageToMemory`, `driver.rs:1578`) matching what was written. Absent that, the 41 stay
pinned as "zero or unstable" and non-zero is not a pass.

### Phase 6 — the GL scanout path (1–2 weeks)

- `EGL_IOSURFACE_LIMINA` (0x3B9A, `egl.rs:170` — a forked-Mesa private extension) becomes standard
  `EGL_EXT_image_dma_buf_import`; classic export becomes `EGL_MESA_image_dma_buf_export`. This
  drops the dependency on our Mesa fork on this host.
- `image_from_iosurface`/`image_from_iosurface_plane` (`egl.rs:463/473`) are the call sites; the
  planar path maps onto multi-plane dma-buf import directly.
- Confirm which winsys flag the Linux VMM passes; `egl.rs:96-99` binds GLES only, and the classic
  caps probe is live (`caps.rs:304-322`) so it adapts.

**The lines this turns green are already named.** Six fixture lines are skipped on Linux for want
of a surface to read — five scanout IOSurfaces in `vrend.score`, one in `blit.score` — and the
replay reports them as skipped on every run, so the count is the gate's own progress bar. Two of
them are the other half of `vrend-nodraw`'s positive control, which on Linux currently moves only
its 19 offscreens. `blit.score`'s is the red/blue variant, and it is skipped twice over: nothing
reads its destination, and `needs_redblue_swizzle` cannot fire either, because it is predicated on
a BGRA resource that cannot be viewed and only a surface-backed one qualifies. Whether it comes
back depends on whether a dma-buf-imported BGRA EGLImage supports a view here — which the import
work will answer directly.

**Gate:** the replayer's own dma-buf read, hashed after a flush, matching the C leg, and the
skipped count reaching zero. The read is the harness's, so both legs are scored by the same code
and a difference is the renderer's.

### Phase 7 — a guest, and a pixel (1 week; the only real gate)

Everything above compiles and scores. None of it says the transport works.

Boot the C leg first so there is a reference frame, then the Rust one. Capture and look.
`harness/vm/frame.py` prints facts and renders no verdict; that discipline carries over unchanged.

**Gate:** a frame, compared against the C leg's frame.

## A new unsafe module is allowed, by decision

CLAUDE.md's unsafe list is exhaustive on purpose. A dma-buf backend runs through `driver.rs` and
`egl.rs`, both already on it, so nothing is forced — but a Linux scanout read for the replayer may
want one. VideoToolbox is the precedent: it was added deliberately and named on the list in the
same change. Adding one by decision is the process working; adding one by accident is the design
failure. Say which in the commit.

## The VMM, settled in phase 0

- **libkrun + KVM** — our fork already links virglrs through rutabaga on macOS, so only the
  platform changes. Cheapest path to a boot, and the consumer the crate was written for.
- **crosvm** — exercises the C ABI as a third party rather than as our own caller.

libkrun first, crosvm as a follow-on. The host's distro, arch and Mesa version decide how painful
each is.

## Missing infrastructure

**There is no CI** (no `.github`, no `.gitlab-ci.yml`). Two hosts and no CI means macOS regresses
silently while Linux lands. A build-and-unit-test job per host is the minimum, and it wants to
exist before phase 1's first cfg.

## Estimate

**10 weeks is the floor, and there is no ceiling until phase 4's enumeration exists.**

Phases 0–3 are ~5 weeks before a single score exists: the replayers need porting before anything
can be scored at all, and phase 1 carries the `Held` rework. Phase 5 is a rewrite of the allocate
path rather than a new `Storage` arm.

Phase 4 is cheap and early precisely because it is the term that dominates. **Do not commit to a
delivery date before it runs.**

## Risks

- **The fork on Linux is unrun code.** Phase 0's inventory is the mitigation; if it comes back bad
  the reference-leg strategy needs rethinking before phase 2 starts.
- **Phase 3 turns up a large bucket (a).** Phase 4 is what bounds it.
- **`vrend/shader/glsl/mod.rs:843`** pins `1 << n` to the C's behaviour on this host's arm64. Both
  arches mask the shift count identically so this is near-certainly a no-op on x86_64 — but it is
  behaviour pinned to a reference whose compiler may fold it differently. One test, not a risk.
- **No pixel oracle until phase 7.** Phases 1–6 produce a compiling, scoring renderer and say
  nothing about whether a desktop appears.
