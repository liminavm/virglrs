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

- `virgl_renderer_resource_export_blob` exports for real: the blob's share of `Storage::Texture`
  *is* the descriptor on this host, so `BlobStorage` needed no fd-holding variant after all. It
  answers `EINVAL` where there is no descriptor to hand back, which is every host that mints
  rather than exports.
- `execute`'s `EXPORT_QUERY` answers fourcc, modifier, per-plane stride and offset, and one
  duplicated fd per plane, from that same descriptor.
- `virgl_renderer_get_fd_for_texture`/`2` stay refused. Not for want of the machinery: `tex_id` is
  a GL name, so only a VMM sharing this renderer's context could call them, and none does —
  QEMU's `virtio-gpu-gl` module imports 28 of these entry points and neither of these two.
- The fence trio (`export_fence`, `export_signalled_fence`, `attach_fence`) stays refused, and its
  reason is about the ABI's shape rather than the host's: a `client_fence_id` names a retirement,
  not a fence object. Venus exports fence fds through `VK_KHR_external_fence_fd` where a fence
  object exists.
- `driver.rs`'s `EMULATED_ON_THE_HOST` advertised `VK_KHR_external_memory_fd` and
  `VK_EXT_external_memory_dma_buf` to the guest and stripped them from the device, because
  KosmicKrisp has neither. The switch is conditioned on
  `VK_EXT_external_memory_metal && !VK_KHR_external_memory_fd`, so it turns itself off here, and
  both branches are pinned by a test that asserts each host's answer rather than the one it is
  compiled on.

## Out of scope

**limina on Linux** — the Mach-port id transport is a VMM port. The boundary is not where it first
appears, though: `resource_iosurface_id`, `resource_read_iosurface` and `resource_sync_iosurface`
are `Renderer` methods, i.e. the **Rust API**, not only `ffi.rs`. All three stayed and all three
answer on both hosts; the only signature that moved is `resource_read_iosurface`, which took
`&mut self` because the exporting host's read is a GPU round trip. Their *names* are booked below.

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
wayland and from kmscube on KMS alike. A census still stops early for a reason of its own: the
first unserved command kills the context, so one boot discovers one command and says nothing
about the rest. So the list below is cut from what the guest's venus device advertises rather
than from what one boot survived to send.

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

The inversion, made real. An allocation the guest **declares for export** gets a descriptor: the
driver lays the memory out and `vkGetMemoryFdKHR` hands back an fd for it; nothing is minted, and
the allocate path prepends no `VkImportMemoryHostPointerInfoEXT`. `Planned` grew an `Exporting`
arm for the moment that takes it — after `vkAllocateMemory`, over the driver's own storage — and
`src/dmabuf.rs` is what holds one.

**Exportable and presentable are two questions, and only the minting host could answer them
together.** There, the host has to *make* storage of a concrete pixel layout, so a window buffer
is recognised by shape: exported, and dedicated to one image of a presentable format. Here the
driver made the storage, and the shape test is the wrong question — Mesa's Wayland WSI renders
into an OPTIMAL image and blits into a linear `VkBuffer`, and it is the **buffer** it declares for
export and shares with the compositor. A buffer has no format, no tiling and no
`vkGetImageSubresourceLayout`: there is nothing to ask the driver.

So the two are asked separately. Exportable is the declaration alone, and yields
`Storage::Exported` — a `Descriptor`, which is an fd and the kernel's own size for it (`lseek`),
and which stays one for life. Presentable is a layout, which only a dedicated image can answer
for; when one does the descriptor comes back as a `Surface` and the driver's answer is an oracle.

**A descriptor's layout arrives from the guest, at `PIPE_RESOURCE_SET_TYPE`**, which already
carried the modifier and the per-plane strides and offsets. `Descriptor::describe` mints a
*separate* `Surface` over a second reference to the buffer, so two contexts may read one
descriptor under different layouts and neither can disturb the other — which matters because a
guest may send two descriptions that disagree. That is also the trust boundary: every plane is
bounded against the kernel's size in checked arithmetic, and any modifier but `LINEAR` and
`INVALID` is refused by name, because a compressed one carries an auxiliary plane whose extent
follows a different rule. A layout the bounds accept and the driver still will not import is
refused too, not asserted — the numbers are the guest's.

**Adopting exported storage is not minting it.** `Winsys::adopts_shared_storage` asks whether a
context can take storage another one exported (an IOSurface here, a dma-buf there — both hosts
do); `Features::adopts_iosurfaces` asks whether this host mints storage of its own, which only
the minting host does. They were one predicate, and while they were, a venus client's window on
Linux was adopted by nothing and composited as a blank texture.

- `EMULATED_ON_THE_HOST` stops emulating, and both branches of the extension switch are pinned by
  a test that asserts each host's answer rather than the one it is compiled on.
- `export_query` answers fourcc, fds, strides, offsets and modifier from the descriptor; a partial
  failure trims the fd count rather than reporting fds it did not produce.
- `export_blob` reverses. `BlobStorage` needed no fd-holding variant after all: a blob already
  holds a share of `Storage::Texture`, and on this host that share *is* the descriptor.
- **force-LINEAR is gone on the exporting host.** It is not parity — on KosmicKrisp an OPTIMAL
  image's texels are not in imported pages, so the minting host must still force it. Here the
  driver keeps its own storage and tiling costs nothing, so the tiling check is not carried across.
- Fence export is `VK_KHR_external_fence_fd`, served through `vkGetFenceFdKHR`. The C ABI's
  sync-file trio stays refused, and its reason is corrected rather than removed: a
  `client_fence_id` names a retirement, not a fence object.
- `resource_read_iosurface` takes `&mut self`, because the exporting host's read is a GPU round
  trip and needs a context switch. limina builds against it.
- `get_fd_for_texture`/`2` stay refused, and the doc comment says why: `tex_id` is a GL name, so
  only a VMM sharing our context could call them, and none does — QEMU's `virtio-gpu-gl` module
  imports 28 of these entry points and neither of these two.

**CPU access to a dma-buf is a favour, not a property.** Whether the fd maps is the driver's
choice, and a tiled buffer's bytes are not pixels even if it does. Measured on this host: scanouts
export with modifier `0x0100000000000001` (`I915_FORMAT_MOD_X_TILED`). So `Surface::readable()` is
the question every
CPU path asks first, a share with no host address is refused at the venus import rather than passed
to a driver as a null pointer, and the harness read goes through the GPU instead.

**What is not written yet: a venus client importing a classic resource.** The share it resolves
to is now a descriptor rather than nothing, and importing one means a dma-buf handle type at
`vkAllocateMemory` instead of a host pointer. Until that is written the import is refused by name.
Measured against a live guest: vkmark runs its whole scene set through venus and scores, hitting
this refusal four times without failing.

**Gate.** The census turning non-zero is *not* a control: it hashes an allocation's pages, and with
force-LINEAR dropped a tiled image's pages are non-zero driver-specific noise. The control is a
known-content image read back through a copy (`vkCopyImageToMemory`) matching what was written, and
until that exists the census entries stay pinned as "zero or unstable".

### Phase 6 — the GL scanout path (1–2 weeks)

`EGL_IOSURFACE_LIMINA` (0x3B9A, a forked-Mesa private extension) is `EGL_EXT_image_dma_buf_import`
here, and the classic export is `EGL_MESA_image_dma_buf_export` — so this host needs no Mesa fork.
`image_from_iosurface`/`image_from_iosurface_plane` are the call sites, and the planar path maps
onto multi-plane dma-buf import directly. Modifiers are sent only when the query produced one.

The export takes every plane the modifier reports, not just the colour one: a compressed modifier
describes a single-plane format with an auxiliary plane beside it, and a descriptor missing it
promises a compression plane that is not in the layout. All of them must be one allocation, which
is what a `Surface` can hold, and that is checked against the descriptors' inodes rather than
assumed. The allocation's size is the kernel's `lseek(SEEK_END)` answer, because `pitch * height`
measures the colour plane and stops short of anything past it.

No such buffer reaches the renderer on this host -- an ICL scanout is plain `X_TILED` with one
plane -- so the export path is pinned by unit tests over stubbed export calls, using the layout
GBM reports for a real `Y_TILED_CCS` buffer here, and not by a boot. The import side is measured
directly: the attribute list `image_of_iosurface` builds for such a buffer, one descriptor named
by both planes, is accepted by iris.

The size of a single-plane export changed with it. `alloc_size` is now the whole allocation rather
than `pitch * height`, which is what the driver actually reserved, so the budget charges the real
figure and a `LINEAR` mapping covers the whole buffer.

**Gate:** the replayer's own scanout read, matching the fixture, and the skipped count reaching
zero. **Met.** Forty scanout lines across six corpora are scored where all of them were compiled
out, the skipped count is zero on nine corpora, and `vrend-nodraw`'s positive control now moves 23
lines rather than 20 — the three scanout surfaces are the other half of it, as predicted.

**The read is a round trip, not a mapping.** The exported buffer is tiled, so the replayer's
`read_iosurface` imports the descriptor back as an EGL image, takes it as a texture's storage and
reads it through a framebuffer. Reading the resource's texture would be cheaper, would give the
same pixels, and would pass just as happily if the descriptor named the wrong memory — which is
the whole point of the line. Armed both ways: moving the exported offset by a page changes the
scanout hash and not the texture readback; moving the pitch by 64 makes the import fail outright.

**What differs from the macOS fixtures is the GL driver.** Every line that moved is a large
desktop surface, of the same kind the readback overlays already carry. Two of them were checked
against this leg's own texture readback of the same resource — `blit` res=21 and `vrend` res=372 —
and were byte-identical to it, which is what says the export path is not what moved them. The
blank 1x1 and 48x48 surfaces match the macOS fixture exactly, on every corpus. The moved lines
live in the per-driver overlays, which `make-overlay.py` now derives for `iosurface` lines as it
always did for readbacks.

### Phase 7 — a guest, and a pixel (1 week; the only real gate)

Everything above compiles and scores. None of it says the transport works.

Boot the C leg first so there is a reference frame, then the Rust one. Capture and look.
`harness/vm/frame.py` prints facts and renders no verdict; that discipline carries over unchanged.

**Gate:** a frame, compared against the C leg's frame. **Met.** Both legs seat a GNOME session
under `cage` on this host and draw the overview -- wallpaper, workspace thumbnails, dock, search
-- and the two captures differ in **16 pixels of 921600**, a 5x6 box that is the clock's minute
digit. Two independent boots, so anything nondeterministic in window placement or damage would
have shown and did not.

**Both legs must be booted under the same host GL API, and the desktop is where that is easiest to
get wrong.** GTK hands QEMU a desktop GL 4.6 core context unless `GDK_GL=gles` is in the
environment, and `gl=es` does not reach it because that path goes through GDK. The C leg accepts
such a context and this renderer refuses it by name, so a comparison run without the variable is
the C on desktop GL against a guest that quietly fell back to llvmpipe -- with a seated session,
a running shell and a captured frame to say everything is fine. `renderer:` naming `virgl` is the
control, and `grep -c virglrs` on the log is what says which leg drew it.

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

## Booked for after the port

Neither is on the path to a working Linux renderer, and doing them mid-port is churn against a
tree still moving.

**The `iosurface` names.** They no longer say what is Apple's and what is everyone's, and the two
halves want opposite treatment.

The C symbols are the limina fork's own additions -- upstream virglrenderer exports none of them
(47 `virgl_renderer_*` symbols, zero matching `iosurface`), and QEMU's `virtio-gpu-gl` module
imports none of them either. Their entire caller set is this tree's two replayers, and
`virgl_renderer_republish_iosurface` has no caller anywhere while returning `EINVAL` on both
hosts. So the C spelling is a private debug ABI: rename it only if the fork is being touched for
another reason, and delete `republish` outright. The `iosurface res=…` score-line key is separate
again, and costs about a thousand recorded fixture lines across both hosts to move.

The **Rust API** is the half that misleads, because it is limina's real presentation path rather
than a debug surface. `resource_read_iosurface` and `resource_sync_iosurface` are generic in
substance and want neutral names. `resource_iosurface_id` is not a rename: on Apple the value is
an `IOSurfaceGetID`, a system-global handle another process can resolve, and here it is a
process-local counter that names nothing outside this renderer. Every generic call site uses it as
a boolean -- both replayers test it for non-zero -- and the transportable handle already exists as
`resource_export`. So the doc has to say which of the two it is handing back, whatever it is
called. The `image_*_iosurface` winsys entry points are internal and rename freely.

**Multi-plane export on the venus side.** A classic export takes every plane a compressed
modifier reports; a venus one still describes every image as having a single memory plane.
`export_dmabuf` cannot do better yet: the real count is
`VkDrmFormatModifierPropertiesListEXT::drmFormatModifierPlaneCount`, which needs a physical device
that function is not given and a struct the wire bindings do not carry. The modifier there is the
guest's image's, so a guest that asks for `Y_TILED_CCS` gets a descriptor claiming one plane under
a two-plane modifier. Measured against iris here, that pairing is refused at `eglCreateImageKHR`
with `EGL_BAD_MATCH`, so the failure is an importer's refusal and not a picture read from the
wrong bytes.

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
