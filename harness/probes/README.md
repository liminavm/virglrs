# Command probes

One program per `wanted` group of [`src/venus/unserved.txt`](../../src/venus/unserved.txt), and
one per command served while no guest can send it, whose positive control is then the only run it
has and whose handler's unit test scores the renderer.

A workload cannot enumerate that ledger. The first unserved command poisons its context, so a
boot discovers exactly one of them and then reports the consequence in the guest driver's
vocabulary — `vkQueueSubmit2` came out as `VK_ERROR_OUT_OF_HOST_MEMORY` from a different call,
then an OOM, then a deadlock. These programs reach the commands deliberately instead, one at a
time, and say which one failed.

## What a probe owes

**Name every command in its group.** The group is the unit of work; a probe that exercises six
of eight leaves two that nothing will ever reach on purpose.

**Check consequences, not survival.** "It did not crash" is not a result: an unserved command
poisons the context, and the poison surfaces somewhere else entirely. Each command needs an
observable effect — a state read back, a buffer's contents, a query's answer. Where the API
offers no way to observe one (a destroy), say so in a comment and score what can be scored.

**Report cleanly.** Exit 0 when every check passed, 1 when one failed with the failing line above
it, 2 when the device could not be brought up at all — which is not a verdict on the group and
must not be counted as one.

**Run headless.** No surface, no swapchain, no window: a probe runs over ssh, and a probe that
needs a seated session can only run where a seated session already is.

**Never hang.** A wait with nothing to satisfy it scores as somebody else's timeout and names
nothing. Satisfy it from the host before the submit, so a miss is a failed check.

## How a group gets done

1. **Positive control first.** Build the probe and run it against the host driver directly
   (`VK_ICD_FILENAMES=…kosmickrisp…`). It must pass. This proves the program is right and that
   the host really serves the group — which is what `wanted` claims, and is otherwise only an
   assertion. A probe that has never passed anywhere cannot produce a meaningful RED.
2. **RED against this build**, in a guest, through venus — scored on the worker log, not on the
   probe. See below.
3. **Serve the commands**, and delete their lines from the ledger — the test
   `every_command_the_protocol_defines_is_served_or_on_the_ledger` holds the file to it.
4. **GREEN**: the probe passes *and* the log shows no refusal from the group.

## The probe is not the oracle

**A probe can print `ok` for a command that was refused.** A zeroed reply is shaped exactly like a
successful one, so the guest driver reads a refusal as success; and venus answers some commands
from a guest-side slot without asking the host at all. Both were measured on the events group:
`vkCreateEvent` was refused, and the probe reported it `ok` along with the next four checks —
`vkSetEvent` and `vkGetEventStatus` among them, which never left the guest. The abort came three
commands later and named nothing.

So a green probe is half a result. The other half is `[virglrs] refused:` in the worker log, which
is the only place a command is named — and it needs its own positive control, because a log with
no `[virglrs]` line at all reads exactly like a log with no refusals. Grep for the prefix first,
then for the refusal.

## Building

Against the guest's own loader, in the guest:

    cc -O1 -o events events.c -lvulkan && ./events

On the host, for the positive control, the loader and headers are Homebrew's:

    cc -O1 -I/opt/homebrew/include -o events events.c -L/opt/homebrew/lib -lvulkan
    VK_ICD_FILENAMES=/Volumes/mesa-cs/build-kk/src/kosmickrisp/vulkan/kosmickrisp_mesa_devenv_icd.aarch64.json ./events

## The probes

| Probe | Group | Commands | Positive control | Through venus |
|---|---|---|---|---|
| `events.c` | `events` | 8 | 26/26 on KosmicKrisp, Apple M1 Max, Vulkan 1.4 | 26/26 on anv via QEMU; 26/26 on KosmicKrisp via limina. No refusal (Fedora 44, 2026-09-25) |
| `buffer_view.c` | `buffer-view` | 2 | 3/3 on KosmicKrisp, Apple M1 Max, Vulkan 1.4 | 3/3 on anv via QEMU; 3/3 on KosmicKrisp via limina. No refusal (Fedora 44, 2026-09-25) |
| `timestamps.c` | `timestamps` | 2 | 4/4 on KosmicKrisp, Apple M1 Max, Vulkan 1.4 | 4/4 on anv via QEMU; 4/4 on KosmicKrisp via limina. No refusal (Fedora 44, 2026-09-25) |
| `maintenance6_binding.c` | `maintenance6-binding` | 3 | 6/6 on KosmicKrisp, Apple M1 Max, Vulkan 1.4 | 6/6 on anv via QEMU; 6/6 on KosmicKrisp via limina. No refusal (Fedora 44, 2026-09-25) |
| `color_write_enable.c` | `color-write-enable` | 1 | 5/5 on anv, Intel Iris Plus (ICL GT2), Vulkan 1.4; KosmicKrisp lacks the extension | 5/5 on anv via QEMU; skipped on KosmicKrisp via limina, as natively. No refusal (Fedora 44, 2026-09-25) |
| `indexed_and_indirect_draw.c` | `indexed-and-indirect-draw` | 8 | 12/12 on KosmicKrisp, Apple M1 Max, and on anv, Intel Iris Plus (ICL GT2), both Vulkan 1.4 | 12/12 on anv via QEMU; 12/12 on KosmicKrisp via limina. No refusal (Fedora 44, 2026-09-25) |
| `image_copy_core.c` | `image-copy-core` | 3 | 6/6 on KosmicKrisp, Apple M1 Max, and on anv, Intel Iris Plus (ICL GT2), both Vulkan 1.4 | 6/6 on anv via QEMU; 6/6 on KosmicKrisp via limina. No refusal (Fedora 44, 2026-09-25) |
| `core_dynamic_state.c` | `core-dynamic-state` | 4 | 4/4 on anv, Intel Iris Plus (ICL GT2); 3/3 on KosmicKrisp, Apple M1 Max, which lacks stippled lines; both Vulkan 1.4. Depth bounds and device mask are scored by the log only | 4/4 on anv via QEMU; 3/3 on KosmicKrisp via limina. No refusal (Fedora 44, 2026-09-25) |
| `render_pass2.c` | `render-pass2` | 5 | 6/6 on KosmicKrisp, Apple M1 Max, and on anv, Intel Iris Plus (ICL GT2), both Vulkan 1.4 | 6/6 on anv via QEMU; 6/6 on KosmicKrisp via limina. No refusal (Fedora 44, 2026-09-25) |
| `copy_commands2.c` | `copy-commands2` | 6 | 4/4 on KosmicKrisp, Apple M1 Max, and on anv, Intel Iris Plus (ICL GT2), both Vulkan 1.4 | 4/4 on anv via QEMU; 4/4 on KosmicKrisp via limina. No refusal (Fedora 44, 2026-09-25) |
| `secondary_command_buffers.c` | `secondary-command-buffers` | 2 | 14/14 on KosmicKrisp, Apple M1 Max, and on anv, Intel Iris Plus (ICL GT2), both Vulkan 1.4. The trim is scored by the log only | 14/14 on anv via QEMU; 14/14 on KosmicKrisp via limina. No refusal (Fedora 44, 2026-09-25) |
| `depth_bias2.c` | `depth-bias-control` | 1 | 5/5 on anv, Intel Iris Plus (ICL GT2), Vulkan 1.4; KosmicKrisp lacks the extension | 5/5 on anv via QEMU; skipped on KosmicKrisp via limina, as natively. No refusal (Fedora 44, 2026-09-25) |
| `logic_op.c` | `extended-dynamic-state2` | 1 | 3/3 on anv, Intel Iris Plus (ICL GT2), Vulkan 1.4; KosmicKrisp lacks `extendedDynamicState2LogicOp` | 3/3 on anv via QEMU; skipped on KosmicKrisp via limina, as natively. No refusal (Fedora 44, 2026-09-25) |
| `vertex_input.c` | `vertex-input-dynamic-state` | 1 | 7/7 on anv, Intel Iris Plus (ICL GT2), Vulkan 1.4; KosmicKrisp lacks the extension | 7/7 on anv via QEMU; skipped on KosmicKrisp via limina, as natively. No refusal (Fedora 44, 2026-09-25) |
| `fragment_shading_rate.c` | `fragment-shading-rate` | 2 | 14/14 on anv, Intel Iris Plus (ICL GT2), Vulkan 1.4; KosmicKrisp lacks the extension | 14/14 on anv via QEMU; skipped on KosmicKrisp via limina, as natively. No refusal (Fedora 44, 2026-09-25) |
| `transform_feedback.c` | `transform-feedback` | 6 | 15/15 on anv, Intel Iris Plus (ICL GT2), Vulkan 1.4. 8/15 on KosmicKrisp, Apple M1 Max, whose emulated transform feedback writes no counter buffer, captures a draw whole or not at all, and answers no stream query | 15/15 on anv via QEMU; 7/15 on KosmicKrisp via limina: the guest reads stream queries back through `vkCmdCopyQueryPoolResults`, which KosmicKrisp leaves unwritten, so "results available" fails too; with `VN_PERF=no_query_feedback` it is the native 8/15. No refusal (Fedora 44, 2026-09-25) |
| `sample_locations.c` | `sample-locations` | 1 | 5/5 on KosmicKrisp, Apple M1 Max, and on anv, Intel Iris Plus (ICL GT2), both Vulkan 1.4 | 5/5 on anv via QEMU; 5/5 on KosmicKrisp via limina. No refusal (Fedora 44, 2026-09-25) |
| `conditional_rendering.c` | `conditional-rendering` | 2 | 6/6 on KosmicKrisp, Apple M1 Max, and on anv, Intel Iris Plus (ICL GT2), both Vulkan 1.4 | 6/6 on anv via QEMU; 6/6 on KosmicKrisp via limina. No refusal (Fedora 44, 2026-09-25) |
| `local_read.c` | `dynamic-rendering-locations` | 2 | 4/4 on KosmicKrisp, Apple M1 Max, and on anv, Intel Iris Plus (ICL GT2), both Vulkan 1.4. Both commands are scored by the log only: the pipeline must carry the same map and mesa installs it at bind, so the pixels pass without them | 4/4 on anv via QEMU; 4/4 on KosmicKrisp via limina. No refusal (Fedora 44, 2026-09-25) |
| `extended_dynamic_state3.c` | `extended-dynamic-state3` | 21 | 24/24 on anv, Intel Iris Plus (ICL GT2), which serves 19 of the 21; 6/6 on KosmicKrisp, Apple M1 Max, which serves 5; both Vulkan 1.4. Rasterization stream, sample locations enable and line rasterization mode are scored by the log only; extra overestimation size and advanced blend by no host | 24/24 on anv via QEMU; 6/6 on KosmicKrisp via limina. No refusal (Fedora 44, 2026-09-25) |
| `maintenance10.c` | `maintenance10` | 1 | 3/3 on KosmicKrisp, Apple M1 Max, and on anv, Intel Iris Plus (ICL GT2), both Vulkan 1.4 | Not reachable: the guest's venus driver offers no guest `VK_KHR_maintenance10`. The handler is scored by `end_rendering2_and_depth_clamp_range_hand_the_driver_the_guests_structs` |
| `depth_clamp_range.c` | `shader-object` (`vkCmdSetDepthClampRangeEXT` only) | 1 | 4/4 on anv, Intel Iris Plus (ICL GT2), Vulkan 1.4; KosmicKrisp lacks `VK_EXT_depth_clamp_control` | 4/4 on anv via QEMU, through `VK_EXT_depth_clamp_control`; the guest offers no `VK_EXT_shader_object`. No refusal (Fedora 44, 2026-09-26) |
| (none) | `descriptor-update-template` | 2 | The guest venus driver answers both commands guest-side and never sends them; there is nothing to probe | — |
