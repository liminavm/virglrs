# Command probes

One program per `wanted` group of [`src/venus/unserved.txt`](../../src/venus/unserved.txt).

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
| `events.c` | `events` | 8 | 26/26 on KosmicKrisp, Apple M1 Max, Vulkan 1.4 | 26/26, no refusal (Fedora 44, 2026-09-08) |
| `buffer_view.c` | `buffer-view` | 2 | 3/3 on KosmicKrisp, Apple M1 Max, Vulkan 1.4 | not yet run |
| `timestamps.c` | `timestamps` | 2 | 4/4 on KosmicKrisp, Apple M1 Max, Vulkan 1.4 | not yet run |
| `maintenance6_binding.c` | `maintenance6-binding` | 3 | 6/6 on KosmicKrisp, Apple M1 Max, Vulkan 1.4 | 6/6, no refusal (Fedora 44, 2026-09-24) |
| `color_write_enable.c` | `color-write-enable` | 1 | 5/5 on anv, Intel Iris Plus (ICL GT2), Vulkan 1.4; KosmicKrisp lacks the extension | not yet run |
| `indexed_and_indirect_draw.c` | `indexed-and-indirect-draw` | 8 | 12/12 on KosmicKrisp, Apple M1 Max, and on anv, Intel Iris Plus (ICL GT2), both Vulkan 1.4 | not yet run |
| `image_copy_core.c` | `image-copy-core` | 3 | 6/6 on KosmicKrisp, Apple M1 Max, and on anv, Intel Iris Plus (ICL GT2), both Vulkan 1.4 | not yet run |
| `core_dynamic_state.c` | `core-dynamic-state` | 4 | 4/4 on anv, Intel Iris Plus (ICL GT2); 3/3 on KosmicKrisp, Apple M1 Max, which lacks stippled lines; both Vulkan 1.4. Depth bounds and device mask are scored by the log only | not yet run |
| `render_pass2.c` | `render-pass2` | 5 | 6/6 on KosmicKrisp, Apple M1 Max, and on anv, Intel Iris Plus (ICL GT2), both Vulkan 1.4 | not yet run |
| `copy_commands2.c` | `copy-commands2` | 6 | 4/4 on KosmicKrisp, Apple M1 Max, and on anv, Intel Iris Plus (ICL GT2), both Vulkan 1.4 | not yet run |
| `secondary_command_buffers.c` | `secondary-command-buffers` | 2 | 14/14 on KosmicKrisp, Apple M1 Max, and on anv, Intel Iris Plus (ICL GT2), both Vulkan 1.4. The trim is scored by the log only | not yet run |
| `depth_bias2.c` | `depth-bias-control` | 1 | 5/5 on anv, Intel Iris Plus (ICL GT2), Vulkan 1.4; KosmicKrisp lacks the extension | not yet run |
| `logic_op.c` | `extended-dynamic-state2` | 1 | 3/3 on anv, Intel Iris Plus (ICL GT2), Vulkan 1.4; KosmicKrisp lacks `extendedDynamicState2LogicOp` | not yet run |
| `vertex_input.c` | `vertex-input-dynamic-state` | 1 | 7/7 on anv, Intel Iris Plus (ICL GT2), Vulkan 1.4; KosmicKrisp lacks the extension | not yet run |
| `fragment_shading_rate.c` | `fragment-shading-rate` | 2 | 14/14 on anv, Intel Iris Plus (ICL GT2), Vulkan 1.4; KosmicKrisp lacks the extension | not yet run |
| `transform_feedback.c` | `transform-feedback` | 6 | 15/15 on anv, Intel Iris Plus (ICL GT2), Vulkan 1.4. 8/15 on KosmicKrisp, Apple M1 Max, whose emulated transform feedback writes no counter buffer, captures a draw whole or not at all, and answers no stream query; there a run through venus is held to the same 8 | not yet run |
| `sample_locations.c` | `sample-locations` | 1 | 5/5 on KosmicKrisp, Apple M1 Max, and on anv, Intel Iris Plus (ICL GT2), both Vulkan 1.4 | not yet run |
| `conditional_rendering.c` | `conditional-rendering` | 2 | 6/6 on KosmicKrisp, Apple M1 Max, and on anv, Intel Iris Plus (ICL GT2), both Vulkan 1.4 | not yet run |
| `local_read.c` | `dynamic-rendering-locations` | 2 | 4/4 on KosmicKrisp, Apple M1 Max, and on anv, Intel Iris Plus (ICL GT2), both Vulkan 1.4. Both commands are scored by the log only: the pipeline must carry the same map and mesa installs it at bind, so the pixels pass without them | not yet run |
