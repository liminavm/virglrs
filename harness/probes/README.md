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

**Be its own oracle.** Exit 0 when every check passed, 1 when one failed with the failing line
above it, 2 when the device could not be brought up at all — which is not a verdict on the group
and must not be counted as one.

**Run headless.** No surface, no swapchain, no window: a probe runs over ssh, and a probe that
needs a seated session can only run where a seated session already is.

**Never hang.** A wait with nothing to satisfy it scores as somebody else's timeout and names
nothing. Satisfy it from the host before the submit, so a miss is a failed check.

## How a group gets done

1. **Positive control first.** Build the probe and run it against the host driver directly
   (`VK_ICD_FILENAMES=…kosmickrisp…`). It must pass. This proves the program is right and that
   the host really serves the group — which is what `wanted` claims, and is otherwise only an
   assertion. A probe that has never passed anywhere cannot produce a meaningful RED.
2. **RED against this build**, in a guest, through venus. Read the worker log: `[virglrs]
   refused:` is the only place the command is named.
3. **Serve the commands**, and delete their lines from the ledger — the test
   `every_command_the_protocol_defines_is_served_or_on_the_ledger` holds the file to it.
4. **GREEN**, same probe, same guest.

## Building

Against the guest's own loader, in the guest:

    cc -O1 -o events events.c -lvulkan && ./events

On the host, for the positive control, the loader and headers are Homebrew's:

    cc -O1 -I/opt/homebrew/include -o events events.c -L/opt/homebrew/lib -lvulkan
    VK_ICD_FILENAMES=/Volumes/mesa-cs/build-kk/src/kosmickrisp/vulkan/kosmickrisp_mesa_devenv_icd.aarch64.json ./events

## The probes

| Probe | Group | Commands | Positive control |
|---|---|---|---|
| `events.c` | `events` | 8 | 26/26 on KosmicKrisp, Apple M1 Max, Vulkan 1.4 |
