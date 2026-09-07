<!--
SPDX-License-Identifier: MIT
Copyright © 2026 Gustavo Noronha Silva
-->

# Mach memory entries across a process boundary

Two spikes behind the question of running the renderer in its own process, so a
crash in it is a GPU reset rather than a dead VM. Both are about memory, and
neither uses a file descriptor anywhere: a mach memory entry is the handle and a
mach port is the lane.

    clang -O1 -Wall -Wextra -o machmem machmem.c
    clang -O1 -Wall -Wextra -o survive survive.c

## `machmem` — can the renderer reach guest RAM?

The VMM's guest RAM is `MAP_PRIVATE | MAP_ANON` (vm-memory's `MmapRegion`, which
is what `GuestMemoryMmap::from_ranges` gives libkrun), so the entry could
plausibly have copy-on-write semantics rather than aliasing the pages. It does
not. Both sides write *after* the map and each is required to see the other,
because a one-way check passes on a copy taken at map time.

    ./machmem [private|shared] [vmshare|plain]
    MACHMEM_SPARSE=1 MACHMEM_MB=16384 ./machmem private vmshare

**Measured 2026-09-07, macOS 26.6.2 arm64:** coherent both ways in all four
combinations of `MAP_PRIVATE`/`MAP_SHARED` against
`MAP_MEM_VM_SHARE`/plain `VM_PROT_READ|VM_PROT_WRITE`, at 2 MiB, 4 GiB and
16 GiB. One entry and one `mach_vm_map` cover the whole region; the kernel
returns the full size asked for.

`MAP_MEM_VM_SHARE` is not required for a single anonymous region, and the spike
proves nothing about a range that spans regions with differing protections —
which real guest RAM becomes once the hypervisor has mapped it. Use it anyway;
it costs nothing and it is the flag whose job is exactly that case.

## `survive` — do the renderer's own pages outlive it?

A host-visible blob's storage is minted by the renderer. If those pages die with
the process, every guest mapping published through `hv_vm_map` dangles after a
crash and there is nothing to recover to. The child mints, fills, hands the
parent an entry, and `SIGKILL`s itself with no cleanup of any kind.

    ./survive

**Measured 2026-09-07, same host:** the parent's mapping stays readable and
writable after the child dies on signal 9. A held entry is enough — storage does
not have to be minted VMM-side, only *held* there.

## What this settles

Guest RAM can be mapped into a renderer process, and renderer-minted storage can
outlive a renderer crash, both with mach ports alone. What neither spike touches:
a range the hypervisor has already mapped (`hv_vm_map` needs an entitlement the
spike does not carry), and the bootstrap lane under the app's sandbox — here
`bootstrap_check_in` succeeded unentitled, which is the lane a spawned child
looks the service up on.
