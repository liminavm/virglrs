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
## `hvmap` — does any of it survive the hypervisor?

Both spikes above run beside no VM at all, and guest RAM in production is a range
`hv_vm_map` has stage-2 mapped. Two orderings, because they are different
questions:

* **forward**, the guest-RAM path: `hv_vm_map` the region, *then* make the entry
  and share it.
* **reverse**, the blob path — what `get_map_ptr` becomes in a split: the
  renderer mints, the VMM maps the entry, and `hv_vm_map`s that address.

```
clang -O1 -Wall -Wextra -framework Hypervisor -o hvmap hvmap.c
codesign --entitlements <libkrun hvf-entitlements.plist> -s - --force hvmap
./hvmap forward ; ./hvmap reverse
```

**Measured 2026-09-07, same host:** both coherent both ways. An entry can be made
over hv-mapped guest RAM and the second task sees the hypervisor's pages; and
`hv_vm_map` accepts a mapping backed by another task's memory entry, so
renderer-minted storage reaches the guest without the renderer ever being the
process that talks to the hypervisor.

## What this settles

Guest RAM can be mapped into a renderer process, renderer-minted storage can be
published to the guest and outlives a renderer crash, and all of it is mach ports
alone — no descriptor anywhere.

What none of it touches:

* **The released-RAM path.** libkrun returns ballooned pages with
  `MADV_FREE_REUSABLE` + `hv_vm_unmap` and heals on fault with `MADV_FREE_REUSE`
  + `hv_vm_map` (`hvf/src/released_ram.rs`). It never re-`mmap`s over the range,
  so the VM object — and therefore the entry — stays valid. But a renderer
  mapping sits *outside* that fault-and-heal loop: a renderer touching a released
  page gets a zero page and no heal. It should never touch one, because the guest
  only ever names live pages in a ring or an iov, but that is an invariant to
  state rather than one the kernel enforces. The settle sweep's
  `mprotect(PROT_NONE)` is task-side and does not reach a renderer's mapping,
  which also means a second mapping keeps those pages resident.
* **The bootstrap lane, which is the spike's and not the design's.**
  `bootstrap_check_in` on an arbitrary name works here only because limina signs
  with the hardened runtime and no `app-sandbox`. App Store delivery means App
  Sandbox, where that name is refused, so the renderer is reached as an in-bundle
  XPC service instead — which is the better shape anyway: an XPC service carries
  mach send rights natively (`xpc_dictionary_set_mach_send`), gets its own sandbox
  profile rather than inheriting the VMM's, and brings lifecycle and restart with
  it. Nothing above depends on the lane: the entries travel over XPC unchanged,
  and only the rendezvous differs from what these programs do.
