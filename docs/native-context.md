# DRM native contexts

A native context hands the guest's GPU driver the host kernel's own uAPI: guest Mesa runs
radeonsi and RADV, and the host forwards buffer creation, VA mapping and command submission to the
host's `amdgpu` driver. There is no translation layer. What the host owns is the trust boundary
between a guest's requests and a kernel `ioctl`, the objects those requests create, and the fences
their submissions produce.

This is Linux-only, and a virglrenderer replacement feature: limina does not use it.

## Shape

```
src/drm/
  mod.rs        the native-context root: probe, capset, the context table
  context.rs    the shared layer: ccmd framing, the response region, objects, blobs
  timeline.rs   per-ring fences
  sys.rs        UNSAFE: the kernel interface (ioctls, mmap, owned handles)
  uapi.rs       the kernel's structs, #[repr(C)], layout-pinned
  amdgpu/
    mod.rs      the backend: probe, capset, context create, ring layout
    proto.rs    amdgpu_virtio_proto.h, the guest wire
    ccmd.rs     one handler per command
```

Gated on a `drm` feature and `cfg(target_os = "linux")`, with one feature per backend:
`drm-amdgpu` first, `drm-i915` (goiaba) next. Nothing here compiles on macOS.

**No C is linked.** The C reaches the kernel through libdrm and libdrm_amdgpu; this calls the
`ioctl`s directly. `uapi.rs` carries the structs from the pinned tree's `src/drm/drm-uapi/`
headers, and the subset of libdrm_amdgpu's own structs that the capset puts on the wire
(`amdgpu_gpu_info`, `amdgpu_buffer_size_alignments`, `amdgpu_heap_info`). Every struct is pinned
by size and field offsets against the C headers (the `harness/abi/` fixture pattern), because a
wrong offset there is a wrong `ioctl`, not a compile error. The libdrm_amdgpu structs are not in
the vendored `drm-uapi/`: their layout is pinned against `libdrm/amdgpu.h` on a Linux host, and the
capset as a whole against the C built **with** amdgpu, since the capset union's size depends on
which backends the C was built with.

## The unsafe module

`sys.rs` is the only unsafe code in the tree that this adds, and it joins CLAUDE.md's list. It
holds the `ioctl` calls, the `mmap` of a buffer through its fake offset, and the handle types that
make those calls safe to use from everywhere else:

- `Device`: an owned render-node fd. One per guest context, because amdgpu gives each open file
  its own GPU VM, and the C's `deduplicate = false` exists for the same reason.
- `Gem`: a GEM handle on one `Device`, which closes itself on drop. It holds its `Device`, so a
  handle cannot outlive the file it is a number in.
- `Syncobj` likewise; and a fence fd is an `OwnedFd`, never an `int`.
- Each `ioctl` is a typed function taking those types: `gem_create(&Device, &GemCreate) -> Gem`,
  `cs_submit(&Device, &Submit) -> ...`. A backend never sees a raw fd, a raw handle or a pointer.

Backends and the shared layer are safe Rust.

## The shared layer

**Requests are copied once.** The C reads each request header from the guest's buffer several
times, and checks `rsp_off` on one read and uses it on another. Here a submit's bytes are copied
into an owned buffer, and each request is decoded from that buffer into a typed request before any
handler sees it. A request shorter than its struct is zero-extended, as the C does, because an
older guest sends shorter ones.

**The response region** is the shared-memory blob the guest creates first (`blob_id` 0): a sealed
memfd, mapped by the host. It is one value, the mapping and its length together; offsets into it
are validated where they are decoded. Responses are staged and copied in at the end of a command,
never written to the guest's memory by a handler, and the host writes no more than the length the
guest pre-set, as the C does. `seqno` is published with a release store after the response.

The amdgpu header in that region has two host-written fields outside the response path, and they
are the only other writes: the heap usage (three `amdgpu_heap_info` blocks, rewritten after
allocations) and `async_error`, the guest's only signal that a command with no response, such as
`GEM_NEW`, failed. The C increments `async_error` in place, a read-modify-write of bytes the guest
can also write. Here the count is the host's, and it is stored into the region; it is never read
back from it.

**A buffer is a `Gem`, shared by whoever uses it.** A buffer created by `GEM_NEW` is pending,
held by the context under its `blob_id`, until `get_blob` binds it to a resource. From then on:

- the resource's storage, a `BlobStorage::Native` arm, owns it, for mapping and export.
- the context names it in **one** map, resource to a weak reference, which is where `BO_VA_OP`
  and `CS_SUBMIT` resolve a resource id. A buffer the context created enters that map at bind; a
  buffer another context created enters it at attach, imported from its dma-buf onto this
  context's file, as the C does. Either way it is one lookup and one kind of entry. A detach
  removes the name, and a resource that is gone stops resolving on its own: there is no purge to
  remember at the unref.
- an import is owned by the resource too, beside the original: one copy per device file. Attach
  reuses the copy already on the context's file before importing, because the kernel answers a
  buffer a file already holds with the handle it already has, and a GEM handle closes once. A
  guest that closes a buffer and reopens it in the same context reaches exactly that.

The bind refuses a blob larger than its buffer. The C does not check, and maps the buffer's size
for a blob of any size.

The storage holds the `Gem`, not a dma-buf, because most of radeonsi's buffers cannot be exported
at all: the kernel refuses a dma-buf for a buffer created `VM_ALWAYS_VALID`
(`amdgpu_dma_buf.c`, `-EPERM`), and the capset tells the guest it may use that flag. So a buffer
is mapped through `GEM_MMAP` on its context's file, and exported to a dma-buf only when something
asks for one, as the C's lazy `export_opaque_handle` does.

One consequence, chosen rather than inherited: a resource that outlives its context keeps that
context's render-node file, and so its GPU VM, open until the resource dies. The C closes the file
at context destroy and leaves the resource naming a handle on a closed file.

**Buffers are counted.** A buffer is charged to the renderer's budget from `GEM_NEW` until its
last owner drops it, as classic's driver memory is: counted, never refused, because the kernel has
allocated it by the time there is a size to count. A guest that creates buffers and never binds
them still pins memory until its context dies, as in the C; a per-context account that can refuse
is open below.

## Fences

One timeline per hardware ring, numbered as the guest numbers them: ring 0 is the CPU, and rings
1.. are the device's rings in (IP type, instance) order from `AMDGPU_INFO_HW_IP_INFO`. The host
never ties a `ring_idx` to the ring a submission really targets; the guest derives the same
numbering, so getting the order wrong is a guest that waits on the wrong ring.

Each `CS_SUBMIT` produces a sync_file. A fence on a ring waits for the newest submission on that
ring, and retires through the existing `fence::Handle`, so the VMM's callback runs on the
renderer's fence thread as it does for vrend and venus.

One waiter thread per context watches the head of every ring's queue at once, since a sync_file
is pollable: FIFO per ring, no thread per ring, and a stop flag
checked between polls, so teardown never waits on the GPU.

Two departures from the C, both ordering fixes:

- **A fence never overtakes an earlier one on its ring.** The C retires a fence immediately when no
  submission has happened since the last fence, even when that earlier fence is still pending. Here
  such a fence queues behind it.
- **Teardown does not block on the GPU.** The C's ring threads wait forever on a fence, and
  destroying the context joins them.

## The amdgpu backend

Ten commands: `QUERY_INFO`, `GEM_NEW`, `BO_VA_OP`, `CS_SUBMIT`, `SET_METADATA`, `BO_QUERY_INFO`,
`CREATE_CTX`, `RESERVE_VMID`, `SET_PSTATE`, `CS_QUERY_FENCE_STATUS`. Each takes a decoded request
and the context's state, and returns a response or an error.

The capset is the hard part to match, not the commands. libdrm_amdgpu derives `amdgpu_gpu_info`
and the heap info from several `AMDGPU_INFO` queries plus its own arithmetic, and the marketing name
from `amdgpu.ids`, a data file read at run time, which this reads too. A guest Mesa is configured by
those bytes, so they are scored byte for byte against the C leg.

The C has two holes this closes at the boundary:

- `CS_SUBMIT` and `SET_PSTATE` with an unknown context id hand the C a null context, which libdrm
  dereferences. Here that is a rejected request.
- `CS_QUERY_FENCE_STATUS` takes a guest-chosen timeout and blocks the submitting thread for it.
  Refusing a non-zero timeout would break RADV, whose wait-idle sends one second. The timeout is
  bounded instead; see below.

## The C ABI

- `VIRGL_RENDERER_DRM` becomes `Config::native_context`.
- Capset 6 (`DRM`) becomes `CapsetId::Drm` and a `Capset::Drm` arm.
- `Config::native_context` asks for native contexts; the device is the first render node whose
  driver has a backend, as the C's `drmOpenWithType` finds it. Before cross-context import lands
  it becomes the node the renderer already resolved for its shared buffers, so native, classic and
  venus resources live on one device. A VMM's own fd (`get_drm_fd`, which QEMU 11.1 does not
  give) would arrive as an optional owned fd in `Config`, not a callback.
- Asked for on a host with no such GPU, the renderer serves everything else and says why not this.
  The C fails its whole init instead, taking classic and venus with it.
- `resource_map_fixed` stays refused: QEMU 11.1 falls back to `resource_map` on `-EOPNOTSUPP`.
- In-fences (`submit_cmd2`, the C's `supports_fence_sharing`) are deferred: QEMU 11.1 never calls
  `submit_cmd2`.

## The renderer

`Bound` gains a third arm, `Native`. The routes it needs: context create and destroy, submit, the
`InContext` blob source (`get_blob`, including the response region), attach and detach,
`context_create_fence`, `resource_export` (lazy dma-buf) and `resource_host_mapping`. The other
match sites refuse it as they refuse an unserved context today.

`docs/design.md` drops `src/drm/` from "What we delete rather than port" in the same change.

## Testing

- **Unit:** the decoder, the response region and the object stages are pure and run anywhere.
- **Fuzz:** a target for the ccmd decoder, which is the trust boundary.
- **Layout:** `uapi.rs` against the C headers, on every host.
- **Boot, on banana** (`harness/README.md` gains the rig): `cube.sh` with the guest's renderer line
  naming `radeonsi`, not `virgl (...)`, on both legs; the capset bytes from the guest compared
  between legs; then RADV (vkcube), then glmark2 and a GNOME session. Every gate is armed against
  Fedora's stock renderer, which offers no native context, so a boot that quietly fell back to
  classic virgl cannot pass.

## Order

1. Probe, capset, context create, the response region and `QUERY_INFO`: the guest's radeonsi
   initialises, and the capset matches the C's.
2. `GEM_NEW`, `get_blob`, `BO_VA_OP`, mapping and export: buffers exist and are shared.
3. `CREATE_CTX`, `CS_SUBMIT`, the timelines, `SET_METADATA`, `BO_QUERY_INFO`, and importing a
   native buffer into another native context: kmscube draws, textured.
4. `RESERVE_VMID`, `SET_PSTATE`, `CS_QUERY_FENCE_STATUS`, and importing classic and venus
   resources: RADV, glmark2, a desktop.

## Open

- Refusing buffers over budget. It needs a per-context account, which today only venus mints.

- The `CS_QUERY_FENCE_STATUS` bound. Recommended: one second, the most the guest's driver asks
  for, so no guest that Mesa ships is changed by it, and a hostile one stalls its device for at
  most that.
- Whether a failed command's response is delivered. The C drops it, along with the seqno update,
  and aborts the rest of the batch; matching that is the default, since the guest was written
  against it.
