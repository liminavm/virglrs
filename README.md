# virglrs

A virtio-gpu host renderer in Rust: the venus (Vulkan) and vrend (GL) paths a guest reaches
through virtio-gpu, on macOS, against KosmicKrisp and zink-on-KosmicKrisp.

It is a from-scratch reimplementation of [virglrenderer][], not a binding to it, written for
[limina][] and consumed as a Rust crate — rutabaga links it directly. The C-ABI dylib it can also
build is a compatibility shim with an end date.

[virglrenderer]: https://gitlab.freedesktop.org/virgl/virglrenderer
[limina]: https://github.com/liminavm/limina

## Building

```sh
scripts/vendor.sh     # third_party/virglrenderer at the pinned rev, + its meson subprojects
cargo build
```

`VIRGLRENDERER_SRC=/path/to/a/clone` sources the C from a local clone instead of the network —
the pinned rev still decides what is checked out, only where it is fetched from changes.

The vendored C tree is a build input, not optional scaffolding: the format tables are generated
from its `virgl_hw.h` and from Mesa's `u_format.yaml`, and the venus wire from the venus-protocol
its meson wrap pins. `build.rs` says so by name if it is missing. You also need `python3` with
mako, and — at runtime — Mesa's libEGL and a Vulkan loader.

## Testing

`cargo test` is the unit floor. Above it, `harness/` replays recorded workloads through both this
implementation and the pinned C one and compares scores; `harness/README.md` describes the three
layers and what each can and cannot catch. The C leg needs a built virglrenderer prefix, which
`harness/replay/build.sh` takes from `VIRGL_PREFIX`.

**The tests that open a display are not opted into, and on macOS they need the Mesa environment.**
Nothing is `#[ignore]`d any more, so a plain `cargo test` runs them and they *fail* rather than
skip where no display can be opened. On this platform that failure is `eglInitialize` returning
`EGL_NOT_INITIALIZED`, which means zink found no Vulkan driver rather than that the host has no
GPU. Export these and the whole suite passes:

```sh
MESA_PREFIX=/Volumes/mesa-cs/zink-kk-prefix
export VK_DRIVER_FILES="$MESA_PREFIX/share/vulkan/icd.d/kosmickrisp_mesa_icd.aarch64.json"
export DYLD_LIBRARY_PATH="$MESA_PREFIX/vulkan-rpath"
export DYLD_FALLBACK_LIBRARY_PATH="$MESA_PREFIX/lib:$EPOXY_PREFIX/lib:$(brew --prefix)/lib"
export MESA_LOADER_DRIVER_OVERRIDE=zink GALLIUM_DRIVER=zink EGL_PLATFORM=surfaceless
export LIBGL_DRIVERS_PATH="$MESA_PREFIX/lib"
cargo test
```

**`DYLD_*` must be exported inside the script that runs `cargo`**, not on the command line that
invokes it: `/bin/bash` is SIP-restricted and strips them at launch, so a caller's copy never
reaches the test binaries. `harness/ctests/ctests.sh` carries the same recipe for the same reason.

## Licence

MIT — see `LICENSES/MIT.txt` and `NOTICE`.
