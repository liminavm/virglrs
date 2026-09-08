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

## Licence

MIT — see `LICENSES/MIT.txt` and `NOTICE`.
