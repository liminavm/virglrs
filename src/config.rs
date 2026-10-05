// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! How the renderer and its contexts were configured.

/// What the caller asked this renderer to be.
///
/// The C ABI says this in a bitmask of eleven `virgl_renderer_init` flags, of which exactly four
/// mean anything here; the rest select a winsys this build does not use. So the Rust API asks for
/// the four, by name, and the shim does the decoding -- a caller should not have to know which
/// bit is which, nor that one of them is spelled inside out.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Config {
    /// Serve venus.
    pub venus: bool,
    /// Serve vrend, the classic virgl renderer. Spelled positively; the ABI spells it `NO_VIRGL`.
    pub vrend: bool,
    /// The VMM cannot inject memory pages, so a blob must come from the guest's own heap. The
    /// guest reads this back out of the venus capset and allocates accordingly.
    pub guest_vram: bool,
    /// Serve hardware video decode. Off by default, and the host advertises no codec until it is
    /// asked for: bringing VideoToolbox up registers supplemental decoders process-wide, which is
    /// not a thing to do to a caller who never asked for video.
    pub video: bool,
    /// Allocate every shared classic buffer linear, and tell the guest each shared buffer's
    /// layout. What it buys is a guest that can scan a shared buffer out through virtio-gpu KMS,
    /// which takes `LINEAR` only, and a venus context that can import one -- which together are
    /// what a Vulkan compositor in the guest needs. Has an effect only with `venus`, and only on a
    /// host that exports rather than mints its storage.
    ///
    /// On by default, where the C has it off: Vulkan compositors are common enough that a guest
    /// which cannot run one is the worse default. The cost is every shared buffer's tiling --
    /// 11.4% of glmark2 under GNOME, measured in `docs/linux-port.md` -- and a caller that would
    /// rather have that back turns this off.
    pub linear_shared: bool,
    /// The GL vrend may run on. On a display of its own it is the API vrend's contexts are made
    /// in; under an embedder the embedder makes them, and this is whether a desktop context it
    /// hands over is taken or refused.
    pub host_gl: HostGl,
}

/// Which GL vrend runs on. See [`Config::host_gl`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HostGl {
    Gles,
    /// A core-profile desktop context, 3.3 or newer. The C ABI's default, as it is the C's, but
    /// not this API's: limina takes the default, and its hosts are gated on GLES.
    Desktop,
}

impl Default for Config {
    /// Nothing served, nothing optional asked for -- except linear shared buffers, which are on
    /// by default; see [`Config::linear_shared`].
    fn default() -> Config {
        Config {
            venus: false,
            vrend: false,
            guest_vram: false,
            video: false,
            linear_shared: true,
            host_gl: HostGl::Gles,
        }
    }
}

/// The renderer a context bound when it was created.
///
/// virtio-gpu calls this the context's capset id. The ABI carries it in the low byte of a flag
/// word whose remaining bits are unused, which is a detail the shim absorbs -- here it is the
/// choice itself, and a context has exactly one for its whole life.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CapsetId {
    /// Classic virgl, the original 3D protocol. Served by vrend, which arrives in P3.
    Virgl,
    /// virgl 2. Likewise vrend's.
    Virgl2,
    /// Vulkan, over the venus wire.
    Venus,
    /// A capset id this build has no name for.
    ///
    /// Not a rejection: a guest binding a renderer we do not serve still gets its context, and
    /// learns on its first submission. Refusing at creation time would fail a real desktop, which
    /// binds virgl2 contexts long before it binds a venus one.
    Unknown(UnnamedCapset),
}

/// The id behind [`CapsetId::Unknown`], which only [`CapsetId::from_raw`] can make.
///
/// The payload is a separate type because a variant's fields are as public as its enum, and a
/// public `Unknown(u8)` lets anyone write `Unknown(4)` -- a second, unequal spelling of
/// [`CapsetId::Venus`]. One number, one variant: with the field private to this module, the
/// alias cannot be written at all rather than merely being avoided by everyone who remembers.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct UnnamedCapset(u8);

impl core::fmt::Debug for UnnamedCapset {
    /// The wrapper is bookkeeping, not information: an id prints as the number it is.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.fmt(f)
    }
}

impl CapsetId {
    /// The capset a virtio-gpu id names.
    ///
    /// Total, and the only way to reach [`CapsetId::Unknown`]. The three ids are virtio-gpu's own
    /// numbers and live here rather than beside the C header's spelling of them, because they are
    /// the protocol's and would outlive the shim.
    pub fn from_raw(id: u8) -> CapsetId {
        match id {
            1 => CapsetId::Virgl,
            2 => CapsetId::Virgl2,
            4 => CapsetId::Venus,
            other => CapsetId::Unknown(UnnamedCapset(other)),
        }
    }
}
