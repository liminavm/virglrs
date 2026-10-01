// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! What a Linux host's VA-API driver decodes, and the decoded picture handed back from it.
//!
//! The decode itself is `vrend::video`'s VA-API backend, which keeps a display, a context and a
//! surface per decode target on each codec's decode thread. This module is the part the rest of
//! the renderer sees whatever the backend: the probe the capset is built from, and a [`Picture`]
//! the delivery path copies planes out of.
//!
//! The bindings are `cros-libva`'s, which are safe: nothing here needs `unsafe`, and the module is
//! not on CLAUDE.md's list for that reason. Its types are `Rc`, not `Send`, so a VA object never
//! leaves the thread that made it: the probe opens and drops its own display, and a [`Picture`] is
//! EGL images over a surface, or a copy of its pixels, never the driver's surface itself.
//!
//! **Any driver, not only one.** The C refuses every VA driver that is not Mesa's, because it
//! leaves slice parameters uninitialised and only Mesa ignores them. Nothing here does that, so
//! the probe asks the driver what it decodes and believes the answer.

use std::sync::Arc;

use cros_libva as va;

use crate::vrend::egl::Image;

/// A codec a stream can be in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Codec {
    H264,
    Hevc,
    Vp9,
    Av1,
}

impl Codec {
    /// Every codec, in the order [`Support`] stores them.
    pub const ALL: [Codec; 4] = [Codec::H264, Codec::Hevc, Codec::Vp9, Codec::Av1];

    /// What this codec is called in a log line.
    pub fn name(self) -> &'static str {
        match self {
            Codec::H264 => "h264",
            Codec::Hevc => "hevc",
            Codec::Vp9 => "vp9",
            Codec::Av1 => "av1",
        }
    }

    /// The VA profile this build decodes the codec with, or `None` for a codec the backend has
    /// no leg for yet.
    ///
    /// The probe asks the driver about these and nothing else, so a codec this returns `None`
    /// for is never advertised, whatever the silicon -- a profile advertised without a leg
    /// behind it poisons the guest's context at its first frame.
    pub fn va_profile(self) -> Option<va::VAProfile::Type> {
        match self {
            Codec::Vp9 => Some(va::VAProfile::VAProfileVP9Profile0),
            // High decodes the Main and Constrained Baseline streams too, which is every H.264
            // profile advertised.
            Codec::H264 => Some(va::VAProfile::VAProfileH264High),
            // Main, which is the one HEVC profile advertised.
            Codec::Hevc => Some(va::VAProfile::VAProfileHEVCMain),
            Codec::Av1 => None,
        }
    }
}

/// What this host's VA driver decodes, of what this build serves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Support {
    hardware: [bool; Codec::ALL.len()],
}

impl Support {
    /// Open the first VA display that initialises and ask it which profiles it decodes.
    ///
    /// No display -- no render node, no driver, a driver that will not initialise -- is a host
    /// that decodes nothing, which is an answer and not an error: it advertises no video and the
    /// guest never sends any.
    pub fn probe() -> Support {
        let mut hardware = [false; Codec::ALL.len()];
        let Some(display) = va::Display::open() else {
            eprintln!("[virglrs] video: no VA-API display opens on this host");
            return Support { hardware };
        };
        let profiles = display.query_config_profiles().unwrap_or_default();
        for (slot, codec) in Codec::ALL.into_iter().enumerate() {
            let Some(profile) = codec.va_profile() else {
                continue;
            };
            hardware[slot] = profiles.contains(&profile)
                && display
                    .query_config_entrypoints(profile)
                    .is_ok_and(|e| e.contains(&va::VAEntrypoint::VAEntrypointVLD));
        }
        Support { hardware }
    }

    /// Whether this host decodes `codec` through a leg this build has.
    pub fn decodes(&self, codec: Codec) -> bool {
        let slot = Codec::ALL.iter().position(|c| *c == codec).expect("every codec is in ALL");
        self.hardware[slot]
    }
}

/// Warm the decoder ahead of the first frame. A VA context costs nothing worth hiding.
pub fn warm_up(_support: &Support) -> Option<std::thread::JoinHandle<()>> {
    None
}

/// The layout a decoded picture is handed back in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelFormat {
    /// Two planes, Y then CbCr: NV12, which is what a VA surface holds.
    BiPlanar420,
    /// Three planes, Y, Cb, Cr: split out of NV12 on the way back.
    Planar420,
}

/// A decoded picture, on its way from the decode thread to a target's planes.
///
/// The surface itself cannot leave the decode thread -- it is a reference picture for the frames
/// after this one, and the driver's types are not `Send` -- so what travels is one of two things
/// that can: EGL images over the surface's planes, which delivery copies from on the GPU, or a
/// copy of its pixels in memory of ours, which delivery uploads. The images are the usual case;
/// the copy is for a target whose planes the GPU copy cannot fill.
pub struct Picture {
    width: u32,
    height: u32,
    pixels: Pixels,
}

enum Pixels {
    Copied(Vec<OwnedPlane>),
    Imaged(Arc<Imaged>),
}

/// EGL images over a surface's two NV12 planes, made once per surface on the decode thread.
pub struct Imaged {
    pub luma: Image,
    pub chroma: Image,
}

/// One plane of an imaged picture: the image, and the extent of it the picture fills.
pub struct ImagedPlane<'a> {
    pub image: &'a Image,
    pub width: u32,
    pub height: u32,
}

struct OwnedPlane {
    width: u32,
    height: u32,
    pitch: usize,
    bytes: Vec<u8>,
}

impl Picture {
    /// Copy a decoded NV12 image out, `width` by `height`, in the layout asked for.
    ///
    /// `None` when the image does not hold what it says it does: a plane whose rows run past
    /// the data the driver mapped, or fewer than two planes. That is the driver's answer about
    /// its own image, and a frame refused here keeps whatever its target held.
    pub(crate) fn from_nv12(
        image: &va::Image<'_>,
        width: u32,
        height: u32,
        pixels: PixelFormat,
    ) -> Option<Picture> {
        let layout = image.image();
        if layout.num_planes < 2 {
            return None;
        }
        let data: &[u8] = image.as_ref();
        let rows = |plane: usize, count: u32| -> Option<(usize, &[u8])> {
            let pitch = layout.pitches[plane] as usize;
            let start = layout.offsets[plane] as usize;
            let len = pitch.checked_mul(count as usize)?;
            Some((pitch, data.get(start..start.checked_add(len)?)?))
        };
        let chroma_width = width.div_ceil(2);
        let chroma_height = height.div_ceil(2);
        let (luma_pitch, luma) = rows(0, height)?;
        let (chroma_pitch, chroma) = rows(1, chroma_height)?;
        // Each chroma row holds a CbCr pair per two luma columns.
        if chroma_pitch < 2 * chroma_width as usize {
            return None;
        }
        let mut planes =
            vec![OwnedPlane { width, height, pitch: luma_pitch, bytes: luma.to_vec() }];
        match pixels {
            PixelFormat::BiPlanar420 => planes.push(OwnedPlane {
                width: chroma_width,
                height: chroma_height,
                pitch: chroma_pitch,
                bytes: chroma.to_vec(),
            }),
            PixelFormat::Planar420 => {
                let pitch = chroma_width as usize;
                let mut cb = Vec::with_capacity(pitch * chroma_height as usize);
                let mut cr = Vec::with_capacity(pitch * chroma_height as usize);
                for row in chroma.chunks_exact(chroma_pitch) {
                    for pair in row[..2 * pitch].as_chunks::<2>().0 {
                        cb.push(pair[0]);
                        cr.push(pair[1]);
                    }
                }
                for bytes in [cb, cr] {
                    planes.push(OwnedPlane {
                        width: chroma_width,
                        height: chroma_height,
                        pitch,
                        bytes,
                    });
                }
            }
        }
        Some(Picture { width, height, pixels: Pixels::Copied(planes) })
    }

    /// A picture that is the surface `planes` images, `width` by `height` of it.
    pub(crate) fn imaged(planes: Arc<Imaged>, width: u32, height: u32) -> Picture {
        Picture { width, height, pixels: Pixels::Imaged(planes) }
    }

    /// Plane `index` as an image, for a picture delivered on the GPU; `None` for one copied
    /// through memory, or past the second plane.
    pub fn image(&self, index: usize) -> Option<ImagedPlane<'_>> {
        let Pixels::Imaged(planes) = &self.pixels else {
            return None;
        };
        let (image, width, height) = match index {
            0 => (&planes.luma, self.width, self.height),
            1 => (&planes.chroma, self.width.div_ceil(2), self.height.div_ceil(2)),
            _ => return None,
        };
        Some(ImagedPlane { image, width, height })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// The planes, for reading. `None` for an imaged picture, whose pixels are only on the GPU.
    pub fn lock(&self) -> Option<Locked<'_>> {
        match &self.pixels {
            Pixels::Copied(planes) => Some(Locked(planes)),
            Pixels::Imaged(_) => None,
        }
    }
}

/// A picture whose planes are readable, for as long as this value lives.
pub struct Locked<'a>(&'a [OwnedPlane]);

impl Locked<'_> {
    /// How many planes the picture has: two for NV12, three for I420.
    pub fn plane_count(&self) -> usize {
        self.0.len()
    }

    /// One plane, or `None` past the end.
    pub fn plane(&self, index: usize) -> Option<Plane<'_>> {
        self.0.get(index).map(|plane| Plane {
            width: plane.width,
            height: plane.height,
            pitch: plane.pitch,
            bytes: &plane.bytes,
        })
    }
}

/// One plane of a decoded picture: `pitch * height` bytes, `width` pixels of each row in use.
pub struct Plane<'a> {
    pub width: u32,
    pub height: u32,
    pub pitch: usize,
    pub bytes: &'a [u8],
}

impl Plane<'_> {
    /// Row `y`, as far as the pitch goes. `None` past the last row.
    pub fn row(&self, y: u32) -> Option<&[u8]> {
        self.bytes.get(y as usize * self.pitch..(y as usize + 1) * self.pitch)
    }
}
