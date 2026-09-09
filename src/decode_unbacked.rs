// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! A host with no hardware decoder this build can reach.
//!
//! `vrend::Vrend::video` already draws the distinction this module lives in: "`None` and a
//! support that decodes nothing are different answers and stay different -- the first is a
//! configuration, the second is this machine's silicon." A host whose decoder virglrs has no
//! backend for is the second answer, so nothing above needs a new concept and nothing needs a
//! branch: [`Support::probe`] reports no codec, `video::advertised` returns an empty profile
//! list, the capset carries `num_video_caps` zero, and a guest that reads it never sends a video
//! command at all.
//!
//! The types a decode *session* is made of are uninhabited here, which is what makes that
//! airtight rather than merely likely. A session cannot be created, so a picture cannot exist,
//! so every path that would deliver one is unreachable by construction rather than by a guard
//! somebody has to keep correct. The methods are `match *self {}`: total, because there is no
//! value to answer for.
//!
//! Linux decodes through VA-API, and that is what replaces this module -- not a stub inside it.
//! Until then this is the honest description of the host: it decodes nothing.
//!
//! No unsafe here, and no foreign calls: there is no library to call.

/// A codec a stream can be in.
///
/// No discriminants: the macOS backend numbers these by the `kCMVideoCodecType_*` FourCC because
/// CoreMedia asks for one, and that number means nothing off that platform. What is shared is
/// the vocabulary -- which codecs the renderer has a name for -- and that is all this carries.
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
}

/// What this host decodes in hardware, which is nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Support {
    _private: (),
}

impl Support {
    /// Probe the host. There is no decoder to find.
    pub fn probe() -> Support {
        Support { _private: () }
    }

    /// Whether this host decodes `codec` in hardware. Never.
    pub fn decodes(&self, _codec: Codec) -> bool {
        false
    }
}

/// Why a session could not be built. Opaque: there is no platform status to carry.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Status;

impl Status {
    /// Whether the codec asked for is absent. It always is.
    pub fn is_no_such_decoder(self) -> bool {
        true
    }
}

impl core::fmt::Debug for Status {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "no decode backend")
    }
}

/// The layout a decoded picture would be produced in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelFormat {
    /// Two planes, Y then CbCr.
    BiPlanar420,
    /// Three planes, Y, Cb, Cr.
    Planar420,
}

/// What a session would be configured from.
///
/// Opaque rather than the backend's parsed record: the bytes exist to be handed to a decoder,
/// and there is none here to hand them to. Keeping the shape without the contents is what lets
/// the layers above build a [`SessionKey`] exactly as they do on any host.
#[derive(Clone, PartialEq, Eq)]
pub struct Configuration;

impl Configuration {
    pub fn vp9(_profile: u8, _bit_depth: u8, _subsampling: u8) -> Configuration {
        Configuration
    }

    pub fn h264(_sps: Vec<u8>, _pps: Vec<u8>) -> Configuration {
        Configuration
    }

    pub fn hevc(_vps: Vec<u8>, _sps: Vec<u8>, _pps: Vec<u8>) -> Configuration {
        Configuration
    }

    pub fn av1c(_box_: Vec<u8>) -> Configuration {
        Configuration
    }

    pub fn is_parameter_sets(&self) -> bool {
        false
    }
}

/// Everything a session is identified by.
#[derive(Clone, PartialEq, Eq)]
pub struct SessionKey {
    pub width: u32,
    pub height: u32,
    pub pixels: PixelFormat,
    pub config: Configuration,
}

/// Why a decode produced no picture.
#[derive(Debug)]
pub enum DecodeError {
    Rejected(Status),
    NoPicture,
    NoSample(Status),
}

/// A decoded picture. Uninhabited: nothing here decodes.
pub enum Picture {}

impl Picture {
    pub fn width(&self) -> u32 {
        match *self {}
    }

    pub fn height(&self) -> u32 {
        match *self {}
    }

    pub fn lock(&self) -> Option<Locked<'_>> {
        match *self {}
    }
}

/// A picture whose planes are mapped. Uninhabited: there is no picture to map.
pub enum Locked<'a> {
    #[doc(hidden)]
    Never(std::marker::PhantomData<&'a ()>, Void),
}

/// The uninhabited half of [`Locked`], which is what makes the whole of it uninhabited while it
/// still carries the lifetime its callers name.
#[doc(hidden)]
pub enum Void {}

impl Locked<'_> {
    pub fn plane_count(&self) -> usize {
        match *self {
            Locked::Never(_, ref void) => match *void {},
        }
    }

    pub fn plane(&self, _index: usize) -> Option<Plane<'_>> {
        match *self {
            Locked::Never(_, ref void) => match *void {},
        }
    }
}

/// One mapped plane of a decoded picture.
pub struct Plane<'a> {
    pub width: u32,
    pub height: u32,
    pub pitch: usize,
    pub bytes: &'a [u8],
}

impl Plane<'_> {
    pub fn row(&self, y: u32) -> Option<&[u8]> {
        self.bytes.get(y as usize * self.pitch..(y as usize + 1) * self.pitch)
    }
}

/// A live decode session. Uninhabited: one cannot be built here.
pub enum Session {}

impl Session {
    /// Build a session. There is no decoder, so this is the one answer.
    pub fn create(_key: SessionKey) -> Result<Session, Status> {
        Err(Status)
    }

    pub fn pixels(&self) -> PixelFormat {
        match *self {}
    }

    pub fn serves(&self, _key: &SessionKey) -> bool {
        match *self {}
    }

    pub fn adopt(&mut self, _key: &SessionKey) -> bool {
        match *self {}
    }

    pub fn decode(&mut self, _unit: &[u8]) -> Result<Picture, DecodeError> {
        match *self {}
    }
}
