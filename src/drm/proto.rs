// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! The framing every native context shares: a submission is a run of requests, each led by a
//! [`ReqHeader`], and a request with an answer names where in the response region to put it.
//! `drm_hw.h`'s `vdrm_ccmd_req`, `vdrm_ccmd_rsp` and `vdrm_shmem`.

use bytemuck::{Pod, Zeroable};

/// `struct vdrm_ccmd_req`: what leads every request.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default, PartialEq, Eq)]
pub struct ReqHeader {
    /// Which command; the backend's numbering.
    pub cmd: u32,
    /// The request's whole length, this header included.
    pub len: u32,
    /// What the host publishes once the request is done, so the guest can tell how far it got.
    pub seqno: u32,
    /// Where in the response region an answer goes, for a command that has one.
    pub rsp_off: u32,
}

pub const HEADER_LEN: usize = size_of::<ReqHeader>();

/// The response region's own header, `struct vdrm_shmem`: the last seqno the host finished, and
/// where the responses start. A backend's header extends it.
pub mod shmem {
    /// The word the host publishes each finished request's seqno into.
    pub const SEQNO: usize = 0;
    /// The word the host writes the responses' start into, once, when the region is attached.
    pub const RSP_MEM_OFFSET: usize = 4;
}

/// Every response starts with its length, `struct vdrm_ccmd_rsp`.
pub const RSP_LEN: usize = 0;

/// A request of type `T` from its bytes, zero-extended when the guest sent fewer.
///
/// An older guest sends the request as its own header knew it, without the fields a newer host
/// added at the end; those read as zero, which is what every such field is defined to mean. A
/// request longer than `T` is a newer guest's, and its tail is the handler's to read or ignore.
pub fn decode<T: Pod>(bytes: &[u8]) -> T {
    let mut out = T::zeroed();
    let dst = bytemuck::bytes_of_mut(&mut out);
    let n = dst.len().min(bytes.len());
    dst[..n].copy_from_slice(&bytes[..n]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable, Debug, PartialEq, Eq)]
    struct Req {
        hdr: ReqHeader,
        old: u64,
        new: u64,
    }

    #[test]
    fn a_short_request_reads_its_missing_fields_as_zero() {
        let mut bytes = vec![0u8; 24];
        bytes[16..24].copy_from_slice(&7u64.to_le_bytes());
        let r: Req = decode(&bytes);
        assert_eq!((r.old, r.new), (7, 0));
    }

    #[test]
    fn a_long_request_is_read_up_to_the_struct() {
        let mut bytes = vec![0xffu8; 64];
        bytes[..16].copy_from_slice(bytemuck::bytes_of(&ReqHeader {
            cmd: 1,
            len: 64,
            ..Default::default()
        }));
        let r: Req = decode(&bytes);
        assert_eq!(r.hdr.cmd, 1);
        assert_eq!(r.new, u64::MAX);
    }
}
