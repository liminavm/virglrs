// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! What a pipeline cache's data has to be before a driver is handed it back.
//!
//! A guest saves `vkGetPipelineCacheData` to disk and passes it back as a new cache's initial
//! data, on a later run or a later boot. The driver parses that data assuming it wrote it, and its
//! parsers are built on that assumption: KosmicKrisp deserializes the shaders inside with no
//! bounds of its own, and anv indexes tables by the object types and counts the data names. A
//! guest that sends data no driver wrote is a guest steering those parsers.
//!
//! So the data a guest is handed carries a tag at its end, an HMAC over everything before it,
//! and initial data is forwarded only when its tag is one this key made. The guest's venus driver
//! puts its own header in front and passes the host's bytes through untouched in both directions,
//! so the tag comes back where it was put. Data that does not open -- another key's, a truncated
//! copy, a forgery -- makes an empty cache, which is an answer Vulkan allows a driver to give
//! for any initial data.
//!
//! Behind the tag, the data must also be the Mesa runtime's framing: its header, then entries
//! whose sizes tile the rest exactly. That is a second line, should the key ever leak, and it
//! holds what can be held without a driver's private formats. It cannot hold an entry's type to
//! the driver's table of them, whose length is the driver's own; the tag is what does that. A
//! host driver outside Mesa writes a different framing, and gets every cache empty.

use hmac::{Hmac, KeyInit, Mac};
use sha2::Sha256;

use crate::config::PipelineCacheKey;

/// Bytes of tag at the end of the data a guest is handed.
pub const TAG_LEN: usize = 32;

/// What the tag is over besides the data, so it means nothing as any other HMAC this key makes.
const DOMAIN: &[u8] = b"virglrs pipeline cache data, version 1\0";

/// `VkPipelineCacheHeaderVersionOne`: four words and a UUID.
const HEADER_LEN: usize = 32;

/// `VK_PIPELINE_CACHE_HEADER_VERSION_ONE`.
const HEADER_VERSION_ONE: u32 = 1;

/// `VK_PIPELINE_CACHE_BLOB_ALIGN`: an entry's data starts on this, counted from the start.
const ENTRY_ALIGN: usize = 8;

/// Mesa's blob writes every word on its own alignment, so an entry's header starts on this after
/// data of any length.
const WORD_ALIGN: usize = 4;

fn mac(key: &PipelineCacheKey, data: &[u8]) -> Hmac<Sha256> {
    let mut mac =
        Hmac::<Sha256>::new_from_slice(key.bytes()).expect("HMAC takes a key of any length");
    mac.update(DOMAIN);
    mac.update(data);
    mac
}

/// The tag `data` is handed to the guest with, under `key`.
pub fn tag(key: &PipelineCacheKey, data: &[u8]) -> [u8; TAG_LEN] {
    mac(key, data).finalize().into_bytes().into()
}

/// The driver's data inside `sealed`, if `key` tagged it and it is framed as Mesa frames it.
pub fn opened<'a>(key: &PipelineCacheKey, sealed: &'a [u8]) -> Option<&'a [u8]> {
    let split = sealed.len().checked_sub(TAG_LEN)?;
    let (data, tag) = sealed.split_at(split);
    // Constant time, so a guest timing the refusal learns nothing about the tag it guessed.
    mac(key, data).verify_slice(tag).ok()?;
    framed(data).then_some(data)
}

/// Whether `data` is the Mesa runtime's framing: its header, a count, and that many entries --
/// padding to [`WORD_ALIGN`], a type, a key size, a data size, the key, padding to
/// [`ENTRY_ALIGN`], the data -- ending exactly at its end.
pub fn framed(data: &[u8]) -> bool {
    let word = |at: usize| -> Option<u32> {
        let bytes = data.get(at..at.checked_add(4)?)?;
        Some(u32::from_le_bytes(bytes.try_into().expect("four bytes")))
    };
    if word(0) != Some(HEADER_LEN as u32) || word(4) != Some(HEADER_VERSION_ONE) {
        return false;
    }
    let Some(count) = word(HEADER_LEN) else { return false };
    let mut at = HEADER_LEN + 4;
    for _ in 0..count {
        let Some(start) = at.checked_next_multiple_of(WORD_ALIGN) else { return false };
        at = start;
        let (Some(ty), Some(key_len), Some(data_len)) = (word(at), word(at + 4), word(at + 8))
        else {
            return false;
        };
        // -1 is an object the driver has no import for, which it keeps as raw bytes.
        if (ty as i32) < -1 {
            return false;
        }
        let Some(key_end) = (at + 12).checked_add(key_len as usize) else { return false };
        let Some(data_start) = key_end.checked_next_multiple_of(ENTRY_ALIGN) else {
            return false;
        };
        let Some(data_end) = data_start.checked_add(data_len as usize) else { return false };
        if data_end > data.len() {
            return false;
        }
        at = data_end;
    }
    at == data.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Data framed as the Mesa runtime frames it, with one entry of each kind of type.
    fn mesa_data() -> Vec<u8> {
        let mut d = Vec::new();
        let words = |d: &mut Vec<u8>, ws: &[u32]| {
            for w in ws {
                d.extend_from_slice(&w.to_le_bytes());
            }
        };
        words(&mut d, &[HEADER_LEN as u32, HEADER_VERSION_ONE, 0x106b, 0x1234]);
        d.extend_from_slice(&[7; 16]);
        words(&mut d, &[2]);
        // An entry the driver imports: a five-byte key, so three bytes of padding.
        words(&mut d, &[0, 5, 6]);
        d.extend_from_slice(&[1; 5]);
        d.resize(d.len().next_multiple_of(ENTRY_ALIGN), 0);
        d.extend_from_slice(&[2; 6]);
        // One it keeps raw, its header on the next word after the six bytes above.
        d.resize(d.len().next_multiple_of(WORD_ALIGN), 0);
        words(&mut d, &[u32::MAX, 8, 3]);
        d.extend_from_slice(&[3; 8]);
        d.resize(d.len().next_multiple_of(ENTRY_ALIGN), 0);
        d.extend_from_slice(&[4; 3]);
        d
    }

    fn sealed(key: &PipelineCacheKey, data: &[u8]) -> Vec<u8> {
        let mut s = data.to_vec();
        s.extend_from_slice(&tag(key, data));
        s
    }

    /// Data this key tagged opens to the driver's own bytes; anything else does not open.
    #[test]
    fn only_data_this_key_tagged_opens() {
        let key = PipelineCacheKey::new([9; PipelineCacheKey::LEN]);
        let data = mesa_data();
        let s = sealed(&key, &data);
        assert_eq!(opened(&key, &s), Some(&data[..]));

        let other = PipelineCacheKey::new([8; PipelineCacheKey::LEN]);
        assert_eq!(opened(&other, &s), None, "another key's data");
        let mut flipped = s.clone();
        flipped[40] ^= 1;
        assert_eq!(opened(&key, &flipped), None, "a byte changed after tagging");
        assert_eq!(opened(&key, &s[..s.len() - 1]), None, "a truncated copy");
        assert_eq!(opened(&key, &data), None, "no tag at all");
        assert_eq!(opened(&key, &[]), None, "nothing");
    }

    /// Behind a good tag, the data must still be framed as Mesa frames it.
    #[test]
    fn tagged_data_must_still_be_mesa_framed() {
        let key = PipelineCacheKey::new([9; PipelineCacheKey::LEN]);
        let data = mesa_data();
        assert!(framed(&data));
        let fails = |edit: &dyn Fn(&mut Vec<u8>)| {
            let mut d = data.clone();
            edit(&mut d);
            assert!(!framed(&d));
            assert_eq!(opened(&key, &sealed(&key, &d)), None);
        };
        fails(&|d| d[0] = 31);
        fails(&|d| d[4] = 2);
        fails(&|d| d[HEADER_LEN] = 3);
        fails(&|d| d.push(0));
        fails(&|d| {
            d.pop();
        });
        // A type below -1.
        fails(&|d| d[HEADER_LEN + 4..HEADER_LEN + 8].copy_from_slice(&(-2i32).to_le_bytes()));
        // A size that runs past the end, and one that wraps.
        fails(&|d| d[HEADER_LEN + 12..HEADER_LEN + 16].copy_from_slice(&u32::MAX.to_le_bytes()));
        fails(&|d| d[HEADER_LEN + 8..HEADER_LEN + 12].copy_from_slice(&u32::MAX.to_le_bytes()));
        // An empty cache is framed.
        assert!(framed(
            &data[..HEADER_LEN + 4]
                .iter()
                .copied()
                .enumerate()
                .map(|(i, b)| { if (HEADER_LEN..HEADER_LEN + 4).contains(&i) { 0 } else { b } })
                .collect::<Vec<_>>()
        ));
    }

    /// The key never prints.
    #[test]
    fn a_key_never_prints() {
        let key = PipelineCacheKey::new([0xab; PipelineCacheKey::LEN]);
        let shown = format!("{key:?}");
        assert!(!shown.contains("ab") && !shown.contains("171"), "{shown}");
    }
}
