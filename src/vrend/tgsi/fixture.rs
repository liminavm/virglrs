// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! The C's shader logs as fixtures: every shader the classic corpus creates, as `tgsi_dump`
//! printed it and as `vrend_convert_shader` translated it (`harness/replay/vrend-shader-log.py`
//! records them). One log is a GLES host's and one a desktop GL host's, the same corpus on each.
//! A third holds the distinct shaders of every pinned capture. `tgsi_dump`'s spelling drops
//! `_PRECISE` and rounds float immediates, so text parsed back from it can translate differently
//! from what the guest sent; each of its blocks also carries the TGSI the Rust leg printed exactly
//! (`VIRGLRS_DEBUG=shader-exact`) from the same replay, which is what the translator is given.
//! The C's dump stays the parser's oracle. The tests here hold the Rust side to those blocks.

#![cfg(test)]

use super::*;

const LOG: &str = include_str!("../../../harness/replay/fixtures/vrend-shaders.txt");
const DESKTOP_LOG: &str =
    include_str!("../../../harness/replay/fixtures/vrend-shaders-desktop.iris.txt");
const CAPTURES_LOG: &str =
    include_str!("../../../harness/replay/fixtures/vrend-shaders-captures.txt");

/// One shader of the log: the C's TGSI dump, the exact text where the log has one, and the GLSL.
pub struct Block {
    pub tgsi: &'static str,
    pub exact: Option<&'static str>,
    pub glsl: &'static str,
}

impl Block {
    /// The text to translate: the exact TGSI where there is one, else the C's dump.
    pub fn input(&self) -> &'static str {
        self.exact.unwrap_or(self.tgsi)
    }
}

/// The GLES log's blocks, in creation order.
pub fn blocks() -> Vec<Block> {
    blocks_of(LOG)
}

/// The desktop GL log's blocks, in creation order.
pub fn desktop_blocks() -> Vec<Block> {
    blocks_of(DESKTOP_LOG)
}

/// The distinct blocks of every pinned capture's GLES log, in first-seen order.
pub fn capture_blocks() -> Vec<Block> {
    blocks_of(CAPTURES_LOG)
}

fn blocks_of(log: &'static str) -> Vec<Block> {
    log.split("TGSI received:\n")
        .skip(1)
        .map(|block| {
            let (tgsi, glsl) = block.split_once("\nGLSL:\n").expect("a GLSL block follows");
            let glsl = glsl.strip_suffix('\n').expect("a block ends in an empty line");
            let (tgsi, exact) = match tgsi.split_once("\nTGSI exact:\n") {
                Some((tgsi, exact)) => (tgsi, Some(exact)),
                None => (tgsi, None),
            };
            Block { tgsi, exact, glsl }
        })
        .collect()
}

#[test]
fn the_corpus_has_its_shaders() {
    assert_eq!(blocks().len(), 33);
    assert_eq!(desktop_blocks().len(), 30);
    assert_eq!(capture_blocks().len(), 151);
}

/// A dump is text `tgsi_text` reads back; parsing the C's dump and dumping the result must print
/// the C's dump again, which holds both the parser and the dump to the C on every shader of the
/// corpus. Where a block has exact text, that is what is parsed: the C's spelling of it must
/// still be the C's dump.
#[test]
fn every_corpus_dump_reads_back_and_prints_the_same() {
    for (i, block) in blocks().iter().chain(&desktop_blocks()).chain(&capture_blocks()).enumerate()
    {
        let shader = text::parse(block.input().as_bytes(), u32::MAX)
            .unwrap_or_else(|e| panic!("shader {i}: {e}"));
        let printed = dump::dump(&shader);
        assert!(
            printed == block.tgsi,
            "shader {i} prints differently:\n--- C\n{}\n--- Rust\n{printed}",
            block.tgsi
        );
        scan::scan(&shader).unwrap_or_else(|e| panic!("shader {i}: {e}"));
    }
}
