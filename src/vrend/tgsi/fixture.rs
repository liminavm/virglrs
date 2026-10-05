// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! The C's shader logs as fixtures: every shader the classic corpus creates, as `tgsi_dump`
//! printed it and as `vrend_convert_shader` translated it (`harness/replay/vrend-shader-log.py`
//! records them). One log is a GLES host's and one a desktop GL host's, the same corpus on each.
//! The tests here hold the Rust side to those blocks.

#![cfg(test)]

use super::*;

const LOG: &str = include_str!("../../../harness/replay/fixtures/vrend-shaders.txt");
const DESKTOP_LOG: &str =
    include_str!("../../../harness/replay/fixtures/vrend-shaders-desktop.iris.txt");

/// One shader of the log: its TGSI dump and its GLSL.
pub struct Block {
    pub tgsi: &'static str,
    pub glsl: &'static str,
}

/// The GLES log's blocks, in creation order.
pub fn blocks() -> Vec<Block> {
    blocks_of(LOG)
}

/// The desktop GL log's blocks, in creation order.
pub fn desktop_blocks() -> Vec<Block> {
    blocks_of(DESKTOP_LOG)
}

fn blocks_of(log: &'static str) -> Vec<Block> {
    log.split("TGSI received:\n")
        .skip(1)
        .map(|block| {
            let (tgsi, glsl) = block.split_once("\nGLSL:\n").expect("a GLSL block follows");
            let glsl = glsl.strip_suffix('\n').expect("a block ends in an empty line");
            Block { tgsi, glsl }
        })
        .collect()
}

#[test]
fn the_corpus_has_its_shaders() {
    assert_eq!(blocks().len(), 33);
    assert_eq!(desktop_blocks().len(), 30);
}

/// A dump is text `tgsi_text` reads back; parsing the C's dump and dumping the result must print
/// the C's dump again, which holds both the parser and the dump to the C on every shader of the
/// corpus.
#[test]
fn every_corpus_dump_reads_back_and_prints_the_same() {
    for (i, block) in blocks().iter().chain(&desktop_blocks()).enumerate() {
        let shader = text::parse(block.tgsi.as_bytes(), u32::MAX)
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
