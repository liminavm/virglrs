// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! The C's shader logs as fixtures: every shader the classic corpus creates, as `tgsi_dump`
//! printed it and as `vrend_convert_shader` translated it (`harness/replay/vrend-shader-log.py`
//! records them). One log is a GLES host's and one a desktop GL host's, the same corpus on each.
//! A third holds the distinct shaders of every pinned capture. `tgsi_dump`'s spelling drops
//! `_PRECISE` and rounds float immediates, so text parsed back from it can translate differently
//! from what the guest sent; each of its blocks also carries the TGSI the Rust leg printed exactly
//! (`VIRGLRS_DEBUG=shader-exact`) from the same replay, which is what the translator is given.
//! The C's dump stays the parser's oracle. A fourth, in the same shape, holds a piglit session's
//! shaders of every stage. The tests here hold the Rust side to those blocks.

#![cfg(test)]

use super::*;

const LOG: &str = include_str!("../../../harness/replay/fixtures/vrend-shaders.txt");
const DESKTOP_LOG: &str =
    include_str!("../../../harness/replay/fixtures/vrend-shaders-desktop.iris.txt");
const CAPTURES_LOG: &str =
    include_str!("../../../harness/replay/fixtures/vrend-shaders-captures.txt");
const PIGLIT_LOG: &str = include_str!("../../../harness/replay/fixtures/vrend-shaders-piglit.txt");

/// One shader of the log: the C's TGSI dump, the exact text and the translation's other inputs
/// where the log has them, and the GLSL.
pub struct Block {
    pub tgsi: &'static str,
    pub exact: Option<&'static str>,
    pub inputs: Option<Inputs>,
    pub glsl: &'static str,
}

/// Everything a translation takes besides its program, as `shader::keytext` printed it.
pub struct Inputs {
    pub config: &'static str,
    pub key: &'static str,
    pub stream_output: &'static str,
    pub local_mem: u32,
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

/// The distinct shaders of a piglit session both legs translated alike, in first-seen order.
pub fn piglit_blocks() -> Vec<Block> {
    blocks_of(PIGLIT_LOG)
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
            // The other inputs follow the exact text as `CFG:`, `KEY:`, `SO:` and `LOCAL_MEM:`.
            let (exact, inputs) = match exact.and_then(|e| e.split_once("CFG: ")) {
                Some((exact, rest)) => {
                    let line = |l: Option<&'static str>, name: &str| {
                        l.and_then(|l| l.strip_prefix(name))
                            .unwrap_or_else(|| panic!("no {name} line after CFG"))
                    };
                    let mut lines = rest.lines();
                    let config = lines.next().expect("the CFG line");
                    let key = line(lines.next(), "KEY: ");
                    let stream_output = line(lines.next(), "SO: ");
                    let local_mem = line(lines.next(), "LOCAL_MEM: ").parse().expect("a number");
                    (Some(exact), Some(Inputs { config, key, stream_output, local_mem }))
                }
                None => (exact, None),
            };
            Block { tgsi, exact, inputs, glsl }
        })
        .collect()
}

#[test]
fn the_corpus_has_its_shaders() {
    assert_eq!(blocks().len(), 33);
    assert_eq!(desktop_blocks().len(), 30);
    assert_eq!(capture_blocks().len(), 151);
    assert_eq!(piglit_blocks().len(), 1935);
}

/// A dump is text `tgsi_text` reads back; parsing the C's dump and dumping the result must print
/// the C's dump again, which holds both the parser and the dump to the C on every shader of the
/// corpus. Where a block has exact text, that is what is parsed: the C's spelling of it must
/// still be the C's dump.
#[test]
fn every_corpus_dump_reads_back_and_prints_the_same() {
    let all = blocks().into_iter().chain(desktop_blocks()).chain(capture_blocks());
    for (i, block) in all.chain(piglit_blocks()).enumerate() {
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
