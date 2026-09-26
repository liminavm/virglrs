// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! Cut the recorded corpora into fuzzer seeds: one file per venus command, from every capture
//! under `harness/vm/captures`, and one per shader in the classic corpus's TGSI log. The venus
//! commands no capture holds are added from `src/venus/wire_samples.rs`.
//!
//! Run from the repository root: `cargo run --manifest-path fuzz/Cargo.toml --bin seed-corpora`.
//! A random input almost never names a real command with plausible arguments, so starting from
//! what guests actually send is what lets the fuzzer reach past the header.

#[allow(dead_code)]
#[path = "../../../harness/replay/rs/src/corpus.rs"]
mod corpus;

#[allow(dead_code)]
#[path = "../../../src/venus/wire_samples.rs"]
mod wire_samples;

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;

fn write_seed(dir: &Path, bytes: &[u8]) -> bool {
    let mut h = DefaultHasher::new();
    bytes.hash(&mut h);
    let path = dir.join(format!("{:016x}", h.finish()));
    if path.exists() {
        return false;
    }
    std::fs::write(&path, bytes).expect("the seed directory is writable");
    true
}

fn main() {
    let venus = Path::new("fuzz/corpus/venus_command");
    std::fs::create_dir_all(venus).expect("the corpus directory can be made");
    let mut captures: Vec<_> = std::fs::read_dir("harness/vm/captures")
        .expect("run from the repository root")
        .map(|e| e.expect("a directory entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "vkrc"))
        .collect();
    captures.sort();
    let mut commands = 0;
    for path in &captures {
        let blob = std::fs::read(path).expect("a capture reads");
        let parsed = corpus::parse(&blob).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let prologue = parsed.prologues.iter().flat_map(|p| &p.entries).map(|e| &e.wire);
        let stream = parsed.records.iter().filter_map(|r| match r {
            corpus::Record::Cmd { wire, .. } => Some(wire),
            _ => None,
        });
        for wire in prologue.chain(stream) {
            commands += usize::from(write_seed(venus, wire));
        }
    }
    println!("venus_command: {commands} distinct commands from {} captures", captures.len());

    // The shapes no capture holds, written out by hand. With no flags, as a ring sends them.
    let mut samples = 0;
    for (ty, args) in wire_samples::commands() {
        let wire = [ty.to_le_bytes().to_vec(), 0u32.to_le_bytes().to_vec(), args].concat();
        samples += usize::from(write_seed(venus, &wire));
    }
    println!("venus_command: {samples} hand-written commands");

    let tgsi = Path::new("fuzz/corpus/tgsi_translate");
    std::fs::create_dir_all(tgsi).expect("the corpus directory can be made");
    let log = std::fs::read_to_string("harness/replay/fixtures/vrend-shaders.txt")
        .expect("the shader log reads");
    let mut shaders = 0;
    for block in log.split("TGSI received:\n").skip(1) {
        let (text, _) = block.split_once("\nGLSL:\n").expect("a GLSL block follows");
        shaders += usize::from(write_seed(tgsi, text.as_bytes()));
    }
    println!("tgsi_translate: {shaders} distinct shaders");
}
