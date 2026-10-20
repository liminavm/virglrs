// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! The translator's SPIR-V side (docs: limina's `docs/design/vrend-spirv.md`): a shader handed
//! to the host as a SPIR-V module, so the host's GLSL front end never runs.

pub mod emit;
pub mod probe;
pub mod writer;

use std::collections::BTreeMap;

/// The switch that keeps every program GLSL where the probe passed: `VIRGLRS_SPIRV=0`. SPIR-V
/// is the default wherever the host takes it.
const ENV: &str = "VIRGLRS_SPIRV";

/// Whether this renderer hands its programs over as SPIR-V, and what came of it: how many were
/// linked each way, and why shaders stayed GLSL. Each reason is said once as it first appears,
/// the running counts each time the total reaches a power of two, and the whole tally when the
/// renderer goes. The counts are said as they grow because a VM's worker is torn down by the
/// guest's power-off without dropping anything, so a tally kept for the end would never be read
/// there; powers of two keep a long session's log to a few dozen lines.
#[derive(Debug, Default)]
pub struct Use {
    enabled: bool,
    spirv_programs: u64,
    glsl_programs: u64,
    declined: BTreeMap<emit::Decline, u64>,
    failed: u64,
}

impl Use {
    /// SPIR-V programs where `probe` passed, unless `VIRGLRS_SPIRV=0` turns them off.
    /// `attrib_binding` is whether the host binds vertex attributes by location, which is how a
    /// module's inputs meet the draw's vertex arrays.
    pub fn new(probe: &Result<(), String>, attrib_binding: bool) -> Use {
        let declined = std::env::var(ENV).as_deref() == Ok("0");
        let enabled = !declined && probe.is_ok() && attrib_binding;
        if declined {
            eprintln!("[virglrs] vrend: SPIR-V programs off ({ENV}=0)");
        } else if enabled {
            eprintln!("[virglrs] vrend: SPIR-V programs on");
        } else {
            eprintln!("[virglrs] vrend: this host cannot take SPIR-V programs; every one is GLSL");
        }
        Use { enabled, spirv_programs: 0, glsl_programs: 0, declined: BTreeMap::new(), failed: 0 }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// A shader the emitter would not write.
    pub fn declined(&mut self, why: emit::Decline) {
        let n = self.declined.entry(why).or_default();
        if *n == 0 {
            eprintln!("[virglrs] vrend: a shader stays GLSL: {why}");
        }
        *n += 1;
    }

    /// A module, or a program of modules, the host refused; its programs go back to GLSL.
    pub fn failed(&mut self, what: &str, why: &str) {
        self.failed += 1;
        eprintln!("[virglrs] vrend: SPIR-V {what} refused, the program goes back to GLSL: {why}");
    }

    /// A program linked, from SPIR-V modules or GLSL.
    pub fn linked(&mut self, spirv: bool) {
        if !self.enabled {
            return;
        }
        if spirv {
            self.spirv_programs += 1;
        } else {
            self.glsl_programs += 1;
        }
        if (self.spirv_programs + self.glsl_programs).is_power_of_two() {
            eprintln!(
                "[virglrs] vrend: {} programs linked from SPIR-V and {} from GLSL so far",
                self.spirv_programs, self.glsl_programs
            );
        }
    }
}

impl Drop for Use {
    fn drop(&mut self) {
        if !self.enabled {
            return;
        }
        let mut declined: Vec<_> = self.declined.iter().map(|(d, n)| (d.to_string(), *n)).collect();
        declined.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
        eprintln!(
            "[virglrs] vrend: {} programs linked from SPIR-V, {} from GLSL, {} refused by the \
             host; shaders kept GLSL: {declined:?}",
            self.spirv_programs, self.glsl_programs, self.failed
        );
    }
}
