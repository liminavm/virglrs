// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! `VIRGLRS_DEBUG`: what the classic renderer prints beyond its errors. A comma-separated list
//! of switches, read once; the C's is `VREND_DEBUG`, and the switches keep its names where they
//! print the same thing.

use std::sync::OnceLock;

/// One switch.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Switch {
    /// Every shader as it is created: the TGSI parsed from the guest's text, printed back, and
    /// the GLSL it became -- the C's `shader` switch, in the C's format, so the two logs diff.
    Shader,
    /// Every resource as it comes and goes at the C ABI, and every miss when one is asked for.
    ///
    /// A create the VMM ignores the return of is a resource that silently never exists, and the
    /// first anyone hears of it is a later command naming a handle nothing holds. This switch is
    /// what turns that into two lines next to each other instead of an archaeology exercise.
    Resource,
}

impl Switch {
    const ALL: &'static [(Switch, &'static str)] =
        &[(Switch::Shader, "shader"), (Switch::Resource, "resource")];
}

fn switches() -> &'static [Switch] {
    static SWITCHES: OnceLock<Vec<Switch>> = OnceLock::new();
    SWITCHES.get_or_init(|| {
        let Ok(spec) = std::env::var("VIRGLRS_DEBUG") else {
            return Vec::new();
        };
        spec.split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .filter_map(|name| {
                let found = Switch::ALL.iter().find(|(_, n)| *n == name).map(|(s, _)| *s);
                if found.is_none() {
                    eprintln!("[virglrs] VIRGLRS_DEBUG: no switch named {name:?}");
                }
                found
            })
            .collect()
    })
}

/// Whether `switch` was asked for.
pub fn enabled(switch: Switch) -> bool {
    switches().contains(&switch)
}
