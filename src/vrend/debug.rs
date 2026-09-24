// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! `VIRGLRS_DEBUG`: what the classic renderer prints beyond its errors. A comma-separated list
//! of switches, read once; the C's is `VREND_DEBUG`, and the switches keep its names where they
//! print the same thing.

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
    /// Every fence, from the ABI call that asks for one to the callback that retires it.
    ///
    /// A fence that is never retired hangs the guest forever and looks exactly like a slow
    /// renderer from outside. This is what tells the two apart: a create with no matching retire
    /// names the step that dropped it.
    Fence,
}

impl Switch {
    const ALL: &'static [(Switch, &'static str)] =
        &[(Switch::Shader, "shader"), (Switch::Resource, "resource"), (Switch::Fence, "fence")];
}

/// The switches `VIRGLRS_DEBUG` asked for, read once when the renderer is built and carried by
/// it from there, as every other knob is.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Switches(u8);

impl Switches {
    pub fn from_env() -> Switches {
        let Ok(spec) = std::env::var("VIRGLRS_DEBUG") else {
            return Switches::default();
        };
        let mut on = Switches::default();
        for name in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            match Switch::ALL.iter().position(|(_, n)| *n == name) {
                Some(i) => on.0 |= 1 << i,
                None => eprintln!("[virglrs] VIRGLRS_DEBUG: no switch named {name:?}"),
            }
        }
        on
    }

    /// Whether `switch` was asked for.
    pub fn enabled(self, switch: Switch) -> bool {
        let i = Switch::ALL.iter().position(|(s, _)| *s == switch).expect("every switch is listed");
        self.0 & (1 << i) != 0
    }
}

/// limina's two trace knobs, read once when the renderer is built and carried by it from there.
///
/// `LIMINA_READBACK_TRACE` names the scanout surfaces a readback found blank and the writes that
/// reached them; `LIMINA_GL_TRACE` drains GL errors after the sampler-view calls so a failing one
/// is named. Neither is a switch in [`Switch`]'s list: limina sets them by these names.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Traces {
    pub readback: bool,
    pub gl: bool,
}

impl Traces {
    pub fn from_env() -> Traces {
        let traces = Traces {
            readback: std::env::var_os("LIMINA_READBACK_TRACE").is_some(),
            gl: std::env::var_os("LIMINA_GL_TRACE").is_some(),
        };
        // Said at startup, not left to the first hit: a diagnostic that only ever speaks when it
        // finds something cannot be told, from its silence, from one that was never compiled
        // in. limina builds `third_party/virglrs`, not whichever clone the change was written
        // in, so "the trace printed nothing" is a claim about the build before it is one about
        // the run.
        if traces.readback {
            eprintln!("[virglrs] readback trace armed: blank scanout readbacks will be named");
        }
        traces
    }
}
