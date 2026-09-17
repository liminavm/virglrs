// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! Which GL context this thread has current, and what this renderer shadows of its state.
//!
//! Its own module so that the record of a switch cannot be made without the switch: the only
//! way to tell [`Current`] a context is current is [`Current::switch_to`], which makes it so.

use super::egl::{self, EglError, Winsys};
use super::gl::BoundProgram;
use super::proto::SubContextId;
use crate::ids::ContextId;

/// Which GL context a shadow of GL state belongs to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GlContext {
    Ctx0,
    Sub(ContextId, SubContextId),
    /// The blitter's own GL context, for the length of one blit. It is a state of this enum and
    /// not a flag beside it because it is the same fact: the blitter's context is one more
    /// context whose bound program is its own.
    Blitter,
}

/// What this renderer shadows of the current GL context's state rather than ask GL for it, and
/// the name of the context that shadow describes.
///
/// The name is the shadow's key and never a reason to skip work. Which context the thread has
/// current is EGL's to answer and a belief about it is falsifiable: the VMM makes its own
/// contexts current between our calls, so every switch here is unconditional. What the VMM
/// cannot reach is the state *inside* our contexts -- a program bound in one of ours is still
/// bound when we come back to it, which is what makes the shadow worth keeping.
///
/// A switch clears it, so what is shadowed is never another context's: a context destroyed takes
/// its shadow with the switch away from it, and a program deleted in one context cannot leave a
/// sibling in the share group believing it bound. The price is one redundant bind after each
/// switch, which no draw pays.
#[derive(Debug)]
pub struct Current {
    on: GlContext,
    program: BoundProgram,
}

impl Current {
    /// Ctx0 with nothing bound, as [`Vrend::new`](super::vrend::Vrend) leaves the thread.
    pub fn ctx0() -> Current {
        Current { on: GlContext::Ctx0, program: BoundProgram::default() }
    }

    /// Make `ctx` current on this thread and record it as `on`.
    ///
    /// The one way the record moves. A refused switch leaves it where it was, because the thread
    /// is still on the context it was on and the shadow is still that context's.
    pub fn switch_to(
        &mut self,
        winsys: &Winsys,
        ctx: &egl::Context,
        on: GlContext,
    ) -> Result<(), EglError> {
        winsys.make_current(ctx)?;
        self.switched_to(on);
        Ok(())
    }

    fn switched_to(&mut self, on: GlContext) {
        if self.on != on {
            self.on = on;
            self.program = BoundProgram::default();
        }
    }

    /// The program bound on the current context, which is what [`Gl::use_program`](super::gl::Gl::use_program)
    /// needs to skip a bind that would change nothing.
    pub fn program(&mut self) -> &mut BoundProgram {
        &mut self.program
    }

    /// Which context the shadow describes.
    #[cfg(test)]
    pub fn on(&self) -> GlContext {
        self.on
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_switch_forgets_the_program_the_other_context_had_bound() {
        let ctx = ContextId::new(1).expect("a context id is non-zero");
        let a = GlContext::Sub(ctx, SubContextId(0));
        let b = GlContext::Sub(ctx, SubContextId(1));
        let mut current = Current::ctx0();
        current.switched_to(a);
        let bound = *current.program();

        // Being told about the context already current changes nothing: a switch that did not
        // move must not throw away a shadow that is still true, or every one of them would.
        current.switched_to(a);
        assert_eq!(*current.program(), bound);

        // A real switch does. GL's current program is per-context, so what was bound on `a` says
        // nothing about `b` -- and `b` may be a context this thread has never had current, or a
        // brand new one that happens to reuse a name.
        current.switched_to(b);
        assert_eq!(*current.program(), BoundProgram::default());

        current.switched_to(GlContext::Blitter);
        assert_eq!(*current.program(), BoundProgram::default());
    }
}
