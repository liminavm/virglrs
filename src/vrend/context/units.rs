// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! One shader stage's sampler units: the view and the sampler state in each slot, and which
//! slots the next draw has to bind again.

use super::*;

/// The view in a slot, the sampler state in the same slot, and whether the unit they make up
/// must be re-bound, as one value.
///
/// A draw binds a unit only when it is marked, so anything that changes what a unit samples --
/// or samples with -- has to mark it. Kept as three fields, a binder could change one of the
/// first two and forget the third, and the change would reach the hardware only by luck: some
/// other change marking the unit for its own reason, which on a desktop is nearly always. Here
/// every mutation marks the slots it touches, and there is no other way to mutate them.
///
/// View slots outrun the units: the decoder admits [`MAX_SHADER_SAMPLER_VIEWS`] of them and the
/// draw samples through the first [`MAX_SAMPLERS`]. A view above the units is held, because the
/// key selection reads every view, and never marked, because no draw binds it. Sampler slots are
/// bounded to the units by the decoder, so a sampler slot the mask cannot hold is a host bug and
/// the mask says so.
#[derive(Default)]
pub struct Units {
    views: BTreeMap<u32, ObjectHandle>,
    samplers: BTreeMap<u32, ObjectHandle>,
    dirty: Dirty<MAX_SAMPLERS>,
}

impl Units {
    /// The view in `slot`, if one is set.
    pub fn view(&self, slot: u32) -> Option<ObjectHandle> {
        self.views.get(&slot).copied()
    }

    /// Every view set, in slot order, the ones above the units included.
    pub fn views(&self) -> impl Iterator<Item = (u32, ObjectHandle)> + '_ {
        self.views.iter().map(|(s, h)| (*s, *h))
    }

    /// The sampler state in `slot`, if one is bound.
    pub fn sampler(&self, slot: u32) -> Option<ObjectHandle> {
        self.samplers.get(&slot).copied()
    }

    /// The units a draw has to bind again.
    pub fn dirty(&self) -> Dirty<MAX_SAMPLERS> {
        self.dirty
    }

    /// Every unit, as a new program takes them: nothing bound for it is still bound.
    pub fn mark_all(&mut self) {
        self.dirty = Dirty::all();
    }

    /// The draw bound every marked unit; nothing is owed until the next change.
    pub fn bound(&mut self) {
        self.dirty.clear();
    }

    /// Whether `slot` already holds `view`, which is what lets a binder skip the GL work.
    pub fn holds_view(&self, slot: u32, view: ObjectHandle) -> bool {
        self.views.get(&slot) == Some(&view)
    }

    /// Put `view` in `slot`, or empty it. The unit is marked either way: an emptied slot has
    /// nothing for the draw to bind, so the mark is idle there, and one rule is better than a
    /// rule with an exception a caller has to know.
    pub fn set_view(&mut self, slot: u32, view: Option<ObjectHandle>) {
        match view {
            Some(h) => self.views.insert(slot, h),
            None => self.views.remove(&slot),
        };
        self.mark_view_slot(slot);
    }

    /// Empty every view slot from `end` up, as `vrend_set_num_sampler_views` does after the
    /// slots a set names.
    pub fn drop_views_from(&mut self, end: u32) {
        let gone: Vec<u32> = self.views.range(end..).map(|(s, _)| *s).collect();
        for slot in gone {
            self.views.remove(&slot);
            self.mark_view_slot(slot);
        }
    }

    /// Empty every view slot naming `handle` and mark each. Whether any did.
    ///
    /// A slot names its view by handle, and a handle the guest has freed is reused by its next
    /// create: left in place, the slot would answer [`Units::holds_view`] for the new view and
    /// keep the old texture on the unit.
    pub fn evict_view(&mut self, handle: ObjectHandle) -> bool {
        let gone: Vec<u32> =
            self.views.iter().filter(|(_, h)| **h == handle).map(|(s, _)| *s).collect();
        for slot in &gone {
            self.views.remove(slot);
            self.mark_view_slot(*slot);
        }
        !gone.is_empty()
    }

    /// Bind `state` to `slot`, or unbind whatever is there. Marked either way, as
    /// `vrend_bind_sampler_states` marks every slot it is handed: the unit samples with
    /// something else now, under a view that may not have moved.
    pub fn bind_sampler(&mut self, slot: u32, state: Option<ObjectHandle>) {
        match state {
            Some(h) => self.samplers.insert(slot, h),
            None => self.samplers.remove(&slot),
        };
        self.dirty.mark(slot);
    }

    /// Unbind `handle` from every slot holding it and close the gaps, the slots after each
    /// moving down one, as `vrend_destroy_sampler_state_object` does. Every slot that changed
    /// is marked, the moved ones included: each of those units samples with a different state
    /// now.
    pub fn forget_sampler(&mut self, handle: ObjectHandle) {
        let held: Vec<(u32, ObjectHandle)> = self.samplers.iter().map(|(s, h)| (*s, *h)).collect();
        let mut shift = 0;
        for (slot, h) in held {
            if h == handle {
                shift += 1;
                self.samplers.remove(&slot);
                self.dirty.mark(slot);
            } else if shift != 0 {
                self.samplers.remove(&slot);
                self.samplers.insert(slot - shift, h);
                self.dirty.mark(slot);
                self.dirty.mark(slot - shift);
            }
        }
    }

    fn mark_view_slot(&mut self, slot: u32) {
        // A view above the units is held but never sampled, so nothing rebinds it.
        if (slot as usize) < MAX_SAMPLERS {
            self.dirty.mark(slot);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(n: u32) -> ObjectHandle {
        ObjectHandle::new(n).expect("a test handle is non-zero")
    }

    #[test]
    fn binding_a_sampler_state_marks_its_unit() {
        let mut u = Units::default();
        u.set_view(2, Some(h(9)));
        u.bound();
        assert!(u.dirty().is_empty(), "the draw bound it; nothing is owed");
        // The view stays; only what the unit samples with changes.
        u.bind_sampler(2, Some(h(5)));
        assert_eq!(u.sampler(2), Some(h(5)));
        assert_eq!(u.view(2), Some(h(9)));
        assert!(u.dirty().contains(2), "a sampler change under an unchanged view is a re-bind");
        assert!(!u.dirty().contains(1) && !u.dirty().contains(3));
        // Unbinding is a change too.
        u.bound();
        u.bind_sampler(2, None);
        assert_eq!(u.sampler(2), None);
        assert!(u.dirty().contains(2));
    }

    #[test]
    fn a_destroyed_view_leaves_every_slot_it_held_and_marks_them() {
        let mut a = Units::default();
        a.set_view(3, Some(h(9)));
        let mut b = Units::default();
        b.set_view(0, Some(h(9)));
        b.set_view(1, Some(h(4)));
        b.set_view(7, Some(h(9)));
        a.bound();
        b.bound();
        assert!(a.evict_view(h(9)));
        assert!(b.evict_view(h(9)));
        assert_eq!(a.views().count(), 0);
        assert_eq!(b.views().map(|(s, _)| s).collect::<Vec<_>>(), vec![1]);
        assert!(a.dirty().contains(3));
        assert!(b.dirty().contains(0) && b.dirty().contains(7) && !b.dirty().contains(1));
        // A handle the guest reuses for a new view then binds afresh, instead of reading as
        // already bound.
        assert!(!a.evict_view(h(9)));
        assert!(!a.holds_view(3, h(9)));
    }

    #[test]
    fn a_destroyed_sampler_state_closes_its_gap_and_marks_every_slot_that_moved() {
        let mut u = Units::default();
        u.bind_sampler(0, Some(h(1)));
        u.bind_sampler(2, Some(h(3)));
        u.bind_sampler(3, Some(h(4)));
        u.bind_sampler(5, Some(h(1)));
        u.bound();
        u.forget_sampler(h(1));
        // The C's compaction: the slot emptied stays empty, and every state after it moves
        // down one -- what was in 2 is now in 1, what was in 3 is now in 2.
        assert_eq!(u.sampler(0), None);
        assert_eq!(u.sampler(1), Some(h(3)));
        assert_eq!(u.sampler(2), Some(h(4)));
        assert_eq!(u.sampler(3), None);
        assert_eq!(u.sampler(5), None);
        // Slot 1 held nothing and lost nothing; it is owed only because a state moved INTO it.
        assert!(u.dirty().contains(1), "a unit a state moved into samples with it now");
        assert!(u.dirty().contains(0) && u.dirty().contains(2) && u.dirty().contains(3));
        assert!(u.dirty().contains(5));
        assert!(!u.dirty().contains(4), "neither held nor moved into, so not owed");
    }

    #[test]
    fn a_view_above_the_units_is_held_and_never_marked() {
        let mut u = Units::default();
        u.set_view(MAX_SAMPLERS as u32 + 3, Some(h(9)));
        assert_eq!(u.view(MAX_SAMPLERS as u32 + 3), Some(h(9)));
        assert!(u.dirty().is_empty());
        assert!(u.evict_view(h(9)));
        assert!(u.dirty().is_empty());
    }

    #[test]
    fn views_past_a_set_are_dropped_and_the_units_among_them_marked() {
        let mut u = Units::default();
        u.set_view(0, Some(h(1)));
        u.set_view(5, Some(h(2)));
        u.set_view(40, Some(h(3)));
        u.bound();
        u.drop_views_from(1);
        assert_eq!(u.views().map(|(s, _)| s).collect::<Vec<_>>(), vec![0]);
        assert!(!u.dirty().contains(0) && u.dirty().contains(5));
    }

    #[test]
    fn a_new_program_owes_every_unit() {
        let mut u = Units::default();
        u.mark_all();
        assert_eq!(u.dirty(), Dirty::all());
        u.bound();
        assert!(u.dirty().is_empty());
    }
}
