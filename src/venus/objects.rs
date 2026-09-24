// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! The per-context object table: guest object ids to host Vulkan handles.
//!
//! The guest chooses every id and sends it on the wire, so this table is the only thing standing
//! between a guest and a host handle it invented. Every id it names is checked here, against both
//! existence and Vulkan object type -- a guest does not get to create a buffer and then use the id
//! as a device.
//!
//! One table per context, and a context owns exactly one `VkInstance`, so every object in it hangs
//! off one instance tree. Nothing is shared between contexts: two guests may pick the same id and
//! never see each other's objects.
//!
//! **A guest id and a host handle are never the same member.** A command that creates an object
//! carries the id the guest picked, and the reply encoder sends that same field back -- a handler
//! that overwrote it with the host handle would leak a host pointer into the guest. Input handles
//! go the other way: the generated `_lookup` decode replaces the id with the host handle in place,
//! which is what makes an argument struct callable by the driver with no conversion, and is safe
//! because a reply never re-encodes an input member.
//!
//! Each direction is therefore missing one half of the pairing by the time this table is written,
//! and the generator supplies it as a shadow member alongside: `handle_<name>` is where the driver
//! writes a created object's handle, and `id_<name>` is the guest id the decoder kept as the
//! lookup overwrote it. The two arrive together at `object_created` and `object_destroyed`.

use std::cell::RefCell;
use std::collections::BTreeMap;

use super::cs::{Handle, HostHandle, Lookup, ObjectId, Objects};
use super::proto::types::{VkDevice, VkObjectType};

/// One live object: the host handle, and the Vulkan type the guest must name it by.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Object {
    /// The id the guest gave it.
    ///
    /// The arena is reached by [`Key`], but a cascade takes objects out that no command ever
    /// named, and their ids are what the rest of the context files them under. Carried on the
    /// object rather than looked back up because there is no key-to-id direction to look up:
    /// `slots` only goes the other way, and scanning it could not tell an id this cascade just
    /// orphaned from one orphaned by an earlier destroy. Written once, in [`Table::add`], beside
    /// the `slots` entry it mirrors -- the pair `Pools` keeps for the same reason.
    pub id: ObjectId,
    /// What kind of object it is. Named rather than a bare i32 because a destroy has to switch on
    /// it to pick the right `vkDestroyX`, and a match on an integer is a match nobody can check.
    pub ty: VkObjectType,
    /// The host handle, whatever Vulkan gave us for it.
    pub handle: HostHandle,
    /// What this object was created under, or `None` for the instance, which is the root.
    ///
    /// A key and not an id: an id is what the *guest* calls the parent, and the guest is free to
    /// destroy a parent and name a fresh object by the same id. The key stops pointing at anything
    /// the moment the parent dies, so a new parent cannot inherit a dead one's children.
    parent: Option<Key>,
}

/// A [`Key`] as something outside this module may hold: opaque, and answered only by
/// [`Table::holds`]. What a record keeps when it has to know, later, whether the object it was
/// made from is still the same object -- rather than whether the id still names *something*.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct ObjectKey(Key);

/// Where an object lives in the [`Arena`], and which occupant of that place it is.
///
/// The generation is the whole mechanism. A key names a slot *and* the object that was in it when
/// the key was made, so a key to something destroyed resolves to nothing even after the slot has
/// been handed to an unrelated object. That is what makes a stale reference fail on its own,
/// rather than by someone remembering to go and delete it.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
struct Key {
    index: usize,
    /// Wide on purpose. A slot is reused every time the object in it is destroyed, so this counts
    /// create/destroy pairs on one slot -- a number a guest chooses. At 32 bits a guest could cycle
    /// a slot back to a generation an orphaned id still holds and have that dead id name a live
    /// object again; the cycles cost it nothing but wire commands. At 64 there is no rate at which
    /// that finishes.
    generation: u64,
}

/// One place in the arena, occupied or not.
#[cfg_attr(test, derive(Clone))]
struct Entry {
    /// Bumped every time the slot is emptied, which is what invalidates every key to it.
    generation: u64,
    object: Option<Object>,
}

/// The one owner of every live object.
///
/// Everything else -- the guest's ids here, and in time the driver's own maps -- holds [`Key`]s
/// into this and nothing else. There is exactly one copy of an object's handle, so there is no
/// second copy to forget to update.
#[derive(Default)]
#[cfg_attr(test, derive(Clone))]
struct Arena {
    entries: Vec<Entry>,
    /// Slots emptied by a removal, to be handed out again.
    free: Vec<usize>,
}

impl Arena {
    fn insert(&mut self, object: Object) -> Key {
        if let Some(index) = self.free.pop() {
            let e = &mut self.entries[index];
            e.object = Some(object);
            return Key { index, generation: e.generation };
        }
        // Indexed by `usize` rather than a narrower type so there is no count to check here: a
        // guest cannot reach a length the machine could not hold the entries for anyway, and the
        // allocation is what fails if it tries.
        let index = self.entries.len();
        self.entries.push(Entry { generation: 0, object: Some(object) });
        Key { index, generation: 0 }
    }

    fn get(&self, key: Key) -> Option<&Object> {
        let e = self.entries.get(key.index)?;
        if e.generation != key.generation {
            return None;
        }
        e.object.as_ref()
    }

    fn remove(&mut self, key: Key) -> Option<Object> {
        let e = self.entries.get_mut(key.index)?;
        if e.generation != key.generation {
            return None;
        }
        let object = e.object.take()?;
        // Every key to this slot, including the one just used, now names an occupant that is gone.
        // Not a wrapping add: coming back around is the one thing that would make a stale key live
        // again, so it is left to overflow loudly rather than quietly, and 64 bits puts it out of
        // reach of anything a guest can send.
        e.generation += 1;
        self.free.push(key.index);
        Some(object)
    }

    /// Remove an object and everything created under it, however deep.
    ///
    /// Vulkan destroys a device's objects with the device and a pool's contents with the pool,
    /// naming none of them. Walking the arena is what makes that a fact about the one place
    /// objects live, rather than a child list kept beside them that a destroy path has to
    /// remember to visit -- the list that goes stale is the bug this replaces.
    fn take_tree(&mut self, key: Key) -> Vec<Doomed> {
        let Some(root) = self.remove(key) else {
            return Vec::new();
        };
        // The device an object has to be destroyed *on* is not its parent -- a fence's parent is
        // the device, but a framebuffer three levels under an instance still needs the device
        // handle from further up. So the walk carries it down: passing a device sets it for
        // everything below, and everything else inherits what it was handed.
        // The `ty` check is what makes the conversion honest: this is the one place that knows
        // the handle is a device's, so it is the one place allowed to say so.
        let under = |o: &Object, inherited| match o.ty {
            VkObjectType::VK_OBJECT_TYPE_DEVICE => Some(VkDevice::from_host(o.handle)),
            _ => inherited,
        };
        let mut taken =
            vec![Doomed { id: root.id, ty: root.ty, handle: root.handle, device: None }];
        let mut walk = vec![(key, under(&root, None))];
        while let Some((parent, device)) = walk.pop() {
            let children: Vec<Key> = self
                .entries
                .iter()
                .enumerate()
                .filter(|(_, e)| e.object.as_ref().is_some_and(|o| o.parent == Some(parent)))
                .map(|(i, e)| Key { index: i, generation: e.generation })
                .collect();
            for child in children {
                if let Some(o) = self.remove(child) {
                    walk.push((child, under(&o, device)));
                    taken.push(Doomed { id: o.id, ty: o.ty, handle: o.handle, device });
                }
            }
        }
        taken
    }

    fn remove_tree(&mut self, key: Key) -> Option<Object> {
        let object = self.get(key).copied()?;
        // Dropping the descent's result would strand host handles anywhere else; here it cannot,
        // because the only caller is `Table::remove`, which serves a leaf destroy the guest asked
        // for by name. A parent's destroy never comes this way -- it goes through
        // `Table::take_tree`, whose `#[must_use]` hands the children to the driver.
        let _ = self.take_tree(key);
        Some(object)
    }

    fn len(&self) -> usize {
        self.entries.iter().filter(|e| e.object.is_some()).count()
    }
}

/// Why an insert was refused. Each is a guest that broke the protocol, not a host mistake.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AddError {
    /// Id zero is not an object. The C asserts on it; here the guest picked it, so it is rejected.
    ZeroId,
    /// The guest reused a live id. Accepting it would orphan the object already under it.
    Duplicate,
}

/// What one id names. An id names exactly one of these, which is the point of the enum: a live
/// object and a refused creation are mutually exclusive states, and keeping them in two containers
/// meant nothing stopped an id from being written into both.
///
/// It happened along one real path -- a guest naming a live id in a create the driver refused
/// ghosted an object that was still there. Nothing went wrong while both held, because the lookup
/// consulted the live side first; the damage was deferred to the destroy, after which the stale
/// ghost went on answering for an id that named nothing, turning every later command naming it into
/// a silent drop where the ring should have stopped.
#[cfg_attr(test, derive(Clone))]
enum Slot {
    Live(Key),
    /// The host refused to create this id. A guest pipelines: it sends the create and the commands
    /// using it without waiting for an answer, so those later commands are already in flight when
    /// the create fails. Remembering the id turns each of them into one lost command instead of a
    /// poisoned ring -- the guest's own error handling then unwinds, as it would on real hardware.
    Ghost,
    /// No handler ran, so there is no host object -- only the guest's id, which stands in for a
    /// handle so that the rest of the stream still decodes. See `context`'s `object_created`.
    ///
    /// It is a slot and never an arena entry, which is the whole point: **the arena holds host
    /// handles and nothing else**, so a number the guest invented cannot reach a driver entry
    /// point by any path. It reached one before this existed -- the fiction went in as an ordinary
    /// object, and a context teardown handed it to `vkDestroyPipeline` as if the driver had
    /// produced it, which segfaults the host on a guest's say-so.
    ///
    /// `under` is what keeps it from outliving its world: it resolves only while the object it was
    /// created beneath is still there, so a device's destroy invalidates every fiction below it
    /// the same way it invalidates every real key, with nothing to remember to purge.
    Fiction {
        ty: VkObjectType,
        under: Option<Key>,
    },
}

impl Slot {
    fn key(&self) -> Option<Key> {
        match self {
            Slot::Live(k) => Some(*k),
            Slot::Ghost | Slot::Fiction { .. } => None,
        }
    }
}

/// One object on its way out, and the device whose entry points can destroy it.
///
/// **Every handle here came from the driver.** These are made only out of arena entries, and the
/// arena is only ever written with what a `vkCreateX` returned -- so an id no handler decided (see
/// [`Slot::Fiction`]) cannot appear, and the destroy that consumes this cannot be handed a number
/// the guest invented.
///
/// Not an [`Object`]: an object in the table knows its parent, which for anything below a device
/// is not the device itself. What a destroy needs is the `VkDevice` to call *on*, and that is
/// worked out once by the walk that takes the tree apart rather than by each caller guessing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Doomed {
    /// The name the rest of the context knows it by, for the records keyed on the guest's id
    /// rather than on a host handle -- the memory census is the one that matters.
    pub id: ObjectId,
    pub ty: VkObjectType,
    pub handle: HostHandle,
    /// `None` for the instance, its physical devices, and the devices themselves -- none of which
    /// is destroyed by a device's entry points.
    pub device: Option<VkDevice>,
}

#[derive(Default)]
#[cfg_attr(test, derive(Clone))]
pub struct Table {
    /// What the guest calls each object. Entries here are allowed to go stale: a key whose object
    /// the arena has dropped resolves to nothing, so a parent's destroy does not have to come back
    /// and tidy this map. That is the point of the whole shape -- the bookkeeping that could be
    /// forgotten no longer exists.
    slots: BTreeMap<ObjectId, Slot>,
    arena: Arena,
    /// What has been created since the journal last looked, so a retained command can be keyed on
    /// the objects it made. Only additions: a destroy needs no counterpart here, because a record
    /// holds an [`ObjectKey`] and a key to a destroyed object already resolves to nothing.
    added: Vec<ObjectKey>,
    /// What has stopped existing since the journal last looked.
    ///
    /// The counterpart of `added`, and needed for one reason only: a command that *frees* objects
    /// has to be journaled against the creates it undoes, and by the time the recorder sees it the
    /// ids it named resolve to nothing. Everything else about a destroy still needs no bookkeeping
    /// -- the entries describing those objects stop being true on their own.
    removed: Vec<ObjectKey>,
}

impl Table {
    pub fn new() -> Table {
        Table::default()
    }

    /// Register a host handle under the id the guest chose, beneath the object that owns it.
    ///
    /// `owner` is the guest's id for the parent -- the device for most objects, the physical
    /// device for a device, `None` for the instance. An owner that no longer resolves leaves the
    /// object parentless rather than refusing it: the id came off the wire, so it is the guest's
    /// to get wrong, and the create it belongs to has already happened.
    ///
    /// Parentless means `None`, and never a key to something dead. An owner's id can still hold
    /// the key a cascade left behind, so the owner is resolved through [`Table::key_of`], which
    /// answers only for a live object -- that is what keeps every live object's parent live, and
    /// so every object reachable from a root.
    pub fn add(
        &mut self,
        id: ObjectId,
        ty: VkObjectType,
        handle: HostHandle,
        owner: Option<ObjectId>,
    ) -> Result<(), AddError> {
        if id.0 == 0 {
            return Err(AddError::ZeroId);
        }
        if self.get(id).is_some() {
            return Err(AddError::Duplicate);
        }
        let parent = owner.and_then(|o| self.key_of(o)).map(|k| k.0);
        let key = self.arena.insert(Object { id, ty, handle, parent });
        self.added.push(ObjectKey(key));
        // An id the host once refused can be created for real later, and so can one whose object
        // died with its parent: both leave an entry here that resolves to nothing, and both are
        // overwritten rather than left beside the new object to swallow its commands.
        self.slots.insert(id, Slot::Live(key));
        Ok(())
    }

    /// Whether the host refused this id, so a later registration must not resurrect it.
    pub fn is_ghost(&self, id: ObjectId) -> bool {
        matches!(self.slots.get(&id), Some(Slot::Ghost))
    }

    /// Record that the host refused to create this id. See [`Slot::Ghost`].
    ///
    /// An id that already names a live object keeps it. A guest may name a live id in a create,
    /// and when the driver refuses that create nothing has changed -- the object still there is
    /// the truth, and a ghost written over it would outlive the object's own destroy.
    pub fn add_ghost(&mut self, id: ObjectId) {
        if id.0 == 0 || self.get(id).is_some() {
            return;
        }
        self.slots.insert(id, Slot::Ghost);
    }

    /// Record an id no handler decided: the unserved command's fiction. See [`Slot::Fiction`].
    ///
    /// An id that already names something keeps it, for [`Table::add_ghost`]'s reason -- a
    /// decision already made is not overwritten by the absence of one.
    ///
    /// An owner that is named and not live records nothing. A fiction stands only while its
    /// world does, and this one's is already gone: recorded parentless, it would stand forever
    /// instead. That differs from [`Table::add`] on purpose -- a real object exists whatever the
    /// guest said about its owner and must stay reachable for its destroy, while a fiction has
    /// nothing behind it and its commands can go unresolved.
    pub fn add_fiction(&mut self, id: ObjectId, ty: VkObjectType, owner: Option<ObjectId>) {
        if id.0 == 0 || self.get(id).is_some() || self.is_ghost(id) {
            return;
        }
        let under = match owner.map(|o| self.key_of(o)) {
            None => None,
            Some(Some(k)) => Some(k.0),
            Some(None) => return,
        };
        self.slots.insert(id, Slot::Fiction { ty, under });
    }

    /// Whether this id is a fiction that still stands -- one whose parent is still here.
    pub fn is_fiction(&self, id: ObjectId) -> bool {
        matches!(self.slots.get(&id), Some(Slot::Fiction { under, .. }) if self.fiction_stands(*under))
    }

    /// A fiction under a parent the arena has dropped names nothing, exactly as a stale key does.
    fn fiction_stands(&self, under: Option<Key>) -> bool {
        under.is_none_or(|k| self.arena.get(k).is_some())
    }

    /// Take an object and everything under it, root first, and hand back every one.
    ///
    /// [`Table::remove`] is the same cascade with the descendants dropped, which is right for a
    /// leaf -- its handler destroyed the one host handle there was. A parent's children were never
    /// named by any command, so nobody else has destroyed them and nobody else can: they arrive
    /// here or they leak. That is why this returns them and why the result may not be discarded.
    #[must_use = "these host handles are still alive, and this is the only place that names them"]
    pub fn take_tree(&mut self, id: ObjectId) -> Vec<Doomed> {
        let Some(key) = self.slots.get(&id).and_then(Slot::key) else {
            // A fiction is destroyed by forgetting it: there is no host handle to hand back, and
            // leaving the slot would have the id go on answering a create the guest makes next.
            if self.is_fiction(id) {
                self.slots.remove(&id);
            }
            return Vec::new();
        };
        let doomed = self.arena.take_tree(key);
        self.slots.remove(&id);
        doomed
    }

    /// Forget an object. Returns what was there, so the caller can destroy the host handle.
    ///
    /// Only a live object is taken out. A ghost outlives the destroy that names it: the guest
    /// pipelined that destroy behind the create that failed, and every command in between is still
    /// in flight behind it.
    pub fn remove(&mut self, id: ObjectId) -> Option<Object> {
        let Some(key) = self.slots.get(&id).and_then(Slot::key) else {
            // See `take_tree`: a fiction goes out of the table and hands nothing back.
            if self.is_fiction(id) {
                self.slots.remove(&id);
            }
            return None;
        };
        // Everything created under it goes at the same moment, because Vulkan has just destroyed
        // it all without naming any of it. Their ids stay in `slots` and stop resolving, which is
        // the cheap half of the trade: nothing has to walk back here and delete them.
        let object = self.arena.remove_tree(key)?;
        self.slots.remove(&id);
        self.removed.push(ObjectKey(key));
        Some(object)
    }

    /// The host `VkDevice` an id was created under, or `None` if nothing above it is a device.
    ///
    /// The same ancestry [`Arena::take_tree`] carries down as it destroys, asked without
    /// destroying anything -- so a caller that needs to know which device an object lives on has
    /// one answer to consult rather than a map of its own to keep in step.
    pub fn device_of(&self, id: ObjectId) -> Option<VkDevice> {
        let mut at = self.slots.get(&id)?.key()?;
        loop {
            let o = self.arena.get(at)?;
            if o.ty == VkObjectType::VK_OBJECT_TYPE_DEVICE {
                return Some(VkDevice::from_host(o.handle));
            }
            at = o.parent?;
        }
    }

    /// The id the guest gave a host handle, which is the reply direction of every lookup here.
    ///
    /// A handful of queries hand back handles *inside* a struct the driver filled --
    /// `vkEnumeratePhysicalDeviceGroups` is the one that reaches a guest -- and those are host
    /// handles that must never leave this process. There is no shadow to put them in, because the
    /// generator can only shadow a member and these are buried in an out-struct's fixed array,
    /// so the swap is the handler's and this is what it swaps through.
    ///
    /// A scan rather than a second map, deliberately. A reverse index would be a second container
    /// holding a fact this one already holds, and every destroy path would owe it an entry it
    /// could be forgotten at -- which is the shape of bug this table was rebuilt to remove. What
    /// it costs is a walk over live objects for a query a guest asks once or twice at startup.
    ///
    /// `ty` is part of the question, not a check on the answer: handles are only unique within a
    /// type, and Vulkan is free to give a physical device and a buffer the same number.
    pub fn id_of_handle(&self, ty: VkObjectType, handle: HostHandle) -> Option<ObjectId> {
        self.arena
            .entries
            .iter()
            .filter_map(|e| e.object.as_ref())
            .find(|o| o.ty == ty && o.handle == handle)
            .map(|o| o.id)
    }

    /// Every live object of one type, in the guest's own id order.
    ///
    /// The order is `slots`', which is sorted by id: an export that walked the arena would come
    /// out in whatever order objects happened to land in it, and two captures of the same world
    /// could then differ by nothing but their layout.
    pub fn of_type(&self, ty: VkObjectType) -> impl Iterator<Item = &Object> {
        self.slots
            .values()
            .filter_map(|s| s.key())
            .filter_map(|k| self.arena.get(k))
            .filter(move |o| o.ty == ty)
    }

    pub fn get(&self, id: ObjectId) -> Option<&Object> {
        self.arena.get(self.slots.get(&id)?.key()?)
    }

    /// The key of the object `id` names right now, for a holder that has to know later whether
    /// it is still *that* object. A key outlives nothing: once the object is destroyed it
    /// resolves to nothing, whatever the guest has since named by the same id.
    pub fn key_of(&self, id: ObjectId) -> Option<ObjectKey> {
        let key = self.slots.get(&id)?.key()?;
        self.arena.get(key).map(|_| ObjectKey(key))
    }

    /// Whether the object a key was taken for is still here.
    pub fn holds(&self, key: ObjectKey) -> bool {
        self.arena.get(key.0).is_some()
    }

    /// Take what has been created since the last call, and start counting again.
    ///
    /// Called once per dispatched command, so what comes back is what that command created --
    /// which is how a retained command finds the objects to hang itself on without any handler
    /// having to say. A command that created nothing hands back nothing, and is not a create.
    pub fn take_added(&mut self) -> Vec<ObjectKey> {
        std::mem::take(&mut self.added)
    }

    /// Take what has stopped existing since the last call, and start counting again.
    ///
    /// Drained per command alongside [`Table::take_added`], for the same reason: left behind, one
    /// command's removals would be attributed to the next.
    pub fn take_removed(&mut self) -> Vec<ObjectKey> {
        std::mem::take(&mut self.removed)
    }

    pub fn len(&self) -> usize {
        self.arena.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Empty the table, handing back every live object with the device it must be destroyed on.
    ///
    /// The teardown that actually runs. A context usually dies mid-workload with the guest still
    /// holding everything it made -- it sent no destroy for any of it and never will -- so this is
    /// not a tidy-up after the guest's own teardown but the only one there is.
    ///
    /// Taken root by root rather than slot by slot, because descending from a root is what works
    /// out which device each object belongs to, and destroying one on the wrong device is worse
    /// than leaking it.
    #[must_use = "these host handles are still alive, and this is the only place that names them"]
    pub fn take_all(&mut self) -> Vec<Doomed> {
        let roots: Vec<ObjectId> = self
            .slots
            .iter()
            .filter(|(_, slot)| {
                slot.key().and_then(|k| self.arena.get(k)).is_some_and(|o| o.parent.is_none())
            })
            .map(|(id, _)| *id)
            .collect();
        let doomed: Vec<Doomed> = roots.into_iter().flat_map(|id| self.take_tree(id)).collect();
        // Every live object's parent is live -- `add` takes only a live owner, and a destroy takes
        // the whole tree beneath what it names -- so the descents above reached everything.
        assert!(self.arena.len() == 0, "an object no root reaches survived a teardown");
        self.slots.clear();
        doomed
    }
}

impl Objects for Table {
    fn lookup(&self, id: ObjectId, ty: i32) -> Lookup {
        match self.slots.get(&id) {
            // A key the arena no longer honours is an object that died with its parent. The id is
            // as good as one the guest invented, and is answered the same way.
            Some(Slot::Live(key)) => match self.arena.get(*key) {
                // Compared as the integer the generated decode passes: the wire's number is the
                // guest's to choose, so it is not turned into a `VkObjectType` before it has been
                // matched against one that came from us.
                Some(o) if o.ty.0 == ty => Lookup::Found(o.handle),
                // A live id named by the wrong type is a guest reinterpreting one object as
                // another. That is not a race it can lose; it is a protocol violation, and the
                // ring stops.
                Some(_) => Lookup::Missing,
                None => Lookup::Missing,
            },
            Some(Slot::Ghost) => Lookup::Ghost,
            // The fiction, answered as the id itself. Whether the type matches is checked exactly
            // as it is for a real object: an unserved create does not buy the guest the right to
            // name its id as something else.
            Some(Slot::Fiction { ty: t, under }) if t.0 == ty && self.fiction_stands(*under) => {
                Lookup::Fiction
            }
            Some(Slot::Fiction { .. }) => Lookup::Missing,
            None => Lookup::Missing,
        }
    }

    fn id_of(&self, ty: i32, host: HostHandle) -> Option<ObjectId> {
        self.id_of_handle(VkObjectType(ty), host)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BUFFER: VkObjectType = VkObjectType::VK_OBJECT_TYPE_BUFFER;
    const IMAGE: VkObjectType = VkObjectType::VK_OBJECT_TYPE_IMAGE;
    const DEVICE: VkObjectType = VkObjectType::VK_OBJECT_TYPE_DEVICE;

    /// The two names of one object are two types, and the table keeps both.
    ///
    /// The pair was always here -- [`ObjectId`] has said so in its own doc since it was written --
    /// but only the guest's half had a type. The host's was a bare `u64` from the table through
    /// the driver's maps to the `Handle` trait, whose `raw()` returned the guest id at some call
    /// sites and the host handle at others, and reading a slot the wrong way compiled.
    ///
    /// A handle newtype still holds one word and both readings still see it. What they cannot do
    /// is reach a sink meant for the other: `add_ghost` takes an id, `from_host` takes a handle,
    /// and the three sabotages that swap them no longer build.
    #[test]
    fn an_object_is_known_by_two_names_of_two_types() {
        use super::super::cs::Handle;
        use super::super::proto::types::VkBuffer;

        let slot = VkBuffer::forged(0x1234);
        assert_eq!(slot.host(), HostHandle::forged(0x1234));
        assert_eq!(slot.guest_id(), ObjectId(0x1234));
        assert_eq!(VkBuffer::from_host(HostHandle::forged(0x1234)), slot);
        assert_eq!(VkBuffer::null(), VkBuffer::forged(0));

        // What the table records is the host's name, under the guest's.
        let mut t = Table::new();
        t.add(ObjectId(7), BUFFER, HostHandle::forged(0xfeed), None).unwrap();
        assert_eq!(t.lookup(ObjectId(7), BUFFER.0), Lookup::Found(HostHandle::forged(0xfeed)));
        assert_eq!(t.id_of_handle(BUFFER, HostHandle::forged(0xfeed)), Some(ObjectId(7)));
    }

    #[test]
    fn a_registered_object_resolves_only_under_its_own_type() {
        let mut t = Table::new();
        t.add(ObjectId(7), BUFFER, HostHandle::forged(0xdead_beef), None).unwrap();
        assert_eq!(t.lookup(ObjectId(7), BUFFER.0), Lookup::Found(HostHandle::forged(0xdead_beef)));
        // The whole point of the table: an id is not a capability for every object type.
        assert_eq!(t.lookup(ObjectId(7), IMAGE.0), Lookup::Missing);
    }

    #[test]
    fn an_id_the_guest_invented_stops_the_ring() {
        let t = Table::new();
        assert_eq!(t.lookup(ObjectId(7), BUFFER.0), Lookup::Missing);
    }

    /// A ghost is the one miss that is not the guest's fault: it pipelined commands behind a create
    /// the host refused, and they were already in flight when it failed.
    #[test]
    fn a_refused_creation_becomes_a_ghost_and_then_stops_being_one() {
        let mut t = Table::new();
        t.add_ghost(ObjectId(7));
        assert_eq!(t.lookup(ObjectId(7), BUFFER.0), Lookup::Ghost);

        t.add(ObjectId(7), BUFFER, HostHandle::forged(1), None).unwrap();
        assert_eq!(t.lookup(ObjectId(7), BUFFER.0), Lookup::Found(HostHandle::forged(1)));
    }

    /// The state the two-container shape allowed and this one cannot represent.
    ///
    /// Neither corpus does this -- measured, over every context in both: no id was ever added
    /// while ghosted, and no live id was ever also a ghost. Which is exactly why it needs a test:
    /// a sequence the recordings do not contain is a sequence only the types can rule out.
    #[test]
    fn a_refused_create_never_ghosts_an_id_that_already_names_something() {
        let mut t = Table::new();
        t.add(ObjectId(7), BUFFER, HostHandle::forged(1), None).unwrap();

        // A guest naming a live id in a create, and a driver that refuses that create. Nothing
        // changed, so the object still there is the truth.
        t.add_ghost(ObjectId(7));
        assert!(!t.is_ghost(ObjectId(7)));
        assert_eq!(t.lookup(ObjectId(7), BUFFER.0), Lookup::Found(HostHandle::forged(1)));

        // And after the guest destroys it the id names nothing at all. A ghost surviving here is
        // the whole bug: every later command naming the id would be swallowed as one lost command
        // when the ring should have stopped.
        t.remove(ObjectId(7)).unwrap();
        assert_eq!(t.lookup(ObjectId(7), BUFFER.0), Lookup::Missing);
    }

    /// The other direction, which must keep working: a ghost is not something a destroy clears.
    #[test]
    fn a_ghost_outlives_the_destroy_the_guest_pipelined_behind_it() {
        let mut t = Table::new();
        t.add_ghost(ObjectId(7));
        // The guest sent create, use, use, destroy without waiting for the create's answer. The
        // destroy names nothing, and the uses still in flight behind it are still the guest's to
        // unwind from.
        assert!(t.remove(ObjectId(7)).is_none());
        assert_eq!(t.lookup(ObjectId(7), BUFFER.0), Lookup::Ghost);
    }

    #[test]
    fn an_id_may_not_be_zero_or_reused_while_it_is_live() {
        let mut t = Table::new();
        assert_eq!(t.add(ObjectId(0), BUFFER, HostHandle::forged(1), None), Err(AddError::ZeroId));
        t.add(ObjectId(7), BUFFER, HostHandle::forged(1), None).unwrap();
        assert_eq!(
            t.add(ObjectId(7), IMAGE, HostHandle::forged(2), None),
            Err(AddError::Duplicate)
        );
        // Still the original: a refused insert must not have disturbed it.
        assert_eq!(t.lookup(ObjectId(7), BUFFER.0), Lookup::Found(HostHandle::forged(1)));

        t.remove(ObjectId(7)).unwrap();
        t.add(ObjectId(7), IMAGE, HostHandle::forged(2), None).unwrap();
        assert_eq!(t.lookup(ObjectId(7), IMAGE.0), Lookup::Found(HostHandle::forged(2)));
    }

    /// What the generation is actually for.
    ///
    /// An id the guest destroys by name loses its entry here, so it would answer `Missing`
    /// whatever the arena did. The ids that keep their entry are the ones a *cascade* orphaned --
    /// and the slots that cascade freed are the first ones handed to the next create. Without the
    /// generation beside the index, an orphan's key would come to name whatever moved in.
    #[test]
    fn an_orphans_stale_key_does_not_come_to_name_whatever_reuses_its_slot() {
        let mut t = Table::new();
        t.add(ObjectId(1), BUFFER, HostHandle::forged(10), None).unwrap();
        t.add(ObjectId(2), IMAGE, HostHandle::forged(20), Some(ObjectId(1))).unwrap();
        // Takes the child with it, and hands both their places back to be used again. Id 2 keeps
        // its entry: nothing walked back here to delete it, which is the whole trade.
        t.remove(ObjectId(1)).unwrap();

        let newcomer = t.add(ObjectId(3), IMAGE, HostHandle::forged(30), None);
        assert_eq!(newcomer, Ok(()));
        assert_eq!(t.lookup(ObjectId(3), IMAGE.0), Lookup::Found(HostHandle::forged(30)));
        // Same slot, same object type, different occupant. The generation is the only thing
        // separating them, and a guest naming id 2 must not be handed 30.
        assert_eq!(t.lookup(ObjectId(2), IMAGE.0), Lookup::Missing);
    }

    /// Destroying a parent destroys what hangs off it, however deep -- Vulkan does exactly this
    /// and names none of it.
    #[test]
    fn destroying_a_parent_takes_every_generation_below_it() {
        let mut t = Table::new();
        t.add(ObjectId(1), BUFFER, HostHandle::forged(10), None).unwrap();
        t.add(ObjectId(2), IMAGE, HostHandle::forged(20), Some(ObjectId(1))).unwrap();
        t.add(ObjectId(3), IMAGE, HostHandle::forged(30), Some(ObjectId(2))).unwrap();
        // A sibling tree that must survive: a cascade that took everything would pass a test that
        // only looked down one branch.
        t.add(ObjectId(4), BUFFER, HostHandle::forged(40), None).unwrap();
        t.add(ObjectId(5), IMAGE, HostHandle::forged(50), Some(ObjectId(4))).unwrap();

        t.remove(ObjectId(1)).unwrap();
        for id in [1, 2, 3] {
            assert!(t.get(ObjectId(id)).is_none(), "id {id} died with the root above it");
        }
        assert_eq!(t.lookup(ObjectId(4), BUFFER.0), Lookup::Found(HostHandle::forged(40)));
        assert_eq!(t.lookup(ObjectId(5), IMAGE.0), Lookup::Found(HostHandle::forged(50)));
        assert_eq!(t.len(), 2);
    }

    /// The half that makes the cascade affordable: an id orphaned by its parent's destroy is left
    /// where it is, and stops resolving on its own. Nothing walks back to delete it, which is why
    /// there is nothing to forget to do -- and the id is still free to name something new.
    #[test]
    fn an_id_orphaned_by_its_parents_destroy_is_reusable_without_being_cleaned_up() {
        let mut t = Table::new();
        t.add(ObjectId(1), BUFFER, HostHandle::forged(10), None).unwrap();
        t.add(ObjectId(2), IMAGE, HostHandle::forged(20), Some(ObjectId(1))).unwrap();
        t.remove(ObjectId(1)).unwrap();

        assert_eq!(t.lookup(ObjectId(2), IMAGE.0), Lookup::Missing);
        // A destroy the guest pipelined behind the device's finds nothing left to destroy, and
        // must not report one -- the host handle is already gone.
        assert!(t.remove(ObjectId(2)).is_none());
        // And the id is not burned: the guest may name a new object by it.
        t.add(ObjectId(2), BUFFER, HostHandle::forged(21), None).unwrap();
        assert_eq!(t.lookup(ObjectId(2), BUFFER.0), Lookup::Found(HostHandle::forged(21)));
    }

    /// A slot cycled many times still does not hand a stale key back its object.
    ///
    /// The generation is a counter a guest advances for free: one create and one destroy move it
    /// on by one, and it can do that for as long as it likes. What must never happen is the
    /// counter coming back around to a value some orphaned id is still holding, which is why it is
    /// 64 bits wide rather than 32. This walks a slot far enough to show the mechanism is the
    /// counter and not luck; the width is what puts the wrap out of reach.
    #[test]
    fn a_slot_cycled_over_and_over_never_hands_an_orphan_its_object_back() {
        let mut t = Table::new();
        t.add(ObjectId(1), BUFFER, HostHandle::forged(10), None).unwrap();
        t.add(ObjectId(2), IMAGE, HostHandle::forged(20), Some(ObjectId(1))).unwrap();
        t.remove(ObjectId(1)).unwrap();
        // Id 2 is now an orphan holding a key to a slot that is back in circulation.

        for round in 0..500u64 {
            t.add(ObjectId(3), IMAGE, HostHandle::forged(1000 + round), None).unwrap();
            assert_eq!(
                t.lookup(ObjectId(2), IMAGE.0),
                Lookup::Missing,
                "the orphan resolved again on round {round}"
            );
            t.remove(ObjectId(3)).unwrap();
        }
    }

    /// An owner the guest names that no longer resolves is the guest's mistake, not a reason to
    /// refuse a create the host has already served. The object lands parentless.
    #[test]
    fn a_create_under_an_owner_that_is_already_gone_still_registers() {
        let mut t = Table::new();
        t.add(ObjectId(9), BUFFER, HostHandle::forged(90), Some(ObjectId(1))).unwrap();
        assert_eq!(t.lookup(ObjectId(9), BUFFER.0), Lookup::Found(HostHandle::forged(90)));
    }

    #[test]
    fn draining_hands_back_every_handle_that_needs_destroying() {
        let mut t = Table::new();
        t.add(ObjectId(1), BUFFER, HostHandle::forged(10), None).unwrap();
        t.add(ObjectId(2), IMAGE, HostHandle::forged(20), None).unwrap();
        // One of them hangs off the other, so a drain that walked the tree would hand back only
        // the root. Teardown wants every live handle: Vulkan is being told about each one.
        t.add(ObjectId(3), IMAGE, HostHandle::forged(30), Some(ObjectId(1))).unwrap();
        let mut got: Vec<_> = t.take_all().into_iter().map(|d| (d.handle, d.device)).collect();
        got.sort_unstable();
        // Every live handle, and each with the device it has to be destroyed on: 30 hangs off the
        // buffer, which is not a device, so it inherits the nothing above it.
        assert_eq!(
            got,
            [
                (HostHandle::forged(10), None),
                (HostHandle::forged(20), None),
                (HostHandle::forged(30), None)
            ]
        );
        assert!(t.is_empty());
    }

    /// Teardown reaches an object several levels below the device that owns it, and the device it
    /// names has to be the one it was made on -- destroying an image on the wrong device is worse
    /// than leaking it. The device is worked out by descending, so the depth is the test: the
    /// image here is a grandchild, and nothing above the device is a device at all.
    #[test]
    fn a_drained_object_names_the_device_it_was_made_on_however_deep_it_sits() {
        let mut t = Table::new();
        t.add(ObjectId(1), BUFFER, HostHandle::forged(10), None).unwrap(); // stands in for the instance
        t.add(ObjectId(2), DEVICE, HostHandle::forged(20), Some(ObjectId(1))).unwrap();
        t.add(ObjectId(3), IMAGE, HostHandle::forged(30), Some(ObjectId(2))).unwrap();
        t.add(ObjectId(4), IMAGE, HostHandle::forged(40), Some(ObjectId(3))).unwrap();
        let mut got: Vec<_> = t.take_all().into_iter().map(|d| (d.handle, d.device)).collect();
        got.sort_unstable();
        // The root and the device itself are destroyed as themselves, not on a device; everything
        // under the device, at any depth, carries the device's handle down with it.
        assert_eq!(
            got,
            [
                (HostHandle::forged(10), None),
                (HostHandle::forged(20), None),
                (HostHandle::forged(30), Some(VkDevice::forged(20))),
                (HostHandle::forged(40), Some(VkDevice::forged(20))),
            ]
        );
    }
}

/// Every sequence of table operations up to [`every_sequence::DEPTH`] long, each step checked
/// against what the table promises.
///
/// The tests above each walk one sequence, most of them the one a bug was found on. This walks all
/// of them, over three ids and three types -- few enough that ids collide: a create under a live
/// id, a refusal over one, a slot handed out again, which is where a lifetime bug lives. The
/// domains are that small on purpose, so enumerating them is exhaustive within the bound rather
/// than a sample of it. Three ids is also the fewest a tree two levels deep takes -- an instance,
/// a device under it, an object under that -- so none is spent on id zero, whose refusal has a
/// test of its own.
///
/// The operations are the ones the context makes, under the constraints it makes them with. An
/// object's owner is the first handle its create names, so it is an instance for a device and a
/// device for anything else, and the decoder has already refused a create whose owner resolves to
/// the wrong type. An instance or a device is destroyed through `take_tree`, everything else
/// through `remove`. The handle a create mints is `MINTED` plus its depth, above every id, so a
/// fiction's id reaching a destroy list cannot pass for a driver handle.
#[cfg(test)]
mod every_sequence {
    use super::*;

    const DEPTH: usize = 4;
    const IDS: [ObjectId; 3] = [ObjectId(1), ObjectId(2), ObjectId(3)];
    const MINTED: u64 = 100;
    const INSTANCE: VkObjectType = VkObjectType::VK_OBJECT_TYPE_INSTANCE;
    const DEVICE: VkObjectType = VkObjectType::VK_OBJECT_TYPE_DEVICE;
    const BUFFER: VkObjectType = VkObjectType::VK_OBJECT_TYPE_BUFFER;
    const TYPES: [VkObjectType; 3] = [INSTANCE, DEVICE, BUFFER];

    #[derive(Clone, Copy, Debug)]
    enum Op {
        Add(ObjectId, VkObjectType, Option<ObjectId>),
        Ghost(ObjectId),
        Fiction(ObjectId, VkObjectType, Option<ObjectId>),
        Remove(ObjectId),
        TakeTree(ObjectId),
    }

    fn alphabet() -> Vec<Op> {
        let owners = [None, Some(IDS[0]), Some(IDS[1]), Some(IDS[2])];
        let mut ops = Vec::new();
        for id in IDS {
            for ty in TYPES {
                for owner in owners {
                    ops.push(Op::Add(id, ty, owner));
                    ops.push(Op::Fiction(id, ty, owner));
                }
            }
            ops.extend([Op::Ghost(id), Op::Remove(id), Op::TakeTree(id)]);
        }
        ops
    }

    fn ty_of(t: &Table, id: ObjectId) -> Option<VkObjectType> {
        t.get(id).map(|o| o.ty)
    }

    /// Whether the context could make `op` against `t`.
    fn admissible(t: &Table, op: Op) -> bool {
        let owned = |ty: VkObjectType, owner: Option<ObjectId>| match owner {
            None => true,
            Some(o) => {
                let parent = if ty == DEVICE { INSTANCE } else { DEVICE };
                ty != INSTANCE && ty_of(t, o).is_none_or(|p| p == parent)
            }
        };
        match op {
            Op::Add(_, ty, owner) | Op::Fiction(_, ty, owner) => owned(ty, owner),
            Op::Ghost(_) => true,
            Op::Remove(id) => ty_of(t, id).is_none_or(|ty| ty == BUFFER),
            Op::TakeTree(id) => ty_of(t, id).is_none_or(|ty| ty != BUFFER),
        }
    }

    /// Every live object, by handle, with its key and the device a destroy of it must name.
    fn live(t: &Table) -> Vec<(HostHandle, Key, Option<VkDevice>)> {
        let mut out: Vec<_> = t
            .arena
            .entries
            .iter()
            .enumerate()
            .filter_map(|(index, e)| {
                let key = Key { index, generation: e.generation };
                e.object.map(|o| (o.handle, key, device_above(t, key)))
            })
            .collect();
        out.sort_unstable_by_key(|(h, ..)| *h);
        out
    }

    /// The nearest device strictly above `key`.
    fn device_above(t: &Table, key: Key) -> Option<VkDevice> {
        let mut at = t.arena.get(key)?.parent;
        while let Some(o) = at.and_then(|k| t.arena.get(k)) {
            if o.ty == DEVICE {
                return Some(VkDevice::from_host(o.handle));
            }
            at = o.parent;
        }
        None
    }

    /// Whether `key` is `root` or sits somewhere below it.
    fn in_tree(t: &Table, key: Key, root: Key) -> bool {
        let mut at = Some(key);
        while let Some(k) = at {
            if k == root {
                return true;
            }
            at = t.arena.get(k).and_then(|o| o.parent);
        }
        false
    }

    /// What the walk must have reached, or its passing says nothing about these cases.
    #[derive(Default, Debug)]
    struct Reached {
        nodes: u64,
        cascades: u64,
        /// Cascades that reached a grandchild: a device's objects under a destroyed instance.
        deep_cascades: u64,
        reused_under_stale_key: u64,
        duplicates: u64,
        fictions_resolved: u64,
        /// Fictions named under an owner that was not live.
        orphaned_fictions: u64,
    }

    /// Check a destroy list against the world before it: every handle `taken` names comes back
    /// exactly once and nothing else does, each on the device above it -- except `root`, the
    /// object the caller named, which is destroyed on no device.
    fn destroyed(
        before: &[(HostHandle, Key, Option<VkDevice>)],
        taken: &[(HostHandle, Key, Option<VkDevice>)],
        doomed: &[Doomed],
        root: Option<Key>,
        ops: &[Op],
    ) {
        let mut got: Vec<HostHandle> = doomed.iter().map(|d| d.handle).collect();
        got.sort_unstable();
        let want: Vec<HostHandle> = taken.iter().map(|(h, ..)| *h).collect();
        assert_eq!(got, want, "a destroy list is not exactly what it took, after {ops:?}");
        for d in doomed {
            assert!(d.handle.raw() >= MINTED, "a destroy list names a fiction, after {ops:?}");
            let (_, key, device) = before.iter().find(|(h, ..)| *h == d.handle).unwrap();
            let want = if Some(*key) == root { None } else { *device };
            assert_eq!(d.device, want, "destroyed on the wrong device, after {ops:?}");
        }
    }

    /// Apply `op` and check it did exactly what it promises. Returns the key a create took.
    fn apply(
        t: &mut Table,
        op: Op,
        h: HostHandle,
        ops: &[Op],
        seen: &mut Reached,
    ) -> Option<ObjectKey> {
        let before = live(t);
        let unchanged = |t: &Table| {
            let after: Vec<HostHandle> = live(t).iter().map(|(h, ..)| *h).collect();
            let was: Vec<HostHandle> = before.iter().map(|(h, ..)| *h).collect();
            assert_eq!(after, was, "the live set changed under {op:?}, after {ops:?}");
        };
        match op {
            Op::Add(id, ty, owner) => {
                let prior = t.get(id).copied();
                match t.add(id, ty, h, owner) {
                    Ok(()) => {
                        assert!(prior.is_none(), "a create replaced a live object, after {ops:?}");
                        let after: Vec<HostHandle> = live(t).iter().map(|(h, ..)| *h).collect();
                        let mut want: Vec<HostHandle> = before.iter().map(|(h, ..)| *h).collect();
                        want.push(h);
                        want.sort_unstable();
                        assert_eq!(
                            after, want,
                            "a create did not add exactly its handle, after {ops:?}"
                        );
                        return Some(t.key_of(id).expect("an added object has a key"));
                    }
                    Err(e) => {
                        seen.duplicates += 1;
                        assert_eq!(
                            e,
                            AddError::Duplicate,
                            "a create was refused for the wrong reason, after {ops:?}"
                        );
                        assert!(prior.is_some(), "a free id was refused, after {ops:?}");
                        assert_eq!(
                            t.get(id).copied(),
                            prior,
                            "a refused create moved {id:?}, after {ops:?}"
                        );
                        unchanged(t);
                    }
                }
            }
            Op::Ghost(id) | Op::Fiction(id, ..) => {
                let prior = t.get(id).copied();
                let lookups = |t: &Table| TYPES.map(|ty| t.lookup(id, ty.0));
                let resolved = lookups(t);
                let orphaned = matches!(op, Op::Fiction(_, _, Some(o)) if t.key_of(o).is_none());
                match op {
                    Op::Ghost(_) => t.add_ghost(id),
                    Op::Fiction(_, ty, owner) => t.add_fiction(id, ty, owner),
                    _ => unreachable!(),
                }
                assert_eq!(t.get(id).copied(), prior, "a refusal moved {id:?}, after {ops:?}");
                if orphaned {
                    seen.orphaned_fictions += 1;
                    assert_eq!(
                        lookups(t),
                        resolved,
                        "a fiction under a dead owner was recorded, after {ops:?}"
                    );
                }
                unchanged(t);
            }
            Op::Remove(id) => {
                let root = t.key_of(id).map(|k| k.0);
                let taken: Vec<_> =
                    before.iter().filter(|(_, k, _)| Some(*k) == root).copied().collect();
                let gone = t.remove(id);
                assert_eq!(
                    gone.map(|o| o.handle),
                    taken.first().map(|(h, ..)| *h),
                    "remove returned the wrong object, after {ops:?}"
                );
                let after: Vec<_> = live(t).iter().map(|(h, ..)| *h).collect();
                let want: Vec<_> =
                    before.iter().filter(|(_, k, _)| Some(*k) != root).map(|(h, ..)| *h).collect();
                assert_eq!(after, want, "a leaf destroy took more than its leaf, after {ops:?}");
            }
            Op::TakeTree(id) => {
                let root = t.key_of(id).map(|k| k.0);
                let taken: Vec<_> = before
                    .iter()
                    .filter(|(_, k, _)| root.is_some_and(|r| in_tree(t, *k, r)))
                    .copied()
                    .collect();
                let doomed = t.take_tree(id);
                if doomed.len() >= 2 {
                    seen.cascades += 1;
                }
                if doomed.len() >= 3 {
                    seen.deep_cascades += 1;
                }
                destroyed(&before, &taken, &doomed, root, ops);
                let after: Vec<_> = live(t).iter().map(|(h, ..)| *h).collect();
                let want: Vec<_> =
                    before.iter().filter(|b| !taken.contains(b)).map(|(h, ..)| *h).collect();
                assert_eq!(after, want, "a destroy took something outside its tree, after {ops:?}");
            }
        }
        None
    }

    /// What holds between any two operations, whatever came before.
    fn invariants(t: &Table, keys: &[(ObjectKey, HostHandle)], ops: &[Op], seen: &mut Reached) {
        // A live object is reachable by its own id and sits under a live parent, which is what
        // makes a cascade complete and a teardown reach everything. An empty slot is on the free
        // list exactly once.
        for (index, e) in t.arena.entries.iter().enumerate() {
            let free = t.arena.free.iter().filter(|&&f| f == index).count();
            match e.object {
                Some(o) => {
                    let key = Key { index, generation: e.generation };
                    assert_eq!(free, 0, "a live slot is on the free list, after {ops:?}");
                    assert!(
                        matches!(t.slots.get(&o.id), Some(Slot::Live(k)) if *k == key),
                        "a live object is not reachable by its own id, after {ops:?}"
                    );
                    assert!(
                        o.parent.is_none_or(|p| t.arena.get(p).is_some()),
                        "a live object outlived its parent, after {ops:?}"
                    );
                }
                None => assert_eq!(free, 1, "an empty slot is not free once, after {ops:?}"),
            }
        }
        // A key names the object it was taken for or nothing, whatever holds its slot now.
        for (k, h) in keys {
            if t.holds(*k) {
                let o = t.arena.get(k.0).unwrap();
                assert_eq!(o.handle, *h, "a stale key names a new object, after {ops:?}");
            } else if t.arena.entries[k.0.index].object.is_some() {
                seen.reused_under_stale_key += 1;
            }
        }
        // A lookup finds a live object under its own type, and a fiction only where no live
        // object stands -- as the guest's own id, never as a driver handle.
        for id in IDS {
            for ty in TYPES {
                let live = t.get(id).filter(|o| o.ty == ty).map(|o| o.handle);
                match t.lookup(id, ty.0) {
                    Lookup::Found(h) => {
                        assert_eq!(Some(h), live, "a lookup found a stranger, after {ops:?}")
                    }
                    Lookup::Fiction => {
                        assert!(
                            live.is_none() && t.get(id).is_none(),
                            "a fiction shadows a live object, after {ops:?}"
                        );
                        seen.fictions_resolved += 1;
                    }
                    _ => assert!(live.is_none(), "a live object does not resolve, after {ops:?}"),
                }
            }
        }
        // A teardown here would take everything, each handle once, on the device above it.
        let before = live(t);
        let mut down = t.clone();
        destroyed(&before, &before, &down.take_all(), None, ops);
        assert!(down.is_empty(), "a teardown left something behind, after {ops:?}");
    }

    fn walk(
        t: &Table,
        alphabet: &[Op],
        keys: &mut Vec<(ObjectKey, HostHandle)>,
        ops: &mut Vec<Op>,
        seen: &mut Reached,
    ) {
        seen.nodes += 1;
        invariants(t, keys, ops, seen);
        if ops.len() == DEPTH {
            return;
        }
        for &op in alphabet {
            if !admissible(t, op) {
                continue;
            }
            let mut next = t.clone();
            let h = HostHandle::forged(MINTED + ops.len() as u64);
            ops.push(op);
            let key = apply(&mut next, op, h, ops, seen);
            if let Some(k) = key {
                keys.push((k, h));
            }
            walk(&next, alphabet, keys, ops, seen);
            if key.is_some() {
                keys.pop();
            }
            ops.pop();
        }
    }

    #[test]
    #[cfg_attr(miri, ignore = "a walk of every sequence, which Miri would run for days")]
    fn every_sequence_keeps_every_promise() {
        let mut seen = Reached::default();
        walk(&Table::new(), &alphabet(), &mut Vec::new(), &mut Vec::new(), &mut seen);
        // Each of these is a case the assertions above are about; a walk that never reached one
        // would pass them vacuously.
        assert!(seen.cascades > 0, "no destroy took a child: {seen:?}");
        assert!(seen.deep_cascades > 0, "no destroy took a grandchild: {seen:?}");
        assert!(seen.reused_under_stale_key > 0, "no slot was reused under a stale key: {seen:?}");
        assert!(seen.duplicates > 0, "no create reused a live id: {seen:?}");
        assert!(seen.fictions_resolved > 0, "no fiction resolved: {seen:?}");
        assert!(seen.orphaned_fictions > 0, "no fiction was named under a dead owner: {seen:?}");
    }
}

/// The table as the decoder and the handlers both see it.
///
/// The decoder holds it while it reads a command, and the handler that command reaches writes to
/// it -- so it cannot be an exclusive borrow on either side. Decoding finishes before the handler
/// runs, so the two never overlap; the cell is what lets the compiler stop caring, and it will
/// panic loudly rather than quietly if that stops being true.
#[derive(Default)]
pub struct Shared(RefCell<Table>);

impl Shared {
    pub fn new() -> Shared {
        Shared::default()
    }

    pub fn borrow(&self) -> std::cell::Ref<'_, Table> {
        self.0.borrow()
    }

    pub fn borrow_mut(&self) -> std::cell::RefMut<'_, Table> {
        self.0.borrow_mut()
    }
}

impl Objects for Shared {
    fn lookup(&self, id: ObjectId, ty: i32) -> Lookup {
        self.0.borrow().lookup(id, ty)
    }

    fn id_of(&self, ty: i32, host: HostHandle) -> Option<ObjectId> {
        self.0.borrow().id_of(ty, host)
    }
}
