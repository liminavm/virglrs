// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! The venus command stream: the trust boundary.
//!
//! Every byte the guest sends the Vulkan renderer arrives here. The generated decoder in
//! [`super::proto`] is written against this module and nothing else, which is what makes the
//! generated code safe Rust: reads are bounds-checked against a slice, allocations come from a
//! capped arena, and a malformed stream sets a poison flag instead of panicking.
//!
//! Poison, not panic, is the rule at this boundary. An `assert!` is for a host invariant we got
//! wrong; a guest that sends a truncated command, an impossible array size or an unknown object id
//! is not that. The C side calls this `fatal` and the name is kept, but it is fatal to the ring,
//! never to the process.
//!
//! Two flavours of poison, mirroring `src/venus/vkr_cs.h`:
//!
//! * **hard** -- the stream is unusable. It is shared with the ring loop, which stops.
//! * **soft** -- this one command named an object whose creation the host refused, and which the
//!   guest had already pipelined commands behind (a ghost). The command is dropped, the ring keeps
//!   going. Generated code cannot tell the two apart, and must not: both mean "do not call the
//!   handler, do not encode a reply". An id the guest simply invented is *hard*, not soft.

use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicBool, Ordering};

use bumpalo::Bump;

/// Cap on the temp arena, matching `VKR_CS_DECODER_TEMP_POOL_MAX_SIZE`. It is not a resource limit
/// so much as an overflow guard: the guest encodes the array sizes, and a plausible one is dozens
/// of megabytes (`vkGetPipelineCacheData`), not a gigabyte.
pub const TEMP_POOL_MAX: usize = 1024 * 1024 * 1024;

/// Round a wire size up to the stream's 4-byte granularity.
#[inline]
pub const fn align4(size: usize) -> usize {
    (size + 3) & !3
}

/// The object id the guest uses to name a Vulkan object. Distinct from the host handle it resolves
/// to: the guest never sees a host handle, and confusing the two is the bug this newtype exists to
/// make unrepresentable.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
#[repr(transparent)]
pub struct ObjectId(pub u64);

/// The handle the host driver gave the object, which is what a Vulkan call is made with.
///
/// The other half of the pair [`ObjectId`] names, and untyped until long after it. Both are one
/// `u64` and they travel together everywhere -- the object table records the two side by side,
/// and the decoder replaces one with the other in place -- so the two readings of the same word
/// were told apart by nothing but the reader.
///
/// Zero is an ordinary value here, unlike a resource handle: Vulkan spells `VK_NULL_HANDLE` as
/// zero, a great many members are optional, and a create the driver refused leaves one behind on
/// purpose. So it is a plain `u64` and not a `NonZeroU64`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
#[repr(transparent)]
pub struct HostHandle(pub u64);

/// A host handle carried with the Vulkan type it is a handle *of*.
///
/// [`HostHandle`] is deliberately kind-erased, and for the object table that is right -- it is
/// keyed by a runtime `ty` because one table holds every type. A container that erases the kind
/// without recording it has a different problem: two Vulkan types are two separate handle spaces,
/// and a driver may hand out the same `u64` for a command pool and a descriptor pool. Keyed by the
/// bare handle they collide; keyed by this they cannot.
///
/// Built from [`Handle::OBJECT_TYPE`], so no call site chooses the tag and none can choose it
/// wrongly.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct TypedHandle {
    ty: i32,
    handle: HostHandle,
}

impl TypedHandle {
    pub fn of<T: Handle>(h: T) -> TypedHandle {
        TypedHandle { ty: T::OBJECT_TYPE, handle: h.host() }
    }

    pub fn host(self) -> HostHandle {
        self.handle
    }
}

/// Resolves guest object ids to host objects. The decoder holds one by reference; what sits behind
/// it is vkr's object table in the renderer and an identity map in the round-trip test.
/// A pointer stored in an arena array -- an array of strings is an array of these.
///
/// It exists because the standard library gives a raw pointer no `Default`, and the arena needs
/// one to hand back a fresh element. `repr(transparent)` is the point: `[Ptr]` and `[*const T]`
/// have the same layout, so the decoded array can be handed to Vulkan as itself.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct Ptr(pub *const core::ffi::c_void);

impl Default for Ptr {
    fn default() -> Self {
        Ptr(core::ptr::null())
    }
}

/// What the object table has to say about an id the guest named.
/// A Vulkan handle, as a value that can be moved between the wire's side and the host's.
///
/// Every handle newtype is one `u64`, but they are deliberately *not* interchangeable -- that is
/// the whole point of the newtypes. A handler still has to move a raw handle from the driver into
/// a typed shadow member, so the conversion is named here rather than done with a cast at every
/// call site, and the generator implements it for every handle type.
/// A handle slot holding the *guest's* id for an object, not a handle the host may be called with.
///
/// Every other handle slot in a decoded command holds the host handle: the decoder replaced the
/// guest's id at lookup, which is what lets the argument struct be handed to the driver unchanged.
/// A create's out-member is the exception -- the guest chooses the id there, the reply re-encodes
/// it, and the driver's answer goes to a shadow member beside it. The two readings were one type
/// and told apart by the reader.
///
/// So the accessor over an out-member hands back this, and it deliberately offers no way to reach
/// a [`HostHandle`]: passing one to the driver would be calling Vulkan with a number the guest
/// made up, and re-encoding a host handle in its place would hand the guest a live host pointer.
/// [`Guest::id`] is the whole interface, because the id is the whole content.
///
/// `repr(transparent)` so the accessor is a cast and the wire layout is untouched.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
#[repr(transparent)]
pub struct Guest<T>(T);

impl<T: Handle> Guest<T> {
    /// The id the guest asked this object to be known by.
    pub fn id(self) -> ObjectId {
        self.0.guest_id()
    }

    /// Name an id the way the decoder would have, for a test that plants one.
    #[cfg(test)]
    pub fn new(handle: T) -> Guest<T> {
        Guest(handle)
    }
}

/// A struct, or an array of them, exactly as the decoder built it in the batch arena.
///
/// A `Vk*` struct carries raw pointers -- its `pNext` chain and every array it names -- and the
/// driver hands it to Vulkan, which follows them. Its fields are all `pub` and it is `Copy`,
/// because it has C's layout and the driver assembles its own; so a bare `&VkBufferCreateInfo`
/// says nothing about where its pointers go, and a driver entry point taking one would be trusting
/// whoever made it. This says where they go: only the decoder mints one, and every driver entry
/// point that passes a guest's struct on to Vulkan takes this rather than a reference. A handler
/// can hand over what the guest sent, and nothing it made or changed -- copying the struct out
/// gives back a bare value no entry point will take.
///
/// `'a` is the arena's, so it cannot outlive the memory its pointers name either.
///
/// `repr(transparent)` over the reference, so a command member of type `Option<Decoded<..>>` keeps
/// the one pointer word C's struct has there.
#[repr(transparent)]
pub struct Decoded<'a, T: ?Sized>(&'a T);

impl<T: ?Sized> Clone for Decoded<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ?Sized> Copy for Decoded<'_, T> {}

impl<T: ?Sized> core::ops::Deref for Decoded<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.0
    }
}

impl<'a, T: ?Sized> Decoded<'a, T> {
    /// Vouch for `r` as something the decoder built.
    ///
    /// # Safety
    ///
    /// Every pointer reachable from `r` -- through `pNext`, through every array member and into
    /// the structs those name -- is null, or names memory that lives for `'a` and holds as many
    /// elements as the count beside it says. The decoder meets that by construction: it allocated
    /// every one of them from the arena `'a` borrows, sized by the count it decoded beside it.
    pub(crate) unsafe fn vouch(r: &'a T) -> Self {
        Decoded(r)
    }

    /// Plant a struct a test built, as though the decoder had.
    #[cfg(test)]
    pub fn planted(r: &'a T) -> Self {
        Decoded(r)
    }

    /// The reference, for reading. Reading is not what needs vouching for; handing a struct on is.
    pub fn get(self) -> &'a T {
        self.0
    }
}

impl<'a, T> Decoded<'a, [T]> {
    /// Each element, vouched for as the array was: an element of a decoded array is decoded.
    pub fn iter(self) -> impl ExactSizeIterator<Item = Decoded<'a, T>> + Clone {
        self.0.iter().map(Decoded)
    }

    /// The element at `i`, if there is one.
    pub fn at(self, i: usize) -> Option<Decoded<'a, T>> {
        self.0.get(i).map(Decoded)
    }
}

/// A struct with no pointer anywhere in it -- an extent, a region, a subresource. Generated.
///
/// There is nothing in one for Vulkan to follow, so there is nothing to vouch for, and any
/// reference to one is as good as a decoded one: see the `From` below.
///
/// # Safety
///
/// Implementing this asserts the type holds no pointer, directly or in any member it embeds. The
/// generator implements it from the same predicate that decides what [`Decoded`] guards.
pub unsafe trait Plain {}

impl<'a, T: Plain> From<&'a T> for Decoded<'a, T> {
    fn from(r: &'a T) -> Self {
        Decoded(r)
    }
}

impl<'a, T: Plain> From<&'a [T]> for Decoded<'a, [T]> {
    fn from(r: &'a [T]) -> Self {
        Decoded(r)
    }
}

impl<T> Default for Decoded<'_, [T]> {
    /// No elements, so no pointers to vouch for.
    fn default() -> Self {
        Decoded(&[])
    }
}

impl<T: ?Sized + core::fmt::Debug> core::fmt::Debug for Decoded<'_, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.fmt(f)
    }
}

/// A struct that heads a `pNext` chain, which is every Vulkan struct with an `sType`. Generated.
///
/// Reading the pointer is safe; following it is what needs the struct to be [`Decoded`].
pub trait Links {
    fn next(&self) -> *const core::ffi::c_void;
}

pub trait Handle: Copy {
    /// The `VkObjectType` discriminant of this handle's Vulkan type.
    ///
    /// An `i32` rather than the generated enum for the reason [`Objects::lookup`] takes one: this
    /// module is the wire runtime and the enum is generated on top of it, so the discriminant is
    /// the widest thing both sides can name. The generator fills it in from the same `c_objtype`
    /// attribute the lookup already used, so the two cannot drift.
    const OBJECT_TYPE: i32;

    /// The host handle in the slot, for a member the decoder has already resolved or the driver
    /// has just written.
    fn host(self) -> HostHandle;

    /// The guest id in the slot, for the out-member of a create -- the one place the guest, not
    /// the host, chooses what the word says.
    fn guest_id(self) -> ObjectId;

    /// Put a host handle in the slot.
    fn from_host(host: HostHandle) -> Self;

    /// `VK_NULL_HANDLE`, for an out-parameter before the driver has written it.
    fn null() -> Self;
}

/// A pool handle, and the one kind of object allocated from it.
///
/// Vulkan's pools each hold exactly one kind: a command pool holds command buffers, a descriptor
/// pool holds descriptor sets. Tying the two together in a type is what stops a command buffer
/// being filed under a descriptor pool -- a transposition the pool bookkeeping cannot otherwise
/// see, because both sides of it are handles.
///
/// The impls are generated from vk.xml's handle parentage, so a pool the renderer starts serving
/// arrives with one and a handle that is merely *named* a pool never does. It lives here rather
/// than beside the bookkeeping that uses it because the generated code can name this module and
/// not the driver.
pub trait PoolOf: Handle {
    type Child: Handle;
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Lookup {
    /// The host handle.
    Found(HostHandle),
    /// An id whose creation the host refused, and which the guest had already pipelined commands
    /// behind. Those commands are lost; the ring is not.
    Ghost,
    /// No such object, or one of a different Vulkan type. Either way the guest named something it
    /// was never given, which is a protocol violation and stops the ring.
    Missing,
}

pub trait Objects {
    /// Resolve `id`, which the wire claims is of `VkObjectType` `ty`. A live object under a
    /// different type is `Missing`, not `Found`: the guest does not get to reinterpret one object
    /// as another by naming its id in the wrong command.
    fn lookup(&self, id: ObjectId, ty: i32) -> Lookup;
}

/// An object table that resolves every id to itself. Used by the wire round-trip, where the point
/// is to reproduce the guest's bytes and no host object exists.
pub struct IdentityObjects;

impl Objects for IdentityObjects {
    fn lookup(&self, id: ObjectId, _ty: i32) -> Lookup {
        Lookup::Found(HostHandle(id.0))
    }
}

/// Reads a command stream.
///
/// The lifetime `'a` ties the decoded structures to both the guest bytes and the arena they were
/// allocated from, so a decoded command cannot outlive either. `temp` is a shared reference on
/// purpose: allocating must not borrow the decoder, because the generated code allocates an array
/// and keeps reading in the same expression.
pub struct Decoder<'a> {
    buf: &'a [u8],
    pos: usize,
    temp: &'a Bump,
    temp_used: Cell<usize>,
    objects: &'a dyn Objects,
    /// Shared with the ring loop, which stops when it is set.
    hard: &'a AtomicBool,
    /// The ghost this command named, if it named one. Per-command: the dispatch that skips
    /// the command takes it, and says which object it was.
    ghost: Cell<Option<ObjectId>>,
    /// Every object this command resolved, in the order it named them.
    ///
    /// The recorder's half of the journal: an entry has to know what its command referenced, and
    /// this is the one place every handle on the wire passes through. Collecting here rather than
    /// re-reading the arguments is the point -- a second parse would be a second opinion about a
    /// value already reconciled, and the two could disagree.
    ///
    /// Interior mutability for the same reason `ghost` has it: the generated decoders reach this
    /// holding only a shared borrow.
    resolved: RefCell<Vec<ObjectId>>,
}

/// What became of one command, as the generated dispatch reports it.
///
/// One verdict, read in one place. The loop that runs a batch reads nothing else off the
/// decoder to learn this: a verdict inferred from a side channel -- a poison flag, a reply's
/// byte count -- is one that can be inferred wrongly, and "skipped" read off a byte count is
/// indistinguishable from "answered with nothing".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dispatched {
    /// The handler ran; when a reply was asked for, it is in the encoder.
    Served,
    /// The arguments did not decode. The stream is poisoned and no handler ran.
    Undecodable,
    /// The command named this ghost -- an object whose create the host refused -- and was skipped
    /// whole, as containment asks: no handler ran and nothing was encoded.
    Ghosted(ObjectId),
    /// Not a command type this protocol defines. It cannot even be skipped: its length is only
    /// knowable by decoding it.
    Undefined,
}

impl<'a> Decoder<'a> {
    pub fn new(
        buf: &'a [u8],
        temp: &'a Bump,
        objects: &'a dyn Objects,
        hard: &'a AtomicBool,
    ) -> Self {
        Decoder {
            buf,
            pos: 0,
            temp,
            temp_used: Cell::new(0),
            objects,
            hard,
            ghost: Cell::new(None),
            resolved: RefCell::new(Vec::new()),
        }
    }

    /// Take the objects this command named, and start collecting again.
    ///
    /// Drained unconditionally after every dispatch, whatever the verdict. A command that ghosted
    /// or was rejected still resolved ids on its way there, and leaving them would hand them to
    /// whichever command is recorded next -- an entry claiming to reference objects it never
    /// mentioned, which the export would then drag creates in for.
    pub fn take_resolved(&self) -> Vec<ObjectId> {
        std::mem::take(&mut self.resolved.borrow_mut())
    }

    /// Bytes consumed so far. The recorder stamps a command as the slice between the position
    /// before and after its dispatch, so this is what the round-trip compares against.
    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn has_command(&self) -> bool {
        self.pos < self.buf.len()
    }

    /// Poison the stream. Takes `&self` because the generated code reaches it from paths that only
    /// hold a shared borrow, exactly as the C does.
    pub fn set_fatal(&self) {
        self.hard.store(true, Ordering::Release);
    }

    /// Whether this command can go no further: the stream is poisoned, or the command named a
    /// ghost and is to be skipped. The generated dispatch asks this and then [`verdict`] for
    /// which; a test asks it alone.
    ///
    /// [`verdict`]: Decoder::verdict
    pub fn fatal(&self) -> bool {
        self.hard.load(Ordering::Acquire) || self.ghost.get().is_some()
    }

    /// Why the command cannot run, taking the ghost with it so it does not outlive the command.
    /// A stream poison outranks a ghost: a command that both named a ghost and failed to decode
    /// is a stream we cannot follow, whatever it named.
    pub fn verdict(&self) -> Dispatched {
        let ghost = self.ghost.take();
        if self.hard.load(Ordering::Acquire) {
            Dispatched::Undecodable
        } else if let Some(id) = ghost {
            Dispatched::Ghosted(id)
        } else {
            Dispatched::Served
        }
    }

    /// What the ring loop asks: is the stream itself unusable?
    pub fn hard_fatal(&self) -> bool {
        self.hard.load(Ordering::Acquire)
    }

    /// The next `n` bytes without consuming them, or `None` (having poisoned the stream) if the
    /// stream is shorter than that.
    pub fn peek_bytes(&self, n: usize) -> Option<&'a [u8]> {
        match self.pos.checked_add(n).and_then(|end| self.buf.get(self.pos..end)) {
            Some(b) => Some(b),
            None => {
                self.set_fatal();
                None
            }
        }
    }

    /// Consume `n` bytes of payload plus the padding that follows them, and hand back the payload.
    ///
    /// The stream advances further than the payload -- every field is padded to the wire's
    /// four-byte granularity -- but both numbers are derived here from the one the caller gave,
    /// rather than passed in as a pair. That is not tidiness: `n` is guest-controlled on the
    /// string and blob paths, padding it up is an addition that can wrap, and a caller that did
    /// the rounding itself would hand over a padded size *smaller* than its payload. Everything
    /// after that trusts the pair, and the slice below is where a wrapped one panicked -- which
    /// under `panic = "abort"` takes every guest's context with it, not just this command.
    ///
    /// So the rounding happens once, checked, and a length that cannot be padded is refused like
    /// any other malformed input.
    pub fn read_bytes(&mut self, n: usize) -> Option<&'a [u8]> {
        let Some(advance) = n.checked_next_multiple_of(4) else {
            self.set_fatal();
            return None;
        };
        let b = self.peek_bytes(advance)?;
        self.pos += advance;
        // `n <= advance` by construction, so this cannot be out of range.
        Some(&b[..n])
    }

    /// Allocate zeroed storage for one `T`, or `None` (poisoning) if the arena is exhausted.
    ///
    /// Exhausted means either bound: the cap the guest is held to, or what the host allocator
    /// will actually give. The second is the guest's problem too -- an implausible size that
    /// clears the cap is still the guest's size -- so it poisons the stream where the arena's
    /// infallible allocators would abort the process.
    pub fn alloc_temp<T: Default>(&self) -> Option<&'a mut T> {
        self.charge(size_of::<T>())?;
        match self.temp.try_alloc(T::default()) {
            Ok(p) => Some(p),
            Err(_) => {
                self.set_fatal();
                None
            }
        }
    }

    /// Allocate a zeroed array of `count` `T`, or `None` (poisoning) if the size is implausible or
    /// the arena is exhausted. A zero count still yields an empty slice, never `None`: an empty
    /// array is a thing the guest is allowed to send.
    pub fn alloc_temp_array<T: Default + Clone>(&self, count: usize) -> Option<&'a mut [T]> {
        let Some(bytes) = size_of::<T>().checked_mul(count) else {
            self.set_fatal();
            return None;
        };
        self.charge(bytes)?;
        if count == 0 {
            // The arena's cursor for a zero-byte request is wherever the last allocation left it,
            // which need not be aligned for `T`. An empty array is a thing the guest sends often
            // enough that this is the common path, not a corner.
            return Some(&mut []);
        }
        match self.temp.try_alloc_slice_fill_with(count, |_| T::default()) {
            Ok(a) => Some(a),
            Err(_) => {
                self.set_fatal();
                None
            }
        }
    }

    fn charge(&self, bytes: usize) -> Option<()> {
        let used = self.temp_used.get().saturating_add(bytes);
        if used > TEMP_POOL_MAX {
            self.set_fatal();
            return None;
        }
        self.temp_used.set(used);
        Some(())
    }

    /// Resolve a guest object id to its host handle.
    ///
    /// The two failures are not the same failure. A ghost is a create the host refused with the
    /// guest's later commands already in flight behind it -- it costs those commands and nothing
    /// more. Anything else is a guest naming an object it was never given, or naming one it has
    /// under the wrong type, and that is a stream we can no longer trust: the ring stops.
    pub fn lookup_object(&self, id: ObjectId, ty: i32) -> HostHandle {
        // `VK_NULL_HANDLE`. Vulkan spells "no object" as zero and a great many members are
        // optional, so this is an ordinary value on the wire and not a miss at all -- an optional
        // `VkPipelineCache` left null is what found this.
        if id.0 == 0 {
            return HostHandle(0);
        }
        match self.objects.lookup(id, ty) {
            Lookup::Found(handle) => {
                self.resolved.borrow_mut().push(id);
                handle
            }
            Lookup::Ghost => {
                self.ghost.set(Some(id));
                HostHandle(0)
            }
            Lookup::Missing => {
                self.set_fatal();
                HostHandle(0)
            }
        }
    }
}

/// The array a decoded command carries, reconciled into a slice.
///
/// A decoded command holds an array the way the wire spells it: the guest's count, and separately
/// a pointer that is null when the guest encoded the array as absent. Whether the two may disagree
/// depends on the array, and both kinds exist:
///
/// * A **required** array's size is checked against the count as it decodes, and a guest that
///   disagrees with itself poisons before any handler runs. Here the pair cannot arrive split, and
///   this function's `None` is defence in depth.
/// * An **optional** array -- `pResolveAttachments`, `pWaitSemaphoreValues`, `vkFreeCommandBuffers`
///   and some eighty other members and arguments -- decodes its size *unchecked*, because "null,
///   with a count of seven" is precisely what Vulkan means by an optional array sharing another
///   field's count. Here the pair genuinely arrives split, and reconciling it is the whole job.
///
/// Which of the two a given array is, is the generator's business and not a handler's. Past this
/// point an array is a slice, and a length that disagrees with its contents cannot be written down.
///
/// `None` is a count with no array behind it. What that means is the caller's to say, and it is
/// not the same answer everywhere: for a required array it is a command that cannot be carried out
/// and must be refused, because calling it empty would report a bind or a write that never
/// happened; for a `noautovalidity` array like `vkFreeCommandBuffers`' it names nothing to act on,
/// which for a free is simply nothing to do. What this function will not do is decide for them.
///
/// Lives here rather than beside the handlers because the invariant it rests on is the decoder's:
/// this module allocated the array, and knows how long it lives.
///
/// # Safety
///
/// `'a` is unconstrained, so the caller is the one choosing how long the slice lives, and nothing
/// in the signature stops it choosing badly. The only callers are the accessors the generator
/// emits on `vn_command_*`, where `'a` is the struct's own -- the lifetime the decode tied to the
/// arena. `ptr` must be null, or an allocation of at least `count` elements that lives for `'a`.
pub unsafe fn wire_array<'a, T>(count: usize, ptr: *const T) -> Option<&'a [T]> {
    if ptr.is_null() {
        return (count == 0).then_some(&[]);
    }
    // SAFETY: the caller's, above.
    Some(unsafe { core::slice::from_raw_parts(ptr, count) })
}

/// [`wire_array`] for an array a command writes back into -- the shadow the generated lifecycle
/// hook reads host handles out of.
///
/// # Safety
///
/// As [`wire_array`], and nothing else may hold the array while the returned slice lives. The
/// generated accessor borrows the command struct mutably, which is what enforces it -- and only
/// because a command struct is neither `Clone` nor `Copy`, so there is no second struct to borrow.
pub unsafe fn wire_array_mut<'a, T>(count: usize, ptr: *mut T) -> Option<&'a mut [T]> {
    if ptr.is_null() {
        return (count == 0).then_some(&mut []);
    }
    // SAFETY: the caller's, above.
    Some(unsafe { core::slice::from_raw_parts_mut(ptr, count) })
}

/// [`wire_array`] for a member that points at one value rather than an array.
///
/// The single-value half of the same wall. A command's `*const T` members are the ids the guest
/// named -- the handle a create is to be filed under -- and reading one is the same question as
/// reading an array of one: sound only because this module allocated it and knows how long it
/// lives.
///
/// `None` is the guest having sent nothing, which Vulkan gives its own meaning to often enough
/// that it cannot be folded into the value. What it means is the caller's to say.
///
/// # Safety
///
/// As [`wire_array`]: `'a` is unconstrained and the only callers are the generated accessors,
/// where it is the command struct's own. `ptr` must be null, or one element that lives for `'a`.
pub unsafe fn wire_ref<'a, T>(ptr: *const T) -> Option<&'a T> {
    // SAFETY: the caller's, above.
    unsafe { ptr.as_ref() }
}

/// [`wire_ref`] for a member that points at a NUL-terminated string.
///
/// The one member on the wire that is *not* a split pair, and the reason it gets its own door
/// rather than a `&[u8]` one. A counted array arrives as a length and a pointer that two layers
/// could disagree about; a C string carries its length inside itself, and [`Decoder::decode_c_string`]
/// makes that true rather than hoping: it copies the guest's bytes into the arena and writes the
/// terminator itself, over whatever the guest put in the last byte. So there is nothing here to
/// reconcile -- storing the wire's length beside the pointer would be a second copy of a fact the
/// bytes already carry, which is the pair this codebase spends its time removing.
///
/// `None` is the guest having sent no string at all, which `vkEnumerateDeviceExtensionProperties`
/// gives a meaning to: no layer named, so the implementation's own extensions.
///
/// # Safety
///
/// As [`wire_array`]: `'a` is unconstrained and the only callers are the generated accessors,
/// where it is the command struct's own. `ptr` must be null, or a NUL-terminated allocation that
/// lives for `'a` -- which for every caller means one `decode_c_string` produced.
pub unsafe fn wire_c_string<'a>(ptr: *const core::ffi::c_char) -> Option<&'a core::ffi::CStr> {
    if ptr.is_null() {
        return None;
    }
    // SAFETY: the caller's, above: the decoder allocated this from the arena and forced its last
    // byte to NUL, so the scan terminates inside the allocation.
    Some(unsafe { core::ffi::CStr::from_ptr(ptr) })
}

/// [`wire_ref`] for the one value a command writes its answer back into.
///
/// This is what a handler is given instead of a raw out-parameter. A query that must answer the
/// guest -- how many physical devices there are, what a format supports -- writes through this and
/// through nothing else, which is what keeps `venus/context.rs` free of unsafe.
///
/// # Safety
///
/// As [`wire_ref`], and nothing else may hold the value while the returned reference lives. The
/// generated accessor borrows the command struct mutably, which is what enforces it -- and only
/// because a command struct is neither `Clone` nor `Copy`, so there is no second struct to borrow.
pub unsafe fn wire_out<'a, T>(ptr: *mut T) -> Option<&'a mut T> {
    // SAFETY: the caller's, above.
    unsafe { ptr.as_mut() }
}

/// An out-count that sizes arrays the same command carries: `pPropertyCount` beside
/// `pProperties`, and every enumeration like it.
///
/// A handler has to write it, because how many there are is the answer. But the decoder allocated
/// the arrays from the value the guest sent, and their accessors and the reply encoder read their
/// length from this same value afterwards -- so a count raised past the allocation is a slice
/// past it. While any of those arrays was sent, the count may be lowered and never raised, which
/// is also all Vulkan lets an enumeration do. With none of them sent, the guest is asking how many
/// there are, nothing was allocated from the value, and any answer fits.
///
/// A raise is this renderer's bug, not the guest's -- the guest never writes through here -- so it
/// asserts.
pub struct OutCount<'c, T> {
    value: &'c mut T,
    /// The most the count may say: what it said when the door was opened, if it sizes an array.
    most: Option<T>,
}

impl<'c, T: Copy + PartialOrd + core::fmt::Display> OutCount<'c, T> {
    pub(crate) fn new(value: &'c mut T, sized: bool) -> Self {
        let most = sized.then_some(*value);
        OutCount { value, most }
    }

    /// What the count says now: the guest's capacity until a handler answers.
    pub fn get(&self) -> T {
        *self.value
    }

    /// Answer with `n`.
    pub fn set(&mut self, n: T) {
        if let Some(most) = self.most {
            assert!(
                n <= most,
                "an out-count raised to {n} past the {most} its arrays were sized to"
            );
        }
        *self.value = n;
    }
}

/// What a venus protocol supports, asked whenever a `pNext` chain must skip a struct the far side
/// cannot parse.
///
/// **Nothing generated for this renderer asks it.** venus-protocol emits that gate on the *driver*
/// side only -- the guest skips what the renderer cannot read, and by the time bytes arrive here
/// the filtering has happened. What this build can serialize is fixed when it is generated, and it
/// is published to the guest once, in the capset, from `proto::info`; there is no per-context
/// negotiation to answer from.
///
/// The trait stays because it is threaded through every generated signature, and because a
/// driver-side encoder generated from the same model -- the differential oracle for the shapes no
/// corpus reaches -- would be the caller that needs it.
pub trait Protocol {
    fn has_extension(&self, number: u32) -> bool;
    fn has_api_version(&self, version: u32) -> bool;
}

/// A protocol that supports everything -- which is every caller this renderer has.
///
/// Re-encoding a chain the guest itself sent cannot need to skip any of it, and no generated path
/// consults this anyway. See [`Protocol`].
pub struct AllOfIt;

impl Protocol for AllOfIt {
    fn has_extension(&self, _number: u32) -> bool {
        true
    }
    fn has_api_version(&self, _version: u32) -> bool {
        true
    }
}

/// Writes a reply stream.
///
/// Unlike the decoder this side is trusted -- the bytes come from us -- so the only failure it can
/// have is running out of room in the guest's reply buffer, which poisons the ring for the same
/// reason a short read does: the guest gave us a buffer that cannot hold the answer.
/// Where an encoder puts its bytes.
///
/// The two exist for two callers with genuinely different needs. The oracles hold the encoder to a
/// buffer of exactly the size `vn_sizeof_*` predicted, so running off the end is the failure they
/// are looking for. A reply, by contrast, is sized by the command's own shape, and the host has no
/// reason to guess it in advance -- so it grows, and the only bound that matters is applied later,
/// against the window the guest actually offered. Nothing is written toward a guest here either
/// way: this is host memory, and [`ReplyStream::write`] is the one place it crosses over.
enum Buffer<'a> {
    Fixed(&'a mut [u8]),
    Grow(&'a mut Vec<u8>),
}

pub struct Encoder<'a> {
    buf: Buffer<'a>,
    pos: usize,
    fatal: bool,
    protocol: &'a dyn Protocol,
    /// Offsets this encoder filled as padding rather than payload, once someone asks for them.
    ///
    /// Only the differential test does. A guest's encoder leaves its padding uninitialised, so no
    /// faithful reproduction of its wire can match those bytes -- and this renderer will not copy
    /// them, because padding a reply with whatever the host buffer held is how host memory leaks
    /// into a guest. Knowing exactly which bytes those are is what keeps the comparison exact
    /// instead of merely lenient.
    padding: Option<Vec<core::ops::Range<usize>>>,
}

impl<'a> Encoder<'a> {
    /// An encoder that must fit what it is given. Overflowing is `fatal`.
    pub fn new(buf: &'a mut [u8], protocol: &'a dyn Protocol) -> Self {
        Encoder { buf: Buffer::Fixed(buf), pos: 0, fatal: false, protocol, padding: None }
    }

    /// An encoder that takes as much room as it needs from `buf`, which it empties first.
    ///
    /// The clear is not for correctness -- every write covers its whole advance, payload then
    /// zero-fill, and only `..pos` is ever read back -- but for the invariant to be visible at the
    /// one place that establishes it. A future write path that skipped bytes would otherwise leak
    /// the previous reply into this one, silently and only sometimes.
    pub fn growing(buf: &'a mut Vec<u8>, protocol: &'a dyn Protocol) -> Self {
        buf.clear();
        Encoder { buf: Buffer::Grow(buf), pos: 0, fatal: false, protocol, padding: None }
    }

    /// The next `advance` bytes to write into, growing the buffer if it is allowed to.
    ///
    /// `None` is a fixed buffer that has run out, which is the only way an encoder fails.
    fn room(&mut self, advance: usize) -> Option<&mut [u8]> {
        let end = self.pos.checked_add(advance)?;
        match &mut self.buf {
            Buffer::Fixed(b) => b.get_mut(self.pos..end),
            Buffer::Grow(v) => {
                if v.len() < end {
                    v.resize(end, 0);
                }
                v.get_mut(self.pos..end)
            }
        }
    }

    /// Start recording padded byte ranges. See the `padding` field.
    pub fn record_padding(&mut self) {
        self.padding = Some(Vec::new());
    }

    pub fn padding(&self) -> &[core::ops::Range<usize>] {
        self.padding.as_deref().unwrap_or(&[])
    }

    fn note_padding(&mut self, from: usize, to: usize) {
        if from < to
            && let Some(p) = &mut self.padding
        {
            p.push(from..to);
        }
    }

    pub fn protocol(&self) -> &dyn Protocol {
        self.protocol
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn fatal(&self) -> bool {
        self.fatal
    }

    /// The bytes written so far.
    pub fn written(&self) -> &[u8] {
        match &self.buf {
            Buffer::Fixed(b) => &b[..self.pos],
            Buffer::Grow(v) => &v[..self.pos],
        }
    }

    /// Write `val` and advance `advance` bytes, zero-filling the padding. `advance` is the wire
    /// size; `val` is the payload.
    pub fn write(&mut self, advance: usize, val: &[u8]) {
        debug_assert!(val.len() <= advance);
        let Some(dst) = self.room(advance) else {
            self.fatal = true;
            return;
        };
        dst[..val.len()].copy_from_slice(val);
        dst[val.len()..].fill(0);
        self.note_padding(self.pos + val.len(), self.pos + advance);
        self.pos += advance;
    }
}

/// Scalars on the wire.
///
/// Every scalar occupies a multiple of four bytes -- a `u8` costs four, a `u64` costs eight -- and
/// arrays of sub-word scalars are packed and then padded as a whole. `vn_sizeof_*` in the C
/// generator is this rule; implementing it once here keeps the generated code free of it.
pub trait Scalar: Copy + Default {
    fn from_le_bytes(b: &[u8]) -> Self;
    fn write_le(self, out: &mut [u8]);
}

macro_rules! scalar {
    ($($t:ty),* $(,)?) => {$(
        impl Scalar for $t {
            fn from_le_bytes(b: &[u8]) -> Self {
                <$t>::from_le_bytes(b.try_into().expect("caller sized the slice"))
            }
            fn write_le(self, out: &mut [u8]) {
                out.copy_from_slice(&self.to_le_bytes());
            }
        }
    )*};
}

// `usize` is here because vk.xml has `size_t` members. It is eight bytes on every target
// this renderer supports, which the wire assumes.
scalar!(u8, i8, u16, i16, u32, i32, u64, i64, usize, isize, f32, f64);

/// Wire size of a single scalar: padded up to a word.
#[inline]
pub const fn sizeof_scalar<T: Scalar>() -> usize {
    if size_of::<T>() >= 4 { size_of::<T>() } else { 4 }
}

/// Wire size of a packed scalar array, padded as a whole.
#[inline]
pub fn sizeof_scalar_array<T: Scalar>(count: usize) -> usize {
    align4(size_of::<T>() * count)
}

impl<'a> Decoder<'a> {
    pub fn decode_scalar<T: Scalar>(&mut self) -> T {
        match self.read_bytes(size_of::<T>()) {
            Some(b) => T::from_le_bytes(b),
            None => T::default(),
        }
    }

    pub fn peek_scalar<T: Scalar>(&self) -> T {
        match self.peek_bytes(sizeof_scalar::<T>()) {
            Some(b) => T::from_le_bytes(&b[..size_of::<T>()]),
            None => T::default(),
        }
    }

    /// Fill `out` from the stream. On a short stream `out` is left zeroed and the stream poisoned,
    /// so the caller never reads uninitialised data.
    #[allow(clippy::chunks_exact_to_as_chunks)] // the chunk size is generic, not const
    pub fn decode_scalar_array<T: Scalar>(&mut self, out: &mut [T]) {
        let packed = size_of_val(out);
        let Some(b) = self.read_bytes(packed) else {
            out.fill(T::default());
            return;
        };
        for (dst, src) in out.iter_mut().zip(b.chunks_exact(size_of::<T>())) {
            *dst = T::from_le_bytes(src);
        }
    }
}

impl<'a> Encoder<'a> {
    pub fn encode_scalar<T: Scalar>(&mut self, val: T) {
        let mut b = [0u8; 8];
        val.write_le(&mut b[..size_of::<T>()]);
        self.write(sizeof_scalar::<T>(), &b[..size_of::<T>()]);
    }

    #[allow(clippy::chunks_exact_to_as_chunks)] // the chunk size is generic, not const
    pub fn encode_scalar_array<T: Scalar>(&mut self, vals: &[T]) {
        let packed = size_of_val(vals);
        let Some(dst) = self.room(align4(packed)) else {
            self.fatal = true;
            return;
        };
        for (chunk, v) in dst.chunks_exact_mut(size_of::<T>()).zip(vals) {
            v.write_le(chunk);
        }
        dst[packed..].fill(0);
        self.note_padding(self.pos + packed, self.pos + align4(packed));
        self.pos += align4(packed);
    }
}

/// A pointer on the wire is an array size of one or zero: present or not. The pointee follows only
/// if present, so it costs the same eight bytes an element count does.
impl<'a> Decoder<'a> {
    pub fn decode_simple_pointer(&mut self) -> bool {
        self.decode_array_size_unchecked() != 0
    }

    /// An array's element count, as the guest claims it. Nothing here validates it against what the
    /// command said the count should be -- that is `decode_array_size`'s job.
    pub fn peek_array_size(&self) -> u64 {
        self.peek_scalar::<u64>()
    }

    pub fn decode_array_size_unchecked(&mut self) -> u64 {
        self.decode_scalar::<u64>()
    }

    /// An array's element count, checked against the count the command already gave us. A guest
    /// that disagrees with itself is poisoned rather than trusted: this is the check that keeps an
    /// oversized array from being allocated and filled.
    pub fn decode_array_size(&mut self, expected: u64) -> u64 {
        let size = self.decode_scalar::<u64>();
        if size != expected {
            self.set_fatal();
            return 0;
        }
        size
    }

    /// `size` bytes of opaque payload, borrowed from the stream rather than copied. The renderer
    /// only ever reads a blob, and the stream outlives the command that named it.
    pub fn decode_blob(&mut self, size: usize) -> Option<&'a [u8]> {
        self.read_bytes(size)
    }

    /// `size` bytes of string, copied into the arena and forced NUL-terminated. The copy is what
    /// buys the terminator: a guest that sends an unterminated string gets it truncated here
    /// rather than run off the end of the stream in whatever host call receives it.
    pub fn decode_c_string(&mut self, size: usize) -> Option<&'a mut [u8]> {
        if size == 0 {
            self.set_fatal();
            return None;
        }
        let bytes = self.read_bytes(size)?;
        let out = self.alloc_temp_array::<u8>(size)?;
        out.copy_from_slice(bytes);
        out[size - 1] = 0;
        Some(out)
    }
}

/// The wire size of `size` bytes of opaque payload: padded to a word, with no count of its own.
pub const fn sizeof_blob(size: usize) -> usize {
    align4(size)
}

/// The length a NUL-terminated string occupies on the wire, terminator included.
///
/// # Safety
/// `p` must point at a NUL-terminated string that outlives the call.
pub unsafe fn c_string_len(p: *const core::ffi::c_char) -> usize {
    let mut n = 0;
    // SAFETY: the caller promises a terminator, so the walk stops inside the allocation.
    while unsafe { *p.add(n) } != 0 {
        n += 1;
    }
    n + 1
}

impl<'a> Encoder<'a> {
    pub fn encode_simple_pointer(&mut self, present: bool) -> bool {
        self.encode_array_size(present as u64);
        present
    }

    pub fn encode_array_size(&mut self, count: u64) {
        self.encode_scalar::<u64>(count);
    }

    /// Opaque payload, padded to a word. The count, when the wire carries one, is the caller's.
    pub fn encode_blob(&mut self, val: &[u8]) {
        self.write(align4(val.len()), val);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::venus::proto::serialize::{
        vn_decode_vkEnumeratePhysicalDevices_args_temp, vn_encode_vkEnumeratePhysicalDevices_reply,
        vn_sizeof_vkEnumeratePhysicalDevices_reply,
    };
    use crate::venus::proto::types::{
        VkPhysicalDevice, vn_command_vkEnumeratePhysicalDevices, vn_command_vkGetQueryPoolResults,
    };

    #[test]
    fn scalars_are_word_padded_on_the_wire() {
        assert_eq!(sizeof_scalar::<u8>(), 4);
        assert_eq!(sizeof_scalar::<u16>(), 4);
        assert_eq!(sizeof_scalar::<u32>(), 4);
        assert_eq!(sizeof_scalar::<u64>(), 8);
        assert_eq!(sizeof_scalar_array::<u8>(5), 8);
        assert_eq!(sizeof_scalar_array::<u32>(3), 12);
    }

    /// An empty array is common on the wire, and the arena's cursor after an odd-sized allocation
    /// is not aligned for anything. Handing that cursor back as a zero-length slice makes every
    /// later `from_raw_parts` on it a misaligned read.
    #[test]
    fn a_zero_length_array_is_still_aligned() {
        let temp = Bump::new();
        let hard = AtomicBool::new(false);
        let dec = Decoder::new(&[], &temp, &IdentityObjects, &hard);
        dec.alloc_temp_array::<u8>(1).unwrap();
        let empty = dec.alloc_temp_array::<u64>(0).unwrap();
        assert!(empty.is_empty());
        assert_eq!(empty.as_ptr().align_offset(align_of::<u64>()), 0);
    }

    /// A `pNext` chain is decoded by recursion of its own depth, and the depth is the guest's to
    /// choose: the wire carries every link's header before any link's body. A chain that names
    /// more links than there are structs it may name has necessarily named one twice, which
    /// Vulkan forbids, so it is refused at that link -- long before it can choose how deep the
    /// ring thread's stack goes. The chain here is a quarter of a million links of one struct a
    /// buffer create admits; unbounded, decoding it overruns any thread's stack.
    #[test]
    #[cfg_attr(miri, ignore = "a quarter-million-link chain, which Miri would run for hours")]
    fn a_pnext_chain_deeper_than_the_structs_it_may_name_is_refused() {
        use crate::venus::proto::serialize::vn_decode_VkBufferCreateInfo_pnext_temp;
        use crate::venus::proto::types::VkStructureType;

        const LINKS: usize = 1 << 18;
        let proto = AllOfIt;
        let mut buf = vec![0u8; LINKS * (8 + 4) + 8];
        let mut enc = Encoder::new(&mut buf, &proto);
        for _ in 0..LINKS {
            enc.encode_simple_pointer(true);
            enc.encode_scalar::<VkStructureType>(
                VkStructureType::VK_STRUCTURE_TYPE_BUFFER_OPAQUE_CAPTURE_ADDRESS_CREATE_INFO,
            );
        }
        enc.encode_simple_pointer(false);

        let temp = Bump::new();
        let hard = AtomicBool::new(false);
        let mut dec = Decoder::new(&buf, &temp, &IdentityObjects, &hard);
        // The links before the refusal are still handed back, as every partial decode is: the
        // poison is what the dispatch reads, and it is what stops the command.
        vn_decode_VkBufferCreateInfo_pnext_temp(&mut dec);
        assert!(dec.hard_fatal(), "a chain longer than the structs it may name is refused");
    }

    /// The arena's cap bounds what a guest may ask for; it is not a promise the host can pay it.
    /// A request the allocator refuses is the guest's problem, and it costs the stream, never the
    /// process.
    #[test]
    fn an_allocation_the_host_refuses_poisons_instead_of_aborting() {
        let temp = Bump::new();
        temp.set_allocation_limit(Some(64));
        let hard = AtomicBool::new(false);
        let dec = Decoder::new(&[], &temp, &IdentityObjects, &hard);
        assert!(dec.alloc_temp_array::<u64>(1024).is_none());
        assert!(dec.hard_fatal());

        let temp = Bump::new();
        temp.set_allocation_limit(Some(0));
        let hard = AtomicBool::new(false);
        let dec = Decoder::new(&[], &temp, &IdentityObjects, &hard);
        assert!(dec.alloc_temp::<u64>().is_none());
        assert!(dec.hard_fatal());
    }

    /// `vkEnumeratePhysicalDevices` asking for `capacity` devices, decoded as the ring would:
    /// the count, the guest's ids, and the shadow the host handles go in, all in `temp`.
    fn enumeration<'a>(
        buf: &'a mut Vec<u8>,
        temp: &'a Bump,
        hard: &'a AtomicBool,
        capacity: u32,
    ) -> vn_command_vkEnumeratePhysicalDevices<'a> {
        buf.extend(7u64.to_le_bytes()); // the instance
        buf.extend(1u64.to_le_bytes()); // the count is there
        buf.extend(capacity.to_le_bytes());
        buf.extend(u64::from(capacity).to_le_bytes()); // the array is there, this long
        for id in 0..u64::from(capacity) {
            buf.extend((100 + id).to_le_bytes());
        }
        let mut dec = Decoder::new(buf, temp, &IdentityObjects, hard);
        let mut args = vn_command_vkEnumeratePhysicalDevices::default();
        vn_decode_vkEnumeratePhysicalDevices_args_temp(&mut dec, &mut args);
        assert!(!dec.fatal(), "a well-formed enumeration decodes");
        args
    }

    /// The whole life of an out-array, which is where a handler holds arena memory as `&mut`: the
    /// decode, the handler filling the shadow and answering fewer than the guest had room for, and
    /// the reply read back out. Run under Miri, this is what checks that the references the
    /// accessors make from arena pointers obey the aliasing rules.
    #[test]
    fn an_enumeration_answers_fewer_than_it_had_room_for() {
        let (mut buf, temp, hard) = (Vec::new(), Bump::new(), AtomicBool::new(false));
        let mut args = enumeration(&mut buf, &temp, &hard, 3);
        let ids = args.pPhysicalDevices().expect("the guest sent its ids");
        assert_eq!(ids.len(), 3);
        let shadow = args.handle_pPhysicalDevices_mut().expect("a shadow beside the ids");
        shadow[0] = VkPhysicalDevice(0xaa);
        shadow[1] = VkPhysicalDevice(0xbb);
        let mut count = args.pPhysicalDeviceCount_mut().expect("the guest sent a count");
        assert_eq!(count.get(), 3, "until answered, the count is the guest's capacity");
        count.set(2);
        assert_eq!(
            args.pPhysicalDevices().map(<[_]>::len),
            Some(2),
            "the array follows the answer"
        );

        let proto = AllOfIt;
        let mut out = vec![0u8; vn_sizeof_vkEnumeratePhysicalDevices_reply(&proto, &args)];
        let mut enc = Encoder::new(&mut out, &proto);
        vn_encode_vkEnumeratePhysicalDevices_reply(&mut enc, &args);
        // The type, the result, the count's presence and value, then an array of two.
        assert_eq!(&out[16..20], &2u32.to_le_bytes(), "the reply carries the answer");
        assert_eq!(&out[20..28], &2u64.to_le_bytes(), "and an array that long");
    }

    /// The arrays were allocated from the count the guest sent, so a handler that answered with
    /// more would have their accessors, and the reply, slice past the allocation.
    #[test]
    #[should_panic(expected = "an out-count raised to 4 past the 3 its arrays were sized to")]
    fn an_out_count_cannot_be_raised_past_the_arrays_it_sized() {
        let (mut buf, temp, hard) = (Vec::new(), Bump::new(), AtomicBool::new(false));
        let mut args = enumeration(&mut buf, &temp, &hard, 3);
        args.pPhysicalDeviceCount_mut().expect("the guest sent a count").set(4);
    }

    /// With no array sent, the guest is asking how many there are: nothing was allocated from the
    /// count, and any answer fits.
    #[test]
    fn a_count_query_takes_any_answer() {
        let mut buf = Vec::new();
        buf.extend(7u64.to_le_bytes());
        buf.extend(1u64.to_le_bytes());
        buf.extend(0u32.to_le_bytes());
        buf.extend(0u64.to_le_bytes()); // no array
        let (temp, hard) = (Bump::new(), AtomicBool::new(false));
        let mut dec = Decoder::new(&buf, &temp, &IdentityObjects, &hard);
        let mut args = vn_command_vkEnumeratePhysicalDevices::default();
        vn_decode_vkEnumeratePhysicalDevices_args_temp(&mut dec, &mut args);
        assert!(!dec.fatal());
        args.pPhysicalDeviceCount_mut().expect("the guest sent a count").set(4);
        assert!(!args.has_pPhysicalDevices());
    }

    /// A command's `_mut` accessors hand out arena memory as `&mut` borrowed from the struct,
    /// which is one borrow only while there is one struct. A `Copy` or `Clone` command would be
    /// two, each lending the same memory.
    #[test]
    fn a_command_cannot_be_duplicated() {
        use core::marker::PhantomData;
        struct Probe<T>(PhantomData<T>);
        // Autoref specialisation: the by-value candidates are tried first, and only apply where
        // the bound holds; otherwise resolution falls through to the impls on `&Probe`.
        trait Copies {
            fn copies(&self) -> bool {
                true
            }
        }
        impl<T: Copy> Copies for Probe<T> {}
        trait CopiesNot {
            fn copies(&self) -> bool {
                false
            }
        }
        impl<T> CopiesNot for &Probe<T> {}
        trait Clones {
            fn clones(&self) -> bool {
                true
            }
        }
        impl<T: Clone> Clones for Probe<T> {}
        trait ClonesNot {
            fn clones(&self) -> bool {
                false
            }
        }
        impl<T> ClonesNot for &Probe<T> {}

        // The positive control: the probe does see a copy where there is one.
        let control = &Probe::<VkPhysicalDevice>(PhantomData);
        assert!(control.copies() && control.clones());
        let e = &Probe::<vn_command_vkEnumeratePhysicalDevices<'static>>(PhantomData);
        assert!(!e.copies() && !e.clones(), "an enumeration can be duplicated");
        let q = &Probe::<vn_command_vkGetQueryPoolResults<'static>>(PhantomData);
        assert!(!q.copies() && !q.clones(), "a blob query can be duplicated");
    }

    /// `Plain` is the one safe way to a [`Decoded`], so it must never be on a struct Vulkan follows
    /// a pointer out of -- whether the pointer is its own member, a function pointer, or inside a
    /// struct it embeds.
    #[test]
    fn a_struct_with_a_pointer_anywhere_in_it_is_never_plain() {
        use crate::venus::proto::types::{
            VkAllocationCallbacks, VkAttachmentSampleLocationsEXT, VkBufferCreateInfo, VkExtent3D,
            VkImageSubresource,
        };
        use core::marker::PhantomData;
        struct Probe<T>(PhantomData<T>);
        trait IsPlain {
            fn plain(&self) -> bool {
                true
            }
        }
        impl<T: Plain> IsPlain for Probe<T> {}
        trait NotPlain {
            fn plain(&self) -> bool {
                false
            }
        }
        impl<T> NotPlain for &Probe<T> {}

        // The positive control: plain values are plain.
        let (extent, sub) =
            (&Probe::<VkExtent3D>(PhantomData), &Probe::<VkImageSubresource>(PhantomData));
        assert!(extent.plain() && sub.plain());
        let buffer = &Probe::<VkBufferCreateInfo>(PhantomData);
        assert!(!buffer.plain(), "a pNext and an array");
        let callbacks = &Probe::<VkAllocationCallbacks>(PhantomData);
        assert!(!callbacks.plain(), "function pointers");
        let embeds = &Probe::<VkAttachmentSampleLocationsEXT>(PhantomData);
        assert!(!embeds.plain(), "a pointer inside an embedded struct");
    }

    #[test]
    fn a_short_stream_poisons_instead_of_panicking() {
        let temp = Bump::new();
        let hard = AtomicBool::new(false);
        let buf = [1u8, 0, 0, 0];
        let mut dec = Decoder::new(&buf, &temp, &IdentityObjects, &hard);
        assert_eq!(dec.decode_scalar::<u32>(), 1);
        assert!(!dec.fatal());
        assert_eq!(dec.decode_scalar::<u32>(), 0);
        assert!(dec.hard_fatal());
    }

    /// The invariant `wire_c_string`'s whole soundness rests on, tested without going near a
    /// `CStr` -- because the obvious way to falsify that accessor is to remove this forcing, and
    /// then the scan runs off the allocation instead of failing.
    ///
    /// A guest that sends an unterminated string is not a guest we may believe: the decoder
    /// overwrites the last byte rather than searching for a NUL it has no guarantee of finding.
    #[test]
    fn a_string_the_guest_left_unterminated_is_terminated_anyway() {
        let temp = Bump::new();
        let hard = AtomicBool::new(false);
        let buf = *b"VK_KHR_a";
        let mut dec = Decoder::new(&buf, &temp, &IdentityObjects, &hard);
        let out = dec.decode_c_string(8).expect("eight bytes are there to read");
        assert_eq!(out, b"VK_KHR_\0", "the last byte is the terminator, whatever the guest sent");
        assert!(!dec.fatal());
    }

    #[test]
    fn an_array_size_the_guest_disagrees_with_poisons() {
        let temp = Bump::new();
        let hard = AtomicBool::new(false);
        let buf = 7u64.to_le_bytes();
        let mut dec = Decoder::new(&buf, &temp, &IdentityObjects, &hard);
        assert_eq!(dec.decode_array_size(3), 0);
        assert!(dec.hard_fatal());
    }

    #[test]
    fn an_implausible_array_size_is_refused_before_it_is_allocated() {
        let temp = Bump::new();
        let hard = AtomicBool::new(false);
        let buf = [0u8; 0];
        let dec = Decoder::new(&buf, &temp, &IdentityObjects, &hard);
        assert!(dec.alloc_temp_array::<u32>(usize::MAX / 2).is_none());
        assert!(dec.hard_fatal());
    }

    /// A string or blob length near `usize::MAX` poisons; it does not take the process down.
    ///
    /// The length is the guest's, decoded unchecked -- the generator emits
    /// `decode_array_size_unchecked` for every string, because a string carries no separate count
    /// to check against. Padding it up to the wire's four-byte granularity is therefore an
    /// addition on a number the guest chose, and one that wraps leaves a padded size *smaller*
    /// than the payload. Everything downstream then trusts the pair: the release build compiles
    /// the debug assert out, the short read succeeds, and slicing the payload out of it panics.
    ///
    /// Under `panic = "abort"` that is not one lost command -- it is the worker gone, and with it
    /// every other guest's context. A guest is never allowed to do that (CLAUDE.md), so the wire
    /// arithmetic has to be total.
    #[test]
    fn a_string_length_that_cannot_be_padded_is_refused_rather_than_fatal_to_the_process() {
        let temp = Bump::new();
        let buf = [0u8; 16];

        for size in [usize::MAX, usize::MAX - 1, usize::MAX - 2, usize::MAX - 3] {
            let hard = AtomicBool::new(false);
            let mut dec = Decoder::new(&buf, &temp, &IdentityObjects, &hard);
            assert!(dec.decode_c_string(size).is_none(), "{size:#x} must not be read");
            assert!(dec.hard_fatal(), "{size:#x} must poison the stream");

            let hard = AtomicBool::new(false);
            let mut dec = Decoder::new(&buf, &temp, &IdentityObjects, &hard);
            assert!(dec.decode_blob(size).is_none(), "{size:#x} must not be read");
            assert!(dec.hard_fatal(), "{size:#x} must poison the stream");
        }
    }

    #[test]
    fn a_ghost_object_poisons_the_command_and_not_the_ring() {
        struct AllGhosts;
        impl Objects for AllGhosts {
            fn lookup(&self, _id: ObjectId, _ty: i32) -> Lookup {
                Lookup::Ghost
            }
        }
        let temp = Bump::new();
        let hard = AtomicBool::new(false);
        let buf = [0u8; 0];
        let dec = Decoder::new(&buf, &temp, &AllGhosts, &hard);
        assert_eq!(dec.lookup_object(ObjectId(42), 0), HostHandle(0));
        assert!(dec.fatal());
        assert!(!dec.hard_fatal());
        assert_eq!(dec.verdict(), Dispatched::Ghosted(ObjectId(42)), "and the verdict names it");
        assert!(!dec.fatal(), "a ghost does not outlive its command");
    }

    /// `VK_NULL_HANDLE` is an ordinary value, not a missing object: Vulkan spells "no object" as
    /// zero, and every optional handle member on the wire carries it.
    #[test]
    fn the_null_handle_is_not_a_lookup_failure() {
        struct Nothing;
        impl Objects for Nothing {
            fn lookup(&self, _id: ObjectId, _ty: i32) -> Lookup {
                Lookup::Missing
            }
        }
        let temp = Bump::new();
        let hard = AtomicBool::new(false);
        let buf = [0u8; 0];
        let dec = Decoder::new(&buf, &temp, &Nothing, &hard);
        assert_eq!(dec.lookup_object(ObjectId(0), 0), HostHandle(0));
        assert!(!dec.fatal());
    }

    /// An id the guest never had is not a ghost. Losing one command per bad id would let a guest
    /// name ids forever; the stream is untrustworthy from here, so the ring stops.
    #[test]
    fn an_object_the_guest_invented_stops_the_ring() {
        struct Nothing;
        impl Objects for Nothing {
            fn lookup(&self, _id: ObjectId, _ty: i32) -> Lookup {
                Lookup::Missing
            }
        }
        let temp = Bump::new();
        let hard = AtomicBool::new(false);
        let buf = [0u8; 0];
        let dec = Decoder::new(&buf, &temp, &Nothing, &hard);
        assert_eq!(dec.lookup_object(ObjectId(42), 0), HostHandle(0));
        assert!(dec.hard_fatal());
        assert_eq!(dec.verdict(), Dispatched::Undecodable);
        assert!(dec.fatal(), "a hard poison does not clear with the command");
    }

    #[test]
    fn scalar_round_trip_reproduces_the_wire_including_padding() {
        let mut buf = [0xaau8; 16];
        let mut enc = Encoder::new(&mut buf, &AllOfIt);
        enc.encode_scalar::<u8>(0x5a);
        enc.encode_scalar::<u64>(0x0102_0304_0506_0708);
        enc.encode_scalar_array::<u8>(&[1, 2, 3]);
        assert_eq!(enc.pos(), 16);
        assert_eq!(&buf[..4], &[0x5a, 0, 0, 0]);
        assert_eq!(&buf[12..], &[1, 2, 3, 0]);
    }
}

/// Proofs over every length a guest can put on the wire, run by `cargo kani`.
///
/// A string or blob length is a `usize` the guest chooses, so these are exactly the wide-data,
/// fixed-control-flow shape Kani settles quickly. Three reads in a row, each of any length, over
/// a stream whose bytes are left to the guest too.
#[cfg(kani)]
mod proofs {
    use super::*;

    const STREAM: usize = 16;

    /// Whether `got` is the `n` bytes of `buf` at `at`. One index, chosen by Kani, stands for
    /// all of them: a comparison of the whole slice unrolls over a symbolic length.
    fn is_window(got: &[u8], buf: &[u8], at: usize, n: usize) -> bool {
        let i: usize = kani::any();
        got.len() == n && (i >= n || got[i] == buf[at + i])
    }

    /// A read hands back exactly the `n` bytes at the position, advances by `n` padded to four,
    /// and never past the stream; a refused one poisons the stream and moves nothing. No length
    /// panics, however near the top of `usize` it is.
    #[kani::proof]
    #[kani::unwind(4)]
    fn a_read_of_any_length_stays_inside_the_stream() {
        let buf: [u8; STREAM] = kani::any();
        let (temp, hard) = (Bump::new(), AtomicBool::new(false));
        let mut dec = Decoder::new(&buf, &temp, &IdentityObjects, &hard);

        for _ in 0..3 {
            let (n, before, poisoned) = (kani::any::<usize>(), dec.pos(), dec.hard_fatal());
            let peeked = dec.peek_bytes(n);
            assert!(dec.pos() == before, "a peek moved the stream");
            assert!(peeked.is_none_or(|b| is_window(b, &buf, before, n)), "a peek read elsewhere");
            match dec.read_bytes(n) {
                Some(b) => {
                    assert!(is_window(b, &buf, before, n), "a read handed back other bytes");
                    assert!(dec.pos() == before + n.next_multiple_of(4), "a read moved wrongly");
                    kani::cover!(n % 4 != 0, "a padded read");
                }
                None => {
                    assert!(dec.pos() == before, "a refused read moved the stream");
                    assert!(dec.hard_fatal(), "a refused read left the stream usable");
                    kani::cover!(!poisoned && n > usize::MAX - 3, "a length that cannot be padded");
                }
            }
            assert!(dec.pos() <= STREAM, "the position passed the end of the stream");
        }
    }

    /// The arena charge never lets the total pass its cap, whatever sizes the guest asks for or
    /// in what order, and a refused charge poisons the stream and charges nothing.
    #[kani::proof]
    #[kani::unwind(4)]
    fn the_arena_charge_never_passes_its_cap() {
        let (temp, hard) = (Bump::new(), AtomicBool::new(false));
        let dec = Decoder::new(&[], &temp, &IdentityObjects, &hard);
        for _ in 0..3 {
            let (bytes, before) = (kani::any::<usize>(), dec.temp_used.get());
            match dec.charge(bytes) {
                Some(()) => assert!(dec.temp_used.get() == before + bytes, "a charge miscounted"),
                None => {
                    assert!(dec.temp_used.get() == before, "a refused charge was counted");
                    assert!(dec.hard_fatal(), "a refused charge left the stream usable");
                }
            }
            assert!(dec.temp_used.get() <= TEMP_POOL_MAX, "the arena passed its cap");
        }
        kani::cover!(dec.temp_used.get() == TEMP_POOL_MAX, "the cap reached exactly");
    }
}
