// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! What the guest has made this process hold, and the cap on it.
//!
//! Every allocation this renderer makes on the guest's behalf lands in the *renderer's* address
//! space, where the guest's own accounting cannot see it: a guest that leaks `VkDeviceMemory` or
//! never unrefs a window buffer grows the host worker, not itself. On macOS that ends with the
//! kernel picking the worker as the largest compressed process and killing it -- the whole VM, no
//! guest backtrace, no crash report, at a moment unrelated to the allocation that caused it.
//! Measured 2026-08-06: a Vulkan compositor re-allocated a 4K backdrop instead of reusing it,
//! ~51 GB/hour, and jetsam took the VM at 142 GB.
//!
//! **One ledger, both arms.** venus charges through an [`Account`], which is a context's key to
//! its own slot; classic charges through [`Classic`], which is one bucket and cannot refuse. The
//! ledger is the renderer's rather than either arm's because the cap is on the process total, and
//! a per-arm ledger would be blind to the half of that total the host actually kills for.
//!
//! **What it does not see, and should never be claimed to.** Only the allocations this process
//! makes with its own hands: venus's device memory and exported pages, and classic's IOSurfaces.
//! Ordinary `glTexStorage`/`glBufferData` storage is the driver's and this process cannot size
//! it; nor are `Shadow::fresh`, `GuestPixels.staging` or the VideoToolbox output pool counted.
//! Only a `SCANOUT` or `SHARED` bind mints a surface, so "the ledger sees what classic holds" is
//! never going to be true -- it sees what classic holds *in IOSurfaces*.
//!
//! Accounting is always on and enforcement is opt-in, because the two answer different questions.
//! The ledger alone says *which* allocation is growing -- one repeated call site reads as
//! "767 x 31.6 MiB device memory" in a single line -- and it is also the only thing that separates
//! a guest leak from ours: a flat ledger under a climbing process footprint means the leak is in
//! this renderer's release path.
//!
//! **A refusal has to kill the context, because nothing else reaches the guest.** venus allocates
//! asynchronously: mesa's `vn_device_memory_alloc_simple` submits `vkAllocateMemory` and returns
//! `VK_SUCCESS` as soon as the command is on the ring, and `vn_device_memory_wait_alloc` waits on
//! a ring seqno without ever reading a `VkResult` back. The answer is discarded. Stopping the
//! context is not the consolation prize either: a host allocation failure already kills it today,
//! by leaving the guest a handle to memory that was never created and poisoning the ring on the
//! next command that names it. Doing it deliberately closes the window and puts the reason in the
//! log. `ret` is still set, because it is correct under `VN_PERF=no_async_mem_alloc` and costs
//! nothing.
//!
//! **A charge is a value, and crediting it is dropping it.** [`Charge`] lives in the record of
//! whatever it paid for, so every path that retires that record -- a free, a device teardown, a
//! context destroy -- credits the ledger without knowing the ledger exists. There is no release
//! call site to forget, which is the whole reason this is not a pair of `charge`/`credit`
//! functions: a ledger with manual credits has one more place to be wrong every time a new
//! destroy path is added.
//!
//! **Who is charged is structural, not ambient.** The C binds the billing context to the calling
//! *thread*, so an allocation made deeper in the stack attributes to whoever that thread served
//! last -- which is why it also needs a pseudo-context for vrend. Here the context is an argument,
//! because the caller always has one, and there is nothing for a second caller to get wrong.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use crate::ids::ContextId;
use crate::venus::vkr::ContextKey;

/// How the cap is configured, and what it is called.
const CAP_ENV: &str = "LIMINA_GPU_MEM_BUDGET_MIB";
/// Whether a refusal stops at the error, instead of stopping the context.
const SOFT_ENV: &str = "LIMINA_GPU_MEM_BUDGET_SOFT";
/// Whether every budget answer is logged, not only a clamp transition.
const TRACE_ENV: &str = "LIMINA_GPU_MEM_BUDGET_TRACE";
/// How often the whole breakdown is printed anyway, in seconds.
const CENSUS_ENV: &str = "LIMINA_GPU_MEM_BUDGET_CENSUS";

/// The share of the cap that is worth a line before anything has been refused, and the share the
/// total has to fall back below before that line is worth printing again.
///
/// Two thresholds and not one: a workload sitting at the mark crosses it on every other charge,
/// and a single edge would reprint the whole breakdown each time.
const WARN_AT: u64 = 80;
const REARM_BELOW: u64 = 70;

/// How every line about the cap begins.
///
/// It says "limina" because the cap is limina's policy, named in `LIMINA_GPU_MEM_BUDGET_MIB` and
/// documented as this grammar in limina's `docs/design/gpu-memory-budget.md` -- an operator reads
/// these lines with that page open, and the harness parses them. This is the one place this
/// renderer speaks a word from the layer above, and it does so because the log is that layer's
/// interface rather than ours.
const GRAMMAR: &str = "limina GPU budget";

/// While a clamp holds, one line a minute rather than one per query.
const HEARTBEAT: std::time::Duration = std::time::Duration::from_secs(60);

/// The live host memory this renderer holds on guests' behalf, and the cap on it.
pub struct Budget {
    /// Bytes, or `None` for accounting with no enforcement -- which is what a bare
    /// virglrenderer, and every replay, runs as.
    cap: Option<u64>,
    /// Whether a refusal leaves the context alive with an error it will not read.
    ///
    /// Only useful paired with a guest running `VN_PERF=no_async_mem_alloc`, which makes its
    /// `vkAllocateMemory` synchronous so the error actually arrives somewhere with a backtrace.
    /// Without that, soft mode is worse than the default: the guest ignores the refusal, keeps a
    /// handle to memory that does not exist, and poisons its ring on the next use anyway -- the
    /// same context death, with the reason several commands in the past.
    soft: bool,
    /// Whether every `VK_EXT_memory_budget` answer is logged, and not only a clamp transition.
    trace: bool,
    /// How often to print the breakdown regardless of what is happening, cap or no cap. This is
    /// the instrument for telling a guest leak from ours: a host total that climbs while the
    /// guest's own census stays flat is memory this renderer is retaining.
    census: Option<std::time::Duration>,
    ledger: Mutex<Ledger>,
    /// What was last said about each heap, so the clamp is logged as an event and not as a
    /// sample. This is state, not configuration, and it lives here for the reason the module
    /// exists: the C kept it in file-scope statics, which is precisely the shape this renderer
    /// does not have.
    heaps: Mutex<[HeapTrace; MAX_HEAPS]>,
}

/// Heaps a `VkPhysicalDeviceMemoryProperties` can describe, and so the width of the budget
/// arrays -- `VK_MAX_MEMORY_HEAPS`, spelled here because the generated types have it as an array
/// length rather than a constant.
const MAX_HEAPS: usize = 16;

/// The last thing said about one heap, which decides whether the next answer is worth a line.
#[derive(Clone, Copy, Default)]
struct HeapTrace {
    /// The driver's figure last time, to tell a real move from jitter.
    driver: u64,
    /// Whether the driver's answer won last time.
    clamped: bool,
    /// When this heap last printed, for the heartbeat while a clamp holds.
    logged: Option<std::time::Instant>,
}

/// Everything charged, by who is answerable for it.
///
/// A charge is attributed to the context that made it for as long as that context lives: its
/// slot is what a per-context leak reads off, and what `VK_EXT_memory_budget` answers from. The
/// slot is retired with the context, and whatever is still charged to it then -- storage a
/// resource holds a share of and a compositor is still sampling, or a charge this renderer
/// failed to drop -- moves to the shared bucket rather than out of the total. Those bytes are
/// resident either way, and the cap is enforced against what is resident.
struct Ledger {
    ctxs: BTreeMap<ContextId, Slot>,
    /// What outlived the context it was charged to.
    shared: PerContext,
    /// What never had a context to be charged to. Classic's IOSurfaces: a described texture is
    /// unmappable and dies at the claim, so no vrend context ever holds a standing charge, and a
    /// slot per vrend context would attribute nothing. Kept apart from `shared` because the two
    /// say different things -- that one means "outlived its owner", this one means "never had
    /// one" -- and a leak reads differently under each.
    classic: PerContext,
    /// Whether the watermark has already been reported for the climb the total is on. See
    /// [`watermark`] for the two thresholds this latch sits between.
    warned: bool,
    /// When the census last printed. Both this and `warned` are sampled on the charge path,
    /// which already holds this lock -- which is why they live here and not beside the
    /// configuration on [`Budget`].
    census: std::time::Instant,
}

impl Ledger {
    fn new() -> Ledger {
        Ledger {
            ctxs: BTreeMap::new(),
            shared: PerContext::default(),
            classic: PerContext::default(),
            warned: false,
            // From now, not from the epoch: the first census is due an interval into the run,
            // rather than on the first allocation of a process that has nothing to say yet.
            census: std::time::Instant::now(),
        }
    }

    /// Total live bytes, whoever holds them. This is the number the cap is enforced against.
    fn bytes(&self) -> u64 {
        self.ctxs.values().map(|s| s.live.bytes()).sum::<u64>()
            + self.shared.bytes()
            + self.classic.bytes()
    }

    /// What the context holding this id calls itself, for a log line. Empty for a context that
    /// gave no name, and for one that is already gone.
    fn name_of(&self, ctx: ContextId) -> &str {
        self.ctxs.get(&ctx).map_or("", |s| s.name.as_str())
    }

    /// The slot a charge was made against, if it is still that slot.
    fn slot_of(&mut self, ctx: ContextKey) -> Option<&mut PerContext> {
        self.ctxs.get_mut(&ctx.id()).filter(|s| s.ctx == ctx).map(|s| &mut s.live)
    }
}

/// One context's slot in the ledger.
struct Slot {
    /// Which occupant of this context id the slot belongs to. A guest reuses context ids, and a
    /// charge made under one context can be credited after the next one with the same id has
    /// opened -- a surface a compositor released after the client that minted it was gone. The
    /// key is what keeps that credit off the new occupant's slot: a charge names the occupant it
    /// was made against, not merely the id, and a slot that is gone is credited to the shared
    /// bucket the retire moved it into.
    ///
    /// The occupant is [`ContextKey`], minted where a context is stood up, rather than a number
    /// this ledger counts for itself -- one fact, one owner. A second counter here would say the
    /// same thing as the key and be free to disagree with it.
    ctx: ContextKey,
    /// What the guest called this context, as the VMM passed it to `context_create` -- `python3`,
    /// `synoik`. A number alone names the culprit only to someone holding a process table from
    /// the same second; this is what makes a breakdown readable after the fact.
    ///
    /// It is here because it belongs to the context, and it arrives as an argument from the one
    /// place a context is stood up. The C reaches the same name through a thread-local set at
    /// each dispatch entry, which is the ambient binding the module doc rejects.
    name: String,
    live: PerContext,
}

/// One context's live allocations, by what they are and how big.
///
/// Only the histogram. A running total beside it would be a second answer to one question, and
/// the two would disagree the first time a path updated one of them -- so the totals below are
/// derived from this every time they are asked for. There are a handful of contexts and a handful
/// of distinct sizes; the cost is not worth a number that can drift.
#[derive(Default)]
struct PerContext {
    live: BTreeMap<(&'static str, u64), u32>,
}

impl PerContext {
    fn bytes(&self) -> u64 {
        self.live.iter().map(|((_, size), n)| size * u64::from(*n)).sum()
    }

    /// Move everything here into `other`. What a retire does with a slot's residue.
    fn drain_into(&mut self, other: &mut PerContext) {
        for ((what, size), n) in std::mem::take(&mut self.live) {
            *other.live.entry((what, size)).or_insert(0) += n;
        }
    }

    fn take(&mut self, what: &'static str, size: u64) {
        *self.live.entry((what, size)).or_insert(0) += 1;
    }

    /// Credit one charge. Every charge was taken against the bucket it is credited to -- a slot
    /// by its key, or the shared bucket a retire drained that slot into -- so an absent entry
    /// is a ledger that has lost count, and says so.
    fn credit(&mut self, what: &'static str, size: u64) {
        let n = self.live.get_mut(&(what, size)).expect("a charge is credited where it was taken");
        *n -= 1;
        if *n == 0 {
            self.live.remove(&(what, size));
        }
    }

    /// The biggest buckets first, which is what names a leak.
    fn worst(&self) -> Vec<(&'static str, u64, u32)> {
        let mut rows: Vec<_> = self.live.iter().map(|((w, size), n)| (*w, *size, *n)).collect();
        rows.sort_by_key(|(_, size, n)| std::cmp::Reverse(size * u64::from(*n)));
        rows
    }
}

/// A charge against the ledger, which credits itself when it is dropped.
///
/// Held by the record of whatever it paid for. Never constructed except by
/// [`Account::try_charge`], so a charge that exists was admitted, and a charge that is gone has
/// been credited -- there is no third state and nothing to keep in step. Who it is charged to is
/// fixed when it is made and never changes: where the credit lands when the context is already
/// gone is the ledger's decision, not the charge's.
pub struct Charge {
    budget: Arc<Budget>,
    payer: Payer,
    what: &'static str,
    size: u64,
}

/// Which bucket a charge was taken from, and so which one credits it.
///
/// Fixed when the charge is made and never revisited. Where the credit lands when a context is
/// already gone is the ledger's decision, not the charge's -- see [`Budget::retire`].
#[derive(Clone, Copy)]
enum Payer {
    /// The venus context that made it, by the occupant rather than the bare id.
    Ctx(ContextKey),
    /// Classic, which has one bucket and no per-context attribution to make.
    Classic,
}

impl Charge {
    pub fn size(&self) -> u64 {
        self.size
    }
}

impl std::fmt::Debug for Charge {
    /// Never the ledger behind it: a charge is interesting for what it is, and a `Debug` that
    /// walked to the budget would print every other context's business beside it.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.payer {
            Payer::Ctx(ctx) => {
                write!(f, "Charge(ctx {}, {} {})", ctx.id().get(), size(self.size), self.what)
            }
            Payer::Classic => write!(f, "Charge(classic, {} {})", size(self.size), self.what),
        }
    }
}

impl Drop for Charge {
    fn drop(&mut self) {
        let mut ledger = self.budget.ledger.lock().expect("the budget ledger");
        // The slot this was taken against, or -- if that context has since retired -- the shared
        // bucket the retire drained it into. Never the slot of a later context with the same id.
        match self.payer {
            Payer::Ctx(ctx) => match ledger.slot_of(ctx) {
                Some(per) => per.credit(self.what, self.size),
                None => ledger.shared.credit(self.what, self.size),
            },
            Payer::Classic => ledger.classic.credit(self.what, self.size),
        }
    }
}

/// Storage this renderer minted, and what it cost -- one value, because they have one lifetime.
///
/// The charge lives with the storage rather than on the record of whatever asked for it, so that
/// it is credited when the *storage* goes and not when the request does. A venus resource holding
/// a share keeps the storage alive past the context that made it, and a classic texture the guest
/// has unreffed sits in `Vrend.doomed` until a GL context can delete it; those bytes are the
/// host's to count in both cases, and a charge on the record would have been credited while the
/// memory stood.
pub struct Charged<T> {
    it: T,
    #[expect(dead_code, reason = "held for its Drop -- crediting the ledger is this going away")]
    charge: Charge,
}

impl<T> Charged<T> {
    pub fn new(it: T, charge: Charge) -> Charged<T> {
        Charged { it, charge }
    }

    pub fn it(&self) -> &T {
        &self.it
    }
}

/// A charged thing is the thing, to whoever only wanted the thing. This is what lets an IOSurface
/// keepalive be handed out as [`Held`](crate::surface::Held) without the holder learning that a
/// ledger exists, or being able to separate the surface from what it cost.
impl<T: crate::surface::Held> crate::surface::Held for Charged<T> {
    fn surface(&self) -> &crate::surface::Surface {
        self.it.surface()
    }
}
/// One context's key to the ledger.
///
/// Charges are made through this and never against the [`Budget`] directly, which is what makes
/// attribution structural: there is no context argument for a caller to pass wrongly, because the
/// only handle a context can reach is its own. Opening it opens the context's slot, and dropping
/// it retires the slot -- dropping the account is the destroy, so no teardown path has to
/// remember to say so.
pub struct Account {
    budget: Arc<Budget>,
    ctx: ContextKey,
}

impl Account {
    /// Open `ctx`'s slot. One at a time per id: a second opening while the first stands is not a
    /// guest's doing -- the VMM names contexts -- but this renderer holding two accounts for one.
    /// `name` is what the guest called this context; it is only ever a log line, so an empty one
    /// costs nothing but a less readable breakdown.
    pub fn open(budget: &Arc<Budget>, ctx: ContextKey, name: String) -> Account {
        let mut ledger = budget.ledger.lock().expect("the budget ledger");
        let prev = ledger.ctxs.insert(ctx.id(), Slot { ctx, name, live: PerContext::default() });
        assert!(prev.is_none(), "ctx {} opened a second budget account", ctx.id().get());
        Account { budget: Arc::clone(budget), ctx }
    }

    /// Take `size` bytes for this context, or say why not.
    ///
    /// Admission and charge are one step on purpose. As two -- ask, then take -- they are two
    /// values that must agree, and ring threads allocate concurrently, so between the two
    /// another context can take the room this one was told it had.
    pub fn try_charge(&self, what: &'static str, size: u64) -> Result<Charge, Refused> {
        let mut ledger = self.budget.ledger.lock().expect("the budget ledger");
        let live = ledger.bytes();
        if let Some(cap) = self.budget.cap
            && live.saturating_add(size) > cap
        {
            return Err(Refused { what, wanted: size, live, cap });
        }
        ledger
            .slot_of(self.ctx)
            .expect("an account's slot stands for as long as the account does")
            .take(what, size);
        // Both samples are taken here rather than from a timer thread: the ledger only changes on
        // this path, so a workload that has stopped allocating has nothing new to report, and
        // there is no thread to own. The cost is that a context born and gone between two ticks
        // is never seen by the census -- it is a sampler, and says so.
        self.budget.sample_locked(&mut ledger);
        Ok(Charge { budget: Arc::clone(&self.budget), payer: Payer::Ctx(self.ctx), what, size })
    }

    /// Whether a refusal should stop this context. See the module doc for why it must, by default.
    pub fn kills_context(&self) -> bool {
        self.budget.kills_context()
    }

    /// What this context holds. The cap is on the total, not on this -- see
    /// [`Budget::live`] for the number that is actually enforced.
    pub fn live(&self) -> u64 {
        self.budget.live_for(self.ctx.id())
    }

    /// Where this context stands against the cap, read in one lock.
    ///
    /// `None` with no cap configured: there is no budget to answer from, and the driver's own
    /// reply is then the only true one. The three numbers come back together because they are
    /// one observation -- read separately they are three values that must agree, and a
    /// concurrent charge between two of them makes them disagree.
    pub fn standing(&self) -> Option<Standing> {
        let cap = self.budget.cap?;
        let ledger = self.budget.ledger.lock().expect("the budget ledger");
        let own = ledger.ctxs.get(&self.ctx.id()).map_or(0, |s| s.live.bytes());
        Some(Standing { own, others: ledger.bytes().saturating_sub(own), cap })
    }

    /// What to report for one heap: the budget the guest is told, and the usage beside it.
    ///
    /// `driver` is what the host driver already wrote there, and it gets a vote: we never promise
    /// more than the hardware has. Zero means it declined to answer, which is not a promise of
    /// nothing.
    pub fn heap_answer(&self, st: Standing, heap: u32, heap_size: u64, driver: u64) -> (u64, u64) {
        let ours = st.ours(heap_size);
        // A budget of zero is not a legal answer for a heap that exists
        // (`VkPhysicalDeviceMemoryBudgetPropertiesEXT`), and `usage <= budget` is the other half
        // of the same requirement -- hence the floor here and the `min` below.
        let answer = if driver != 0 { ours.min(driver) } else { ours }.max(1);
        self.budget.trace_heap(self.ctx.id(), heap, st, ours, driver, answer);
        (answer, st.own.min(answer))
    }

    /// Say what was refused and what the process is holding, which is the whole reason the
    /// histogram exists: one repeated call site reads as a single line naming its size and count.
    /// Verbose on purpose. What the guest sees is a lost device or an aborted process, which is
    /// also what a dozen unrelated venus transport failures look like -- so a refusal that did
    /// not name itself would be read as a transport bug.
    pub fn report_refusal(&self, refused: Refused) {
        let ledger = self.budget.ledger.lock().expect("the budget ledger");
        eprintln!(
            "[virglrs] {GRAMMAR}: REFUSING a {} {} allocation for {}",
            size(refused.wanted),
            refused.what,
            named(self.ctx.id(), ledger.name_of(self.ctx.id())),
        );
        eprintln!(
            "[virglrs] {GRAMMAR}:   and {}: this is limina's host-memory cap ({CAP_ENV}), not \
             the GPU running out of memory",
            if self.budget.kills_context() {
                "killing this context deliberately"
            } else {
                "returning an error the guest will not read"
            },
        );
        self.budget.report_locked(&ledger, "at refusal");
    }

    /// An account on a ledger of its own, for a test with no renderer around it.
    ///
    /// Tests never read the environment: the harness runs them in one process in parallel, so a
    /// test that set `LIMINA_GPU_MEM_BUDGET_MIB` would be setting it for every other test at once.
    #[cfg(test)]
    pub fn for_test(cap: Option<u64>) -> Account {
        Account::open(
            &Budget::with_cap(cap, false),
            ContextKey::for_test(ContextId::new(1).expect("1 is not zero")),
            String::new(),
        )
    }
}

impl Drop for Account {
    fn drop(&mut self) {
        self.budget.retire(self.ctx);
    }
}

/// Classic's handle to the ledger, which is nothing like an [`Account`].
///
/// It has no `standing()`, no `heap_answer()` and no retiring `Drop`, because none of the three
/// means anything here: classic answers no `VK_EXT_memory_budget` query, and there is no context
/// whose destroy would retire a slot. It also cannot refuse. A refused `resource_create` becomes
/// `RESP_ERR` after the guest kernel has already handed the handle out, so the guest goes on to
/// use a resource that was never made and poisons itself several commands later -- the same blind
/// refusal venus has, with the cause further away. This counts, and that is the whole job.
pub struct Classic {
    budget: Arc<Budget>,
}

impl Classic {
    pub fn open(budget: &Arc<Budget>) -> Classic {
        Classic { budget: Arc::clone(budget) }
    }

    /// Take `size` bytes for classic. Infallible by design -- see the type's doc.
    pub fn charge(&self, what: &'static str, size: u64) -> Charge {
        let mut ledger = self.budget.ledger.lock().expect("the budget ledger");
        ledger.classic.take(what, size);
        // Sampled here for the reason a context's charge is: this is the only path that moves the
        // ledger, so a renderer that has stopped allocating has nothing new to report.
        self.budget.sample_locked(&mut ledger);
        Charge { budget: Arc::clone(&self.budget), payer: Payer::Classic, what, size }
    }

    /// A handle on a ledger of its own, for a test with no renderer around it.
    #[cfg(test)]
    pub fn for_test() -> Classic {
        Classic::open(&Budget::with_cap(None, false))
    }
}

/// Where one context stands against the cap: what it holds, what everyone else holds, and the
/// cap itself.
///
/// The split matters because the cap is global while the ledger is per-context. "What is left for
/// me" is the cap minus what *others* hold -- excluding this context's own bytes, so a client's
/// own allocating cannot lower the budget it is told, which is the only thing that makes the
/// number something a client can back off against.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Standing {
    /// What this context holds.
    pub own: u64,
    /// What every other holder holds, including what outlived the context charged for it.
    pub others: u64,
    /// The configured cap, on the total of the two.
    pub cap: u64,
}

impl Standing {
    /// What we answer for one heap, before the driver gets a vote.
    ///
    /// Floored at our own usage because memory already held is inside one's budget, and because
    /// an exhausted cap would otherwise report a budget below the usage beside it. Clamped to the
    /// heap because no arithmetic here can conjure memory the hardware does not have.
    pub fn ours(&self, heap_size: u64) -> u64 {
        self.cap.saturating_sub(self.others).max(self.own).min(heap_size)
    }
}

/// A refusal, carrying what it takes to say so.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Refused {
    /// What the allocation was for, in the words the histogram files it under.
    pub what: &'static str,
    pub wanted: u64,
    pub live: u64,
    pub cap: u64,
}

impl Budget {
    /// The budget limina configured, read once.
    ///
    /// Unset or zero is accounting with no cap, so a bare virglrenderer and every replay are
    /// unaffected. A value that is not a number is a misconfiguration worth saying out loud
    /// rather than silently running uncapped -- but not worth refusing to start over.
    pub fn from_env() -> Arc<Budget> {
        let cap = match std::env::var(CAP_ENV) {
            Err(_) => None,
            Ok(v) => match v.trim().parse::<u64>() {
                Ok(0) => None,
                Ok(mib) => Some(mib * 1024 * 1024),
                Err(_) => {
                    eprintln!("[virglrs] {CAP_ENV}={v:?} is not a number of MiB; no cap");
                    None
                }
            },
        };
        let soft = std::env::var(SOFT_ENV).is_ok_and(|v| v == "1");
        let trace = std::env::var(TRACE_ENV).is_ok_and(|v| !v.is_empty() && v != "0");
        // Read independently of the cap: the census is the leak-hunting instrument, and the run
        // being hunted is usually the one deliberately left uncapped.
        let census = match std::env::var(CENSUS_ENV) {
            Err(_) => None,
            Ok(v) => match v.trim().parse::<u64>() {
                Ok(0) => None,
                Ok(secs) => Some(std::time::Duration::from_secs(secs)),
                Err(_) => {
                    eprintln!("[virglrs] {CENSUS_ENV}={v:?} is not a number of seconds; no census");
                    None
                }
            },
        };
        if let Some(bytes) = cap {
            eprintln!(
                "[virglrs] {GRAMMAR}: cap {} MiB, refusal {}",
                bytes / (1024 * 1024),
                if soft { "returns an error the guest will not read" } else { "stops the context" }
            );
        }
        if let Some(every) = census {
            eprintln!("[virglrs] {GRAMMAR}: census every {} s", every.as_secs());
        }
        Budget::new(cap, soft, trace, census)
    }

    /// A budget with the cap given rather than the one configured. The tests' way in: reading the
    /// environment from a test would make every other test in the process depend on it.
    pub fn with_cap(cap: Option<u64>, soft: bool) -> Arc<Budget> {
        Budget::new(cap, soft, false, None)
    }

    /// A budget that reports on a timer, for a test that wants the sampling path taken.
    #[cfg(test)]
    fn with_census(cap: Option<u64>, every: std::time::Duration) -> Arc<Budget> {
        Budget::new(cap, false, false, Some(every))
    }

    /// As [`Budget::with_cap`], saying also whether every budget answer is traced and how often
    /// the breakdown prints anyway.
    fn new(
        cap: Option<u64>,
        soft: bool,
        trace: bool,
        census: Option<std::time::Duration>,
    ) -> Arc<Budget> {
        Arc::new(Budget {
            cap,
            soft,
            trace,
            census,
            ledger: Mutex::new(Ledger::new()),
            heaps: Mutex::new([HeapTrace::default(); MAX_HEAPS]),
        })
    }

    /// The two things a charge is the occasion to look at: whether the total has climbed far
    /// enough to be worth warning about, and whether the census is due.
    ///
    /// Takes the ledger it is to read rather than locking, because its only caller is holding
    /// that lock already -- the charge it is reporting on is the one that just landed, and a
    /// second acquisition here would deadlock on a mutex that does not recurse.
    fn sample_locked(&self, ledger: &mut Ledger) {
        if let Some(cap) = self.cap {
            let (report, warned) = watermark(ledger.warned, ledger.bytes(), cap);
            ledger.warned = warned;
            if report {
                self.report_locked(ledger, &format!("{WARN_AT}% watermark crossed"));
            }
        }
        if let Some(every) = self.census {
            let now = std::time::Instant::now();
            if now.duration_since(ledger.census) >= every {
                ledger.census = now;
                self.report_locked(ledger, "census");
            }
        }
    }

    /// Explain one heap's reported budget, because what the guest finally sees is
    /// `min(our answer, the driver's)` and only the host can tell those two apart.
    ///
    /// That is not academic: our answer does not move as the asking client allocates, so a
    /// guest-visible budget that *grows* across a client's own allocations can only be the
    /// driver's number rising -- it tracks real host GPU pressure. Without this line the two are
    /// indistinguishable from our ledger being wrong.
    ///
    /// Emission is deliberately asymmetric. A client may query the budget every frame, so the
    /// full line is gated on the trace flag, while a clamp -- which changes what the guest is
    /// told -- is logged untraced, but only on the transition plus a slow heartbeat while it
    /// holds. Logging the sample instead of the event is what once put ten lines a second into a
    /// supervisor log.
    fn trace_heap(
        &self,
        ctx: ContextId,
        heap: u32,
        st: Standing,
        ours: u64,
        driver: u64,
        answer: u64,
    ) {
        let clamped = driver != 0 && ours > driver;
        let transition = {
            let mut heaps = self.heaps.lock().expect("the budget heap trace");
            // A heap index past the array is a driver describing more heaps than Vulkan allows;
            // there is nothing to remember about it, so it only ever prints when traced.
            match heaps.get_mut(heap as usize) {
                None => false,
                Some(h) => {
                    // Only a material move counts, or the driver's jitter alone would reprint.
                    let moved = h.driver == 0 || driver.abs_diff(h.driver) > h.driver / 8;
                    let due = h.logged.is_none_or(|t| t.elapsed() >= HEARTBEAT);
                    let transition = clamped != h.clamped || (clamped && moved && due);
                    if transition {
                        h.logged = Some(std::time::Instant::now());
                    }
                    h.clamped = clamped;
                    h.driver = driver;
                    transition
                }
            }
        };
        if !self.trace && !(clamped && transition) {
            return;
        }
        eprintln!(
            "[virglrs] {GRAMMAR}: memory_budget ctx {} heap {heap} own={} others={} cap={} \
             ours={ours} driver={driver} final={answer} clamped={}",
            ctx.get(),
            st.own,
            st.others,
            st.cap,
            u8::from(clamped),
        );
    }

    /// Whether a refusal should stop the context. See the module doc for why it must, by default.
    pub fn kills_context(&self) -> bool {
        !self.soft
    }

    /// Total live bytes across every context.
    pub fn live(&self) -> u64 {
        self.ledger.lock().expect("the budget ledger").bytes()
    }

    /// What outlived the context it was charged to, and is still resident.
    pub fn shared(&self) -> u64 {
        self.ledger.lock().expect("the budget ledger").shared.bytes()
    }

    /// What classic holds, which is every IOSurface it has minted.
    pub fn classic(&self) -> u64 {
        self.ledger.lock().expect("the budget ledger").classic.bytes()
    }

    /// What one context holds.
    pub fn live_for(&self, ctx: ContextId) -> u64 {
        self.ledger.lock().expect("the budget ledger").ctxs.get(&ctx).map_or(0, |s| s.live.bytes())
    }

    /// Retire a context's slot, because the context is gone.
    ///
    /// A guest reuses context ids, so a slot left behind would bill a dead context's bytes to
    /// whoever takes its number next. What is still charged to it is not uncounted, though: the
    /// bytes are resident -- storage a resource holds a share of that a compositor is still
    /// sampling, which is the ordinary case, or a [`Charge`] this renderer failed to drop, which
    /// is a leak -- and the two cannot be told apart here. Both move to the shared bucket, where
    /// the cap goes on seeing them and the report names them, and each is credited from there
    /// when its holder lets go.
    fn retire(&self, ctx: ContextKey) {
        let mut ledger = self.ledger.lock().expect("the budget ledger");
        // Only this account's slot: `open` refuses a second account for a live id, so a slot
        // under another occupant here is one a later account owns, and stays.
        if ledger.ctxs.get(&ctx.id()).is_none_or(|s| s.ctx != ctx) {
            return;
        }
        let mut slot = ledger.ctxs.remove(&ctx.id()).expect("just found");
        let residual = slot.live.bytes();
        if residual != 0 {
            eprintln!(
                "[virglrs] {GRAMMAR}: {} destroyed with {} still charged, now counted as shared \
                 -- storage a resource still holds a share of, or a charge this renderer did not \
                 drop",
                named(ctx.id(), &slot.name),
                size(residual),
            );
            eprintln!("[virglrs] {GRAMMAR}:   {}", histogram(&slot.live));
        }
        slot.live.drain_into(&mut ledger.shared);
    }

    /// Say what everything holds, biggest first. What makes a leak name itself.
    pub fn report(&self, reason: &str) {
        let ledger = self.ledger.lock().expect("the budget ledger");
        self.report_locked(&ledger, reason);
    }

    /// [`Budget::report`] against a ledger the caller is already holding.
    ///
    /// The split exists because the watermark and the census report from inside a charge, which
    /// holds this lock for the duration -- see [`Budget::sample_locked`].
    fn report_locked(&self, ledger: &Ledger, reason: &str) {
        let live = ledger.bytes();
        let against = match self.cap {
            Some(cap) => format!("{} cap ({}%)", size(cap), percent(live, cap)),
            None => String::from("no cap"),
        };
        eprintln!("[virglrs] {GRAMMAR}: {reason} — {} live of {against}", size(live));
        for (ctx, slot) in ledger.ctxs.iter() {
            eprintln!(
                "[virglrs] {GRAMMAR}:   {}: {} live — {}",
                named(*ctx, &slot.name),
                size(slot.live.bytes()),
                histogram(&slot.live),
            );
        }
        if ledger.classic.bytes() != 0 {
            eprintln!(
                "[virglrs] {GRAMMAR}:   classic, which has no context to name: {} live — {}",
                size(ledger.classic.bytes()),
                histogram(&ledger.classic),
            );
        }
        if ledger.shared.bytes() != 0 {
            eprintln!(
                "[virglrs] {GRAMMAR}:   outliving their contexts: {} live — {}",
                size(ledger.shared.bytes()),
                histogram(&ledger.shared),
            );
        }
    }
}

/// Whether the watermark is worth a line, and what the latch becomes.
///
/// Split out from the reporting so it can be tested for what it decides rather than for what it
/// prints. Returns `(report, warned)`: it arms once at [`WARN_AT`] and re-arms only once the
/// total has fallen back below [`REARM_BELOW`], so a workload hovering at the mark says it once.
fn watermark(warned: bool, live: u64, cap: u64) -> (bool, bool) {
    match percent(live, cap) {
        pct if !warned && pct >= WARN_AT => (true, true),
        pct if warned && pct < REARM_BELOW => (false, false),
        _ => (false, warned),
    }
}

/// How full the cap is, in whole percent. Widened because `live` is a byte count and a cap can be
/// tens of gigabytes: `live * 100` is not a `u64` multiplication anyone should have to think about.
fn percent(live: u64, cap: u64) -> u64 {
    if cap == 0 {
        return 0;
    }
    (u128::from(live) * 100 / u128::from(cap)) as u64
}

/// A context as a log line names it: the id, and what it called itself if it said.
fn named(ctx: ContextId, name: &str) -> String {
    match name {
        "" => format!("ctx {}", ctx.get()),
        name => format!("ctx {} [{name}]", ctx.get()),
    }
}

/// The live allocations on one line, biggest bucket first -- `4 x 14.1 MiB (IOSurface)`.
///
/// One line rather than one per bucket because this is what a leak looks like when it is read:
/// `767 x 31.6 MiB (device memory)` is a repeated call site named outright, and it should not need
/// scrolling to find. Bounded, because a pathological context could otherwise put a hundred
/// buckets into a supervisor log -- the tail is the part that never mattered.
fn histogram(per: &PerContext) -> String {
    /// Enough to see the shape; anything past this is noise beside the buckets above it.
    const MOST: usize = 6;
    let worst = per.worst();
    if worst.is_empty() {
        return String::from("nothing live");
    }
    let mut line = worst
        .iter()
        .take(MOST)
        .map(|(what, bytes, n)| format!("{n} x {} ({what})", size(*bytes)))
        .collect::<Vec<_>>()
        .join(", ");
    if worst.len() > MOST {
        line.push_str(&format!(", and {} smaller", worst.len() - MOST));
    }
    line
}

/// Bytes as a human reads them. A budget line is read while something is on fire.
fn size(bytes: u64) -> String {
    const MIB: f64 = 1024.0 * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    if bytes < 1024 * 1024 {
        format!("{bytes} B")
    } else if (bytes as f64) < GIB {
        format!("{:.1} MiB", bytes as f64 / MIB)
    } else {
        format!("{:.1} GiB", bytes as f64 / GIB)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(n: u32) -> ContextId {
        ContextId::new(n).expect("not zero")
    }

    /// A fresh occupant of id `n`. Twice for one id is two occupants, which is what the reuse
    /// tests are about.
    fn key(n: u32) -> ContextKey {
        ContextKey::for_test(ctx(n))
    }

    /// Classic's bucket and the shared one are different answers, and stay apart.
    ///
    /// They are both "bytes with no live context behind them", which is exactly why folding them
    /// together would be easy and wrong: shared means a context held this and is gone, classic
    /// means no context ever held it. A leak reads differently under each -- the first names a
    /// destroy path that did not credit, the second names a resource nobody unreffed -- and the
    /// cap counts both either way.
    #[test]
    fn classic_and_shared_are_two_buckets_of_one_total() {
        let budget = Budget::with_cap(None, false);
        let classic = Classic::open(&budget);
        let surface = classic.charge("IOSurface", 4096);

        let account = Account::open(&budget, key(1), String::from("synoik"));
        let outliving = account.try_charge("device memory", 8192).expect("no cap");
        drop(account);

        assert_eq!(budget.classic(), 4096, "classic's, and not moved by a context retiring");
        assert_eq!(budget.shared(), 8192, "the retired context's residue, and only that");
        assert_eq!(budget.live(), 4096 + 8192, "the cap sees both");

        drop(surface);
        assert_eq!(budget.classic(), 0, "credited to the bucket it was taken from");
        assert_eq!(budget.shared(), 8192, "and nothing else moved");
        drop(outliving);
        assert_eq!(budget.live(), 0);
    }

    #[cfg(target_os = "macos")]
    /// What a classic mint builds, and what finally credits it.
    ///
    /// The charge rides inside the `Arc` the EGL image holds, so the thing that credits it is the
    /// last holder of the *surface* letting go -- a venus context that imported the share, or a
    /// texture still sitting in `Vrend.doomed` after the guest unreffed its resource. A charge
    /// kept on the resource record instead would be credited at the unref, while the memory stood.
    #[test]
    fn a_charged_share_is_credited_by_its_last_holder_and_not_its_first() {
        let budget = Budget::with_cap(None, false);
        let classic = Classic::open(&budget);
        let surface = crate::surface::Surface::plain(64, 64, crate::surface::PixelFormat::Bgra)
            .expect("minted");
        let id = surface.id();
        let bytes = surface.alloc_size();
        assert!(bytes >= 64 * 64 * 4, "a 64x64 BGRA surface is at least its pixels");

        let charge = classic.charge("IOSurface", bytes);
        let held: Arc<dyn crate::surface::Held> = Arc::new(Charged::new(surface, charge));
        assert_eq!(budget.classic(), bytes, "charged once, when it was minted");

        // What `Storage::lent` hands a venus context: a second holder of the one share, never a
        // second charge over the same surface.
        let lent = Arc::clone(&held);
        assert_eq!(budget.classic(), bytes, "an import counts nothing new");
        assert_eq!(lent.surface().id(), id, "and reaches the same surface through the charge");

        drop(held);
        assert_eq!(budget.classic(), bytes, "the first holder letting go credits nothing");
        drop(lent);
        assert_eq!(budget.classic(), 0, "the last one does");
    }

    /// A charge outlives the context that made it, and the ledger goes on counting it for as
    /// long as it stands -- attributed to the context while the context lives, and to nobody
    /// after, but never uncounted.
    ///
    /// This is a compositor still sampling a client's last frame after the client has exited:
    /// the client's context is gone, its surface is not, and the cap has to know.
    #[test]
    fn a_charge_that_outlives_its_context_stays_counted() {
        let budget = Budget::with_cap(Some(8192), false);
        let account = Account::open(&budget, key(1), String::new());
        let charge = account.try_charge("IOSurface", 4096).expect("under the cap");
        assert_eq!(budget.live_for(ctx(1)), 4096, "the context's own, for as long as it lives");
        assert_eq!(budget.shared(), 0);

        // The context is destroyed while the surface is still held.
        drop(account);
        assert_eq!(budget.live_for(ctx(1)), 0, "the slot is gone with the context");
        assert_eq!(budget.shared(), 4096, "what it held is nobody's now, and still resident");
        assert_eq!(budget.live(), 4096, "a context's destroy does not uncount what outlives it");
        let other = Account::open(&budget, key(2), String::new());
        let _at_cap = other.try_charge("device memory", 4096).expect("exactly at the cap");
        assert!(
            other.try_charge("device memory", 1).is_err(),
            "the cap saw the outliving bytes, not just ctx 2's own"
        );

        drop(charge);
        assert_eq!(budget.shared(), 0, "the last holder going is what credits it");
        assert_eq!(budget.live(), 4096, "and ctx 2's own is untouched");
    }

    /// A guest reuses context ids. A charge made under one opening of an id and credited after
    /// the next opening has to credit what it was taken against -- the shared bucket the retire
    /// moved it into -- and not the new occupant's slot, which never took it.
    #[test]
    fn a_late_credit_never_lands_on_the_next_context_with_the_same_id() {
        let budget = Budget::with_cap(None, false);
        let first = Account::open(&budget, key(8), String::new());
        let held = first.try_charge("IOSurface", 600).expect("no cap");
        drop(first);
        assert_eq!(budget.shared(), 600);

        let second = Account::open(&budget, key(8), String::new());
        let _own = second.try_charge("device memory", 900).expect("no cap");
        assert_eq!(budget.live_for(ctx(8)), 900, "the next generation starts clean");

        drop(held);
        assert_eq!(budget.live_for(ctx(8)), 900, "the survivor's own charge is untouched");
        assert_eq!(budget.shared(), 0, "the old one was credited where it had gone");
        assert_eq!(budget.live(), 900);
    }

    /// A charge is credited by being dropped, and by nothing else. That is the whole design: the
    /// record that owns the allocation owns the charge, so every path that retires the record --
    /// including ones not written yet -- credits the ledger without knowing it exists.
    #[test]
    fn a_charge_is_credited_by_going_out_of_scope() {
        let budget = Budget::with_cap(None, false);
        assert_eq!(budget.live(), 0);

        let account = Account::open(&budget, key(1), String::new());
        let a = account.try_charge("device memory", 4096).expect("no cap");
        let b = account.try_charge("device memory", 4096).expect("no cap");
        assert_eq!(budget.live(), 8192, "two of the same size are two, not one");

        drop(a);
        assert_eq!(budget.live(), 4096);
        drop(b);
        assert_eq!(budget.live(), 0, "and the bucket is gone, not left at zero");
    }

    /// The cap is on the process, not on a context: the host OS kills the worker for the total.
    /// So one context's allocations have to count against another's headroom.
    #[test]
    fn the_cap_is_what_every_context_holds_together() {
        let budget = Budget::with_cap(Some(1000), false);
        let one = Account::open(&budget, key(1), String::new());
        let two = Account::open(&budget, key(2), String::new());
        let held = one.try_charge("device memory", 800).expect("fits");

        let refused = two.try_charge("device memory", 300).expect_err("does not fit");
        assert_eq!(refused, Refused { what: "device memory", wanted: 300, live: 800, cap: 1000 });

        // Bound, not dropped on the spot: a charge that goes out of scope at the end of the
        // statement that made it is credited before the next line reads the ledger.
        let _exact =
            two.try_charge("device memory", 200).expect("exactly the room left is room enough");

        drop(held);
        let _reused = one
            .try_charge("device memory", 800)
            .expect("freeing reopens the headroom it was using");
        assert_eq!(budget.live(), 1000, "and the ledger is at the cap, not over it");
    }

    /// Retiring a context takes its slot, and the residue goes to the shared bucket rather than
    /// out of the total: the bytes are still resident, and the cap is on what is resident. The
    /// next generation of the same id starts with a clean slot, and with the room that is
    /// actually there.
    #[test]
    fn retiring_a_context_takes_its_slot_and_not_its_bytes() {
        let budget = Budget::with_cap(Some(1000), false);
        let first = Account::open(&budget, key(8), String::new());
        let charge = first.try_charge("device memory", 600).expect("fits");

        drop(first);
        assert_eq!(budget.live_for(ctx(8)), 0, "the slot is gone with the context");
        assert_eq!(budget.live(), 600, "the bytes are not: something still holds them");
        let next = Account::open(&budget, key(8), String::new());
        assert!(
            next.try_charge("device memory", 900).is_err(),
            "the room is what is actually free, not what the new context alone is using"
        );
        drop(charge);
        let _fits = next.try_charge("device memory", 900).expect("and now it is free");
    }

    /// What a context is told it has is the cap minus what *others* hold -- so its own allocating
    /// does not shrink the number it is backing off against, which is the only thing that makes
    /// the number usable.
    #[test]
    fn the_budget_answered_is_what_is_left_for_this_context() {
        const HEAP: u64 = 64 * 1024;
        let budget = Budget::with_cap(Some(1000), false);
        let mine = Account::open(&budget, key(1), String::new());
        let other = Account::open(&budget, key(2), String::new());

        let st = mine.standing().expect("a cap is configured");
        assert_eq!(st, Standing { own: 0, others: 0, cap: 1000 });
        assert_eq!(st.ours(HEAP), 1000, "an empty ledger offers the whole cap");

        let _theirs = other.try_charge("device memory", 400).expect("fits");
        let _mine = mine.try_charge("device memory", 200).expect("fits");
        let st = mine.standing().expect("a cap is configured");
        assert_eq!(st, Standing { own: 200, others: 400, cap: 1000 });
        assert_eq!(st.ours(HEAP), 600, "what is left for me counts my own bytes as still mine");

        // The heap is the ceiling no arithmetic here may exceed.
        assert_eq!(st.ours(500), 500, "we never promise more than the hardware has");

        // An exhausted cap still owes a budget at least as big as the usage beside it.
        let _rest = other.try_charge("device memory", 400).expect("takes the cap to the line");
        let st = mine.standing().expect("a cap is configured");
        assert_eq!(st.others, 1000 - 200);
        assert_eq!(st.ours(HEAP), 200, "floored at what I hold, because I do hold it");
    }

    /// The driver gets a vote and the spec gets the last word: never more than the host offers,
    /// never zero, and never a usage above the budget printed beside it.
    #[test]
    fn the_drivers_answer_clamps_ours_and_the_spec_floors_both() {
        const HEAP: u64 = 1 << 40;
        let budget = Budget::with_cap(Some(1000), false);
        let mine = Account::open(&budget, key(1), String::new());
        let _held = mine.try_charge("device memory", 200).expect("fits");
        let st = mine.standing().expect("a cap is configured");

        let (answer, usage) = mine.heap_answer(st, 0, HEAP, 0);
        assert_eq!((answer, usage), (1000, 200), "a silent driver leaves our answer standing");

        let (answer, usage) = mine.heap_answer(st, 0, HEAP, 900);
        assert_eq!((answer, usage), (900, 200), "a tighter driver wins");

        let (answer, usage) = mine.heap_answer(st, 0, HEAP, 5000);
        assert_eq!((answer, usage), (1000, 200), "a looser one does not");

        // A heap index past what Vulkan allows has no trace slot; it must still answer.
        let (answer, usage) = mine.heap_answer(st, MAX_HEAPS as u32 + 3, HEAP, 1);
        assert_eq!((answer, usage), (1, 1), "budget floored at one, and usage never above it");
    }

    /// The watermark says it once on the way up, and again only after the total has come back
    /// down. A workload that sits at the mark -- which is exactly what a leaking one does, one
    /// allocation at a time -- would otherwise reprint the whole breakdown on every charge.
    #[test]
    fn the_watermark_fires_on_the_climb_and_re_arms_only_after_a_fall() {
        const CAP: u64 = 1000;
        // Below the mark, nothing is said and nothing is armed.
        assert_eq!(watermark(false, 700, CAP), (false, false));
        assert_eq!(watermark(false, 799, CAP), (false, false));

        // Crossing it reports, once.
        assert_eq!(watermark(false, 800, CAP), (true, true), "80% is the mark, and it is crossed");
        assert_eq!(watermark(true, 850, CAP), (false, true), "climbing further says nothing more");
        assert_eq!(watermark(true, 999, CAP), (false, true), "nor does approaching the cap");

        // Falling back to just under the mark is not a fall: it is the hovering the second
        // threshold exists for.
        assert_eq!(watermark(true, 750, CAP), (false, true), "still latched between the marks");
        assert_eq!(watermark(true, 699, CAP), (false, false), "below 70% it re-arms");
        assert_eq!(watermark(false, 800, CAP), (true, true), "and the next climb reports again");

        // A cap of zero is no cap at all; the caller does not reach here, but the arithmetic
        // must not divide by it.
        assert_eq!(watermark(false, 800, 0), (false, false));
    }

    /// Both reports are printed from inside the charge that occasioned them, which is holding the
    /// ledger lock -- so both have to read the ledger they were handed rather than take it again.
    /// A `Mutex` does not recurse, so getting this wrong is not a wrong number but a worker that
    /// stops: this test hangs rather than fails if the reporting path ever locks for itself.
    #[test]
    fn reporting_from_inside_a_charge_does_not_take_the_lock_it_is_already_holding() {
        // Due on the first charge, so the census path is taken rather than merely available.
        let budget = Budget::with_census(Some(1000), std::time::Duration::ZERO);
        let account = Account::open(&budget, key(1), String::from("python3"));

        // Under the mark: census only.
        let small = account.try_charge("device memory", 100).expect("fits");
        // Over it: the watermark reports too, from the same held lock.
        let _crossing = account.try_charge("device memory", 800).expect("fits");
        assert_eq!(budget.live(), 900, "and the charges landed, both of them");

        // The refusal path reports twice more while holding it once.
        let refused = account.try_charge("device memory", 200).expect_err("over the cap");
        account.report_refusal(refused);

        // Falling back below the re-arm mark and climbing again takes the same path a second
        // time, which a latch left set by a report that unwound would not.
        drop(small);
        let _again = account.try_charge("device memory", 100).expect("fits under the cap");
        assert_eq!(budget.live(), 900);
    }

    /// The name is what makes a breakdown readable an hour later, and it is an argument the whole
    /// way down -- from the VMM's `context_create`, through the account, into the slot. Nothing
    /// consults a thread to find out who is being charged.
    #[test]
    fn a_context_is_named_in_the_ledger_by_the_name_it_was_opened_with() {
        let budget = Budget::with_cap(None, false);
        let named_ctx = Account::open(&budget, key(1), String::from("synoik"));
        let anonymous = Account::open(&budget, key(2), String::new());
        let _theirs = named_ctx.try_charge("IOSurface", 4096).expect("no cap");

        {
            let ledger = budget.ledger.lock().expect("the budget ledger");
            assert_eq!(ledger.name_of(ctx(1)), "synoik");
            assert_eq!(ledger.name_of(ctx(2)), "", "a context that gave no name has none");
            assert_eq!(ledger.name_of(ctx(3)), "", "and neither has one that does not exist");
        }
        assert_eq!(named(ctx(1), "synoik"), "ctx 1 [synoik]");
        assert_eq!(named(ctx(2), ""), "ctx 2", "no name is no brackets, not empty ones");

        // The name goes with the context: a slot retired into the shared bucket keeps no
        // claim on it, because the bytes there are nobody's by then.
        drop(named_ctx);
        let ledger = budget.ledger.lock().expect("the budget ledger");
        assert_eq!(ledger.name_of(ctx(1)), "");
        drop(ledger);
        drop(anonymous);
    }

    /// The breakdown is one line per context, biggest bucket first, and bounded: a context with a
    /// hundred distinct sizes must not put a hundred lines into a supervisor log.
    #[test]
    fn the_breakdown_names_the_biggest_bucket_first_and_stays_one_line() {
        let mut per = PerContext::default();
        assert_eq!(histogram(&per), "nothing live");

        per.take("device memory", 4 * 1024 * 1024);
        per.take("device memory", 4 * 1024 * 1024);
        per.take("IOSurface", 1024 * 1024);
        assert_eq!(
            histogram(&per),
            "2 x 4.0 MiB (device memory), 1 x 1.0 MiB (IOSurface)",
            "by total bytes held, which is what names a leak"
        );

        for n in 1..20u64 {
            per.take("device memory", n * 1024);
        }
        let line = histogram(&per);
        assert_eq!(line.matches(" x ").count(), 6, "bounded at the buckets worth reading");
        assert!(line.ends_with("and 15 smaller"), "and the tail is counted, not dropped: {line}");
    }

    /// Sizes are read while something is on fire, and a cap is quoted in the units it was set in.
    #[test]
    fn a_size_is_printed_in_the_unit_a_reader_expects() {
        assert_eq!(size(4096), "4096 B", "below a MiB, exact bytes: a page is a page");
        assert_eq!(size(33_177_600), "31.6 MiB", "the backdrop that named the 2026-08-06 leak");
        assert_eq!(size(2048 * 1024 * 1024), "2.0 GiB", "and a cap in the units it was set in");
        assert_eq!(percent(1800, 2048), 87);
        assert_eq!(percent(0, 2048), 0);
        assert_eq!(percent(5, 0), 0, "an absent cap is never divided by");
    }

    /// Accounting is always on and enforcement is not. Without a cap nothing is ever refused --
    /// which is what a bare virglrenderer and every replay run as, and why the corpora do not
    /// move when this lands.
    #[test]
    fn without_a_cap_nothing_is_refused_and_everything_is_counted() {
        let budget = Budget::with_cap(None, false);
        let account = Account::open(&budget, key(1), String::new());
        let _huge = account.try_charge("device memory", u64::MAX / 2).expect("no cap");
        let _more = account.try_charge("IOSurface", u64::MAX / 2).expect("still no cap");
        assert_eq!(budget.live(), u64::MAX - 1, "counted, both of them");
        assert!(budget.kills_context(), "and a refusal would still stop the context");
    }
}
