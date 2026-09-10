// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! What the command path costs per guest command, for scoring a change to it.
//!
//! **Why this exists rather than a profiler.** The `gpu worker` thread is saturated: it runs at
//! ~90% of one core under a live desktop, and every guest context's commands go through it. A
//! sampling profile of a saturated thread reports *proportions*, and proportions cannot see a
//! throughput win — make each command cheaper and the thread stays pegged, doing more commands in
//! the same second, with every percentage roughly where it was. Measured that way, replacing the
//! command path's `BTreeMap`s with hash tables read as "no change" in the profile while the real
//! workload moved (vkmark 25-28 → 33-39 fps). The number that would have said so directly is
//! **microseconds per command**, and nothing was reporting it.
//!
//! So this counts work done, not time spent in functions, and divides.
//!
//! **What `busy` does and does not cover.** It is the time inside `Vrend::submit` and nothing else.
//! The fence path is a different entry point — `fence_context` and `fence_global` call
//! `take_fence` outside that window — so `us/cmd` and the submit line's `% of wall` are **blind to
//! what a fence costs**, and an A/B of the fence path that reads them sees nothing and means
//! nothing. That was learned by reporting exactly that. The fence line below carries its own timer
//! for the same reason, and the two shares are of the same wall clock, so they can be added.
//!
//! **Fences and presents are counted for the same reason, one level up.** A sampling profile says
//! what *share* of the worker a sync or a present wait took; it cannot say how many the guest asked
//! for. That is the question a change to the fence path leaves open — removing the drain may have
//! made each fence cheaper, or let the guest ask for more of them, and the profile reads the same
//! either way. Each is one add, on a path that already costs a GL call.
//!
//! **What it costs when armed.** A predictable branch and an add per command; two `Instant::now()`
//! calls per *submit*, never per command. A batch carries hundreds of commands, so the clock is
//! amortised to nothing. When it is not armed there is no clock at all and the per-command cost is
//! one test of an `Option` discriminant that is already in cache.
//!
//! **Printing is the hazard, not counting.** A line per submit would put `write(2)` on the hot
//! path and measure the logging. Nothing is printed until the reporting interval has elapsed, and
//! that deadline is checked once per submit.
//!
//! Armed by `VIRGLRS_SUBMIT_STATS`: `1` reports every 5 s, any other positive integer is the
//! interval in seconds. Unset, this is inert.

use std::time::{Duration, Instant};

/// Per-command cost of the classic command path, or nothing at all when unarmed.
#[derive(Default)]
pub struct Tally {
    /// `None` is the whole of "switched off": no clock is read, no field is updated, and there is
    /// nothing to flush at teardown.
    on: Option<Armed>,
}

/// How many positions in a fence's walk are reported separately. A context with more
/// sub-contexts than this has its tail folded into the last bucket -- the question is whether cost
/// is flat or front-loaded, and a tail bucket answers that as well as a longer array would.
const HOPS: usize = 8;

struct Armed {
    every: Duration,
    window_began: Instant,
    submits: u64,
    commands: u64,
    dwords: u64,
    /// Fences answered with a GL sync: the ones that cost a flush. On this share group that is a
    /// full handshake with mesa's threaded context -- see [`super::waiter`].
    syncs: u64,
    /// Fences answered by the queue's order alone, which cost nothing.
    ordered: u64,
    /// Sync objects taken, which is not the same as fences: one fence syncs every sub-context of
    /// its context plus ctx0. Divided by `syncs` this says how many queues a fence covers — the
    /// number a deduction about the sub-context count needs and cannot otherwise get.
    sync_objects: u64,
    /// Wall time inside `Vrend::take_fence`, the share of the worker the fence path costs. Kept
    /// apart from `busy` because they are different entry points; see the note above.
    fence_busy: Duration,
    /// The two halves of a hop, apart: time inside `eglMakeCurrent`, and time inside
    /// `glFenceSync`. They are separated because their fixes are opposite -- a cost in the switch
    /// is paid per context the walk visits and would be answered by visiting fewer of them, while
    /// a cost in the sync is the driver's and would be answered by asking it differently. One
    /// number covering both cannot choose.
    current_busy: Duration,
    sync_busy: Duration,
    /// Hops taken, and what each cost by its position in the walk. A walk whose per-hop cost is
    /// flat in the index is paying for the *visiting*; one where the first hop holds nearly all of
    /// it is paying for work that was queued, and visiting the others is free. That distinction is
    /// the whole question, and no aggregate answers it.
    hops: u64,
    hop_busy: [Duration; HOPS],
    hop_n: [u64; HOPS],
    /// Fences the driver refused a sync for, which fall back to finishing every context that could
    /// hold the work. Counted apart from `ordered` -- both answer `Answer::Ordered`, and one is a
    /// full `glFinish` while the other is free, so a single counter cannot tell an expensive window
    /// from a cheap one.
    drained: u64,
    /// Calls to `resource_sync_iosurface`: one blocking wait on the surface's shared event each.
    presents: u64,
    /// Wall time inside `Vrend::submit`. Against the window's own length this also says what
    /// share of the worker's second the command path took, which is the other half of the
    /// question: a cheap command path that is still 90% of the thread has not finished the job.
    busy: Duration,
}

impl Armed {
    fn new(every: Duration) -> Self {
        Self {
            every,
            window_began: Instant::now(),
            submits: 0,
            commands: 0,
            dwords: 0,
            syncs: 0,
            ordered: 0,
            sync_objects: 0,
            fence_busy: Duration::ZERO,
            current_busy: Duration::ZERO,
            sync_busy: Duration::ZERO,
            hops: 0,
            hop_busy: [Duration::ZERO; HOPS],
            hop_n: [0; HOPS],
            drained: 0,
            presents: 0,
            busy: Duration::ZERO,
        }
    }
}

impl Tally {
    /// Read the environment once, at renderer construction.
    ///
    /// Deliberately not a lazy `OnceLock` consulted on the hot path: this hangs off the renderer
    /// like every other piece of state, so there is no global to race on and no per-command
    /// `getenv` to regret. A renderer built without the variable set can never start reporting.
    pub fn from_env() -> Self {
        let Ok(v) = std::env::var("VIRGLRS_SUBMIT_STATS") else {
            return Self { on: None };
        };
        let secs = match v.trim() {
            "" | "0" => return Self { on: None },
            "1" => 5,
            other => match other.parse::<u64>() {
                Ok(n) if n > 0 => n,
                // A misspelled interval must not read as "off" -- that is a silent needle, and a
                // needle nothing arms is worse than no needle. Take the default and say so.
                _ => {
                    eprintln!(
                        "[virglrs] vrend: VIRGLRS_SUBMIT_STATS={other:?} is not a positive \
                         number of seconds; reporting every 5s"
                    );
                    5
                }
            },
        };
        eprintln!("[virglrs] vrend: submit stats on, reporting every {secs}s");
        Self { on: Some(Armed::new(Duration::from_secs(secs))) }
    }

    /// One guest command ran. The whole per-command cost of this instrument.
    #[inline]
    pub fn command(&mut self) {
        if let Some(a) = &mut self.on {
            a.commands += 1;
        }
    }

    /// The clock, or `None` when unarmed -- which is what keeps an unarmed build from reading it
    /// at all. Every duration below is a difference of two of these.
    #[inline]
    pub fn mark(&self) -> Option<Instant> {
        self.on.as_ref().map(|_| Instant::now())
    }

    /// One hop of a fence's walk: the context switch, then the sync taken on it.
    ///
    /// `index` is the hop's position in the walk, with ctx0 last. Takes the three marks rather
    /// than two durations so the split cannot be computed one way here and another way at the next
    /// call site.
    pub fn fence_hop(&mut self, index: usize, marks: Option<(Instant, Instant, Instant)>) {
        let (Some(a), Some((began, switched, synced))) = (&mut self.on, marks) else { return };
        let at = index.min(HOPS - 1);
        a.hops += 1;
        a.hop_n[at] += 1;
        a.hop_busy[at] += synced - began;
        a.current_busy += switched - began;
        a.sync_busy += synced - switched;
    }

    /// This fence was answered by finishing rather than by syncs: the driver refused one.
    pub fn fence_drained(&mut self) {
        if let Some(a) = &mut self.on {
            a.drained += 1;
        }
    }

    /// One guest fence was answered, and at what cost: a sync is a flush, an ordering is free.
    pub fn fence(&mut self, answer: &super::waiter::Answer, began: Option<Instant>) {
        let Some(a) = &mut self.on else { return };
        match answer {
            super::waiter::Answer::Syncs(f) => {
                a.syncs += 1;
                a.sync_objects += f.len() as u64;
            }
            super::waiter::Answer::Ordered => a.ordered += 1,
        }
        if let Some(began) = began {
            a.fence_busy += Instant::now() - began;
        }
    }

    /// One surface-backed resource was made whole for a present.
    #[inline]
    pub fn present(&mut self) {
        if let Some(a) = &mut self.on {
            a.presents += 1;
        }
    }

    /// Start of one batch, or `None` when unarmed -- which is also what stops the clock being read.
    #[inline]
    pub fn batch_began(&self) -> Option<Instant> {
        self.on.as_ref().map(|_| Instant::now())
    }

    /// End of one batch. Reports if the window is up, so the check happens per batch and never
    /// per command.
    pub fn batch_ended(&mut self, began: Option<Instant>, dwords: usize) {
        let (Some(a), Some(began)) = (&mut self.on, began) else { return };
        a.submits += 1;
        a.dwords += dwords as u64;
        let now = Instant::now();
        a.busy += now - began;
        let window = now - a.window_began;
        if window < a.every {
            return;
        }
        let secs = window.as_secs_f64();
        // Per *command*, because that is the unit a change to the command path moves. Per batch
        // would move when the guest changed how it packs them, which is not us.
        let per_cmd_us =
            if a.commands == 0 { 0.0 } else { a.busy.as_secs_f64() * 1e6 / a.commands as f64 };
        // The raw counts and the window go out beside the rates, because a rate alone cannot say
        // how much it is built on: an idle window rounded to "0 cmd/s" next to a us/cmd figure
        // derived from three commands reads as a contradiction, and an instrument that prints
        // nonsense once is an instrument nobody believes when it matters.
        eprintln!(
            "[virglrs] vrend submit: {:.0} cmd/s  {:.0} dw/s  {:.0} batch/s  \
             {per_cmd_us:.2} us/cmd  {:.1}% of wall  \
             (n={} cmd in {} batch over {secs:.1}s)",
            a.commands as f64 / secs,
            a.dwords as f64 / secs,
            a.submits as f64 / secs,
            100.0 * a.busy.as_secs_f64() / secs,
            a.commands,
            a.submits,
        );
        // Its own line, because the one above is the shape earlier runs were read with. Per submit
        // as well as per second: a guest that fences every batch and one that fences every tenth
        // are a different problem at the same rate.
        let fences = a.syncs + a.ordered;
        let per_submit = if a.submits == 0 { 0.0 } else { fences as f64 / a.submits as f64 };
        // Per fence and not per sync object: the unit a change to the fence path moves. The ratio
        // between the two is printed beside it, because it is how many GL queues one fence covers
        // and nothing else reports it.
        let per_fence_us =
            if fences == 0 { 0.0 } else { a.fence_busy.as_secs_f64() * 1e6 / fences as f64 };
        let syncs_each = if a.syncs == 0 { 0.0 } else { a.sync_objects as f64 / a.syncs as f64 };
        eprintln!(
            "[virglrs] vrend fences: {:.1} fence/s ({:.1} sync/s  {:.1} ordered/s)  \
             {per_submit:.2} fence/submit  {syncs_each:.2} sync/fenced  \
             {per_fence_us:.1} us/fence  {:.1}% of wall  {:.1} present/s  \
             (n={} sync, {} ordered, {} present over {secs:.1}s)",
            fences as f64 / secs,
            a.syncs as f64 / secs,
            a.ordered as f64 / secs,
            100.0 * a.fence_busy.as_secs_f64() / secs,
            a.presents as f64 / secs,
            a.syncs,
            a.ordered,
            a.presents,
        );
        // The decomposition, and the line that exists to choose between two fixes. Printed only
        // when there were hops: a window with none has nothing to say here and a row of zeroes
        // reads like an answer.
        if a.hops > 0 {
            let us =
                |d: Duration, n: u64| if n == 0 { 0.0 } else { d.as_secs_f64() * 1e6 / n as f64 };
            let mut by_index = String::new();
            for at in 0..HOPS {
                if a.hop_n[at] == 0 {
                    continue;
                }
                by_index.push_str(&format!(
                    " {}{}:{:.0}",
                    at,
                    if at == HOPS - 1 { "+" } else { "" },
                    us(a.hop_busy[at], a.hop_n[at])
                ));
            }
            eprintln!(
                "[virglrs] vrend fence hops: {:.1} hop/s  {:.1} us/hop                   ({:.1} us current  {:.1} us sync)  {:.1} drain/s  us/hop by index:{by_index}",
                a.hops as f64 / secs,
                us(a.current_busy + a.sync_busy, a.hops),
                us(a.current_busy, a.hops),
                us(a.sync_busy, a.hops),
                a.drained as f64 / secs,
            );
        }
        *a = Armed::new(a.every);
        a.window_began = now;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unarmed is inert: nothing counts, and no clock is offered to read.
    #[test]
    fn an_unarmed_tally_counts_nothing() {
        let mut t = Tally { on: None };
        assert!(t.batch_began().is_none());
        t.command();
        t.batch_ended(None, 4096);
        assert!(t.on.is_none(), "an unarmed tally has nothing to hold");
    }

    /// Armed, it accumulates -- and does not report before its window is up, which is what keeps
    /// `write(2)` off the hot path.
    #[test]
    fn an_armed_tally_accumulates_without_reporting() {
        let mut t = Tally { on: Some(Armed::new(Duration::from_secs(3600))) };
        for _ in 0..10 {
            let b = t.batch_began();
            assert!(b.is_some(), "an armed tally reads the clock once per batch");
            for _ in 0..100 {
                t.command();
            }
            t.fence(&super::super::waiter::Answer::Ordered, None);
            t.present();
            t.batch_ended(b, 512);
        }
        let a = t.on.as_ref().expect("armed");
        assert_eq!(a.commands, 1000);
        assert_eq!(a.dwords, 5120);
        assert_eq!(a.submits, 10, "the window has not elapsed, so nothing was flushed");
        assert_eq!((a.ordered, a.presents), (10, 10));
        assert_eq!(a.syncs, 0, "no sync was offered, and none may be invented");
    }

    /// A window that has elapsed reports and starts a fresh one, so a rate is never diluted by
    /// the window before it.
    #[test]
    fn a_report_resets_the_window() {
        let mut t = Tally { on: Some(Armed::new(Duration::ZERO)) };
        {
            let a = t.on.as_mut().expect("armed");
            a.submits = 7;
            a.commands = 700;
            a.dwords = 70;
            a.syncs = 3;
            a.ordered = 4;
            a.sync_objects = 9;
            a.fence_busy = Duration::from_millis(2);
            a.current_busy = Duration::from_millis(1);
            a.sync_busy = Duration::from_millis(1);
            a.hops = 9;
            a.hop_busy[0] = Duration::from_millis(1);
            a.hop_n[0] = 9;
            a.drained = 1;
            a.presents = 2;
            a.busy = Duration::from_millis(5);
        }
        let b = t.batch_began();
        t.command();
        t.batch_ended(b, 1);
        let a = t.on.as_ref().expect("armed");
        assert_eq!((a.commands, a.dwords, a.submits), (0, 0, 0));
        assert_eq!(
            (a.syncs, a.ordered, a.presents),
            (0, 0, 0),
            "every counter, or the next \
                    window's rate carries this one's"
        );
        assert_eq!(a.busy, Duration::ZERO);
        assert_eq!(a.sync_objects, 0);
        assert_eq!(a.fence_busy, Duration::ZERO, "the fence clock resets with the rest");
        // The decomposition too. A counter that survives its window does not read as a bug: it
        // reads as a rate, and the next window reports this one's cost as its own.
        assert_eq!((a.current_busy, a.sync_busy), (Duration::ZERO, Duration::ZERO));
        assert_eq!((a.hops, a.drained), (0, 0));
        assert_eq!(a.hop_busy[0], Duration::ZERO);
        assert_eq!(a.hop_n[0], 0, "the per-index buckets as well");
        assert_eq!(a.every, Duration::ZERO, "the interval is not a counter and must survive");
    }

    /// A junk interval arms at the default rather than silently disarming.
    #[test]
    fn a_junk_interval_still_arms() {
        // SAFETY: single-threaded test, and the variable is read once, below.
        unsafe { std::env::set_var("VIRGLRS_SUBMIT_STATS", "banana") };
        let t = Tally::from_env();
        unsafe { std::env::remove_var("VIRGLRS_SUBMIT_STATS") };
        assert!(t.on.is_some(), "a misspelled interval must not read as off");
    }
}
