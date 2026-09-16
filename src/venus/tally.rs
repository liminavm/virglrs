// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! What the venus command path costs per guest command, for scoring a change to it.
//!
//! The vrend tally's argument ([`crate::vrend::tally`]) holds here unchanged: a saturated thread
//! profiles as proportions, and proportions cannot see a throughput win. The number that can is
//! microseconds per command, so this counts commands and divides.
//!
//! **Two origins, counted apart.** A batch reaches a context from the virtqueue -- the ABI's
//! `submit` and its `resume` -- or from a ring, which is where a desktop's traffic is: mesa puts
//! nearly every `vkCmd*` on a ring and the virtqueue carries little but the ring's creation. A
//! change to the ring loop and a change to the decoder show up on different lines, and one line
//! covering both would hide a ring regression behind the virtqueue's noise.
//!
//! **No lock.** Ring batches run concurrently on different contexts, and the one lock every ring
//! used to contend on was removed on purpose. The counters are atomics reached through an `Arc`
//! that every ring's dispatcher holds beside the census; a batch adds to them once at its end, and
//! whichever thread's batch finds the window elapsed claims the report with a compare-exchange
//! and swaps every counter to zero. Nothing here is taken for a batch's length.
//!
//! **What a number means.** `busy` is wall time from the context lock being held to the batch
//! returning: decode, dispatch, the driver, the record into the journal, and the reply. It is
//! reported as *thread-seconds per second*, not a share of wall, because several rings busy at
//! once sum past one. `cmd/batch` is printed beside `us/cmd` for the same reason the vrend tally
//! never had to: a ring batch is small, so the two clock reads a batch costs are not amortised
//! across hundreds of commands, and the reader should see how much of `us/cmd` is the instrument.
//!
//! **What a replay number is.** Under the replayer every wait blocks inline, no ring thread runs,
//! and no reply is encoded. Its `us/cmd` covers decode, dispatch, the driver and the record, and
//! nothing of the reply path or the ring loop -- enough to score the decoder's copies, not the
//! desktop. The corpora also hold one command per record, so a replay's `cmd/batch` reads 1.0
//! and everything a batch costs -- the locks, the journal record, this clock -- lands on each
//! command; a desktop batch carries more and spreads it. A ring batch replayed from the journal
//! is still counted as a ring batch: the origin names the stream the commands arrived on, not
//! the thread that ran them.
//!
//! **Printing is the hazard, not counting.** Nothing is printed until the interval has elapsed,
//! checked once per batch. A run shorter than the interval reports once at teardown instead, so
//! a corpus replay set to a long interval prints exactly one line per origin covering the whole
//! run.
//!
//! Armed by `VIRGLRS_SUBMIT_STATS`, the knob [`crate::stats`] describes; unset, this is inert.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Per-command cost of the venus command path, or nothing at all when unarmed.
#[derive(Default)]
pub struct Tally {
    /// `None` is the whole of "switched off": no clock is read, no counter is touched, and
    /// there is nothing to flush at teardown.
    on: Option<Armed>,
}

/// Which stream a batch arrived on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Origin {
    /// The virtqueue: a `submit` or the `resume` of one.
    Context,
    /// A ring's buffer, whether read by its thread or fed back from the journal.
    Ring,
}

struct Armed {
    every: Duration,
    /// The clock's epoch. Every other instant is nanoseconds after this, so it fits an atomic.
    base: Instant,
    /// When the window being counted began, as nanoseconds after `base`. The claim on a report
    /// is a compare-exchange on it: the thread that moves it forward prints, the rest carry on.
    window_began: AtomicU64,
    context: Counters,
    ring: Counters,
}

#[derive(Default)]
struct Counters {
    batches: AtomicU64,
    commands: AtomicU64,
    bytes: AtomicU64,
    busy_ns: AtomicU64,
}

/// One window's worth of one origin, taken out of the counters in a single pass.
struct Window {
    batches: u64,
    commands: u64,
    bytes: u64,
    busy: Duration,
}

impl Counters {
    fn add(&self, commands: u64, bytes: u64, busy: Duration) {
        // Relaxed throughout: each counter is its own fact, read only to be printed, and a
        // report that splits one batch across two windows is off by one batch, not wrong.
        self.batches.fetch_add(1, Ordering::Relaxed);
        self.commands.fetch_add(commands, Ordering::Relaxed);
        self.bytes.fetch_add(bytes, Ordering::Relaxed);
        self.busy_ns.fetch_add(busy.as_nanos() as u64, Ordering::Relaxed);
    }

    fn take(&self) -> Window {
        Window {
            batches: self.batches.swap(0, Ordering::Relaxed),
            commands: self.commands.swap(0, Ordering::Relaxed),
            bytes: self.bytes.swap(0, Ordering::Relaxed),
            busy: Duration::from_nanos(self.busy_ns.swap(0, Ordering::Relaxed)),
        }
    }
}

impl Armed {
    fn new(every: Duration) -> Self {
        Self {
            every,
            base: Instant::now(),
            window_began: AtomicU64::new(0),
            context: Counters::default(),
            ring: Counters::default(),
        }
    }

    fn of(&self, origin: Origin) -> &Counters {
        match origin {
            Origin::Context => &self.context,
            Origin::Ring => &self.ring,
        }
    }

    fn since_base(&self, now: Instant) -> u64 {
        now.duration_since(self.base).as_nanos() as u64
    }

    /// Print this window if it has elapsed, and only from the one thread that claims it.
    fn report_if_due(&self, now: Instant) {
        let began = self.window_began.load(Ordering::Relaxed);
        let now_ns = self.since_base(now);
        if now_ns.saturating_sub(began) < self.every.as_nanos() as u64 {
            return;
        }
        // AcqRel: the winner's swaps below must not be reordered before its claim, or a
        // loser's adds land in a window that has already been printed.
        if self
            .window_began
            .compare_exchange(began, now_ns, Ordering::AcqRel, Ordering::Relaxed)
            .is_err()
        {
            return;
        }
        self.report(Duration::from_nanos(now_ns - began), "");
    }

    fn report(&self, elapsed: Duration, note: &str) {
        for (origin, counters) in [("ring", &self.ring), ("context", &self.context)] {
            let w = counters.take();
            if w.batches == 0 {
                continue;
            }
            let secs = elapsed.as_secs_f64().max(1e-9);
            let cmds = w.commands.max(1) as f64;
            eprintln!(
                "[virglrs] venus {origin}: {:.1} us/cmd {:.1} cmd/batch {:.0} cmd/s {:.0} B/cmd \
                 {:.2} thread-s/s (n={} cmd in {} batch over {:.1}s{note})",
                w.busy.as_secs_f64() * 1e6 / cmds,
                w.commands as f64 / w.batches as f64,
                w.commands as f64 / secs,
                w.bytes as f64 / cmds,
                w.busy.as_secs_f64() / secs,
                w.commands,
                w.batches,
                elapsed.as_secs_f64(),
            );
        }
    }
}

impl Tally {
    /// Read the environment once, at renderer construction. The knob and its parsing are
    /// [`crate::stats`], shared with the vrend tally.
    pub fn from_env() -> Self {
        Self { on: crate::stats::report_interval("venus").map(Armed::new) }
    }

    /// A batch is about to run with the context lock held. `None` when unarmed, so the caller
    /// never reads a clock nobody will look at.
    #[inline]
    pub fn batch_began(&self) -> Option<Instant> {
        self.on.as_ref().map(|_| Instant::now())
    }

    /// The batch returned. `commands` is how many it dispatched -- the delta of the context's
    /// own count, so nested execute streams are included -- and `bytes` how long its buffer was.
    /// Reports if the window has elapsed; this is the only place a report is considered.
    #[inline]
    pub fn batch_ended(&self, began: Option<Instant>, origin: Origin, commands: u64, bytes: usize) {
        let (Some(a), Some(began)) = (&self.on, began) else {
            return;
        };
        let now = Instant::now();
        a.of(origin).add(commands, bytes as u64, now.duration_since(began));
        a.report_if_due(now);
    }
}

impl Drop for Tally {
    /// Flush the partial window, so a run shorter than the interval still reports. This is the
    /// root's drop: every ring's dispatcher holds a share, and a context joins its rings before
    /// it lets go of them, so by the time the last share dies no batch is running.
    fn drop(&mut self) {
        let Some(a) = &self.on else {
            return;
        };
        let now = Instant::now();
        let began = a.window_began.load(Ordering::Relaxed);
        a.report(Duration::from_nanos(a.since_base(now).saturating_sub(began)), ", teardown");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn armed(every: Duration) -> Tally {
        Tally { on: Some(Armed::new(every)) }
    }

    fn counts(c: &Counters) -> (u64, u64, u64) {
        (
            c.batches.load(Ordering::Relaxed),
            c.commands.load(Ordering::Relaxed),
            c.bytes.load(Ordering::Relaxed),
        )
    }

    /// Unarmed is inert: nothing counts, and no clock is offered to read.
    #[test]
    fn an_unarmed_tally_counts_nothing() {
        let t = Tally::default();
        assert!(t.batch_began().is_none());
        t.batch_ended(None, Origin::Ring, 12, 4096);
        assert!(t.on.is_none(), "an unarmed tally has nothing to hold");
    }

    /// Armed, it accumulates per origin -- and does not report before its window is up, which
    /// is what keeps `write(2)` off the hot path.
    #[test]
    fn an_armed_tally_accumulates_by_origin_without_reporting() {
        let t = armed(Duration::from_secs(3600));
        for _ in 0..10 {
            let b = t.batch_began();
            assert!(b.is_some(), "an armed tally reads the clock once per batch");
            t.batch_ended(b, Origin::Ring, 100, 512);
        }
        let b = t.batch_began();
        t.batch_ended(b, Origin::Context, 3, 64);
        let a = t.on.as_ref().expect("armed");
        assert_eq!(counts(&a.ring), (10, 1000, 5120), "the window has not elapsed");
        assert_eq!(counts(&a.context), (1, 3, 64), "the other origin is its own count");
        assert_eq!(a.window_began.load(Ordering::Relaxed), 0, "no report moved the window");
    }

    /// A window that has elapsed reports and starts a fresh one, so a rate is never diluted by
    /// the window before it.
    #[test]
    fn a_report_resets_every_counter_and_moves_the_window() {
        let t = armed(Duration::ZERO);
        let a = t.on.as_ref().expect("armed");
        a.ring.add(700, 7000, Duration::from_millis(5));
        a.context.add(2, 20, Duration::from_millis(1));
        std::thread::sleep(Duration::from_millis(2));
        let b = t.batch_began();
        t.batch_ended(b, Origin::Ring, 1, 1);
        assert_eq!(counts(&a.ring), (0, 0, 0));
        assert_eq!(
            counts(&a.context),
            (0, 0, 0),
            "every origin, or the next window carries this one"
        );
        assert_eq!(a.ring.busy_ns.load(Ordering::Relaxed), 0, "the clock resets with the rest");
        assert!(a.window_began.load(Ordering::Relaxed) > 0, "the report claimed a new window");
        assert_eq!(a.every, Duration::ZERO, "the interval is not a counter and must survive");
    }

    /// Two threads adding to the same tally lose nothing: the counters are the sum of both.
    #[test]
    fn concurrent_batches_are_all_counted() {
        let t = std::sync::Arc::new(armed(Duration::from_secs(3600)));
        let threads: Vec<_> = (0..4)
            .map(|_| {
                let t = std::sync::Arc::clone(&t);
                std::thread::spawn(move || {
                    for _ in 0..1000 {
                        let b = t.batch_began();
                        t.batch_ended(b, Origin::Ring, 2, 8);
                    }
                })
            })
            .collect();
        for h in threads {
            h.join().expect("counting cannot panic");
        }
        let a = t.on.as_ref().expect("armed");
        assert_eq!(counts(&a.ring), (4000, 8000, 32000));
    }
}
