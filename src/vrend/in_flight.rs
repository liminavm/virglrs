// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! How much fenced GL work one classic context has queued ahead of the GPU, and a bound on it.
//!
//! **Nothing else bounds it.** A classic fence is answered by taking a sync and handing it to the
//! [waiter](super::waiter), so the thread that runs guest batches never waits for the GPU. That is
//! the point of the waiter, and it holds as long as the GPU keeps up. When it does not -- a WebGL
//! page whose frames cost the GPU more than a frame's time -- the guest is told nothing, keeps
//! submitting, and the host driver queues every batch: host zink throttles only at thousands of
//! batches, and KosmicKrisp keeps a command allocator alive for each render pass until its command
//! buffer completes. Measured on wildbrush.vercel.app: ~2000 Metal command buffers in flight, a
//! 22 GB worker, the host swapping, and the virtio-gpu worker blocked for 31 s at a stretch.
//!
//! So each context counts the fences it has in the waiter's queue, and a batch for a context that
//! is at the limit waits for the oldest of them first. The wait is on the thread that runs every
//! context's batches, so it stalls them all -- but only for as long as the GPU takes to retire one
//! fence of the runaway context, instead of for as long as it takes to drain everything that
//! context was allowed to queue. The guest-side alternative, capping each client's unsignalled
//! fences in the guest kernel, would stall only the client, but only on a guest running our kernel.
//!
//! **Why a count of fences and not of GL work.** A fence is the only unit both sides can see: the
//! guest asks for one per execbuffer that touches a buffer, and the waiter knows when its work has
//! run. A context that never asks for a fence is not bounded by this, and has never been the
//! problem -- nothing it submitted could be waited for by anything, the guest included.
//!
//! **The wait gives up.** A sync that never signals would otherwise hang every context behind a
//! context that is itself hung. After `patience` the batch runs anyway and the caller is told, so
//! a stuck GPU degrades to the old unbounded behaviour rather than to a frozen worker.

use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

struct Inner {
    /// Fences of this context the waiter holds and has not yet seen signal.
    queued: Mutex<usize>,
    released: Condvar,
}

/// One context's count of fences in flight. Cloned into whoever needs to wait on it.
#[derive(Clone)]
pub struct Gate(Arc<Inner>);

/// One fence's place in its context's count, given back when dropped.
///
/// Owned by the waiter's job, so the count cannot outlive or undercount what it describes: the job
/// is dropped exactly once, whether its fence retires, the waiter drains at teardown, or the job
/// is thrown away.
pub struct Ticket(Arc<Inner>);

/// What [`Gate::wait_below`] did.
#[derive(Debug, PartialEq, Eq)]
pub enum Waited {
    /// The context was under the limit.
    No,
    /// It was at the limit, for this long, until a fence retired.
    For(Duration),
    /// It was still at the limit when patience ran out, and the caller goes ahead regardless.
    GaveUp(Duration),
}

impl Default for Gate {
    fn default() -> Self {
        Gate(Arc::new(Inner { queued: Mutex::new(0), released: Condvar::new() }))
    }
}

impl Gate {
    /// Count one more fence in flight.
    pub fn ticket(&self) -> Ticket {
        *self.0.queued.lock().expect("the in-flight count is never held across a panic") += 1;
        Ticket(Arc::clone(&self.0))
    }

    /// How many fences are in flight now.
    pub fn queued(&self) -> usize {
        *self.0.queued.lock().expect("the in-flight count is never held across a panic")
    }

    /// Block while `limit` or more fences are in flight, for at most `patience`.
    pub fn wait_below(&self, limit: usize, patience: Duration) -> Waited {
        let began = Instant::now();
        let mut queued =
            self.0.queued.lock().expect("the in-flight count is never held across a panic");
        if *queued < limit {
            return Waited::No;
        }
        while *queued >= limit {
            let left = patience.saturating_sub(began.elapsed());
            if left.is_zero() {
                return Waited::GaveUp(began.elapsed());
            }
            queued = self
                .0
                .released
                .wait_timeout(queued, left)
                .expect("the in-flight count is never held across a panic")
                .0;
        }
        Waited::For(began.elapsed())
    }
}

impl Drop for Ticket {
    fn drop(&mut self) {
        let mut queued =
            self.0.queued.lock().expect("the in-flight count is never held across a panic");
        *queued = queued.checked_sub(1).expect("a ticket is given back once, after it was taken");
        self.0.released.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LONG: Duration = Duration::from_secs(10);

    #[test]
    fn under_the_limit_does_not_wait() {
        let gate = Gate::default();
        let _one = gate.ticket();
        assert_eq!(gate.wait_below(2, LONG), Waited::No);
    }

    #[test]
    fn a_ticket_given_back_counts_down() {
        let gate = Gate::default();
        let one = gate.ticket();
        let two = gate.ticket();
        assert_eq!(gate.queued(), 2);
        drop(one);
        drop(two);
        assert_eq!(gate.queued(), 0);
    }

    /// At the limit, the wait lasts exactly until a ticket is given back on another thread -- which
    /// is where the waiter gives them back.
    #[test]
    fn at_the_limit_waits_for_a_ticket_to_come_back() {
        let gate = Gate::default();
        let held = vec![gate.ticket(), gate.ticket()];
        let hold = Duration::from_millis(200);
        let releaser = std::thread::spawn(move || {
            std::thread::sleep(hold);
            drop(held);
        });
        match gate.wait_below(2, LONG) {
            Waited::For(d) => {
                assert!(d >= hold / 2, "returned after {d:?}, before any ticket came back")
            }
            other => panic!("expected to wait for the release, got {other:?}"),
        }
        releaser.join().expect("the releasing thread");
        assert_eq!(gate.queued(), 0);
    }

    /// A fence that never retires must not hang the caller.
    #[test]
    fn gives_up_when_nothing_comes_back() {
        let gate = Gate::default();
        let _stuck = gate.ticket();
        match gate.wait_below(1, Duration::from_millis(100)) {
            Waited::GaveUp(d) => assert!(d >= Duration::from_millis(100)),
            other => panic!("expected to give up, got {other:?}"),
        }
    }
}
