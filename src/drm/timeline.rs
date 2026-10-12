// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! A native context's fences: one queue per ring, and one thread that waits on the head of every
//! queue at once and retires each fence when the work before it is done.
//!
//! A fence waits for the newest submission on its ring when it was taken: the sync_file the ring
//! last produced. A fence taken with no submission since the last one has nothing of its own to
//! wait for, and retires when the fence ahead of it does -- never before, so a ring's fences
//! retire in the order they were taken. The C retires such a fence at once, overtaking a
//! still-pending one ahead of it.
//!
//! Stopping never waits on the GPU: the thread waits on a socket as well as on the fences, and
//! dropping the timelines writes to it. Fences still pending then are dropped unretired, as a
//! destroyed context's are in the C.

use crate::fence::Handle;
use crate::ids::{ContextId, FenceId, RingIdx};
use std::collections::VecDeque;
use std::io::{Read, Write};
use std::os::fd::{AsFd, OwnedFd};
use std::os::unix::net::UnixStream;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

/// A fence waiting its turn, and what it waits for when it reaches the head of its ring.
struct Pending {
    /// `None` for a fence taken with no new submission, which only waits for the ones ahead.
    sync: Option<Arc<OwnedFd>>,
    fence: FenceId,
}

struct State {
    /// Index `i` is ring `i + 1`: ring 0 is the CPU's, and its fences never queue.
    rings: Vec<VecDeque<Pending>>,
    stop: bool,
}

pub struct Timelines {
    ctx: ContextId,
    state: Arc<Mutex<State>>,
    fences: Handle,
    wake: UnixStream,
    thread: Option<JoinHandle<()>>,
}

impl Timelines {
    /// Start the waiter for `ctx`'s `rings` hardware rings.
    pub fn start(ctx: ContextId, rings: u32, fences: Handle) -> std::io::Result<Timelines> {
        let (wake, woken) = UnixStream::pair()?;
        woken.set_nonblocking(true)?;
        let state = Arc::new(Mutex::new(State {
            rings: (0..rings).map(|_| VecDeque::new()).collect(),
            stop: false,
        }));
        let thread = std::thread::Builder::new().name(format!("virglrs-nctx-{ctx}")).spawn({
            let state = Arc::clone(&state);
            let fences = fences.clone();
            move || wait(ctx, &state, &fences, woken)
        })?;
        Ok(Timelines { ctx, state, fences, wake, thread: Some(thread) })
    }

    /// Queue `fence` on `ring` (1-based), behind `sync` when the ring has submitted since its
    /// last fence. A fence with nothing to wait for retires now.
    pub fn push(&self, ring: RingIdx, sync: Option<OwnedFd>, fence: FenceId) {
        let mut st = self.state.lock().expect("the timeline lock is never held across a panic");
        let queue = &mut st.rings[ring.0 as usize - 1];
        if queue.is_empty() && sync.is_none() {
            self.fences.retire_context(self.ctx, ring, fence);
            return;
        }
        queue.push_back(Pending { sync: sync.map(Arc::new), fence });
        drop(st);
        self.kick();
    }

    fn kick(&self) {
        // A full socket already has a wake-up pending, which is all one more would say.
        if let Err(e) = (&self.wake).write(&[1])
            && e.kind() != std::io::ErrorKind::WouldBlock
        {
            panic!("the timeline thread's socket would not take a wake-up: {e}");
        }
    }
}

impl Drop for Timelines {
    fn drop(&mut self) {
        self.state.lock().expect("the timeline lock is never held across a panic").stop = true;
        self.kick();
        if let Some(t) = self.thread.take() {
            t.join().expect("the timeline thread does not panic");
        }
    }
}

/// Retire the fences at the head of each ring that have nothing left to wait for.
fn retire_ready(ctx: ContextId, st: &mut State, fences: &Handle) {
    for (i, queue) in st.rings.iter_mut().enumerate() {
        while queue.front().is_some_and(|p| p.sync.is_none()) {
            let p = queue.pop_front().expect("checked above");
            fences.retire_context(ctx, RingIdx(i as u32 + 1), p.fence);
        }
    }
}

fn wait(ctx: ContextId, state: &Mutex<State>, fences: &Handle, mut woken: UnixStream) {
    loop {
        let heads: Vec<(usize, Arc<OwnedFd>)> = {
            let mut st = state.lock().expect("the timeline lock is never held across a panic");
            if st.stop {
                return;
            }
            retire_ready(ctx, &mut st, fences);
            st.rings
                .iter()
                .enumerate()
                .filter_map(|(i, q)| Some((i, Arc::clone(q.front()?.sync.as_ref()?))))
                .collect()
        };
        let mut fds: Vec<_> = heads.iter().map(|(_, fd)| fd.as_fd()).collect();
        fds.push(woken.as_fd());
        let ready = super::sys::wait_readable(&fds).expect("poll on descriptors we own");
        if ready[heads.len()] {
            let mut drain = [0u8; 64];
            while matches!(woken.read(&mut drain), Ok(n) if n > 0) {}
        }
        let mut st = state.lock().expect("the timeline lock is never held across a panic");
        for ((i, fd), _) in heads.iter().zip(&ready).filter(|(_, r)| **r) {
            let queue = &mut st.rings[*i];
            // Still the head: nothing but this thread pops, and pushes go to the back.
            let p = queue.pop_front().expect("a head this thread saw");
            assert!(p.sync.as_ref().is_some_and(|s| Arc::ptr_eq(s, fd)), "the head did not move");
            fences.retire_context(ctx, RingIdx(*i as u32 + 1), p.fence);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::ClientFenceId;
    use std::sync::mpsc::{Receiver, Sender, channel};
    use std::time::Duration;

    struct Recorder(Sender<(u32, u64)>);

    impl crate::fence::FenceSink for Recorder {
        fn context_fence(&mut self, _: ContextId, ring: RingIdx, fence: FenceId) {
            self.0.send((ring.0, fence.0)).expect("the test is listening");
        }
        fn global_fence(&mut self, _: ClientFenceId) {}
        fn present_fence(&mut self, _: FenceId) {}
    }

    /// A stand-in for a sync_file: readable once the other end is written to.
    fn sync() -> (UnixStream, OwnedFd) {
        let (signal, sync) = UnixStream::pair().unwrap();
        (signal, sync.into())
    }

    fn start(rings: u32) -> (crate::fence::Retirement, Timelines, Receiver<(u32, u64)>) {
        let (tx, rx) = channel();
        let r = crate::fence::Retirement::start(
            Box::new(Recorder(tx)),
            crate::vrend::debug::Switches::default(),
        );
        let t = Timelines::start(ContextId::new(1).unwrap(), rings, r.handle()).unwrap();
        (r, t, rx)
    }

    fn next(rx: &Receiver<(u32, u64)>) -> (u32, u64) {
        rx.recv_timeout(Duration::from_secs(5)).expect("a fence retires")
    }

    #[test]
    fn a_fence_with_nothing_ahead_and_nothing_to_wait_for_retires_at_once() {
        let (_r, t, rx) = start(1);
        t.push(RingIdx(1), None, FenceId(1));
        assert_eq!(next(&rx), (1, 1));
    }

    #[test]
    fn a_ring_retires_in_order_and_a_fence_never_overtakes_the_one_ahead() {
        let (_r, t, rx) = start(2);
        let (first, a) = sync();
        let (second, b) = sync();
        t.push(RingIdx(1), Some(a), FenceId(1));
        // Taken with no submission since: it waits behind fence 1, where the C retires it now.
        t.push(RingIdx(1), None, FenceId(2));
        t.push(RingIdx(1), Some(b), FenceId(3));
        // The later submission finishing first must not retire anything on its ring.
        (&second).write_all(&[1]).unwrap();
        // Another ring is not held up by this one.
        t.push(RingIdx(2), None, FenceId(9));
        assert_eq!(next(&rx), (2, 9));
        assert!(rx.recv_timeout(Duration::from_millis(200)).is_err(), "ring 1 overtook fence 1");
        (&first).write_all(&[1]).unwrap();
        assert_eq!([next(&rx), next(&rx), next(&rx)], [(1, 1), (1, 2), (1, 3)]);
    }

    #[test]
    fn stopping_does_not_wait_for_a_pending_fence() {
        let (_r, t, rx) = start(1);
        let (_never, a) = sync();
        t.push(RingIdx(1), Some(a), FenceId(1));
        drop(t);
        assert!(rx.try_recv().is_err(), "a pending fence is dropped, not retired");
    }
}
