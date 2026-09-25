// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! The commands this build does not serve, as `unserved.txt` records them.
//!
//! One reader for two users: the test that holds the file to the handlers, and the driver, which
//! declines to advertise an extension a guest could only use by sending one of these. The file is
//! compiled in, so a malformed line is this tree's bug and panics; the test reads every line, so
//! it panics there first.

use std::collections::BTreeSet;

/// Why a command is on the ledger. The file's header says what each one means.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Wanted,
    OutOfReach,
    NoHostFeature,
    NotInReference,
}

impl Status {
    /// Every status, with the name the file spells it with.
    pub const ALL: [(&'static str, Status); 4] = [
        ("wanted", Status::Wanted),
        ("out-of-reach", Status::OutOfReach),
        ("no-host-feature", Status::NoHostFeature),
        ("not-in-reference", Status::NotInReference),
    ];

    /// Whether the command keeps its extension off the list the guest is told.
    ///
    /// Everything but `not-in-reference`. Those are commands the guest's venus driver answers
    /// itself -- private data, `vkUnmapMemory2` -- or that the C refuses while advertising the
    /// extension anyway. Withholding on them would take away `VK_KHR_map_memory2` and
    /// `VK_EXT_private_data`, which work without the renderer, and would part from the reference
    /// where the reference is not wrong.
    pub fn withholds(self) -> bool {
        self != Status::NotInReference
    }
}

/// One line of the ledger.
#[derive(Clone, Copy, Debug)]
pub struct Line {
    pub command: &'static str,
    pub status: Status,
    pub group: &'static str,
}

const LEDGER: &str = include_str!("unserved.txt");

/// Every line of the ledger, in file order.
pub fn lines() -> impl Iterator<Item = Line> {
    LEDGER.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')).map(|l| {
        let (command, rest) = l.split_once(char::is_whitespace).unwrap_or_else(|| {
            panic!("ledger line has no status, so it says nothing about why: {l:?}")
        });
        let (status, group) = rest.trim().split_once(':').unwrap_or_else(|| {
            panic!("a status is `status:group`, and this one names no group: {l:?}")
        });
        let status = Status::ALL.iter().find(|(name, _)| *name == status).map_or_else(
            || panic!("unknown status {status:?} on {command}; the four are {:?}", Status::ALL),
            |(_, s)| *s,
        );
        assert!(!group.is_empty(), "an empty group on {command}");
        Line { command, status, group }
    })
}

/// The commands whose extension is not advertised: every line whose status [`Status::withholds`].
pub fn withheld() -> BTreeSet<&'static str> {
    lines().filter(|l| l.status.withholds()).map(|l| l.command).collect()
}
