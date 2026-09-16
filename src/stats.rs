// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! The one knob that arms the per-command instruments.
//!
//! `VIRGLRS_SUBMIT_STATS` arms both `vrend::tally` and `venus::tally` at once: `1` reports every
//! 5 s, any other positive integer is the interval in seconds, and unset, empty or `0` leaves
//! them inert. Each renderer reads it once at construction and reports under its own prefix, so
//! one boot with the variable set scores both command paths.
//!
//! A misspelled interval arms at the default rather than reading as off. An instrument that
//! silently disarms on a typo is a needle nothing is holding, and the run it was meant to score
//! reads clean for the wrong reason. The rule lives here, once, so the two tallies cannot drift on
//! it.

use std::time::Duration;

/// The environment variable both tallies read.
pub const KNOB: &str = "VIRGLRS_SUBMIT_STATS";

/// How often the instrument named `who` should report, or `None` to stay inert.
///
/// Read once, at renderer construction, and deliberately not from a lazy global consulted on the
/// hot path: the tally hangs off the renderer like every other piece of state, so a renderer built
/// without the variable set can never start reporting.
pub fn report_interval(who: &str) -> Option<Duration> {
    let value = std::env::var(KNOB).ok();
    let every = parse(who, value.as_deref())?;
    eprintln!("[virglrs] {who}: submit stats on, reporting every {}s", every.as_secs());
    Some(every)
}

/// The knob's value to an interval. `None` is off; junk is the default, said aloud.
fn parse(who: &str, value: Option<&str>) -> Option<Duration> {
    let secs = match value?.trim() {
        "" | "0" => return None,
        "1" => 5,
        other => match other.parse::<u64>() {
            Ok(n) if n > 0 => n,
            _ => {
                eprintln!(
                    "[virglrs] {who}: {KNOB}={other:?} is not a positive number of seconds; \
                     reporting every 5s"
                );
                5
            }
        },
    };
    Some(Duration::from_secs(secs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unset_empty_and_zero_are_off() {
        assert_eq!(parse("t", None), None);
        assert_eq!(parse("t", Some("")), None);
        assert_eq!(parse("t", Some(" 0 ")), None);
    }

    #[test]
    fn one_is_the_default_and_a_number_is_seconds() {
        assert_eq!(parse("t", Some("1")), Some(Duration::from_secs(5)));
        assert_eq!(parse("t", Some("30")), Some(Duration::from_secs(30)));
    }

    /// A junk interval arms at the default rather than silently disarming.
    #[test]
    fn a_junk_interval_still_arms() {
        assert_eq!(parse("t", Some("banana")), Some(Duration::from_secs(5)));
        assert_eq!(parse("t", Some("-3")), Some(Duration::from_secs(5)));
    }
}
