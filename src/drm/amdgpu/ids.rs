// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! The GPU's marketing name, found the way libdrm_amdgpu finds it: in the `amdgpu.ids` table by
//! PCI device and revision, else, for an APU, in the CPU's model name, else a generic name.
//!
//! The guest shows this name to its users and its Mesa puts it in the renderer string, and it
//! reaches the guest in the capset -- so it is libdrm's answer, quirks included, that has to come
//! out: a table line that does not parse stops the search, and a name ends at its first comma.
//! Reading the files is the caller's; this only decides.

use std::path::{Path, PathBuf};

/// The name libdrm_amdgpu gives a device it finds nowhere.
pub const GENERIC: &str = "AMD Radeon Graphics";

/// Where libdrm_amdgpu looks for the table: each directory in `AMDGPU_ASIC_ID_TABLE_PATHS` that
/// holds an `amdgpu.ids`, else the path it was built with, which is the distribution's.
pub fn table_path(paths_env: Option<&str>, exists: impl Fn(&Path) -> bool) -> PathBuf {
    const DEFAULT: &str = "/usr/share/libdrm/amdgpu.ids";
    paths_env
        .filter(|v| !v.is_empty())
        .into_iter()
        .flat_map(|v| v.split(':'))
        .map(|dir| Path::new(dir).join("amdgpu.ids"))
        .find(|p| exists(p))
        .unwrap_or_else(|| PathBuf::from(DEFAULT))
}

/// The marketing name for `(device, revision)`.
///
/// `table` is the contents of `amdgpu.ids` and `cpuinfo` those of `/proc/cpuinfo`, either `None`
/// when it could not be read. `apu` is whether the device reports `AMDGPU_IDS_FLAGS_FUSION`.
pub fn marketing_name(
    table: Option<&str>,
    cpuinfo: Option<&str>,
    device: u32,
    revision: u32,
    apu: bool,
) -> String {
    if let Some(name) = table.and_then(|t| from_table(t, device, revision)) {
        return name;
    }
    if apu && let Some(name) = cpuinfo.and_then(from_cpuinfo) {
        return name;
    }
    GENERIC.to_owned()
}

/// The table's answer. The first line that is neither empty nor a comment is the table's version
/// and names nothing; every line after it is `device, revision, name`.
fn from_table(table: &str, device: u32, revision: u32) -> Option<String> {
    let mut lines = table.lines().filter(|l| !l.is_empty() && !l.starts_with('#'));
    lines.next()?;
    for line in lines {
        match parse_line(line, device, revision) {
            Line::Other => continue,
            Line::Match(name) => return Some(name),
            // libdrm stops at a line it cannot read, rather than skipping it.
            Line::Malformed => return None,
        }
    }
    None
}

enum Line {
    Match(String),
    Other,
    Malformed,
}

/// One table line, read as libdrm's `strtok_r` and `strtol` read it: fields are split on commas
/// with empty fields skipped, ids are hexadecimal and must be the whole field, and the name loses
/// its leading blanks and nothing else.
fn parse_line(line: &str, device: u32, revision: u32) -> Line {
    let mut fields = line.split(',').filter(|f| !f.is_empty());
    let Some(did) = fields.next().and_then(hex) else { return Line::Malformed };
    if did != device {
        return Line::Other;
    }
    let Some(rid) = fields.next().and_then(hex) else { return Line::Malformed };
    if rid != revision {
        return Line::Other;
    }
    match fields.next().map(|n| n.trim_start_matches([' ', '\t'])) {
        Some(name) if !name.is_empty() => Line::Match(name.to_owned()),
        _ => Line::Malformed,
    }
}

/// `strtol(s, &end, 16)` with `*end == '\0'` required: leading whitespace and a sign or `0x` are
/// accepted, trailing anything is not.
fn hex(s: &str) -> Option<u32> {
    let s = s.trim_start();
    let (neg, s) = match s.as_bytes().first() {
        Some(b'-') => (true, &s[1..]),
        Some(b'+') => (false, &s[1..]),
        _ => (false, s),
    };
    let s = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")).unwrap_or(s);
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let v = i64::from_str_radix(s, 16).ok()?;
    // `strtol` into a `uint32_t`: the conversion truncates, as the C's assignment does.
    Some((if neg { v.wrapping_neg() } else { v }) as u32)
}

/// An APU's name, from the first `model name` line: from `Radeon` on when the CPU's name carries
/// the GPU's, else the whole value, prefixed with `AMD` when it lacks it.
fn from_cpuinfo(cpuinfo: &str) -> Option<String> {
    let line = cpuinfo.lines().find(|l| l.starts_with("model name"))?;
    let value = match line.find("Radeon") {
        Some(at) => &line[at..],
        None => &line[line.find(':')? + 1..],
    };
    let value = value.trim_start_matches([' ', '\t']);
    Some(if value.starts_with("AMD") { value.to_owned() } else { format!("AMD {value}") })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TABLE: &str = "# comment\n\n1.0.0\n150E, C1, AMD Radeon 890M Graphics\n\
                         150E, C4, AMD Radeon 880M Graphics\n7480,\tC0,\tAMD Radeon RX 7600, XT\n";

    #[test]
    fn the_table_matches_device_and_revision() {
        assert_eq!(from_table(TABLE, 0x150e, 0xc1).as_deref(), Some("AMD Radeon 890M Graphics"));
        assert_eq!(from_table(TABLE, 0x150e, 0xc4).as_deref(), Some("AMD Radeon 880M Graphics"));
        assert_eq!(from_table(TABLE, 0x150e, 0xc2), None);
    }

    #[test]
    fn a_name_ends_at_its_first_comma_and_loses_leading_blanks() {
        assert_eq!(from_table(TABLE, 0x7480, 0xc0).as_deref(), Some("AMD Radeon RX 7600"));
    }

    #[test]
    fn the_version_line_names_nothing() {
        assert_eq!(
            from_table("150E, C1, first\n150E, C1, second\n", 0x150e, 0xc1).as_deref(),
            Some("second")
        );
    }

    #[test]
    fn a_malformed_line_ends_the_search() {
        let t = "1.0\nzz, 00, broken\n150E, C1, unreachable\n";
        assert_eq!(from_table(t, 0x150e, 0xc1), None);
        // An id with trailing junk is malformed, as `*endptr != '\0'` makes it.
        let t = "1.0\n150E , C1, junk\n";
        assert_eq!(from_table(t, 0x150e, 0xc1), None);
    }

    #[test]
    fn an_apu_falls_back_to_the_cpu_name() {
        let cpu = "processor\t: 0\nmodel name\t: AMD Ryzen AI 9 HX 370 w/ Radeon 890M\n";
        assert_eq!(marketing_name(None, Some(cpu), 1, 2, true), "AMD Radeon 890M");
        let cpu = "model name\t: Ryzen 7 5800U\n";
        assert_eq!(marketing_name(None, Some(cpu), 1, 2, true), "AMD Ryzen 7 5800U");
        assert_eq!(marketing_name(None, Some(cpu), 1, 2, false), GENERIC);
        assert_eq!(marketing_name(None, None, 1, 2, true), GENERIC);
    }

    #[test]
    fn the_table_path_honours_the_environment() {
        let p = table_path(Some("/nope:/here"), |p| p == Path::new("/here/amdgpu.ids"));
        assert_eq!(p, Path::new("/here/amdgpu.ids"));
        assert_eq!(table_path(Some("/nope"), |_| false), Path::new("/usr/share/libdrm/amdgpu.ids"));
        assert_eq!(table_path(None, |_| true), Path::new("/usr/share/libdrm/amdgpu.ids"));
    }
}
