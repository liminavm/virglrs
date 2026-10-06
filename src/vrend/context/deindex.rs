// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! An indexed draw rewritten as a non-indexed one, for a GLES host that refuses indexed draws
//! while transform feedback is active.
//!
//! GLES 3.0 and 3.1 make every `glDrawElements*` an `INVALID_OPERATION` while transform feedback
//! is active and not paused; `OES_geometry_shader`, and GLES 3.2 with it, lift that. A guest's GL
//! has no such rule, and its driver draws indexed whenever it likes -- converting quads, fans and
//! polygons, for one. So on such a host the draw is served by gathering, per vertex buffer, the
//! bytes each index names into a buffer of its own, in index order, and drawing that as arrays.
//! The captured vertices are then the ones the indexed draw would have captured, in its order.
//!
//! This half is the arithmetic, kept apart from GL so that it can be tested as such: which
//! vertices the indices name, where primitive restart cuts them into runs, and the bytes a
//! binding's vertices are gathered from.

use super::super::proto::IndexType;
use std::ops::Range;

/// The most vertices a de-indexed draw may name. The indices are resolved into eight bytes each,
/// so this bounds that allocation by what the guest's count asks for.
pub const MAX_VERTICES: u32 = 1 << 24;

/// The most bytes one binding's gathered vertices may take. Its vertex count times its width is
/// the guest's to choose, and a draw past this is refused rather than allocated for.
pub const MAX_GATHERED: u64 = 1 << 28;

/// The vertices an indexed draw names, in order, and the runs primitive restart cuts them into.
#[derive(Debug, PartialEq, Eq)]
pub struct Resolved {
    /// The vertex each drawn index names: the index plus the draw's bias. A negative one names
    /// no vertex.
    pub vertices: Vec<i64>,
    /// The runs of `vertices` drawn as one primitive sequence each.
    pub runs: Vec<Range<usize>>,
}

impl Resolved {
    /// The lowest and highest vertex named, or `None` when no vertex is named at all.
    pub fn span(&self) -> Option<(u64, u64)> {
        let mut named = self.vertices.iter().filter_map(|&v| u64::try_from(v).ok());
        let first = named.next()?;
        Some(named.fold((first, first), |(lo, hi), v| (lo.min(v), hi.max(v))))
    }
}

/// Read `count` indices of `ty` out of `bytes`, as the host would have drawn them.
///
/// `restart` is whether the draw restarts primitives. A GLES host restarts at the index type's
/// maximum alone (`GL_PRIMITIVE_RESTART_FIXED_INDEX`, which is what the draw enables there), and
/// compares the index before the bias is added, as GL does. A restart index is drawn as nothing
/// and ends the run it is in.
///
/// `bytes` must hold the `count` indices; the draw checked the index buffer covers them.
pub fn resolve(bytes: &[u8], ty: IndexType, count: usize, bias: i32, restart: bool) -> Resolved {
    let size = ty.bytes() as usize;
    assert!(bytes.len() >= count * size, "the draw checked its index buffer covers its count");
    let restart_index = match ty {
        IndexType::U8 => u32::from(u8::MAX),
        IndexType::U16 => u32::from(u16::MAX),
        IndexType::U32 => u32::MAX,
    };
    let mut vertices = Vec::with_capacity(count);
    let mut runs = Vec::new();
    let mut run_start = 0;
    for raw in bytes[..count * size].chunks_exact(size) {
        let index = match ty {
            IndexType::U8 => u32::from(raw[0]),
            IndexType::U16 => u32::from(u16::from_le_bytes([raw[0], raw[1]])),
            IndexType::U32 => u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]),
        };
        if restart && index == restart_index {
            if vertices.len() > run_start {
                runs.push(run_start..vertices.len());
            }
            run_start = vertices.len();
            continue;
        }
        vertices.push(i64::from(index) + i64::from(bias));
    }
    if vertices.len() > run_start {
        runs.push(run_start..vertices.len());
    }
    Resolved { vertices, runs }
}

/// Gather one binding's bytes for `vertices`, `width` bytes each, out of `src`.
///
/// `src` is the binding's buffer from vertex `first` on: vertex `v` starts at `(v - first) *
/// stride` in it. A vertex wholly or partly outside `src` -- or a negative one -- reads zeros
/// where it is outside, which is what a host with robust buffer access returns for a fetch out of
/// bounds.
pub fn gather(src: &[u8], first: u64, stride: u64, width: usize, vertices: &[i64]) -> Vec<u8> {
    let mut out = vec![0u8; vertices.len() * width];
    for (slot, &v) in out.chunks_exact_mut(width).zip(vertices) {
        let Some(at) = u64::try_from(v)
            .ok()
            .and_then(|v| v.checked_sub(first))
            .and_then(|v| v.checked_mul(stride))
            .and_then(|at| usize::try_from(at).ok())
        else {
            continue;
        };
        if let Some(have) = src.get(at..) {
            let n = have.len().min(width);
            slot[..n].copy_from_slice(&have[..n]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u16s(v: &[u16]) -> Vec<u8> {
        v.iter().flat_map(|i| i.to_le_bytes()).collect()
    }

    #[test]
    fn indices_name_their_vertices_plus_the_bias_in_order() {
        let r = resolve(&u16s(&[3, 1, 2, 7]), IndexType::U16, 4, 10, false);
        assert_eq!(r.vertices, [13, 11, 12, 17]);
        assert_eq!(r.runs, vec![0..4]);
        assert_eq!(r.span(), Some((11, 17)));
    }

    #[test]
    fn only_the_drawn_count_is_read() {
        let r = resolve(&[5, 6, 7, 8], IndexType::U8, 2, 0, false);
        assert_eq!(r.vertices, [5, 6]);
    }

    #[test]
    fn a_restart_index_ends_a_run_and_draws_nothing() {
        let r =
            resolve(&u16s(&[0, 1, 2, 0xffff, 0xffff, 3, 4, 5, 0xffff]), IndexType::U16, 9, 0, true);
        assert_eq!(r.vertices, [0, 1, 2, 3, 4, 5]);
        assert_eq!(r.runs, [0..3, 3..6]);
    }

    #[test]
    fn the_restart_index_is_compared_before_the_bias() {
        let r = resolve(&[0xff, 1], IndexType::U8, 2, -1, true);
        assert_eq!(r.vertices, [0]);
        let r = resolve(&[0xff, 1], IndexType::U8, 2, -1, false);
        assert_eq!(r.vertices, [254, 0]);
    }

    #[test]
    fn without_restart_the_maximum_is_an_ordinary_index() {
        let r = resolve(&u32::MAX.to_le_bytes(), IndexType::U32, 1, 0, false);
        assert_eq!(r.vertices, [i64::from(u32::MAX)]);
    }

    #[test]
    fn a_negative_vertex_names_nothing() {
        let r = resolve(&[0, 2], IndexType::U8, 2, -1, false);
        assert_eq!(r.vertices, [-1, 1]);
        assert_eq!(r.span(), Some((1, 1)));
        assert_eq!(resolve(&[0], IndexType::U8, 1, -1, false).span(), None);
    }

    #[test]
    fn a_binding_is_gathered_in_index_order_from_its_first_vertex() {
        // Vertices 4..8, stride 4, two bytes wide each.
        let src: Vec<u8> = (0..16).collect();
        assert_eq!(gather(&src, 4, 4, 2, &[6, 4, 7]), [8, 9, 0, 1, 12, 13]);
    }

    #[test]
    fn a_vertex_outside_the_buffer_reads_zeros_where_it_is_outside() {
        let src: Vec<u8> = (1..=10).collect();
        // Vertex 2 starts at byte 8 and has two of its four bytes; 3 and -1 have none.
        assert_eq!(
            gather(&src, 0, 4, 4, &[2, 3, -1, 1]),
            [9, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 6, 7, 8]
        );
        // Below the first vertex mapped is outside too.
        assert_eq!(gather(&src, 1, 4, 1, &[0]), [0]);
    }
}
