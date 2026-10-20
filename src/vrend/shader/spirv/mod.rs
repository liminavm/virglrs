// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! The translator's SPIR-V side (docs: limina's `docs/design/vrend-spirv.md`): a shader handed
//! to the host as a SPIR-V module, so the host's GLSL front end never runs.

pub mod writer;
