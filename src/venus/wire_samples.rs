// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva

//! Hand-written venus commands for the wire shapes no recorded capture contains.
//!
//! Every word is spelled out in the order guest Mesa's encoder writes it, so decoding one is a
//! check against that encoder rather than against our own. Two users, one copy: the protocol tests
//! (`proto.rs`) decode and re-encode them, and `fuzz/src/bin/seed-corpora.rs` writes the whole
//! commands as seeds for `venus_command`, which random bytes would almost never reach. Plain bytes
//! and nothing from the crate, because the fuzz binary includes this file by path.

pub fn u32(v: u32) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}

pub fn u64(v: u64) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}

/// One AABB geometry reading its boxes from `addr`.
pub fn aabbs(addr: u64) -> Vec<u8> {
    [
        u32(1000150006), // sType = ..._GEOMETRY_KHR
        u64(0),          // pNext
        u32(1),          // geometryType = AABBS
        u32(1),          // the geometry union's tag: AABBS
        u32(1000150003), // sType = ..._GEOMETRY_AABBS_DATA_KHR
        u64(0),          // pNext
        u32(0),          // data: a device address
        u64(addr),
        u64(24), // stride
        u32(0),  // flags
    ]
    .concat()
}

/// A build info over `geometries`, sent as `pGeometries` or as `ppGeometries` rows of one.
pub fn info(geometries: &[Vec<u8>], by_row: bool) -> Vec<u8> {
    let n = geometries.len() as u64;
    let mut w = [
        u32(1000150000), // sType = ..._BUILD_GEOMETRY_INFO_KHR
        u64(0),          // pNext
        u32(0),          // type = TOP_LEVEL, no object needed to say so
        u32(0),          // flags
        u32(0),          // mode = BUILD
        u64(0),          // srcAccelerationStructure
        u64(0),          // dstAccelerationStructure
        u32(n as u32),   // geometryCount
    ]
    .concat();
    if by_row {
        w.extend(u64(0)); // pGeometries: absent
        w.extend(u64(n)); // ppGeometries: n rows of one
        for g in geometries {
            w.extend(u64(1));
            w.extend(g);
        }
    } else {
        w.extend(u64(n));
        geometries.iter().for_each(|g| w.extend(g));
        w.extend(u64(0)); // ppGeometries: absent
    }
    w.extend(u32(0)); // scratchData: a device address
    w.extend(u64(0x5c7a_0000));
    w
}

/// A range whose four words are `base` onwards, so a row read in the wrong place shows.
pub fn range(base: u32) -> Vec<u8> {
    [u32(base), u32(base + 1), u32(base + 2), u32(base + 3)].concat()
}

/// Two builds, one of each geometry form, with range rows of one and two.
pub fn build_args() -> Vec<u8> {
    [
        u64(0x11), // commandBuffer
        u32(2),    // infoCount
        u64(2),    // pInfos
        info(&[aabbs(0xa000)], false),
        info(&[aabbs(0xb000), aabbs(0xc000)], true),
        u64(2), // ppBuildRangeInfos: a row per build
        u64(1),
        range(10),
        u64(2),
        range(20),
        range(30),
    ]
    .concat()
}

/// The same two builds, their primitive counts in rows and their ranges on the device.
pub fn build_indirect_args() -> Vec<u8> {
    [
        u64(0x11),
        u32(2),
        u64(2),
        info(&[aabbs(0xa000)], false),
        info(&[aabbs(0xb000), aabbs(0xc000)], true),
        u64(2), // pIndirectDeviceAddresses
        u64(0xd000),
        u64(0xe000),
        u64(2), // pIndirectStrides
        u32(16),
        u32(16),
        u64(2), // ppMaxPrimitiveCounts: a row per build
        u64(1),
        u32(7),
        u64(2),
        u32(8),
        u32(9),
    ]
    .concat()
}

/// The size query for one build of two geometries.
pub fn build_sizes_args() -> Vec<u8> {
    [
        u64(0x3), // device
        u32(1),   // buildType = DEVICE
        u64(1),   // pBuildInfo
        info(&[aabbs(0xb000), aabbs(0xc000)], false),
        u64(2), // pMaxPrimitiveCounts
        u32(8),
        u32(9),
        u64(1),          // pSizeInfo: the room for the answer
        u32(1000150020), // sType = ..._BUILD_SIZES_INFO_KHR
        u64(0),          // pNext
    ]
    .concat()
}

/// A copy of a serialized structure out of device memory, whose source is the address union.
pub fn copy_memory_to_as_args() -> Vec<u8> {
    [
        u64(0x11),       // commandBuffer
        u64(1),          // pInfo
        u32(1000150012), // sType = ..._COPY_MEMORY_TO_ACCELERATION_STRUCTURE_INFO_KHR
        u64(0),          // pNext
        u32(0),          // src: a device address
        u64(0xf000),
        u64(0x51), // dst
        u32(2),    // mode = DESERIALIZE
    ]
    .concat()
}

/// Each sample with the `VkCommandTypeEXT` that heads it. The flags word between the two is the
/// caller's to add.
pub fn commands() -> [(u32, Vec<u8>); 4] {
    [
        (306, build_args()),
        (307, build_indirect_args()),
        (319, build_sizes_args()),
        (315, copy_memory_to_as_args()),
    ]
}
