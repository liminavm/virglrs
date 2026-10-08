#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright © 2026 Gustavo Noronha Silva
"""Break the renderer on purpose, and report which breakages the tests notice.

A passing suite says nothing about what it would catch. This says it directly: each entry below
is a one-line edit that makes the renderer wrong in a way a guest would see, applied to a clean
tree, tested, and reverted. `RED` is the suite catching it. `SURVIVED` is a hole, named.

    harness/sabotage/sweep.py [pattern ...]

Entries are matched by substring against their name; with none, every entry runs. Each is applied
alone, so one sabotage never masks another.

The edits are exact string replacements and every one asserts it matched, so an entry whose target
has been refactored away fails loudly instead of quietly testing nothing -- a sweep that reports
`RED` for an edit it never made is worse than no sweep. Every target is checked up front, before
anything is built, so a refactor costs one message naming all of them.
"""

import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
RS = ROOT

# (name, path relative to the repo root, what to replace, what with, cargo test filter)
#
# A filter of the form `kani:<harness>` runs that one Kani proof instead of `cargo test`: the
# property is stated for every input up to a bound, so the witness is the proof, not a test.
# `loom:<test>` runs one loom model, built with `--cfg loom` in its own target directory so the
# two builds do not evict each other. `doc:<filter>` runs the doctests under that filter, which
# plain `cargo test` does not run in this crate: a `compile_fail` doctest is how a property the
# type system holds is shown to hold.
SABOTAGES = [
    (
        'an array accessor hands its handler one element fewer',
        'venus-gen/rustgen.py',
        "unsafe { cs::%s((%s) as usize, val.%s as *%s _) }'\n                    % (call, n, f, 'mut' if mutable else 'const')",
        "unsafe { cs::%s(((%s) as usize).saturating_sub(1), val.%s as *%s _) }'\n                    % (call, n, f, 'mut' if mutable else 'const')",
        'witness',
    ),
    (
        'an array accessor hands its handler one element more',
        'venus-gen/rustgen.py',
        "unsafe { cs::%s((%s) as usize, val.%s as *%s _) }'\n                    % (call, n, f, 'mut' if mutable else 'const')",
        "unsafe { cs::%s((%s) as usize + 1, val.%s as *%s _) }'\n                    % (call, n, f, 'mut' if mutable else 'const')",
        'witness',
    ),
    (
        'an out-count may be raised past the arrays it sized',
        'src/venus/cs.rs',
        """            assert!(
                n <= most,
                "an out-count raised to {n} past the {most} its arrays were sized to"
            );""",
        """            let _ = most;""",
        'an_out_count_cannot_be_raised_past_the_arrays_it_sized',
    ),
    (
        'a struct embedding a pointerful one counts as plain',
        'venus-gen/rustgen.py',
        """                        if (v.ty.is_pointer() or b.category == VkType.FUNCPOINTER
                                or (b.category in kinds and b.name in found)):""",
        """                        if (v.ty.is_pointer() or b.category == VkType.FUNCPOINTER):""",
        'a_struct_with_a_pointer_anywhere_in_it_is_never_plain',
    ),
    (
        'an answer takes any write, pointers included',
        'src/venus/cs.rs',
        """        assert!(before == after, "an answer's pointers or their counts were rewritten");""",
        """        let _ = (before, after);""",
        'an_answer_keeps',
    ),
    (
        'the shape of an answer leaves out the counts its pointers are sized by',
        'venus-gen/rustgen.py',
        """                    lines.append('out.push(%s);' % shape[1].replace('val.', 'self.'))""",
        """                    pass""",
        'an_answer_keeps_the_size_of_its_blob',
    ),
    (
        'the shape of an answer leaves out its tag',
        'venus-gen/rustgen.py',
        """                lines.append('out.push(self.sType.0 as u64);')""",
        """                pass""",
        'an_answer_keeps_its_tag',
    ),
    (
        'the shape of an answer leaves out the structs it embeds',
        'venus-gen/rustgen.py',
        """                    lines.append('cs::Shape::shape(&self.%s, out);' % f)""",
        """                    pass""",
        'an_answer_keeps_the_pointers_of_what_it_embeds',
    ),
    (
        'a command can be copied, lending its arrays twice',
        'venus-gen/templates/types.rs',
        """% for ty in GEN.supported_types[VkType.COMMAND]:
#[derive(Default)]""",
        """% for ty in GEN.supported_types[VkType.COMMAND]:
#[derive(Clone, Copy, Default)]""",
        'a_command_cannot_be_duplicated',
    ),
    (
        'an unserved command is counted and then continues, as though the host had done it',
        'src/venus/context.rs',
        """        self.todo.note(cmd);
        self.reject("is not a command this build serves");""",
        """        self.todo.note(cmd);""",
        'unserved_command',
    ),
    (
        'a query read-back is handed to the driver with results that run past the room the guest offered',
        'src/venus/driver.rs',
        """        if facts.bytes_for(count, stride, flags)? > out.len() as u64 {
            return Err(QueryRefused::OutOfRoom);
        }""",
        """        facts.bytes_for(count, stride, flags)?;""",
        'query',
    ),
    (
        'a query read-back names queries past the end of the pool',
        'src/venus/driver.rs',
        """        let facts = self.query_facts(pool)?;
        facts.holds(first, count)?;
        if facts.bytes_for""",
        """        let facts = self.query_facts(pool)?;
        if facts.bytes_for""",
        'query',
    ),
    (
        'a host-side query pool reset is handed to the driver past the end of the pool, and the driver zeroes host memory there',
        'src/venus/driver.rs',
        """        self.query_facts(pool)?.holds(first, count)?;
        // SAFETY: a device in this table, a pool recorded on it, and a range of queries the pool
        // holds -- which is what the driver writes host memory at.""",
        """        self.query_facts(pool)?;
        // SAFETY: a device in this table, a pool recorded on it, and a range of queries the pool
        // holds -- which is what the driver writes host memory at.""",
        'query',
    ),
    (
        'a recorded query begin is handed to the driver past the end of the pool',
        'src/venus/driver.rs',
        """        let (d, facts) = self.query_recorder(cb, pool)?;
        facts.holds(query, 1)?;
        facts.is_bracketed()?;
        // SAFETY: as above, and a query the pool holds.
        unsafe { (d.vkCmdBeginQuery())(cb, pool, query, flags) };""",
        """        let (d, facts) = self.query_recorder(cb, pool)?;
        facts.is_bracketed()?;
        // SAFETY: as above, and a query the pool holds.
        unsafe { (d.vkCmdBeginQuery())(cb, pool, query, flags) };""",
        'query',
    ),
    (
        'a recorded query-pool result copy is handed to the driver past the end of the pool',
        'src/venus/driver.rs',
        """        let (d, facts) = self.query_recorder(cb, pool)?;
        facts.holds(first, count)?;
        // SAFETY: as above, and a range of queries the pool holds.
        unsafe {
            (d.vkCmdCopyQueryPoolResults())""",
        """        let (d, _facts) = self.query_recorder(cb, pool)?;
        // SAFETY: as above, and a range of queries the pool holds.
        unsafe {
            (d.vkCmdCopyQueryPoolResults())""",
        'query',
    ),
    (
        'a query kind this host advertises through an extension is sized as one nobody can size, so its read-back is refused',
        'src/venus/driver.rs',
        """            | VkQueryType::VK_QUERY_TYPE_PRIMITIVES_GENERATED_EXT
            | VkQueryType::VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT""",
        """            | VkQueryType::VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT""",
        'query',
    ),
    (
        "a query read-back's status word is not counted in the room it needs",
        'src/venus/driver.rs',
        """            + u64::from(has(VkQueryResultFlagBits::VK_QUERY_RESULT_WITH_AVAILABILITY_BIT))
            + u64::from(has(VkQueryResultFlagBits::VK_QUERY_RESULT_WITH_STATUS_BIT_KHR));""",
        """            + u64::from(has(VkQueryResultFlagBits::VK_QUERY_RESULT_WITH_AVAILABILITY_BIT));""",
        'query',
    ),
    (
        'a destroyed query pool keeps its record, so a recycled handle is measured against a previous life',
        'src/venus/driver.rs',
        """            VkObjectType::VK_OBJECT_TYPE_QUERY_POOL => {
                self.query_pools.remove(&VkQueryPool::from_host(handle));
            }""",
        """            VkObjectType::VK_OBJECT_TYPE_QUERY_POOL => {}""",
        'query_results',
    ),
    (
        "a query pool the device's teardown took keeps its record",
        'src/venus/driver.rs',
        """        for (ty, handle) in recorded {
            self.forget(ty, handle);
        }""",
        """        for (ty, handle) in recorded {
            let _ = (ty, handle);
        }""",
        'query',
    ),
    (
        'a query command hands the driver its arguments in the wrong order',
        'src/venus/context.rs',
        """        let done = self.driver.cmd_copy_query_pool_results(
            args.commandBuffer,
            args.queryPool,
            args.firstQuery,
            args.queryCount,
            args.dstBuffer,
            args.dstOffset,
            args.stride,""",
        """        let done = self.driver.cmd_copy_query_pool_results(
            args.commandBuffer,
            args.queryPool,
            args.firstQuery,
            args.queryCount,
            args.dstBuffer,
            args.stride,
            args.dstOffset,""",
        'query_results',
    ),
    (
        'a ghost absorbs a command the guest is waiting on, and the guest reads a stale reply slot as its answer',
        'src/venus/context.rs',
        """                if wants_reply {
                    poison(
                        id,
                        &dec,
                        cmd,
                        &format!("wanted a reply, and names object {} the host refused", ghost.0),
                    );
                    break;
                }
""",
        '',
        'a_ghost_absorbs_a_command',
    ),
    (
        'a reply is committed before the command that produced it is judged',
        'src/venus/ring.rs',
        '        self.pos += bytes.len();\n        Ok(())',
        '        Ok(())',
        '',
    ),
    (
        'a reply that does not fit is written anyway',
        'src/venus/ring.rs',
        '        if bytes.len() > remaining {\n            return Err(ReplyOverflow { wanted: bytes.len(), remaining });\n        }',
        '        let bytes = &bytes[..bytes.len().min(remaining)];',
        '',
    ),
    (
        'a seek past the reply window is clamped instead of refused',
        'src/venus/ring.rs',
        '        if pos > self.window.size() {\n            return false;\n        }\n        self.pos = pos;',
        '        self.pos = pos.min(self.window.size());',
        '',
    ),
    (
        'a run of pool allocations is filed one object out of step',
        'src/venus/driver.rs',
        '            out.iter().copied().zip(ids.iter().copied()).filter(|(h, _)| h.host().raw() != 0),',
        '            out.iter().copied().zip(ids.iter().skip(1).copied()).filter(|(h, _)| h.host().raw() != 0),',
        '',
    ),
    (
        'a refused run of pool allocations leaves its ids plain missing',
        'src/venus/context.rs',
        '            eprintln!("[virglrs] vkAllocateCommandBuffers refused by the driver");\n            self.ghost_ids(ids);',
        '            eprintln!("[virglrs] vkAllocateCommandBuffers refused by the driver");',
        '',
    ),
    (
        'a short enumeration ghosts the devices it did answer for',
        'src/venus/context.rs',
        '        self.ghost_ids(&ids[got as usize..]);',
        '        self.ghost_ids(ids);',
        '',
    ),
    (
        'a short enumeration reports the length the guest asked for',
        'src/venus/context.rs',
        '        if let Some(mut count) = args.pPhysicalDeviceCount_mut() {\n            count.set(got);\n        }',
        '        let asked = ids.len() as u32;\n        if let Some(mut count) = args.pPhysicalDeviceCount_mut() {\n            count.set(asked);\n        }',
        '',
    ),
    (
        'extensions are learned for slots the driver never filled',
        'src/venus/context.rs',
        '        for pd in out.iter().take(got as usize) {',
        '        for pd in out.iter() {',
        '',
    ),
    (
        'a shader whose code is not a whole number of words is passed on',
        'src/venus/context.rs',
        '        if info.codeSize % 4 != 0 {',
        '        if false {',
        '',
    ),
    (
        'constants are pushed without the bytes that are the command',
        'src/venus/context.rs',
        '        let Some(values) = args.pValues() else {\n            self.reject("pushed constants without saying what they are");\n            return;\n        };',
        '        let values = args.pValues().unwrap_or(&[]);',
        '',
    ),
    (
        'an empty submit is turned away instead of signalling its fence',
        'src/venus/context.rs',
        '        let submits = match args.pSubmits().validate(&self.driver.facts()) {',
        '        if args.pSubmits().is_empty() {\n            return;\n        }\n        let submits = match args.pSubmits().validate(&self.driver.facts()) {',
        '',
    ),
    (
        "a bind sends the dynamic offsets' count with the descriptor sets",
        'src/venus/context.rs',
        '            sets,\n            offsets,\n        );\n        self.held(done);',
        '            sets,\n            &offsets[..offsets.len().min(sets.len())],\n        );\n        self.held(done);',
        '',
    ),
    (
        'a failed pipeline run leaks the pipelines it did make',
        'src/venus/driver.rs',
        "            unsafe { (fns.vkDestroyPipeline())(device, *survivor, ptr(alloc)) };\n            // The guest's reply must not carry a handle that is now gone.\n            *survivor = VkPipeline::NULL;",
        '            *survivor = VkPipeline::NULL;',
        '',
    ),
    (
        'a failed pipeline run leaves destroyed handles in the reply',
        'src/venus/driver.rs',
        "            unsafe { (fns.vkDestroyPipeline())(device, *survivor, ptr(alloc)) };\n            // The guest's reply must not carry a handle that is now gone.\n            *survivor = VkPipeline::NULL;",
        '            unsafe { (fns.vkDestroyPipeline())(device, *survivor, ptr(alloc)) };',
        '',
    ),
    (
        'a capture larger than the allocation is clamped to fit instead of refused',
        'src/venus/driver.rs',
        """        if src.len() as u64 > record.size {
            return Err(MemoryError::LargerThanAllocation);
        }""",
        """        let src = &src[..src.len().min(record.size as usize)];""",
        'a_capture_goes_back_in_by_the_route_it_came_out_of',
    ),
    (
        'a binary semaphore reaches the timeline entry points unchecked',
        'src/venus/driver.rs',
        """        match self.semaphores.get(&sem).map(|f| f.kind) {
            Some(SemaphoreKind::Timeline) => Ok(()),
            Some(SemaphoreKind::Binary) => Err(NotATimeline::Binary),
            None => Err(NotATimeline::Unrecorded),
        }""",
        """        let _ = sem;
        Ok(())""",
        'a_binary_semaphore_in_a_timeline_command_is_refused',
    ),
    (
        "a semaphore's kind is forgotten while the semaphore is still live",
        'src/venus/driver.rs',
        """        self.semaphores.insert(sem, SemaphoreFacts { kind, requested });
        Ok(sem)""",
        """        let _ = kind;
        self.semaphores.insert(sem, SemaphoreFacts { kind: SemaphoreKind::Timeline, requested });
        Ok(sem)""",
        'a_binary_semaphore_in_a_timeline_command_is_refused',
    ),
    (
        'a fence with a submit outstanding is captured as the driver reports it',
        'src/venus/driver.rs',
        """        if self.pending_fences.contains(&fence) {
            return true;
        }""",
        """""",
        'a_captured_sync_state_puts_a_rebuilt_world_back_where_it_was',
    ),
    (
        'a sync restore only ever signals, never resets',
        'src/venus/context.rs',
        """                        (true, false) => self.driver.fast_forward(device, VkSemaphore::NULL, fence),
                        (false, true) => self.driver.unsignal_fence(device, fence),""",
        """                        (true, false) => self.driver.fast_forward(device, VkSemaphore::NULL, fence),
                        (false, true) => true,""",
        'a_captured_sync_state_puts_a_rebuilt_world_back_where_it_was',
    ),
    (
        "a timeline's promised value is dropped in favour of its counter",
        'src/venus/driver.rs',
        """        reached.max(requested)""",
        """        reached""",
        'a_captured_sync_state_puts_a_rebuilt_world_back_where_it_was',
    ),
    # The classic contents blob. Only the codec is reachable from a unit test: a no-op restore is
    # caught by the replay gate's scrub, and "a level that could not be read back is written as
    # zeros" is caught by the composite corpora, whose planar decode targets a texture transfer
    # cannot move -- 41 skipped levels on --ctx 10,11 and 6 on --ctx 8,9. Zeros in their place
    # would change the entry count and the all-zero count, both of which the gate prints.
    (
        'a truncated content blob is parsed as far as it goes instead of refused',
        'src/vrend/content.rs',
        """        if blob.len() - at < size {
            return Err(Malformed::Truncated);
        }""",
        """        let size = size.min(blob.len() - at);""",
        'a_blob_that_is_not_one_is_refused',
    ),
    # The unserved command's fiction. It is a decode convenience with no host object behind it,
    # and the crash it caused was a guest sending one unserved create and then exiting.
    (
        "an unserved create's invented handle goes into the table as an object",
        'src/venus/context.rs',
        """            self.objects.borrow_mut().add_fiction(id, ty, owner);
            return;""",
        """            let _ = self.objects.borrow_mut().add(id, ty, HostHandle(id.0), owner);
            return;""",
        'an_unserved_creates_invented_handle_never_reaches_the_driver',
    ),
    (
        "a compute pipeline run is compiled without the guest's pipeline cache",
        'src/venus/context.rs',
        """        let host = self.driver.create_pipelines(device, cache, infos, alloc, out);
        args.ret = host.err().unwrap_or(VkResult::VK_SUCCESS);
        if host.is_err() {
            eprintln!("[virglrs] vkCreateComputePipelines refused by the driver");""",
        """        let host = self.driver.create_pipelines(device, Default::default(), infos, alloc, out);
        args.ret = host.err().unwrap_or(VkResult::VK_SUCCESS);
        if host.is_err() {
            eprintln!("[virglrs] vkCreateComputePipelines refused by the driver");""",
        'the_compute_pipeline_pair_reaches_the_driver_as_the_guest_sent_it',
    ),
    (
        "a dispatch's group counts are passed in whatever order",
        'src/venus/driver.rs',
        '        unsafe { (d.vkCmdDispatch())(cb, x, y, z) };',
        '        unsafe { (d.vkCmdDispatch())(cb, z, y, x) };',
        'the_compute_pipeline_pair_reaches_the_driver_as_the_guest_sent_it',
    ),
    # The classic blob a context describes for itself. The claim's GL half needs the live host,
    # so what is scored here is the reconciliation and the bookkeeping -- which is where every
    # defect the C carries on this path lives.
    (
        "a blob larger than the resource backing it is trimmed to fit",
        'src/vrend/vrend.rs',
        """    if size > width as u64 {
        return Err(ClaimRefused::Oversize { asked: size, allocated: width });
    }
    Ok(())""",
        """    let _ = (size, width);
    Ok(())""",
        'a_blob_is_never_published_past_the_resource_backing_it',
    ),
    (
        'a described blob id may be described twice, and the first claim wins',
        'src/vrend/context.rs',
        """        } else if self.described.contains_key(&blob_id) {
            NotDescribed::IdTaken""",
        """        } else if false {
            NotDescribed::IdTaken""",
        'a_described_blob_is_claimed_once_and_by_the_context_that_described_it',
    ),
    (
        'a claim drops the command that described it, so a rebuild has nothing to send',
        'src/vrend/context.rs',
        '        self.described.remove(&blob_id)',
        '        self.described.remove(&blob_id).map(|mut r| {\n            r.described_by = None;\n            r\n        })',
        'a_described_blob_is_claimed_once_and_by_the_context_that_described_it',
    ),
    # VK_EXT_host_image_copy. The wire's `...MESA` forms carry the bytes where Vulkan carries a
    # host address, so the renderer rebuilds each region by hand -- and every field copied across
    # by hand is one that can be dropped or transposed into a skewed picture with no error
    # anywhere.
    (
        "an image read out lands somewhere other than the reply's own blob",
        'src/venus/driver.rs',
        '            pHostPointer: out.as_mut_ptr().cast(),',
        '            pHostPointer: core::ptr::null_mut(),',
        'the_host_copy_reshape_hands_the_driver_the_copy_the_guest_sent',
    ),
    (
        "a host copy's row length is dropped on the way to the driver",
        'src/venus/driver.rs',
        """                memoryRowLength: r.memoryRowLength,
                memoryImageHeight: r.memoryImageHeight,
                imageSubresource: r.imageSubresource,""",
        """                memoryRowLength: 0,
                memoryImageHeight: r.memoryImageHeight,
                imageSubresource: r.imageSubresource,""",
        'the_host_copy_reshape_hands_the_driver_the_copy_the_guest_sent',
    ),
    (
        'a layout transition is passed the guest\'s count instead of the array it got',
        'src/venus/driver.rs',
        '        Some(unsafe { f(device, transitions.len() as u32, transitions.as_ptr()) })',
        '        Some(unsafe { f(device, 1, transitions.as_ptr()) })',
        'the_host_copy_reshape_hands_the_driver_the_copy_the_guest_sent',
    ),
    # An extension command's entry point is the guest's choice at `vkCreateDevice`, not ours: the
    # capset advertises everything the pinned vk.xml can serialize. So the fallible accessor is
    # what stands between a guest sending a command its own device never enabled and an abort.
    # The ledger of commands a guest may send and this build does not serve. Invisible until it
    # was written down -- three seated-desktop boots found three of its members one at a time.
    (
        'a command a guest may send reaches no handler and is on no ledger',
        'src/venus/unserved.txt',
        'vkQueueBindSparse                                  no-host-feature:sparse-binding\n',
        '',
        'venus::context::tests::every_command_the_protocol_defines_is_served_or_on_the_ledger',
    ),
    (
        'a command served through a macro is read as unserved',
        'src/venus/context.rs',
        '                here = lines.peek().map_or("", |l| l.trim_start());',
        '                here = "";',
        'venus::context::tests::every_command_the_protocol_defines_is_served_or_on_the_ledger',
    ),
    # A strided array. vk.xml's `stride` describes the guest's own memory and never reaches the
    # wire; refusing to serialize it poisoned a context at a command the desktop sends, and
    # forwarding the guest's number would walk the driver through our arena.
    (
        'a strided array is decoded as one element rather than as the count it carries',
        'venus-gen/rustgen.py',
        "        if var.is_blob():",
        """        if 'stride' in var.attrs:
            return ('dynamic', '1')
        if var.is_blob():""",
        'venus::proto::tests::a_strided_array_is_an_ordinary_counted_array_on_the_wire',
    ),
    (
        "a multi-draw walks the driver by the stride the guest sent, not the array's own",
        'src/venus/driver.rs',
        '        let stride = size_of::<VkMultiDrawIndexedInfoEXT>() as u32;',
        '        let stride = 0u32;',
        'venus::context::tests::a_multi_draw_is_walked_by_the_array_it_was_given',
    ),
    # The recording commands the seated desktop sends. Three shapes nothing else can see: a
    # fixed array parameter (C passes the address, not the aggregate), a count-bearing dynamic
    # state command (no first index for the count to be passed as), and four arrays under one
    # count where an absent one must be null and not empty.
    (
        'a fixed-size array argument is handed to the driver as an aggregate',
        'src/venus/driver.rs',
        '        unsafe { (d.vkCmdSetBlendConstants())(cb, constants.as_ptr()) };',
        '        unsafe { (d.vkCmdSetBlendConstants())(cb, [0.0; 4].as_ptr()) };',
        'the_desktop_recording_shapes_reach_the_driver_as_the_guest_sent_them',
    ),
    (
        "a count-bearing dynamic state command is passed a first index it has no room for",
        'src/venus/driver.rs',
        '        unsafe { f(cb, viewports.len() as u32, viewports.as_ptr()) };',
        '        unsafe { f(cb, 1, viewports.as_ptr()) };',
        'the_desktop_recording_shapes_reach_the_driver_as_the_guest_sent_them',
    ),
    (
        'an optional array the guest did send is dropped on the way to the driver',
        'src/venus/driver.rs',
        '        let (sizes, strides) = (optional(sizes), optional(strides));',
        '        let (sizes, strides) = (optional(sizes), core::ptr::null());',
        'the_desktop_recording_shapes_reach_the_driver_as_the_guest_sent_them',
    ),
    (
        'an extension recording command aborts when the device has no entry point for it',
        'src/venus/driver.rs',
        '        let f = self.recorder(cb)?.try_vkCmdSetAttachmentFeedbackLoopEnableEXT()?;',
        '        let f = self.recorder(cb)?.vkCmdSetAttachmentFeedbackLoopEnableEXT();',
        'an_extension_recording_command_the_device_does_not_export_is_refused_and_not_aborted_on',
    ),
    (
        'a feedback-loop enable reaches the driver with an aspect mask the guest never sent',
        'src/venus/driver.rs',
        '        unsafe { f(cb, aspects) };',
        '        unsafe { f(cb, VkImageAspectFlags(0)) };',
        'an_extension_recording_command_the_device_does_not_export_is_refused_and_not_aborted_on',
    ),
    (
        "a scanout's restore goes through vkMapMemory like any other allocation",
        'src/venus/driver.rs',
        """        if let Some(surface) = record.surface() {
            return Ok(surface.write_from(src));
        }
""",
        """""",
        'a_capture_goes_back_in_by_the_route_it_came_out_of',
    ),
    # Not here any more: "a blob is attributed to whichever context holds its id now". The line it
    # broke matched a resource's exporter against a context key, and it went with the table walk it
    # sat in: a held export is now read from the context's own record, which a new occupant of the
    # id does not inherit because the record dies with the context. There is no comparison left to
    # weaken. `a_reused_context_id_does_not_inherit_the_previous_contexts_blobs` still pins the
    # behaviour, against the design rather than against a line.
    (
        'the census reports storage a guest only borrowed',
        'src/venus/driver.rs',
        '            .filter(|(_, a)| a.censused())',
        '            .filter(|(_, a)| a.censused() || true)',
        '',
    ),
    (
        'an import is billed as though it were fresh storage',
        'src/venus/driver.rs',
        """            (Some(bytes), _, _) => Planned::Ready(Backing::Imported(bytes)),""",
        """            (Some(bytes), _, _) => {
                std::mem::forget(self.admit("device memory", size)?);
                Planned::Ready(Backing::Imported(bytes))
            }""",
        '',
    ),
    (
        'the cap is consulted and then ignored',
        'src/budget.rs',
        """        if let Some(cap) = self.budget.cap
            && live.saturating_add(size) > cap
        {""",
        """        if let Some(cap) = self.budget.cap
            && false
        {""",
        '',
    ),
    (
        'a budget refusal leaves the context running',
        'src/venus/driver.rs',
        """            NoMemory::OverBudget { stop: self.account.kills_context() }""",
        """            NoMemory::OverBudget { stop: false }""",
        '',
    ),
    (
        'a scanout is charged at the size the guest asked for, not the surface it got',
        'src/venus/driver.rs',
        """                let charge = self.admit("IOSurface", surface.alloc_size())?;""",
        """                let charge = self.admit("IOSurface", size)?;""",
        '',
    ),
    (
        'a budget refusal is reported to the guest and to nobody else',
        'src/venus/context.rs',
        '''            if let driver::NoMemory::OverBudget { stop: true } = e {
                self.reject("the host memory budget refused this allocation");
            }
''',
        '',
        '',
    ),
    (
        'an image-to-buffer copy hands the driver no regions',
        'src/venus/driver.rs',
        '''            (d.vkCmdCopyImageToBuffer())(
                cb,
                src,
                layout,
                dst,
                regions.len() as u32,
                regions.as_ptr(),
            )''',
        '''            (d.vkCmdCopyImageToBuffer())(cb, src, layout, dst, 0, regions.as_ptr())''',
        '',
    ),
    (
        'an image-to-buffer copy never reaches the driver at all',
        'src/venus/context.rs',
        '''        let done = self.driver.cmd_copy_image_to_buffer(
            args.commandBuffer,
            args.srcImage,
            args.srcImageLayout,
            args.dstBuffer,
            regions,
        );''',
        '''        let done = Some(());''',
        '',
    ),
    (
        'a monitored ring is registered but never stamped',
        'src/venus/monitor.rs',
        '''                Some(status) => {
                    status.set_bits(STATUS_ALIVE);
                    true
                }''',
        '''                Some(_) => true,''',
        '',
    ),
    (
        'a ring asking to be monitored is quietly not monitored',
        'src/venus/context.rs',
        '''        if let Some(want) = monitor_period(info) {''',
        '''        if let Some(want) = None::<Option<u32>> {''',
        '',
    ),
    (
        'a reporting period of zero is given a default instead of being refused',
        'src/venus/context.rs',
        '''    Some(Some(m.maxReportingPeriodMicroseconds).filter(|&us| us != 0))''',
        '''    Some(Some(m.maxReportingPeriodMicroseconds).filter(|&us| us != 0).or(Some(3_000_000)))''',
        '',
    ),
    (
        'a shortened reporting period does not wake the sleeping monitor',
        'src/venus/monitor.rs',
        '''            self.shared.wake.notify_one();
        }
    }
}''',
        '''        }
    }
}''',
        '',
    ),
    (
        'a ring blocks on a virtqueue seqno without telling the waiter',
        'src/venus/ring_thread.rs',
        '''    state.blocked_on_vq = Some(seqno);''',
        '''    state.blocked_on_vq = None;''',
        '',
    ),
    (
        'a ring wait sleeps without ever checking whether it is the only thing that could end it',
        'src/venus/ring_thread.rs',
        '''        if let Some(want) = self.park.stalled_on() {''',
        '''        if let Some(want) = None::<u64> {''',
        '',
    ),
    (
        'a stalled ring is judged on two values that were never true together',
        'src/venus/ring_thread.rs',
        '''        state.blocked_on_vq.filter(|&want| state.vq_seqno < want)''',
        '''        state.blocked_on_vq''',
        '',
    ),
    (
        "a present barrier inherits the context stream's deadlock verdict",
        'src/venus/ring_thread.rs',
        '''        matches!(self.inner.run(), Stop::Reached)''',
        '''        self.inner.wait()''',
        'a_present_barrier',
    ),
    (
        'a ring wait past everything the guest wrote is waited through',
        'src/venus/ring_thread.rs',
        '''        if head == tail && !seqno_ge(tail, self.seqno) {''',
        '''        if false {''',
        '',
    ),
    (
        'a suspended batch reports the wait command as already run',
        'src/venus/context.rs',
        '''            suspended = Some((at, on));''',
        '''            suspended = Some((dec.pos(), on));''',
        '',
    ),
    (
        'a virtqueue wait suspends even when the seqno is already published',
        'src/venus/context.rs',
        '''        if published < args.seqno {
            self.suspend(Wait::Virtqueue(args.seqno));
        }''',
        '''        let _ = published;
        self.suspend(Wait::Virtqueue(args.seqno));''',
        '',
    ),
    (
        'a ring seqno wider than a ring position is truncated instead of refused',
        'src/venus/context.rs',
        '''        let Ok(seqno) = u32::try_from(args.seqno) else {
            self.reject("waited on a ring seqno too large to be a position in a ring");
            return;
        };''',
        '''        let seqno = args.seqno as u32;''',
        '',
    ),
    (
        'a ring extra write reaches past the extra region',
        'src/venus/ring.rs',
        '''        if end > self.extra.begin() + self.extra.size() {
            return false;
        }''',
        '''''',
        '',
    ),
    (
        'a virtqueue seqno submitted before the ring started is dropped',
        'src/venus/ring_thread.rs',
        '''        state: Mutex::new(ParkState { vq_seqno: ring.virtqueue_seqno, ..ParkState::default() }),''',
        '''        state: Mutex::new(ParkState::default()),''',
        '',
    ),
    (
        'a transport command is served on whichever stream it arrives on',
        'src/venus/context.rs',
        '''        let Some(id) = self.current_ring else {
            self.reject("waited on a virtqueue seqno from the context's own stream");
            return;
        };''',
        '''        let id = self.current_ring.unwrap_or(RingId::new(7).expect("seven is not zero"));''',
        '',
    ),
    (
        'an executed stream is not bounds-checked against the resource holding it',
        'src/venus/context.rs',
        """        let end = s.offset.checked_add(s.size);
        if end.is_none_or(|end| end > map.len()) {""",
        """        let end = s.offset.checked_add(s.size);
        if false {""",
        '',
    ),
    (
        'an executed stream may execute streams of its own, as deep as the guest likes',
        'src/venus/context.rs',
        """            if h.depth > 0 {
                poison(
                    id,
                    &dec,
                    cmd,
                    "executes command streams from inside a command stream it is already executing",
                );
                break;
            }""",
        """""",
        '',
    ),
    (
        'a per-stream reply position is ignored',
        'src/venus/context.rs',
        """        if let Some(&pos) = exec.reply_positions.as_ref().map(|p| &p[i]) {""",
        """        if let Some(&pos) = None::<&usize> {""",
        '',
    ),
    (
        'a reply position outside the window is clamped rather than refused',
        'src/venus/context.rs',
        """            if !stream.seek(pos) {""",
        """            let pos = pos.min(stream.window().size());
            if !stream.seek(pos) {""",
        '',
    ),
    (
        'an empty stream skips before its reply position is honoured',
        'src/venus/context.rs',
        """        if let Some(&pos) = exec.reply_positions.as_ref().map(|p| &p[i]) {""",
        """        if s.size == 0 {
            continue;
        }
        if let Some(&pos) = exec.reply_positions.as_ref().map(|p| &p[i]) {""",
        '',
    ),
    (
        'reply positions are accepted with no window for them to be positions in',
        'src/venus/context.rs',
        """            Some(_) if self.reply.is_none() => {
                self.reject("executed command streams with reply positions and no reply stream");
                return;
            }""",
        """""",
        '',
    ),
    (
        "a ring's shared memory is reachable by any context that guesses its handle",
        'src/renderer.rs',
        """        if !res.attached.contains(&ctx) {
            eprintln!(
                "[virglrs] ctx {}: resource {handle:?} is not attached to this context",
                ctx.get(),
            );
            return None;
        }
        if let Some(map) = res.shm() {
            return Some(ResourceBytes::Host(Arc::clone(map)));
        }""",
        """        if let Some(map) = res.shm() {
            return Some(ResourceBytes::Host(Arc::clone(map)));
        }
        if !res.attached.contains(&ctx) {
            eprintln!(
                "[virglrs] ctx {}: resource {handle:?} is not attached to this context",
                ctx.get(),
            );
            return None;
        }""",
        'a_context_reaches_the_resources_the_guest_attached_to_it',
    ),
    (
        'the gate on what the guest attached to a context is dropped',
        'src/renderer.rs',
        """        if !res.attached.contains(&ctx) {""",
        """        if false {""",
        '',
    ),
    (
        'a scanout export lends no share, so its storage stays trapped in one context',
        'src/venus/driver.rs',
        """        let share = storage.clone();""",
        """        let share = match storage {
            Storage::Texture(_) => return Err(ExportError::NotMappable),
            other => other.clone(),
        };""",
        'memory_is_published_once_and_leaves_the_census_when_it_is',
    ),
    (
        "a context's destroy uncounts what still outlives it",
        'src/budget.rs',
        """        slot.live.drain_into(&mut ledger.shared);""",
        """        drop(slot);""",
        'a_charge_that_outlives_its_context_stays_counted',
    ),
    (
        'the cap stops counting storage the moment its context is gone',
        'src/budget.rs',
        """        self.ctxs.values().map(|s| s.live.bytes()).sum::<u64>()
            + self.shared.bytes()
            + self.classic.bytes()""",
        """        self.ctxs.values().map(|s| s.live.bytes()).sum::<u64>()
            + self.classic.bytes()""",
        'a_charge_that_outlives_its_context_stays_counted',
    ),
    (
        'a late credit lands on whichever context holds the id now',
        'src/budget.rs',
        """        self.ctxs.get_mut(&ctx.id()).filter(|s| s.ctx == ctx).map(|s| &mut s.live)""",
        """        self.ctxs.get_mut(&ctx.id()).map(|s| &mut s.live)""",
        'a_late_credit_never_lands_on_the_next_context_with_the_same_id',
    ),
    (
        'a credit skips the context that took it and lands on the shared bucket',
        'src/budget.rs',
        """            Payer::Ctx(ctx) => match ledger.slot_of(ctx) {""",
        """            Payer::Ctx(ctx) => match ledger.slot_of(ctx).filter(|_| false) {""",
        'budget::every_sequence',
    ),
    (
        'a charge that exactly fills the cap is refused',
        'src/budget.rs',
        """            && live.saturating_add(size) > cap""",
        """            && live.saturating_add(size) >= cap""",
        'budget::every_sequence',
    ),
    (
        'a classic credit lands on the shared bucket',
        'src/budget.rs',
        """            Payer::Classic => ledger.classic.credit(self.what, self.size),""",
        """            Payer::Classic => ledger.shared.credit(self.what, self.size),""",
        'budget::every_sequence',
    ),
    (
        'an image the guest shares keeps the opaque tiling it asked for',
        'src/venus/driver.rs',
        """        info.tiling = VkImageTiling::VK_IMAGE_TILING_LINEAR;
""",
        """""",
        '',
    ),
    (
        'an opaque image gets a surface the driver will never write into',
        'src/venus/driver.rs',
        """        if !matches!(
            facts.tiling,
            VkImageTiling::VK_IMAGE_TILING_LINEAR
                | VkImageTiling::VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT
        ) {""",
        """        if false {""",
        '',
    ),
    (
        'host-visible memory is left to the driver again, so a blob over it has nothing to hold',
        'src/venus/driver.rs',
        """        let host_visible = props.is_some_and(|p| p.0 & HOST_VISIBLE_BIT != 0);""",
        """        let host_visible = false;""",
        'host_addressable_memory_is_backed_by_pages_this_renderer_minted',
    ),
    (
        'an allocation is published twice, and neither resource can learn of the other',
        'src/venus/driver.rs',
        """        *published = true;""",
        """        *published = false;""",
        'memory_is_published_once_and_leaves_the_census_when_it_is',
    ),
    (
        'a blob is published larger than the storage it was minted from',
        'src/venus/driver.rs',
        """        if blob_size > len {
            return Err(ExportError::LargerThanAllocation);
        }""",
        """""",
        'memory_is_published_once_and_leaves_the_census_when_it_is',
    ),
    (
        'minted pages lend no share, so the buffer stays trapped in one context',
        'src/venus/driver.rs',
        """        let share = storage.clone();""",
        """        let share = match storage {
            Storage::Linear(_) => return Err(ExportError::NotMappable),
            other => other.clone(),
        };""",
        '',
    ),
    (
        'freeing a descriptor set is refused again, and every GTK client dies a few frames in',
        'src/venus/context.rs',
        """    fn vkFreeDescriptorSets(&mut self, args: &mut vn_command_vkFreeDescriptorSets<'_>) {
        let sets = self.array_or_empty(args.pDescriptorSets());""",
        """    fn vkFreeDescriptorSets(&mut self, args: &mut vn_command_vkFreeDescriptorSets<'_>) {
        self.unsupported(VkCommandTypeEXT::VK_COMMAND_TYPE_vkFreeDescriptorSets_EXT);
        if true { return; }
        let sets = self.array_or_empty(args.pDescriptorSets());""",
        '',
    ),
    (
        'a free is handed to the driver under whatever pool the guest named',
        'src/venus/driver.rs',
        """        if !self.pools.all_from(pool, objects) {
            return Err(FreeRefused::NotFromThisPool);
        }""",
        """        if !self.pools.is_open(pool) {
            return Err(FreeRefused::NotFromThisPool);
        }""",
        'freeing_descriptor_sets_releases_them_from_their_pool',
    ),
    (
        'saving the pipeline cache is refused again, and every GTK client dies after its first pipeline',
        'src/venus/context.rs',
        """    fn vkGetPipelineCacheData(&mut self, args: &mut vn_command_vkGetPipelineCacheData<'_>) {
        let device = args.device;""",
        """    fn vkGetPipelineCacheData(&mut self, args: &mut vn_command_vkGetPipelineCacheData<'_>) {
        self.unsupported(VkCommandTypeEXT::VK_COMMAND_TYPE_vkGetPipelineCacheData_EXT);
        if true { return; }
        let device = args.device;""",
        '',
    ),
    (
        'an out blob the guest offers room for is decoded as absent, and its handler writes nowhere',
        'venus-gen/rustgen.py',
        """                hit = ['let n = dec.decode_array_size(%s) as usize;' % shape[1],
                       'let Some(a) = dec.alloc_temp_array::<u8>(n) else { return };',
                       '%s = a.as_mut_ptr() as %s _;' % (m, ptr)]""",
        """                hit = ['let n = dec.decode_array_size(%s) as usize;' % shape[1],
                       '%s = %s;' % (m, null)]""",
        'an_out_blob_is_room',
    ),
    (
        'the count call of an out blob is held to the garbage size beside it, and every count call is poisoned',
        'venus-gen/rustgen.py',
        """        if count is not None and not var.is_optional() and var.can_validate():
            miss = ['dec.decode_array_size(%s);' % count]""",
        """        if count is not None:
            miss = ['dec.decode_array_size(%s);' % count]""",
        'an_out_blob_is_room',
    ),
    (
        "an out blob's slice is sized by the length member the handler rewrites, not the room",
        'venus-gen/rustgen.py',
        """                    rows.append((f, 'u8', '(val.room_%s) as u64' % f, True))""",
        """                    rows.append((f, 'u8', shape[1], True))""",
        'an_out_blob_is_bounded_by_the_room',
    ),
    (
        'a reply encodes as many out-blob bytes as the handler claims, past the room',
        'venus-gen/rustgen.py',
        """                        '    assert!(n <= %s, "%s wrote {n} bytes of %s into room for {}", %s);'
                        % (room, ty.name, var.name, room),""",
        """                        '    let _ = %s;' % room,""",
        'an_out_blob_is_bounded_by_the_room',
    ),
    (
        'the fill call of the pipeline cache data reports the room offered, not the bytes written',
        'src/venus/context.rs',
        """        let asked = self.driver.pipeline_cache_data(device, cache, Some(out));
        match asked {
            Ok((n, ret)) => {""",
        """        let room = out.len();
        let asked = self.driver.pipeline_cache_data(device, cache, Some(out));
        match asked {
            Ok((_, ret)) => {
                let n = room;""",
        '',
    ),
    (
        'merging pipeline caches is refused again',
        'src/venus/context.rs',
        """    fn vkMergePipelineCaches(&mut self, args: &mut vn_command_vkMergePipelineCaches<'_>) {
        let srcs = args.pSrcCaches();""",
        """    fn vkMergePipelineCaches(&mut self, args: &mut vn_command_vkMergePipelineCaches<'_>) {
        self.unsupported(VkCommandTypeEXT::VK_COMMAND_TYPE_vkMergePipelineCaches_EXT);
        if true { return; }
        let srcs = args.pSrcCaches();""",
        '',
    ),
    (
        'the image copy the overview blur asks for is refused again',
        'src/venus/context.rs',
        """    fn vkCmdCopyImage(&mut self, args: &mut vn_command_vkCmdCopyImage<'_>) {
        let regions = args.pRegions();
        let done = self.driver.cmd_copy_image(
            args.commandBuffer,
            args.srcImage,
            args.srcImageLayout,
            args.dstImage,
            args.dstImageLayout,
            regions,
        );
        self.held(done);
    }

""",
        """""",
        '',
    ),
    (
        "an image copy hands each image the other one's layout",
        'src/venus/context.rs',
        """    fn vkCmdCopyImage(&mut self, args: &mut vn_command_vkCmdCopyImage<'_>) {
        let regions = args.pRegions();
        let done = self.driver.cmd_copy_image(
            args.commandBuffer,
            args.srcImage,
            args.srcImageLayout,
            args.dstImage,
            args.dstImageLayout,
            regions,
        );
        self.held(done);
    }

""",
        """    fn vkCmdCopyImage(&mut self, args: &mut vn_command_vkCmdCopyImage<'_>) {
        let regions = args.pRegions();
        let done = self.driver.cmd_copy_image(
            args.commandBuffer,
            args.srcImage,
            args.dstImageLayout,
            args.dstImage,
            args.srcImageLayout,
            regions,
        );
        self.held(done);
    }

""",
        '',
    ),
    (
        'only the fd half of the emulated external memory is advertised',
        'src/venus/driver.rs',
        """                EMULATED_ON_THE_HOST
                    .iter()
                    .filter(|n| crate::venus::proto::info::extension(n).is_some()),""",
        """                EMULATED_ON_THE_HOST
                    .iter()
                    .take(1)
                    .filter(|n| crate::venus::proto::info::extension(n).is_some()),""",
        '',
    ),
    (
        'the emulated external memory is advertised on drivers that emulate nothing',
        'src/venus/driver.rs',
        """        self.supports(pd, "VK_EXT_external_memory_metal")
            && !self.supports(pd, "VK_KHR_external_memory_fd")""",
        """        self.supports(pd, "VK_EXT_external_memory_metal")""",
        '',
    ),
    (
        'a transport wait inside an executed stream suspends a batch it cannot resume',
        'src/venus/context.rs',
        """            if h.depth > 0 {
                poison(
                    id,
                    &dec,
                    cmd,
                    "suspends the batch, and a command stream being executed has nowhere to suspend to",
                );
                break;
            }""",
        """""",
        '',
    ),
    (
        'a classic blend factor is read from the bit beside it',
        'src/vrend/decode.rs',
        "Some(RtBlendEq { rgb: eq(1, 4, 9)?, alpha: eq(14, 17, 22)? })",
        "Some(RtBlendEq { rgb: eq(1, 5, 9)?, alpha: eq(14, 17, 22)? })",
        'vrend',
    ),
    (
        'a refused classic command does not end its batch',
        'src/vrend/decode.rs',
        "self.at = if decoded.is_ok() { end } else { self.words.len() };",
        "self.at = end;",
        'vrend',
    ),
    (
        'a classic slot range past the per-stage array is bound',
        'src/vrend/decode.rs',
        "if count > max || start as usize > max - count {",
        "if count > max {",
        'vrend',
    ),
    (
        'a classic shader continuation is taken for a new shader',
        'src/vrend/decode.rs',
        "let chunk = if offlen >> 31 != 0 {",
        "let chunk = if offlen >> 31 == 2 {",
        'vrend',
    ),
    (
        'an inline write hands its bytes over starting one dword early',
        'src/vrend/decode.rs',
        "Command::ResourceInlineWrite { transfer: w.transfer()?, data: w.tail(12) }",
        "Command::ResourceInlineWrite { transfer: w.transfer()?, data: w.tail(11) }",
        'vrend',
    ),
    (
        'a classic resource lends nothing, so a venus context cannot import one',
        'src/renderer.rs',
        'Some(held) => Some(ResourceBytes::Shared(Storage::lent(Arc::clone(held)))),',
        'Some(_) => None,',
        'a_classic_resource_lends',
    ),
    (
        "a resource's claim is dropped without condemning vrend's half of it",
        'src/vrend/resource.rs',
        """impl Drop for Claim {
    fn drop(&mut self) {
        self.condemned.condemn(self.handle);
    }
}""",
        """impl Drop for Claim {
    fn drop(&mut self) {}
}""",
        'a_reset_frees_the_handles_vrend_held',
    ),
    (
        "a venus context resolves to the classic renderer, so vrend gets its blob attach",
        'src/renderer.rs',
        """            CapsetId::Venus => Bound::Venus(VenusCtx(ctx)),""",
        """            CapsetId::Venus => Bound::Classic(ClassicCtx(ctx)),""",
        'a_venus_contexts_own_blob_is_released_by_its_unref_without_classic_work',
    ),
    (
        'a switch records the context before the winsys has made it current',
        'src/vrend/current.rs',
        """        winsys.make_current(ctx)?;
        self.switched_to(on);
        Ok(())""",
        """        self.switched_to(on);
        winsys.make_current(ctx)?;
        Ok(())""",
        'a_refused_switch_leaves_the_shadow_on_the_context_the_thread_kept',
    ),
    (
        'an executed stream runs at the depth of the batch that named it',
        'src/venus/context.rs',
        """        self.depth += 1;
        let out = f(self);
        self.depth -= 1;
        out""",
        """        f(self)""",
        'an_execute_inside_an_executed_stream_is_refused',
    ),
    (
        'a layout with more planes than it holds is trimmed to fit',
        'src/surface.rs',
        """        if planes.len() > MAX_PLANES {
            return Err(BadLayout::TooManyPlanes { said: planes.len(), max: MAX_PLANES });
        }
        let mut at""",
        """        let planes = &planes[..planes.len().min(MAX_PLANES)];
        let mut at""",
        'a_layout_the_guest_got_wrong_is_refused_by_name_and_never_trimmed',
    ),
    (
        'a journal handed to a live context opens a replay of its own',
        'src/venus/context.rs',
        '        let Some(r) = self.replay.as_mut() else { return Err(NOT_REPLAYING) };',
        '        let r = self.replay.get_or_insert_with(Replay::default);',
        'a_journal_fed_to_a_live_context_is_refused',
    ),
    (
        'a journal fed to a live context opens a replay of its own',
        'src/venus/context.rs',
        '        let Some(r) = self.replay.as_mut() else { return Err(NotReplaying) };',
        '        let r = self.replay.get_or_insert_with(Replay::default);',
        'a_journal_fed_to_a_live_context_is_refused',
    ),
    (
        'a journal handed to a live classic context opens a replay of its own',
        'src/vrend/context.rs',
        '        let Some(r) = self.replay.as_mut() else { return Err(NOT_REPLAYING) };',
        '        let r = self.replay.get_or_insert_with(Replay::default);',
        'a_journal_fed_to_a_live_classic_context_is_refused',
    ),
    (
        'a journal fed to a live classic context is fed as though it were replaying',
        'src/vrend/context.rs',
        '        if self.replay.is_none() {\n            return Err(Unfed::NotReplaying);\n        }',
        '        if self.replay.is_none() {\n            return Ok(());\n        }',
        'a_journal_fed_to_a_live_classic_context_is_refused',
    ),
    (
        'a second replay_begin drops the journal the first was handed',
        'src/venus/context.rs',
        '        self.replay.get_or_insert_with(Replay::default);',
        '        self.replay = Some(Replay::default());',
        'a_replay_begun_twice_keeps_the_journal_it_was_handed',
    ),
    (
        'a second classic replay_begin drops the journal the first was handed',
        'src/vrend/context.rs',
        '        self.replay.get_or_insert_with(Replay::default);',
        '        self.replay = Some(Replay::default());',
        'a_classic_replay_begun_twice_keeps_the_journal_it_was_handed',
    ),
    (
        'a slot one past the last is minted, as the C admitted it for an image load',
        'src/vrend/pipe.rs',
        '(index < Self::COUNT).then(|| Self(index as u8))',
        '(index <= Self::COUNT).then(|| Self(index as u8))',
        'an_image_load_past_the_last_slot_loads_zero',
    ),
    (
        'a sampler declared past the last slot wraps onto a low one',
        'src/vrend/shader/glsl/decl.rs',
        'let Some(last) = SamplerSlot::new(last) else {',
        'let Some(last) = SamplerSlot::new(last % MAX_SAMPLERS as u32) else {',
        'a_sampler_declared_past_the_last_slot_is_refused',
    ),
    (
        "a clip distance count past the hardware limit is stored a byte wide and summed",
        'src/vrend/shader/glsl/decl.rs',
        """        Property::NumClipdistEnabled => {
            if data > MAX_CLIP_OR_CULL_DISTANCES {
                return fail(format!(
                    "Clip distance count {data} exceeds the limit of {MAX_CLIP_OR_CULL_DISTANCES}"
                ));
            }
            ctx.shader_req_bits |= req::CLIP_DISTANCE;""",
        """        Property::NumClipdistEnabled => {
            ctx.shader_req_bits |= req::CLIP_DISTANCE;""",
        'a_clip_or_cull_count_past_the_hardware_limit_is_refused',
    ),
    (
        "a cull distance count past the hardware limit is stored a byte wide and summed",
        'src/vrend/shader/glsl/decl.rs',
        """        Property::NumCulldistEnabled => {
            if data > MAX_CLIP_OR_CULL_DISTANCES {
                return fail(format!(
                    "Cull distance count {data} exceeds the limit of {MAX_CLIP_OR_CULL_DISTANCES}"
                ));
            }
            ctx.num_cull_dist_prop = data as u8;
        }""",
        """        Property::NumCulldistEnabled => ctx.num_cull_dist_prop = data as u8,""",
        'a_clip_or_cull_count_past_the_hardware_limit_is_refused',
    ),
    (
        'an array the host allocator refuses aborts the process instead of poisoning the stream',
        'src/venus/cs.rs',
        """        match self.temp.try_alloc_slice_fill_with(count, |_| T::default()) {
            Ok(a) => Some(a),
            Err(_) => {
                self.set_fatal();
                None
            }
        }""",
        """        Some(self.temp.alloc_slice_fill_with(count, |_| T::default()))""",
        'an_allocation_the_host_refuses_poisons_instead_of_aborting',
    ),
    (
        'a create under a live id reaches the driver, and the object it makes is nobody\'s',
        'src/venus/context.rs',
        """        if self.objects.borrow().get(id).is_some() {
            self.reject("created an object under an id that is already an object");""",
        """        if false && self.objects.borrow().get(id).is_some() {
            self.reject("created an object under an id that is already an object");""",
        'a_create_under_a_live_id_never_reaches_the_driver',
    ),
    (
        'an object the driver hands back is refused the second time it is asked for',
        'src/venus/context.rs',
        """        if handed_back(ty) {
            return true;
        }""",
        """        if false && handed_back(ty) {
            return true;
        }""",
        'an_object_handed_back_again_keeps_its_first_name',
    ),
    (
        'a live id handed a different object keeps the first one quietly',
        'src/venus/context.rs',
        'if host.raw() != 0 && (have.ty != ty || have.handle != host) {',
        'if false && host.raw() != 0 && (have.ty != ty || have.handle != host) {',
        'an_object_handed_back_again_keeps_its_first_name',
    ),
    (
        'an object already named is registered again under a second id',
        'src/venus/context.rs',
        '&& objects.id_of_handle(ty, host).is_some_and(|first| first != id)',
        '&& objects.id_of_handle(ty, host).is_some_and(|first| first == id)',
        'an_object_handed_back_under_a_second_name_is_refused',
    ),
    (
        'a queue is a create, and asking for it twice is refused',
        'src/venus/context.rs',
        'ty == VkObjectType::VK_OBJECT_TYPE_PHYSICAL_DEVICE || ty == VkObjectType::VK_OBJECT_TYPE_QUEUE',
        'ty == VkObjectType::VK_OBJECT_TYPE_PHYSICAL_DEVICE',
        'a_queue_asked_for_again_keeps_its_first_name',
    ),
    (
        'a pNext chain is decoded as deep as the guest cares to make it',
        'venus-gen/rustgen.py',
        """                '    if depth >= %d {' % len(next_types),
                '        dec.set_fatal();',
                '        return core::ptr::null_mut();',
                '    }',
""",
        '',
        'a_pnext_chain_deeper_than_the_structs_it_may_name_is_refused',
    ),
    (
        'a sampler state bound under an unchanged view is never re-bound',
        'src/vrend/context/units.rs',
        """            None => self.samplers.remove(&slot),
        };
        self.dirty.mark(slot);""",
        """            None => self.samplers.remove(&slot),
        };""",
        'binding_a_sampler_state_marks_its_unit',
    ),
    (
        'a rasterizer bind stores the state and never marks the shader dirty',
        'src/vrend/context.rs',
        """        self.rs = state;
        self.shader_dirty = true;""",
        """        self.rs = state;""",
        'binding_a_rasterizer_marks_the_shader_dirty',
    ),
    (
        'a shader blit leaves the level range it read on its source for the next draw to sample',
        'src/vrend/blitter.rs',
        """        restore_tex_param(gl, job.src_gl_target, &prior);""",
        """        let _ = &prior;""",
        'a_blit_puts_back_the_level_range_it_confined_its_source_to',
    ),
    (
        'a destroyed sampler state shifts the ones after it down without a re-bind',
        'src/vrend/context/units.rs',
        """                self.dirty.mark(slot);
                self.dirty.mark(slot - shift);""",
        """                self.dirty.mark(slot);""",
        'a_destroyed_sampler_state_closes_its_gap_and_marks_every_slot_that_moved',
    ),
    (
        'one context holds the unimplemented-command census for its whole batch, so every other ring waits it out',
        'src/venus/vkr.rs',
        """        let Ok(mut ctx) = ctx.try_lock() else {
            return Verdict::Busy;
        };
        // Only now, with the batch about to run""",
        """        let Ok(mut ctx) = ctx.try_lock() else {
            return Verdict::Busy;
        };
        let Ok(_census) = self.todo.seen.try_lock() else {
            return Verdict::Busy;
        };
        // Only now, with the batch about to run""",
        'a_context_blocked_in_the_driver_does_not_stop_another_contexts_ring',
    ),
    (
        'a fence the driver reports signalled at once still suspends the batch and round-trips the ring loop',
        'src/venus/context.rs',
        """        let probe = self.driver.wait_for_fences(args.device, fences, args.waitAll, 0);
        if probe != VkResult::VK_TIMEOUT || args.timeout == 0 {""",
        """        let probe = self.driver.wait_for_fences(args.device, fences, args.waitAll, 0);
        if args.timeout == 0 {""",
        'a_fence_the_driver_answers_at_once_never_suspends',
    ),
    (
        'a driver wait blocks inside the batch with the context and the resource table locked',
        'src/venus/context.rs',
        """        self.replaying || self.depth > 0
    }""",
        """        true
    }""",
        'a_ring_inside_a_driver_wait_does_not_hold_its_context',
    ),
    (
        'a fence a ring is waiting on inside the driver can be destroyed from the context stream',
        'src/venus/context.rs',
        """        if self.waited_fence(args.fence) {
            self.reject("destroyed a fence one of its streams is waiting on");
            return;
        }""",
        '',
        'a_fence_being_waited_on_cannot_be_destroyed_until_the_wait_is_over',
    ),
    (
        'a device with a wait in flight can be destroyed, cascading through the fence being waited on',
        'src/venus/context.rs',
        """        if self.waited_device(args.device) {
            self.reject("destroyed a device one of its streams is waiting on");
            return;
        }""",
        '',
        'a_device_with_a_wait_in_flight_cannot_be_destroyed',
    ),
    (
        "a destroy from a ring ignores the driver wait the context's own stream is suspended on",
        'src/venus/context.rs',
        """        self.own_wait.iter().chain(self.rings.values().filter_map(|e| e.wait.as_ref()))""",
        """        None.into_iter().chain(self.rings.values().filter_map(|e| e.wait.as_ref()))""",
        'a_destroy_is_refused_exactly_while_a_live_streams_wait_reads_it',
    ),
    (
        "a ring's driver wait is recorded on the context, and outlives the ring it was made on",
        'src/venus/context.rs',
        """            Some(id) => self.rings.get_mut(&id).map(|e| &mut e.wait),""",
        """            Some(_) => Some(&mut self.own_wait),""",
        'a_destroyed_ring_releases_what_its_wait_was_reading',
    ),
    (
        'a timed driver wait is made in one call, so a ring cannot be stopped until the GPU signals',
        'src/venus/driver.rs',
        """            let slice = left.min(DriverWait::SLICE);""",
        """            let slice = left;""",
        'a_context_with_a_ring_inside_an_endless_wait_is_destroyed_promptly',
    ),
    (
        "a ring the guest named 0 is quietly renamed instead of refused",
        'src/ids.rs',
        """    pub const fn new(raw: u64) -> Option<RingId> {
        match NonZeroU64::new(raw) {""",
        """    pub const fn new(raw: u64) -> Option<RingId> {
        match NonZeroU64::new(if raw == 0 { 1 } else { raw }) {""",
        'a_ring_named_zero_is_refused',
    ),
    (
        'the sample-count ceiling is ignored and the host maximum advertised anyway',
        'src/vrend/caps.rs',
        'Some(c) if max_samples > c => {',
        'Some(c) if max_samples > c && false => {',
        'sample_ceiling',
    ),
    (
        'a ceiling of zero is read as no ceiling, which is what it least means',
        'src/vrend/caps.rs',
        'Ok(n) => Some(n.max(1)),',
        'Ok(0) => None,\n        Ok(n) => Some(n),',
        'sample_ceiling',
    ),
    (
        'a row before the cursor resumes from it instead of restarting, and reads the wrong pages',
        'src/guest_mem.rs',
        """        if at < cursor.base || cursor.list != self.entries.as_ptr() {
            *cursor = Cursor::default();
        }""",
        """        if (at < cursor.base || cursor.list != self.entries.as_ptr()) && false {
            *cursor = Cursor::default();
        }""",
        'ascending_rows_carry_the_cursor_and_a_backwards_row_restarts',
    ),
    (
        'a cursor from another list is trusted on this one, and indexes past its end',
        'src/guest_mem.rs',
        '        if at < cursor.base || cursor.list != self.entries.as_ptr() {\n',
        '        if at < cursor.base {\n',
        'a_cursor_from_another_list_restarts_on_this_one',
    ),
    (
        'an AV1 super-resolution picture from the hardware is delivered, and puts wrong pixels on screen',
        'src/vrend/video/mod.rs',
        '            Some(buffer) if shape.misreturned() && matches!(engine, Engine::Host) => {\n'
        '                Delivery::Withheld(buffer)\n'
        '            }\n',
        '',
        'superres',
    ),
    (
        'a super-resolution frame stays on the hardware with its picture withheld, though dav1d could deliver it',
        'src/vrend/video/mod.rs',
        '        } else if history.lost || history.units.is_empty() {\n',
        '        } else if true {\n',
        'superres',
    ),
    (
        'a 10-bit stream switches to dav1d at super-resolution, whose pictures no target takes',
        'src/vrend/video/mod.rs',
        '        let refused = if desc.seq.bit_depth != 8 {\n',
        '        let refused = if false {\n',
        'superres',
    ),
    (
        'the frame that switches to dav1d is replayed into it and then decoded a second time',
        'src/vrend/video/mod.rs',
        '        units.pop();\n',
        '',
        'superres',
    ),
    (
        'a shown key frame does not restart the replay history, so a switch replays a stale GOP',
        'src/vrend/video/mod.rs',
        '        if shape.key() {\n            *self = History::default();\n        }\n',
        '',
        'the_replay_history',
    ),
    (
        'a host with no AV1 silicon sends AV1 to a hardware decoder it does not have',
        'src/vrend/video/mod.rs',
        '        let route = if hardware { Route::Hardware(History::default()) } else { Route::Software };\n',
        '        let route = Route::Hardware(History::default());\n',
        'without_av1_silicon',
    ),
    (
        'dav1d starts mid-stream without the history it was handed, and predicts from nothing',
        'src/vrend/video/mod.rs',
        '            *self = Software::start(handle, unit.replay.as_deref().unwrap_or_default());\n',
        '            *self = Software::start(handle, &[]);\n',
        'a_decoder_started_with_the_history',
    ),
    (
        'dav1d outputs only shown frames, leaving a hidden frame\'s surface unwritten',
        'src/vrend/video/software.rs',
        '        settings.set_output_invisible_frames(true);\n',
        '',
        'a_hidden_frame_gives_its_picture_too',
    ),
    (
        'an AV1 super-resolution descriptor is refused, dropping the held frame and its own decode',
        'src/vrend/video/mod.rs',
        '        let config = match av1::SeqParams::read(descriptor)',
        '        if desc.superres.is_some() {\n'
        '            return Err(Refusal::HostRefusedFrame);\n'
        '        }\n'
        '        let config = match av1::SeqParams::read(descriptor)',
        'superres',
    ),
    # The object table, walked through every operation sequence to a fixed depth: each of these
    # breaks one promise the walk checks at every step.
    (
        'a destroyed object\'s slot keeps its generation, so a stale key names the next occupant',
        'src/venus/objects.rs',
        '        e.generation += 1;\n',
        '',
        'every_sequence_keeps_every_promise',
    ),
    (
        'a destroy takes its children but not theirs, and the grandchildren leak',
        'src/venus/objects.rs',
        '                    walk.push((child, under(&o, device)));\n',
        '',
        'every_sequence_keeps_every_promise',
    ),
    (
        'a device is not carried down its tree, so its objects are destroyed on no device',
        'src/venus/objects.rs',
        'VkObjectType::VK_OBJECT_TYPE_DEVICE => Some(VkDevice::from_host(o.handle)),',
        'VkObjectType::VK_OBJECT_TYPE_DEVICE => inherited,',
        'every_sequence_keeps_every_promise',
    ),
    (
        'a refused create ghosts an id that still names a live object',
        'src/venus/objects.rs',
        """        if id.0 == 0 || self.get(id).is_some() {
            return;
        }
        self.slots.insert(id, Slot::Ghost);""",
        """        if id.0 == 0 {
            return;
        }
        self.slots.insert(id, Slot::Ghost);""",
        'every_sequence_keeps_every_promise',
    ),
    (
        'a create takes a dead key as its parent, and only a teardown sweep can reach it',
        'src/venus/objects.rs',
        '        let parent = owner.and_then(|o| self.key_of(o)).map(|k| k.0);',
        '        let parent = owner.and_then(|o| self.slots.get(&o)).and_then(Slot::key);',
        'every_sequence_keeps_every_promise',
    ),
    (
        'a fiction named under a dead owner is recorded parentless, and stands forever',
        'src/venus/objects.rs',
        '            Some(None) => return,',
        '            Some(None) => None,',
        'every_sequence_keeps_every_promise',
    ),
    # Fence retirement, under every interleaving loom can produce of a fence retired from another
    # thread against the owner letting go.
    (
        'the retirement thread is let go without waiting for it to drain',
        'src/fence.rs',
        '            let _ = t.join();',
        '            drop(t);',
        'loom:fence::loom_models',
    ),
    (
        'a queued fence does not wake the retirement thread',
        'src/fence.rs',
        '    g.jobs.push_back(job);\n    cv.notify_one();\n',
        '    g.jobs.push_back(job);\n',
        'loom:fence::loom_models',
    ),
    (
        'a fence is queued ahead of the ones before it, and its ring retires out of order',
        'src/fence.rs',
        '    g.jobs.push_back(job);',
        '    g.jobs.push_front(job);',
        'loom:fence::loom_models',
    ),
    # The ring thread's two condvar sleeps, under every interleaving loom can produce of what wakes
    # them. Loom's `wait_timeout` never times out, so a wake lost here is a hang in the model rather
    # than the half-second stall the timeout turns it into on hardware.
    (
        'a ring-seqno waiter checks the head before taking the lock it sleeps under',
        'src/venus/ring_thread.rs',
        """        let held = self.changed.lock().expect("the wait-ring lock is never poisoned");
        if ready() {
            return false;
        }
""",
        """        if ready() {
            return false;
        }
        let held = self.changed.lock().expect("the wait-ring lock is never poisoned");
""",
        'loom:venus::ring_thread::loom_models',
    ),
    (
        'a change to a ring-seqno waiter\'s predicate wakes nobody',
        'src/venus/ring_thread.rs',
        '        self.wake.notify_all();\n',
        '',
        'loom:venus::ring_thread::loom_models',
    ),
    (
        'a published virtqueue seqno does not wake the ring asleep on it',
        'src/venus/ring_thread.rs',
        """        state.vq_seqno = state.vq_seqno.max(seqno);
        self.wake.notify_one();""",
        """        state.vq_seqno = state.vq_seqno.max(seqno);""",
        'loom:venus::ring_thread::loom_models',
    ),
    (
        'a stop does not wake a ring asleep on its park',
        'src/venus/ring_thread.rs',
        """        started.store(false, Ordering::Release);
        self.wake.notify_one();""",
        """        started.store(false, Ordering::Release);""",
        'loom:venus::ring_thread::loom_models',
    ),
    # The decoder's reads and its arena charge, for every length a guest can put on the wire.
    (
        'a string length near the top of usize wraps when padded, and the read slices past it',
        'src/venus/cs.rs',
        '        let Some(advance) = n.checked_next_multiple_of(4) else {',
        '        let Some(advance) = Some(n.wrapping_add(3) & !3) else {',
        'kani:a_read_of_any_length_stays_inside_the_stream',
    ),
    (
        'a read advances by its payload and not its padding, and the next field starts misaligned',
        'src/venus/cs.rs',
        '        let b = self.peek_bytes(advance)?;\n        self.pos += advance;',
        '        let b = self.peek_bytes(advance)?;\n        self.pos += n;',
        'kani:a_read_of_any_length_stays_inside_the_stream',
    ),
    (
        'the arena charge wraps, and a huge request slips under the cap',
        'src/venus/cs.rs',
        '        let used = self.temp_used.get().saturating_add(bytes);',
        '        let used = self.temp_used.get().wrapping_add(bytes);',
        'kani:the_arena_charge_never_passes_its_cap',
    ),
    # The scatter-list walk, for every list of up to three entries and every range over it.
    (
        'a piece of a transfer ignores where in its entry it starts, and runs past the entry',
        'src/guest_mem.rs',
        '            let take = (e.len - skip).min(len - done);',
        '            let take = e.len.min(len - done);',
        'kani:every_piece_lies_inside_its_entry',
    ),
    (
        'a walk leaves its cursor one entry ahead of where it stopped',
        'src/guest_mem.rs',
        '        *cursor = Cursor { list: self.entries.as_ptr(), entry: i, base };',
        '        *cursor = Cursor { list: self.entries.as_ptr(), entry: i + 1, base };',
        'kani:a_resumed_walk_matches_a_fresh_one',
    ),
    # A ring layout is checked against the rules for every value of every field, both ways: a
    # parser that refuses too much fails the proof as surely as one that accepts too much.
    (
        'two ring regions that only touch are refused as overlapping',
        'src/venus/ring.rs',
        """    pub fn is_disjoint(&self, other: &Region) -> bool {
        self.begin >= other.end || self.end <= other.begin""",
        """    pub fn is_disjoint(&self, other: &Region) -> bool {
        self.begin >= other.end || self.end < other.begin""",
        'kani:parse_accepts_exactly_the_layouts_the_rules_allow',
    ),
    (
        'a ring buffer that is not a power of two is accepted, and its size is used as a mask',
        'src/venus/ring.rs',
        'if size == 0 || !size.is_power_of_two() || size > RING_BUFFER_MAX_SIZE {',
        'if size == 0 || size > RING_BUFFER_MAX_SIZE {',
        'kani:parse_accepts_exactly_the_layouts_the_rules_allow',
    ),
    # Asynchronous decode: a picture lands after END_FRAME returns, and every reader of its target
    # has to wait for it -- a fence, a control-queue read, a second decode into the same target.
    (
        'a share a venus context reads through leaves its surface unmarked',
        'src/venus/driver.rs',
        """            held.surface().mark_lent();
            Lent(held)""",
        """            Lent(held)""",
        'a_share_a_venus_context_reads_through_marks_its_surface_lent',
    ),
    (
        'a control-queue read takes a target without waiting for the picture decoding into it',
        'src/vrend/vrend.rs',
        """        let settled = texture.settle(&self.gl, super::video::pending::Wait::Block);""",
        """        let settled = texture.settle(&self.gl, super::video::pending::Wait::IfLanded);""",
        'a_control_queue_read_waits_for_the_picture_in_flight',
    ),
    (
        'a fence retires without waiting for the pictures decoding ahead of it',
        'src/vrend/waiter.rs',
        """        for picture in &job.owed.pictures {
            picture.wait();
        }""",
        """        let _ = &job.owed.pictures;""",
        'a_fence_waits_for_the_pictures_decoding_ahead_of_it',
    ),
    (
        'a fence wait returns before the GPU is past the fence',
        'src/vrend/waiter.rs',
        """        match gl.fence_wait(fence, SLICE_NS) {
            FenceWait::Signalled => return,""",
        """        match FenceWait::Signalled {
            FenceWait::Signalled => return,""",
        'vrend::waiter::tests::a',
    ),
    (
        'a second decode into a target drops the first picture instead of delivering it',
        'src/vrend/video/pending.rs',
        """            deliver(replaced, gl, name, planes);
            if let Some(began) = began {""",
        """            drop(replaced);
            if let Some(began) = began {""",
        'a_target_decoded_into_twice_takes_the_first_picture_before_the_second',
    ),
    (
        'a decode into a target whose picture is still decoding waits without being counted',
        'src/vrend/video/pending.rs',
        """                counters.replaces.record(began.elapsed());""",
        """                let _ = (&counters, began);""",
        'a_target_decoded_into_twice_takes_the_first_picture_before_the_second',
    ),
    (
        'a decode sent into a full queue waits without being counted',
        'src/vrend/video/pending.rs',
        """    unsettled.0.queue.record(began.elapsed());""",
        """    let _ = (unsettled, began);""",
        'a_send_into_a_full_queue_waits_and_is_counted',
    ),
    (
        'a timestamp query is never recorded',
        'src/vrend/context.rs',
        """            host.gl.query_timestamp(q.id);""",
        """            let _ = q.id;""",
        'a_timestamp_query_is_recorded_and_read_back_in_eight_bytes',
    ),
    (
        "a timer query's result is reported in four bytes",
        'src/vrend/context.rs',
        """        (gl.get_query_object_ui64v(id, GL_QUERY_RESULT), 8u32)""",
        """        (gl.get_query_object_ui64v(id, GL_QUERY_RESULT), 4u32)""",
        'a_timestamp_query_is_recorded_and_read_back_in_eight_bytes',
    ),
    (
        'a CUSTOM buffer is held without a charge',
        'src/vrend/resource.rs',
        """        let charge = budget.charge("CUSTOM buffer", size as u64);""",
        """        let charge = budget.charge("CUSTOM buffer", 0);""",
        'a_custom_buffer_is_charged_for_the_bytes_it_holds',
    ),
    (
        "a frame's bitstream grows without its charge following",
        'src/vrend/video/mod.rs',
        """            self.charge = Some(budget.charge("video bitstream", held));""",
        """            let _ = (budget, held);""",
        'a_frames_bitstream_is_charged_for_as_long_as_it_is_held',
    ),
    (
        'a refused vkQueueSubmit still records what it promised',
        'src/venus/driver.rs',
        """        if ret == VkResult::VK_SUCCESS {
            self.note_submit(submits.get(), fence);""",
        """        if ret == ret {
            self.note_submit(submits.get(), fence);""",
        'a_refused_submit_leaves_no_fence_pending_and_no_signal_requested',
    ),
    (
        'a classic replay feeds past the command that poisoned its context',
        'src/vrend/context.rs',
        """            if self.replay_one(host, sub, &chunks).is_err() {""",
        """            if self.replay_one(host, sub, &chunks).is_err() && seq.0 == u64::MAX {""",
        'a_replay_that_poisons_its_context_stops_and_says_so',
    ),
    (
        'a command a replay drops leaves its GL error for the guest\'s first live batch',
        'src/vrend/context.rs',
        """                host.gl.drain_errors();
                continue;""",
        """                continue;""",
        'a_command_a_replay_drops_does_not_refuse_the_next_live_submit',
    ),
    (
        'a command a replay drops leaves its fault stuck on the context',
        'src/vrend/context.rs',
        """                host.gl.drain_errors();
                continue;""",
        """                self.fault = Some(f);
                continue;""",
        'a_command_a_replay_drops_does_not_refuse_the_next_live_submit',
    ),
    (
        'a host offering implicit multisampling refuses the surface that takes the offer',
        'src/vrend/transfer.rs',
        """    if !features.has(Feature::implicit_msaa) {
        return Err(NotMultisampled::Feature(Feature::implicit_msaa));""",
        """    if !features.has(Feature::implicit_msaa) || samples > 0 {
        return Err(NotMultisampled::Feature(Feature::implicit_msaa));""",
        'a_multisampled_surface_renders_into_its_texture_where_the_host_offers_it',
    ),
    (
        'an evaluation shader without a control shader is refused, not given one',
        'src/vrend/context/select.rs',
        """        if self.sub().injects_tcs() {
            self.select_passthrough(host, cmd, vertices_per_patch)""",
        """        if self.sub().injects_tcs() {
            Err(Fault::Unimplemented { cmd, what: "an injected tessellation control shader" })""",
        'an_evaluation_shader_without_a_control_shader_draws_its_patches',
    ),
    (
        'an injected control shader writes levels the guest never set',
        'src/vrend/context.rs',
        """                    sub.tess_factors = factors;""",
        """                    let _ = factors;""",
        'an_evaluation_shader_without_a_control_shader_draws_its_patches',
    ),
    (
        'a desktop context the caller did not ask for is taken anyway',
        'src/vrend/vrend.rs',
        """    if asked != HostGl::Desktop {
        return Err(UnservedGl::NotAskedFor);""",
        """    if asked != HostGl::Desktop && asked == HostGl::Desktop {
        return Err(UnservedGl::NotAskedFor);""",
        'a_desktop_context_is_taken_only_when_asked_for',
    ),
    (
        'a GLES cursor stored as RGBA is handed over unswapped',
        'src/vrend/transfer.rs',
        """        if res.entry(formats).is_some_and(|e| e.stores_bgra_as_rgba()) {
            swizzle_bgra(""",
        """        if res.entry(formats).is_some_and(|e| e.stores_bgra_as_rgba()) && false {
            swizzle_bgra(""",
        'a_bgra_cursor_reads_back_in_the_order_it_was_written_on_either_flavour',
    ),
    (
        'a desktop host refuses pipeline statistics queries',
        'src/vrend/context.rs',
        """            need(Feature::pipeline_statistics_query)?;""",
        """            return Err(Fault::Unimplemented { cmd, what: "that query type" });""",
        'pipeline_statistics_and_overflow_queries_count_where_the_host_has_them',
    ),
    (
        'every pipeline statistic counts the first one',
        'src/vrend/context.rs',
        """                .get(index as usize)""",
        """                .get(0)""",
        'pipeline_statistics_and_overflow_queries_count_where_the_host_has_them',
    ),
    (
        'a query on a vertex stream counts on stream 0',
        'src/vrend/context.rs',
        """        if q.index > 0 {
            host.gl.begin_query_indexed(q.gl_type, q.index, q.id);""",
        """        if q.index > 0 && q.index == 0 {
            host.gl.begin_query_indexed(q.gl_type, q.index, q.id);""",
        'pipeline_statistics_and_overflow_queries_count_where_the_host_has_them',
    ),
    (
        'a query result is never written into the query buffer',
        'src/vrend/context.rs',
        """        gl.query_object_into_buffer(id, pname, width, offset);""",
        """        let _ = (id, pname, width, offset);""",
        'a_query_result_lands_in_a_buffer_where_the_host_has_query_buffers',
    ),
    (
        'a query buffer asked for availability gets the result',
        'src/vrend/context.rs',
        """            (-1, _) => GL_QUERY_RESULT_AVAILABLE,""",
        """            (-1, _) => GL_QUERY_RESULT,""",
        'a_query_result_lands_in_a_buffer_where_the_host_has_query_buffers',
    ),
    (
        'an indirect draw never binds its count buffer',
        'src/vrend/context/draw.rs',
        """                gl.bind_buffer(GL_PARAMETER_BUFFER, count_buffer);""",
        """                gl.bind_buffer(GL_PARAMETER_BUFFER, None);""",
        'an_indirect_draw_count_is_read_from_its_buffer_where_the_host_has_one',
    ),
    (
        'a shader-buffer barrier is read from the bit the port used to give it',
        'src/vrend/context.rs',
        """        (1 << 1, shader_buffer),""",
        """        (1 << 11, shader_buffer),""",
        'each_gallium_barrier_bit_asks_for_its_own_gl_barrier',
    ),
    (
        'the separate-shader-objects extension is asked for under the C\'s misspelling',
        'src/vrend/features.rs',
        """["GL_ARB_separate_shader_objects"]""",
        """["GL_ARB_seperate_shader_objects"]""",
        'every_extension_the_table_names_is_in_the_registry',
    ),
    (
        'a desktop host copies unrenderable formats only through guest pages',
        'src/vrend/context/blit.rs',
        """            if host.features.api().is_gles() {
                let (guest, ctx) = (host.guest, host.ctx);""",
        """            if host.features.api().is_gles() || !host.features.api().is_gles() {
                let (guest, ctx) = (host.guest, host.ctx);""",
        'a_copy_between_unrenderable_textures_lands_on_desktop_gl',
    ),
    (
        'a GLES host copies between unrenderable formats as desktop GL does',
        'src/vrend/context/blit.rs',
        """            if host.features.api().is_gles() {
                let (guest, ctx) = (host.guest, host.ctx);""",
        """            if host.features.api().is_gles() && false {
                let (guest, ctx) = (host.guest, host.ctx);""",
        'a_copy_between_unrenderable_textures_samples_as_the_copied_image',
    ),
    (
        'the GLES copy fallback reads the source with the destination level\'s pitch',
        'src/vrend/transfer.rs',
        """        stride: u32::try_from(from.stride).map_err(|_| Error::IovOutOfRange)?,""",
        """        stride: 0,""",
        'a_copy_between_unrenderable_textures_samples_as_the_copied_image',
    ),
    (
        'the GLES copy fallback takes the source box from the start of its row',
        'src/vrend/transfer.rs',
        """            + desc.blocks_wide(region.x as u32) as u64 * desc.block_bytes() as u64;""",
        """            + 0;""",
        'a_copy_between_unrenderable_textures_samples_as_the_copied_image',
    ),
    (
        'the GLES copy fallback leaves the destination\'s pages as they were',
        'src/vrend/transfer.rs',
        """                || !dst_pages.copy_in_from(&mut written, to_at, &bytes)""",
        """                || !dst_pages.copy_in_from(&mut written, to_at, &bytes[..0])""",
        'a_copy_between_unrenderable_textures_samples_as_the_copied_image',
    ),
    (
        'the copy fallback reads its box from the start of the source level',
        'src/vrend/transfer.rs',
        """    read_box_whole_level(gl, entry, src, src_level, src_box, src_box.y, &l, &mut data)?;""",
        """    let at_start = Box3 { x: 0, y: 0, ..*src_box };
    read_box_whole_level(gl, entry, src, src_level, &at_start, 0, &l, &mut data)?;""",
        'a_copy_between_unrenderable_textures_lands_on_desktop_gl',
    ),
    (
        'a desktop host reads a cursor only through a framebuffer',
        'src/vrend/transfer.rs',
        """    if features.api().is_gles() {
        read_layer(gl, features, formats, res, 0, 0, 0, 0,""",
        """    if !features.api().is_gles() || features.api().is_gles() {
        read_layer(gl, features, formats, res, 0, 0, 0, 0,""",
        'a_cursor_no_framebuffer_reads_reads_back_on_desktop_gl',
    ),
    (
        'a framebuffer read runs at whatever pack state the context was left in',
        'src/vrend/gl.rs',
        """        self.pack_tight();
        // Desktop GL clamps a read""",
        """        // Desktop GL clamps a read""",
        'an_unaligned_cursor_reads_back_on_a_fresh_renderer_on_either_flavour',
    ),
    (
        'a desktop host translates a separable stage as a linked one',
        'src/vrend/shader/glsl/decl.rs',
        """                ctx.separable_program = data != 0;""",
        """                ctx.separable_program = data != 0 && data == 0;""",
        'separable_stages_draw_as_a_pipeline_on_desktop_gl_and_linked_whole_on_gles',
    ),
    (
        'a pipeline is linked but never bound',
        'src/vrend/context/draw.rs',
        """            ProgramObject::Pipeline { pipeline, .. } => gl.use_pipeline(bound, *pipeline),""",
        """            ProgramObject::Pipeline { .. } => gl.use_program(bound, None),""",
        'separable_stages_draw_as_a_pipeline_on_desktop_gl_and_linked_whole_on_gles',
    ),
    (
        'a pipeline stage\'s uniforms land in whichever program was active',
        'src/vrend/context/draw.rs',
        """            if let Some(program) = sub.stage_program(at, stage) {
                sub.program_at(at).object.activate(gl, program);
            }
""",
        """""",
        'separable_stages_draw_as_a_pipeline_on_desktop_gl_and_linked_whole_on_gles',
    ),
    (
        'a pipeline is validated before its draw writes the sampler units it uses',
        'src/vrend/context/draw.rs',
        """        self.draw_bind_objects(host, at, new_program);
        // A pipeline is validated once its stages' sampler units are the ones it draws with. The
        // C reports one the driver refuses and carries on; the report poisons the context, which
        // is what refusing it here does.
        if let ProgramObject::Pipeline { pipeline, validated: false } = object {
            if !gl.validate_program_pipeline(pipeline) {
                return Err(Fault::Shader { cmd, what: "a program pipeline the driver refused" });
            }
            let prog = self.sub_mut().program_at_mut(at);
            prog.object = ProgramObject::Pipeline { pipeline, validated: true };
        }
""",
        """        if let ProgramObject::Pipeline { pipeline, validated: false } = object {
            if !gl.validate_program_pipeline(pipeline) {
                return Err(Fault::Shader { cmd, what: "a program pipeline the driver refused" });
            }
            let prog = self.sub_mut().program_at_mut(at);
            prog.object = ProgramObject::Pipeline { pipeline, validated: true };
        }
        self.draw_bind_objects(host, at, new_program);
""",
        'separable_stages_sampling_different_targets_draw_as_one_pipeline',
    ),
    (
        'a pipeline found again keeps the block bindings another pipeline gave its stages',
        'src/vrend/context/draw.rs',
        """                if let Some(walk) = stale {""",
        """                if let Some(walk) = stale.filter(|_| false) {""",
        'a_pipeline_found_again_renumbers_its_shared_stages_blocks',
    ),
    (
        'a VMM that does not ask for GLES is given GLES anyway',
        'src/ffi.rs',
        """        host_gl: if flags & abi::USE_GLES != 0 { HostGl::Gles } else { HostGl::Desktop },""",
        """        host_gl: HostGl::Gles,""",
        'the_init_flags_decode_into_the_configuration_they_name',
    ),
    (
        'a VMM that asks for GLES is given desktop GL anyway',
        'src/ffi.rs',
        """        host_gl: if flags & abi::USE_GLES != 0 { HostGl::Gles } else { HostGl::Desktop },""",
        """        host_gl: HostGl::Desktop,""",
        'the_init_flags_decode_into_the_configuration_they_name',
    ),
    (
        'a display of ours makes GLES contexts when desktop GL is asked for',
        'src/vrend/vrend.rs',
        """                    HostGl::Desktop => (Flavour::Gl, DESKTOP_VERSIONS),""",
        """                    HostGl::Desktop => (Flavour::Gles, GLES_VERSIONS),""",
        'the_host_gl_is_the_one_asked_for',
    ),
    (
        'the proc table carries no desktop entry point',
        'gl-gen/gen.py',
        """    desktop, _ = gl.requirements('gl', GL_FEATURES, GL_EXTENSIONS)""",
        """    desktop = []""",
        'the_host_gl_is_the_one_asked_for',
    ),
    (
        'the packed-float vertex format is advertised under a number the wire does not give it',
        'src/vrend/caps.rs',
        """        .find(|f| f.name() == "R11G11B10_FLOAT")""",
        """        .find(|f| f.wire() == 135)""",
        'the_packed_float_vertex_format_is_the_wire_one',
    ),
    (
        'a desktop host tells the guest it is GLES',
        'src/vrend/caps.rs',
        """        capbit(api.is_gles(), cap::HOST_IS_GLES);""",
        """        capbit(true, cap::HOST_IS_GLES);""",
        'the_host_gl_is_the_one_asked_for',
    ),
    (
        'a desktop context resolves entry points by their GLES spellings',
        'src/vrend/gl.rs',
        """                    Api::Gl(_) => &[$(Procs::$gl),*],""",
        """                    Api::Gl(_) => &[$(Procs::$es),*],""",
        'a_resolver_asks_only_for_its_apis_spelling',
    ),
    (
        'a desktop context is judged by the GLES column',
        'src/vrend/features.rs',
        """                    $((Feature::$name, Api::Gl(_)) => $gl,""",
        """                    $((Feature::$name, Api::Gl(_)) => $gles,""",
        'a_desktop_version_reads_the_desktop_column',
    ),
    (
        'a desktop host probes the GLES format spellings',
        'src/vrend/formats.rs',
        """                When::Gles => gles,
                When::DesktopGl => !gles,""",
        """                When::Gles => true,
                When::DesktopGl => false,""",
        'the_host_gl_is_the_one_asked_for',
    ),
    (
        'a desktop host answers readonly for a texture no framebuffer reads',
        'src/vrend/transfer.rs',
        """            if !can_readpixels && !desktop {""",
        """            if !can_readpixels {""",
        'a_texture_no_framebuffer_reads_reads_back_on_either_flavour',
    ),
    (
        'a desktop host swaps BGRA on transfer as GLES does',
        'src/vrend/formats.rs',
        """        super::resource::is_bgra(self.gl.format) && self.gl.glformat != GL_BGRA""",
        """        super::resource::is_bgra(self.gl.format)""",
        'the_host_gl_is_the_one_asked_for',
    ),
    (
        'a desktop host swaps a blue-first vertex element in the shader as well',
        'src/vrend/context.rs',
        """        if blue_first && api.is_gles() {""",
        """        if blue_first {""",
        'a_blue_first_vertex_element_is_swapped_once',
    ),
    (
        'a bounded read is chosen by whether its name resolves',
        'src/vrend/gl.rs',
        """        if features.has(Feature::arb_robustness) {
            RobustReads::Arb""",
        """        if features.has(Feature::arb_robustness) || api.gl_at_least(45) {
            RobustReads::Arb""",
        'the_bounded_reads_are_the_ones_the_driver_advertises',
    ),
    (
        'a GLES host with viewport arrays has no spelling for a second viewport',
        'src/vrend/gl.rs',
        """        = gles [try_glViewportIndexedfOES], gl [try_glViewportIndexedf]);""",
        """        = gles [], gl [try_glViewportIndexedf]);""",
        'a_resolver_asks_only_for_its_apis_spelling',
    ),
    (
        'a desktop host is handed GLSL ES',
        'src/vrend/shader/glsl/header.rs',
        """    if ctx.cfg.is_gles() {
        emit_header_es(ctx);""",
        """    if true {
        emit_header_es(ctx);""",
        'every_corpus_shader_translates_to_the_c_desktop_glsl',
    ),
    (
        'a desktop sampler is declared with a GLES precision',
        'src/vrend/shader/glsl/header.rs',
        """    let precision = if ctx.cfg.is_gles() { "highp" } else { "" };""",
        """    let precision = "highp";""",
        'every_corpus_shader_translates_to_the_c_desktop_glsl',
    ),
    (
        'a desktop host leaves the logic op at copy',
        'src/vrend/context/draw.rs',
        """                gl.logic_op(logic_op(new_state.logicop_func));""",
        """                gl.logic_op(logic_op(LogicOp::Copy));""",
        'a_logic_op_replaces_the_fragment_on_either_flavour',
    ),
    (
        'an alpha-only target takes the shader red on desktop GL',
        'src/vrend/context/select.rs',
        """                    key.fs.cbufs_are_a8_bitmask |= 1 << i;""",
        """                    key.fs.cbufs_are_a8_bitmask |= 0;""",
        'an_alpha_only_target_takes_the_shaders_alpha_on_desktop_gl',
    ),
    (
        'an alpha-only target clears its red to the clear red',
        'src/vrend/context.rs',
        """            color = [color[3], 0.0, 0.0, 0.0];""",
        """            color = [color[0], 0.0, 0.0, 0.0];""",
        'an_alpha_only_target_clears_to_the_clears_alpha_on_desktop_gl',
    ),
    (
        'an alpha-only target blends its red with the colour factors',
        'src/vrend/context/draw.rs',
        """                        src: conv_a8_blend(eq.alpha.src),""",
        """                        src: conv_a8_blend(eq.rgb.src),""",
        'an_emulated_alpha_target_blends_its_alpha_in_red',
    ),
    (
        'a desktop host drops the default tessellation levels',
        'src/vrend/context.rs',
        """                    host.gl.patch_parameter_fv(GL_PATCH_DEFAULT_OUTER_LEVEL, &outer);""",
        """                    let _ = outer;""",
        'an_evaluation_shader_without_a_control_shader_draws_its_patches',
    ),
    (
        'a desktop host stores a 1D texture as 2D',
        'src/vrend/resource.rs',
        """        TextureTarget::Texture1d if !gles => GL_TEXTURE_1D,""",
        """        TextureTarget::Texture1d if false => GL_TEXTURE_1D,""",
        'a_1d_texture_is_one_on_desktop_gl_and_round_trips_on_both',
    ),
    (
        'a desktop host stores every rectangle as 2D',
        'src/vrend/resource.rs',
        """        TextureTarget::Rect if rect_ok => GL_TEXTURE_RECTANGLE,""",
        """        TextureTarget::Rect if false => GL_TEXTURE_RECTANGLE,""",
        'desktop_gl_stores_1d_as_1d_and_rect_as_rect_where_the_format_can_be_one',
    ),
    (
        'every layer of a 1D array is uploaded into its first',
        'src/vrend/transfer.rs',
        """            let (y, h) = if target == GL_TEXTURE_1D_ARRAY { (z, d) } else { (y, h) };""",
        """            let (y, h) = (y, h);""",
        'a_1d_texture_is_one_on_desktop_gl_and_round_trips_on_both',
    ),
    (
        'a desktop blitter samples through GLSL ES',
        'src/vrend/blitter.rs',
        """        let dialect = if gl.api().is_gles() { Dialect::Es } else { Dialect::Core };""",
        """        let dialect = Dialect::Es;""",
        'a_1d_source_blits_through_the_sampler_its_flavour_declares',
    ),
    (
        'a sampler view binds whatever kind of resource its handle names now',
        'src/vrend/context.rs',
        """                (Storage::Texture(_), Span::Levels { .. }) => {}""",
        """                (Storage::Texture(_) | Storage::Buffer { .. }, Span::Levels { .. }) => {}""",
        'a_texture_view_is_refused_once_its_handle_names_a_buffer',
    ),
    (
        "a rebuild binds an orphaned shader without creating it",
        'src/vrend/context/select.rs',
        """                    (o.created.seq, Cow::Borrowed(o.created.chunks.as_slice())),""",
        """                    (o.created.seq, Cow::Owned(vec![Vec::new()])),""",
        'a_shader_destroyed_while_bound_is_rebuilt_bound_and_destroyed',
    ),
    (
        "a rebuild leaves an orphaned shader's handle live",
        'src/vrend/context/select.rs',
        """                    (o.destroyed_at, Cow::Owned(vec![destroy])),""",
        """                    (o.destroyed_at, Cow::Owned(vec![Vec::new()])),""",
        'a_shader_destroyed_while_bound_is_rebuilt_bound_and_destroyed',
    ),
    (
        'a replacing create is retained before the object it replaces goes',
        'src/vrend/context.rs',
        """        self.destroy_object(host, handle);
        let at = Retained::new(self.seq.advance(), wire);""",
        """        let at = Retained::new(self.seq.advance(), wire);
        self.destroy_object(host, handle);""",
        'a_create_over_a_bound_shaders_handle_keeps_both_across_a_rebuild',
    ),
    (
        'an image over some of an array\'s layers binds all of them',
        'src/vrend/context/draw.rs',
        """        (true, layers) => ImageLayers::Range { first, layers },""",
        """        (true, _) => ImageLayers::Whole,""",
        'an_image_reads_its_layers_as_the_c_does',
    ),
    (
        'a compute dispatch is refused as unimplemented again',
        'src/vrend/context.rs',
        """                self.launch_grid(host, grid, indirect, indirect_offset)""",
        """                Err(Fault::Unimplemented { cmd: kind, what: "compute dispatch" })""",
        'a_dispatch_runs_its_grid_from_the_wire_or_its_buffer',
    ),
    (
        'an indirect dispatch runs the grid on the wire instead of its buffer\'s',
        'src/vrend/context/draw.rs',
        """            Some(_) => gl.dispatch_compute_indirect(indirect_offset),""",
        """            Some(_) => gl.dispatch_compute(grid),""",
        'a_dispatch_runs_its_grid_from_the_wire_or_its_buffer',
    ),
    (
        'an indirect dispatch reads its grid from the start of its buffer',
        'src/vrend/context/draw.rs',
        """            Some(_) => gl.dispatch_compute_indirect(indirect_offset),""",
        """            Some(_) => gl.dispatch_compute_indirect(0),""",
        'a_dispatch_runs_its_grid_from_the_wire_or_its_buffer',
    ),
    (
        'a query result the pages refused is marked as delivered',
        'src/vrend/context.rs',
        """    let delivered = guest.pages(ctx, resource).is_some_and(|pages| pages.copy_in(0, &state));""",
        """    let delivered = guest.pages(ctx, resource).is_some_and(|pages| pages.copy_in(0, &state) || true);""",
        'a_query_result_the_pages_cannot_take_stays_owed',
    ),
    (
        'a transfer on a context the resource is not attached to answers no resource',
        'src/renderer.rs',
        """                    return Err(Error::NotAttached);""",
        """                    return Err(Error::NoResource);""",
        'a_transfer_on_a_context_the_resource_is_not_attached_to_is_refused_as_such',
    ),
    (
        'a host with only the EGL image storage entry point is taken to bind images',
        'src/vrend/features.rs',
        """    pub fn binds_egl_images(&self) -> bool {
        self.has(Feature::egl_image)""",
        """    pub fn binds_egl_images(&self) -> bool {
        self.has(Feature::egl_image) || self.has(Feature::egl_image_storage)""",
        'only_the_oes_entry_point_makes_a_host_bind_egl_images',
    ),
    (
        'a dma-buf surface label wraps to zero',
        'src/dmabuf.rs',
        """    n.checked_add(1).unwrap_or(1)""",
        """    n.wrapping_add(1)""",
        'a_surface_label_skips_zero_when_the_count_wraps',
    ),
    (
        'a Vulkan handle can be built from any number by its constructor',
        'venus-gen/templates/types.rs',
        """pub struct ${ty.name}(u64);""",
        """pub struct ${ty.name}(pub u64);""",
        'doc:venus::cs::Handle',
    ),
    (
        'a Vulkan handle can be made from a number without unsafe',
        'venus-gen/templates/types.rs',
        """    pub unsafe fn from_raw(raw: u64) -> Self {""",
        """    pub fn from_raw(raw: u64) -> Self {""",
        'doc:venus::cs::Handle',
    ),
    (
        'a host handle can be built from any number, and from_host makes it a handle',
        'src/venus/cs.rs',
        """pub struct HostHandle(u64);""",
        """pub struct HostHandle(pub u64);""",
        'doc:venus::cs::Handle',
    ),
    (
        'temporaries may be declared past the register space',
        'src/vrend/shader/glsl/decl.rs',
        """    if ctx.temps_declared > TEMP_REGISTERS {""",
        """    if false && ctx.temps_declared > TEMP_REGISTERS {""",
        'temporaries_past_the_register_space_are_refused',
    ),
    (
        "a coherent image store marks the image its coordinate register names",
        'src/vrend/shader/glsl/tex.rs',
        'if !set_image_qualifier(ctx, inst, Some(image), dst_reg.indirect) {',
        'if !set_image_qualifier(ctx, inst, ImageSlot::new(inst.src[0].index), inst.src[0].indirect) {',
        'a_coherent_image_store_marks_the_image_it_writes',
    ),
    (
        'a buffer operand is taken for the sampler a texture instruction samples through',
        'src/vrend/shader/glsl/inst.rs',
        """            Binding::Sampler(s) => Some(s),
            _ => None,""",
        """            Binding::Sampler(s) => Some(s),
            Binding::Other(i) => SamplerSlot::new(i),
            _ => None,""",
        'a_texture_instruction_whose_sampler_is_a_buffer_is_refused',
    ),
    (
        'a TXQS whose sampler is not a sampler is emitted',
        'src/vrend/shader/glsl/tex.rs',
        """    ctx.shader_req_bits |= super::req::TXQS;
    if set_texture_reqs(ctx, inst, binding).is_none() {
        ctx.bufs.set_error();
        return;
    }""",
        """    ctx.shader_req_bits |= super::req::TXQS;
    let _ = set_texture_reqs(ctx, inst, binding);""",
        'a_texture_instruction_whose_sampler_is_a_buffer_is_refused',
    ),
    (
        'a LODQ whose sampler is not a sampler is emitted',
        'src/vrend/shader/glsl/tex.rs',
        """    ctx.shader_req_bits |= super::req::LODQ;
    if set_texture_reqs(ctx, inst, sinfo.binding).is_none() {
        ctx.bufs.set_error();
        return;
    }""",
        """    ctx.shader_req_bits |= super::req::LODQ;
    let _ = set_texture_reqs(ctx, inst, sinfo.binding);""",
        'a_texture_instruction_whose_sampler_is_a_buffer_is_refused',
    ),
    (
        'an export of a handle naming nothing is answered as not exportable',
        'src/renderer.rs',
        """        self.with_resource(handle, |_| ()).ok_or(Error::NoResource)?;""",
        """        self.with_resource(handle, |_| ());""",
        'an_export_of_nothing_is_not_an_export_refused',
    ),
    (
        'an shm mapping may run past the end of its descriptor',
        'src/guest_mem.rs',
        """        if len as u64 > held {""",
        """        if len as u64 > held && held == u64::MAX {""",
        'a_mapping_longer_than_its_descriptor_is_refused',
    ),
    (
        "a refused typing drops the exporter's storage",
        'src/vrend/resource.rs',
        """        adopted.map_err(|why| (self, why))""",
        """        adopted.map_err(|why| (Untyped { storage: None }, why))""",
        'a_refused_set_type_keeps_the_exporters_storage',
    ),
    (
        'a refused vkQueueSubmit2 still records what it promised',
        'src/venus/driver.rs',
        """        if ret == VkResult::VK_SUCCESS {
            self.note_submit2(submits.get(), fence);""",
        """        if ret == ret {
            self.note_submit2(submits.get(), fence);""",
        'a_refused_submit_leaves_no_fence_pending_and_no_signal_requested',
    ),
    (
        'a stats window drops the decoder counters when the next one starts',
        'src/vrend/tally.rs',
        """        *a = Armed::new(a.every, a.settles.clone());""",
        """        *a = Armed::new(a.every, Default::default());""",
        'the_decoder_counters_outlive_the_window',
    ),
    (
        'the VideoToolbox warm-up is never started',
        'src/videotoolbox.rs',
        """    if !support.decodes(Codec::Vp9) {
        return None;
    }""",
        """    if support.decodes(Codec::Vp9) {
        return None;
    }""",
        'after_the_warm_up_a_first_session_of_another_codec_is_cheap',
    ),
    (
        'the VideoToolbox warm-up thread builds no session',
        'src/videotoolbox.rs',
        """        if let Err(status) = Session::create(key) {""",
        """        if let Err(status) = Err::<(), _>(Status(0)).map(|()| drop(key)) {""",
        'after_the_warm_up_a_first_session_of_another_codec_is_cheap',
    ),
    (
        'pages the host mints for a context are credited as soon as they are charged',
        'src/renderer.rs',
        """            Ok((fd, map)) => Ok(HostShm { fd, map: Arc::new(map.charged(charge)) }),""",
        """            Ok((fd, map)) => Ok(HostShm { fd, map: Arc::new({ drop(charge); map }) }),""",
        'host_minted_pages_are_charged_to_the_context_that_asked',
    ),
    (
        'a refused host-minted blob stops its context',
        'src/venus/context.rs',
        """            .inspect_err(|refused| account.report_answered_refusal(*refused))""",
        """            .inspect_err(|refused| {
                account.report_answered_refusal(*refused);
                self.fatal.store(true, Ordering::Release);
            })""",
        'host_minted_pages_are_charged_to_the_context_that_asked',
    ),
    (
        'a classic context mints host pages for nothing',
        'src/renderer.rs',
        """                        Ok(crate::budget::Classic::open(&self.budget).charge("host shm", size))""",
        """                        Ok(crate::budget::Classic::open(&crate::budget::Budget::with_cap(None, false)).charge("host shm", size))""",
        'only_a_blob_that_asks_the_host_for_memory_is_given_any',
    ),
    (
        'vkCmdPushDescriptorSet2 is recorded as nothing',
        'src/venus/context.rs',
        """        let done = self.driver.cmd_push_descriptor_set2(args.commandBuffer, info);""",
        """        let done = Some(()).filter(|_| info.get().set == u32::MAX);""",
        'the_maintenance6_binding_commands_hand_the_driver_the_guests_struct',
    ),
    (
        'vkCmdBindDescriptorSets2 goes to the plain entry point',
        'src/venus/driver.rs',
        """        let f = self.recorder(cb)?.try_vkCmdBindDescriptorSets2()?;
        // SAFETY: as `cmd_push_descriptor_set2`.
        unsafe { f(cb, info.get()) };""",
        """        let i = info.get();
        // SAFETY: sabotage -- the same arrays through the older command, stage flags dropped.
        unsafe {
            (self.recorder(cb)?.vkCmdBindDescriptorSets())(
                cb,
                VkPipelineBindPoint::VK_PIPELINE_BIND_POINT_GRAPHICS,
                i.layout,
                i.firstSet,
                i.descriptorSetCount,
                i.pDescriptorSets,
                i.dynamicOffsetCount,
                i.pDynamicOffsets,
            )
        };""",
        'the_maintenance6_binding_commands_hand_the_driver_the_guests_struct',
    ),
    (
        'a host with no multisample arrays still counts as multisampling',
        'src/vrend/features.rs',
        """            && self.has(Feature::storage_multisample)
            && self.has(Feature::storage_multisample_2d_array)""",
        """            && self.has(Feature::storage_multisample)""",
        'multisample_textures_need_the_array_form_too',
    ),
    (
        'the sample count ignores the missing array form',
        'src/vrend/caps.rs',
        """        if features.multisample_textures() {
            c.v1.max_samples =""",
        """        if has(Feature::storage_multisample) {
            c.v1.max_samples =""",
        'without_multisample_arrays_no_multisampling_is_advertised',
    ),
    (
        'the format table multisamples without the array form',
        'src/vrend/formats.rs',
        """        if features.multisample_textures() {""",
        """        if features.has(Feature::multisample) && features.has(Feature::storage_multisample) {""",
        'without_multisample_arrays_no_multisampling_is_advertised',
    ),
    (
        'the planar YUV stand-ins are probed for multisampling like any row',
        'src/vrend/formats.rs',
        """                e.can_multisample = !stand_in
                    && e.can_texture_storage""",
        """                e.can_multisample = (stand_in || true)
                    && e.can_texture_storage""",
        'no_planar_yuv_format_is_advertised_multisampled_on_either_flavour',
    ),
    (
        'a colour output written alone declares no logic-op output',
        'src/vrend/shader/glsl/header.rs',
        """            if output.name == Semantic::Color && ctx.key.fs.logicop_func.is_some() {""",
        """            if false && output.name == Semantic::Color && ctx.key.fs.logicop_func.is_some() {""",
        'a_logic_op_applies_to_a_shader_that_writes_one_colour_buffer',
    ),
    (
        'the logic op numbers colour buffers by output position',
        'src/vrend/shader/glsl/exit.rs',
        """        ctx.outputs.iter().filter(|o| o.name == Semantic::Color).map(|o| o.sid as usize).collect();""",
        """        (0..ctx.outputs.len()).collect();""",
        'a_logic_op_applies_to_a_shader_that_writes_one_colour_buffer',
    ),
    (
        'a texture may ask for one level past its mip chain, as in the C',
        'src/vrend/resource.rs',
        """    deepest.ilog2().min(flat.ilog2() + 1)""",
        """    let _ = deepest;
    flat.ilog2() + 1""",
        'the_checks_refuse_what_the_c_refuses',
    ),
    (
        'a transfer box is sized in 32 bits before it is checked',
        'src/vrend/transfer.rs',
        """    desc.size_2d(desc.stride(extent(r.width)), extent(r.height))""",
        """    Some((desc.blocks_wide(extent(r.width)) * desc.block_bytes()) as u64 * desc.blocks_high(extent(r.height)) as u64)""",
        'a_transfer_box_too_large_to_size_is_refused',
    ),
    (
        'a transfer box too large for a u64 is sized anyway',
        'src/vrend/transfer.rs',
        """        .and_then(|layer| layer.checked_mul(extent(r.depth) as u64))""",
        """        .map(|layer| layer * extent(r.depth) as u64)""",
        'kani:a_box_of_any_size_is_sized_without_overflow',
    ),
    (
        'the texture-target table panics on a buffer',
        'src/vrend/resource.rs',
        """        TextureTarget::Buffer => return None,""",
        """        TextureTarget::Buffer => unreachable!("a buffer has no texture target"),""",
        'a_buffer_sampler_view_is_made_over_a_buffer_and_refused_over_a_texture',
    ),
    (
        'a buffer sampler view looks its target up as a texture\'s',
        'src/vrend/context.rs',
        """        let target = if is_buffer {
            tex_target""",
        """        let target = if false {
            tex_target""",
        'a_buffer_sampler_view_is_made_over_a_buffer_and_refused_over_a_texture',
    ),
    (
        'a blit box out past any texture is served',
        'src/vrend/context/blit.rs',
        """        if !within_reach(&src.region) || !within_reach(&dst.region) {""",
        """        if false && (!within_reach(&src.region) || !within_reach(&dst.region)) {""",
        'a_blit_box_far_past_any_texture_is_refused_and_one_short_of_it_is_survived',
    ),
    (
        'a stretched blit adds its scaled destination offset unchecked',
        'src/vrend/blitter.rs',
        """            x: (dst0.x + dst_width).saturating_add((s1.x as f32 * scale_x) as i32),""",
        """            x: dst0.x + dst_width + (s1.x as f32 * scale_x) as i32,""",
        'a_blit_box_far_past_any_texture_is_refused_and_one_short_of_it_is_survived',
    ),
    (
        'the blitter draws every destination slice from slice 0',
        'src/vrend/blitter.rs',
        """            let layer = job.dst_layer + dst_z;""",
        """            let layer = dst_z;""",
        'a_blit_through_the_blitter_lands_at_the_destination_box_s_layers',
    ),
    (
        'the blitter draws every destination slice at the box\'s first layer',
        'src/vrend/blitter.rs',
        """            let layer = job.dst_layer + dst_z;""",
        """            let layer = job.dst_layer;""",
        'a_blit_through_the_blitter_lands_at_the_destination_box_s_layers',
    ),
    (
        'desktop GL clamps a read-back to [0, 1]',
        'src/vrend/gl.rs',
        """        if !self.api.is_gles() {
            // SAFETY: plain scalars.
            unsafe { self.t.glClampColor()(GL_CLAMP_READ_COLOR, GL_FALSE as GLenum) };""",
        """        if false {
            // SAFETY: plain scalars.
            unsafe { self.t.glClampColor()(GL_CLAMP_READ_COLOR, GL_FALSE as GLenum) };""",
        'a_desktop_texture_reads_back_what_was_uploaded',
    ),
    (
        'a 3-3-2 pixel has no size',
        'src/vrend/gl.rs',
        """        GL_UNSIGNED_BYTE_3_3_2 | GL_UNSIGNED_BYTE_2_3_3_REV => return Some(1),""",
        """""",
        'a_desktop_texture_reads_back_what_was_uploaded',
    ),
    (
        'a rectangle stored as 2D is declared a rectangle sampler',
        'src/vrend/shader/glsl/header.rs',
        """        Texture::Rect if rect_is_2d(ctx, i as usize) => Texture::D2,""",
        """        Texture::Rect if false => Texture::D2,""",
        'a_rectangle_stored_as_2d_samples_as_2d',
    ),
    (
        'a rectangle stored as 2D is sized without a LOD on desktop',
        'src/vrend/shader/glsl/tex.rs',
        """        Texture::Rect | Texture::ShadowRect if rect_is_2d(ctx, sampler.index()) => {""",
        """        Texture::Rect | Texture::ShadowRect if ctx.cfg.is_gles() => {""",
        'a_rectangle_stored_as_2d_samples_as_2d',
    ),
    (
        'a rectangle stored as 2D is sampled without a LOD on desktop',
        'src/vrend/shader/glsl/tex.rs',
        """            Texture::Buffer | Texture::Msaa2d | Texture::Msaa2dArray => "",
            _ => ", 0",""",
        """            Texture::Buffer | Texture::Msaa2d | Texture::Msaa2dArray => "",
            Texture::Rect | Texture::ShadowRect if !gles => "",
            _ => ", 0",""",
        'a_rectangle_stored_as_2d_samples_as_2d',
    ),
    (
        'a rectangle stored as 2D is fetched without a LOD on desktop',
        'src/vrend/shader/glsl/tex.rs',
        """            || (texture == Texture::Rect && rect_is_2d(ctx, slot.index()))""",
        """            || (texture == Texture::Rect && ctx.cfg.is_gles())""",
        'a_rectangle_stored_as_2d_samples_as_2d',
    ),
    (
        'a format the driver rounds is offered',
        'src/vrend/caps.rs',
        """            if entry.bindings.sampler_view && entry.stores_exactly && (rgb32_buffers || !rgb32) {""",
        """            if entry.bindings.sampler_view && (rgb32_buffers || !rgb32) {""",
        'every_format_offered_is_stored_exactly',
    ),
    (
        'the probe takes every format as stored exactly',
        'src/vrend/formats.rs',
        """            stores_exactly,
        })""",
        """            stores_exactly: true || stores_exactly,
        })""",
        'every_format_offered_is_stored_exactly',
    ),
    (
        'an unsigned channel is held exactly at any depth',
        'src/vrend/formats.rs',
        """            ChannelType::Unsigned => held % channel.bits == 0,""",
        """            ChannelType::Unsigned => true,""",
        'a_format_is_held_exactly_only_at_a_depth_that_holds_its_values',
    ),
    (
        'a signed channel is held exactly at any depth',
        'src/vrend/formats.rs',
        """                (held - 1) % (channel.bits - 1) == 0""",
        """                true""",
        'a_format_is_held_exactly_only_at_a_depth_that_holds_its_values',
    ),
    (
        'an unsigned integer target gets a float output',
        'src/vrend/context/select.rs',
        """                        key.fs.cbufs_unsigned_int_bitmask |= 1 << i;""",
        """""",
        'an_integer_target_takes_the_bits_the_shader_writes',
    ),
    (
        'an integer output is cast without integer GLSL',
        'src/vrend/shader/glsl/decl.rs',
        """                        if ctx.outputs[i].ty != VecType::Float {
                            ctx.shader_req_bits |= req::INTS;""",
        """                        if false {
                            ctx.shader_req_bits |= req::INTS;""",
        'an_integer_target_takes_the_bits_the_shader_writes',
    ),
    (
        'an image load into an integer output stays float',
        'src/vrend/shader/glsl/tex.rs',
        """        let (open, close) = if typed_output {""",
        """        let (open, close) = if false && typed_output {""",
        'an_integer_target_takes_the_bits_the_shader_writes',
    ),
    (
        'the other stages are offered the vertex stage images',
        'src/vrend/caps.rs',
        """                .map(|(stage, _)| images[stage.index()])""",
        """                .map(|_| images[ShaderStage::Vertex.index()])""",
        'the_image_count_for_the_other_stages_fits_each_of_them',
    ),
    (
        'an image binds at the whole level word',
        'src/vrend/context.rs',
        """                            ImageSpan::Layers { level: im.level_size & 0xff, first, last }""",
        """                            ImageSpan::Layers { level: im.level_size, first, last }""",
        'an_image_binds_at_the_level_in_the_low_byte',
    ),
    (
        'RGB32 is offered without asking whether its texture buffer works',
        'src/vrend/caps.rs',
        """        let rgb32_buffers = rgb32_texture_buffers_work(gl, features);""",
        """        let rgb32_buffers = true || rgb32_texture_buffers_work(gl, features);""",
        'rgb32_is_offered_only_where_its_texture_buffer_samples',
    ),
    (
        'a buffer view past the texel limit keeps its length',
        'src/vrend/context.rs',
        """        Some(Elements { first, count: count.min(room) })""",
        """        Some(Elements { first, count })""",
        'a_texture_buffer_past_the_texel_limit_is_sized_as_the_limit',
    ),
    (
        'a buffer image past the texel limit keeps its length',
        'src/vrend/context.rs',
        """                            let size = texels.min(limit - first) * bs;""",
        """                            let size = texels * bs;""",
        'a_texture_buffer_past_the_texel_limit_is_sized_as_the_limit',
    ),
    (
        'an emptied image slot keeps the unit an earlier draw bound',
        'src/vrend/context/draw.rs',
        """            let Some(iview) = sub.images[s].get(&i) else {
                gl.bind_image_texture(image_unit, None, 0, false, 0, GL_READ_ONLY, GL_R32UI);
                continue;
            };""",
        """            let Some(iview) = sub.images[s].get(&i) else {
                continue;
            };""",
        'an_emptied_image_slot_reads_nothing',
    ),
    (
        'an indexed draw under transform feedback on GLES is drawn indexed',
        'src/vrend/context/draw.rs',
        """        let deindexed = draw.indexed""",
        """        let deindexed = false && draw.indexed""",
        'an_indexed_draw_under_transform_feedback_captures_its_indexed_vertices',
    ),
    (
        'a de-indexed draw ignores primitive restart',
        'src/vrend/context/draw.rs',
        """            draw.index_bias,
            draw.primitive_restart.then_some(draw.restart_index),
        );""",
        """            draw.index_bias,
            None,
        );""",
        'an_indexed_draw_under_transform_feedback_captures_its_indexed_vertices',
    ),
    (
        'a de-indexed draw takes any vertex count',
        'src/vrend/context/draw.rs',
        """        if draw.count > deindex::MAX_VERTICES {""",
        """        if false && draw.count > deindex::MAX_VERTICES {""",
        'an_indexed_draw_under_transform_feedback_captures_its_indexed_vertices',
    ),
    (
        'a de-indexed draw gathers any number of bytes',
        'src/vrend/context/draw.rs',
        """            if (resolved.vertices.len() as u64).saturating_mul(width) > deindex::MAX_GATHERED {""",
        """            if false && (resolved.vertices.len() as u64).saturating_mul(width) > deindex::MAX_GATHERED {""",
        'an_indexed_draw_under_transform_feedback_captures_its_indexed_vertices',
    ),
    (
        'vkCmdSetColorWriteEnableEXT tells the driver one switch fewer',
        'src/venus/driver.rs',
        """        unsafe { f(cb, enables.len() as u32, enables.as_ptr()) };""",
        """        unsafe { f(cb, enables.len().saturating_sub(1) as u32, enables.as_ptr()) };""",
        'color_write_enable_hands_the_driver_every_switch',
    ),
    (
        'vkCmdDrawIndirectCount swaps its draw cap and its stride',
        'src/venus/driver.rs',
        """        let f = self
            .drawer(cb, BindPoint::Graphics)?
            .try_vkCmdDrawIndirectCount()
            .ok_or(Unrecorded::NoDevice)?;
        // SAFETY: as above.
        unsafe { f(cb, buffer, offset, count_buffer, count_offset, max_draws, stride) };""",
        """        let f = self
            .drawer(cb, BindPoint::Graphics)?
            .try_vkCmdDrawIndirectCount()
            .ok_or(Unrecorded::NoDevice)?;
        // SAFETY: sabotage -- two u32s transposed.
        unsafe { f(cb, buffer, offset, count_buffer, count_offset, stride, max_draws) };""",
        'the_indexed_and_indirect_draws_hand_the_driver_every_argument_in_place',
    ),
    (
        'vkCmdDispatchBase dispatches from its counts instead of its base',
        'src/venus/driver.rs',
        """        let ([bx, by, bz], [x, y, z]) = (base, groups);""",
        """        let ([bx, by, bz], [x, y, z]) = (groups, groups);""",
        'the_indexed_and_indirect_draws_hand_the_driver_every_argument_in_place',
    ),
    (
        'vkCmdResolveImage gives the destination the source layout',
        'src/venus/driver.rs',
        """            (d.vkCmdResolveImage())(
                cb,
                src,
                src_layout,
                dst,
                dst_layout,""",
        """            (d.vkCmdResolveImage())(
                cb,
                src,
                src_layout,
                dst,
                src_layout,""",
        'the_core_clear_resolve_and_update_hand_the_driver_what_the_guest_sent',
    ),
    (
        'vkCmdUpdateBuffer writes one byte short',
        'src/venus/driver.rs',
        """                VkDeviceSize(data.len() as u64),""",
        """                VkDeviceSize(data.len() as u64 - 1),""",
        'the_core_clear_resolve_and_update_hand_the_driver_what_the_guest_sent',
    ),
    (
        'vkCmdSetDepthBounds swaps its minimum and maximum',
        'src/venus/driver.rs',
        """        unsafe { (d.vkCmdSetDepthBounds())(cb, min, max) };""",
        """        unsafe { (d.vkCmdSetDepthBounds())(cb, max, min) };""",
        'the_core_dynamic_state_setters_hand_the_driver_the_guests_values',
    ),
    (
        'vkCreateRenderPass2 goes through the panicking accessor',
        'src/venus/context.rs',
        """            |d| d.try_vkCreateRenderPass2(),""",
        """            |d| Some(d.vkCreateRenderPass2()),""",
        'the_render_pass2_commands_hand_the_driver_the_guests_structs',
    ),
    (
        'vkCmdNextSubpass drops the guest\'s subpass contents',
        'src/venus/driver.rs',
        """        unsafe { (d.vkCmdNextSubpass())(cb, contents) };""",
        """        unsafe { (d.vkCmdNextSubpass())(cb, VkSubpassContents::VK_SUBPASS_CONTENTS_INLINE) };""",
        'the_render_pass2_commands_hand_the_driver_the_guests_structs',
    ),
    (
        'vkCmdBlitImage2 is recorded as nothing',
        'src/venus/context.rs',
        """        let done = self.driver.cmd_blit_image2(args.commandBuffer, info);""",
        """        let done = Some(()).filter(|_| info.get().regionCount != u32::MAX);""",
        'the_copy_commands2_hand_the_driver_the_guests_struct',
    ),
    (
        'vkDestroyDescriptorUpdateTemplate goes through the panicking accessor',
        'src/venus/context.rs',
        """            |d| d.try_vkDestroyDescriptorUpdateTemplate(),""",
        """            |d| Some(d.vkDestroyDescriptorUpdateTemplate()),""",
        'descriptor_update_templates_are_made_and_destroyed_through_the_device',
    ),
    (
        'vkCmdSetDepthBias2EXT is recorded as nothing',
        'src/venus/context.rs',
        """        let done = self.driver.cmd_set_depth_bias2(args.commandBuffer, info);""",
        """        let done = Some(()).filter(|_| info.get().depthBiasClamp.is_finite() || true);""",
        'depth_bias2_hands_the_driver_the_guests_struct',
    ),
    (
        'vkCmdSetLogicOpEXT always sets COPY',
        'src/venus/driver.rs',
        """        let f = self.recorder(cb)?.try_vkCmdSetLogicOpEXT()?;
        // SAFETY: as above.
        unsafe { f(cb, op) };""",
        """        let f = self.recorder(cb)?.try_vkCmdSetLogicOpEXT()?;
        // SAFETY: sabotage -- the guest's op replaced.
        unsafe { f(cb, VkLogicOp::VK_LOGIC_OP_COPY) };""",
        'logic_op_hands_the_driver_the_guests_op',
    ),
    (
        'vkCmdSetSampleLocationsEXT is recorded as nothing',
        'src/venus/context.rs',
        """        let done = self.driver.cmd_set_sample_locations(args.commandBuffer, info);""",
        """        let done = Some(()).filter(|_| info.get().sampleLocationsCount != u32::MAX);""",
        'sample_locations_hand_the_driver_the_guests_struct',
    ),
    (
        'vkCmdSetRenderingInputAttachmentIndices is recorded as nothing',
        'src/venus/context.rs',
        """        let done = self.driver.cmd_set_rendering_input_attachment_indices(args.commandBuffer, info);""",
        """        let done = Some(()).filter(|_| info.get().colorAttachmentCount != u32::MAX);""",
        'the_rendering_location_setters_hand_the_driver_the_guests_structs',
    ),
    (
        'vkCmdEndConditionalRenderingEXT never reaches the driver',
        'src/venus/driver.rs',
        """        let f = self.recorder(cb)?.try_vkCmdEndConditionalRenderingEXT()?;
        // SAFETY: as above.
        unsafe { f(cb) };""",
        """        let _f = self.recorder(cb)?.try_vkCmdEndConditionalRenderingEXT()?;""",
        'conditional_rendering_hands_the_driver_the_guests_struct',
    ),
    (
        'vkCmdSetVertexInputEXT reads the attributes under the binding count',
        'src/venus/driver.rs',
        """                attributes.len() as u32,""",
        """                bindings.len() as u32,""",
        'vertex_input_hands_the_driver_both_arrays_with_their_own_counts',
    ),
    (
        'vkCmdSetFragmentShadingRateKHR swaps its two combiners',
        'src/venus/driver.rs',
        """unsafe { f(cb, &size, ops.as_ptr()) };""",
        """unsafe { f(cb, &size, [ops[1], ops[0]].as_ptr()) };""",
        'fragment_shading_rate_hands_the_driver_the_size_and_both_combiners',
    ),
    (
        "vkGetPhysicalDeviceFragmentShadingRatesKHR writes back a count that is not the driver's",
        'src/venus/context.rs',
        """pFragmentShadingRateCount_mut() {
                    count.set(n);""",
        """pFragmentShadingRateCount_mut() {
                    count.set(n + 1);""",
        'fragment_shading_rates_are_counted_then_written',
    ),
    (
        'vkCmdBeginQueryIndexedEXT is not held to the pool',
        'src/venus/driver.rs',
        """facts.holds(query, 1)?;
        facts.is_bracketed()?;
        let f = d.try_vkCmdBeginQueryIndexedEXT()""",
        """facts.is_bracketed()?;
        let f = d.try_vkCmdBeginQueryIndexedEXT()""",
        'every_query_index_is_held_to_the_pool',
    ),
    (
        'transform feedback begin and end drop the counter offsets',
        'src/venus/driver.rs',
        """optional(self.buffers), optional(self.offsets))""",
        """optional(self.buffers), core::ptr::null())""",
        'transform_feedback_hands_the_driver_the_guests_arrays',
    ),
    (
        'vkCmdBindTransformFeedbackBuffersEXT drops the sizes',
        'src/venus/driver.rs',
        """offsets.as_ptr(), optional(sizes))""",
        """offsets.as_ptr(), core::ptr::null())""",
        'transform_feedback_hands_the_driver_the_guests_arrays',
    ),
    (
        "a computed array length is evaluated in the member's own type",
        'venus-gen/rustgen.py',
        """return '(%s as i128)' % a if expr.strip() != name else a""",
        """return a""",
        'a_computed_length_does_not_overflow_on_the_guests_number',
    ),
    (
        'vkCmdSetExtraPrimitiveOverestimationSizeEXT always sets zero',
        'src/venus/driver.rs',
        """unsafe { f(cb, extra_primitive_overestimation_size) };""",
        """unsafe { f(cb, 0.0) };""",
        'the_extended_dynamic_state3_scalar_setters_hand_the_driver_the_guests_values',
    ),
    (
        'the per-attachment setters start from attachment zero',
        'src/venus/driver.rs',
        """unsafe { f(cb, first, values.len() as u32, values.as_ptr()) };""",
        """unsafe { f(cb, 0, values.len() as u32, values.as_ptr()) };""",
        'the_extended_dynamic_state3_array_setters_hand_the_driver_every_element',
    ),
    (
        "a pool is taken as the device's whatever device owns it",
        'src/venus/driver.rs',
        """self.open.get(&TypedHandle::of(pool)).is_some_and(|p| p.device == device)""",
        """self.open.contains_key(&TypedHandle::of(pool))""",
        'every_pool_command_is_held_to_the_pools_own_device',
    ),
    (
        'a primary replays after the secondary it executed is recorded again',
        'src/venus/journal.rs',
        """Item::Recording(b, _) => replayable.contains(b),""",
        """Item::Recording(b, _) => live.holds(*b),""",
        'recording_the_secondary_again_drops_the_primary_that_executed_it',
    ),
    (
        'an invalid secondary does not invalidate what executed it',
        'src/venus/journal.rs',
        """            if broken.is_empty() {
                break;""",
        """            if !broken.is_empty() || broken.is_empty() {
                break;""",
        'an_invalid_secondary_takes_everything_that_executed_it',
    ),
    (
        'compaction drops the tape of a live buffer it emptied',
        'src/venus/journal.rs',
        """!t.commands.is_empty() || live.holds(*b)""",
        """!t.commands.is_empty()""",
        'compacting_an_invalid_primary_keeps_it_invalid',
    ),
    (
        "vkCmdExecuteCommands runs another device's buffer",
        'src/venus/driver.rs',
        """            self.pools.device_of(*s) == Some(device)
                && self.pools.level_of(*s)""",
        """            self.pools.level_of(*s)""",
        'execute_commands_runs_only_the_primarys_own_devices_buffers',
    ),
    (
        'vkCmdExecuteCommands runs a primary as a secondary',
        'src/venus/driver.rs',
        """                && self.pools.level_of(*s) == Some(Level::Secondary)""",
        """                && self.pools.level_of(*s).is_some()""",
        'execute_commands_runs_only_the_primarys_own_devices_buffers',
    ),
    (
        "a trim is taken for the device's whatever device owns the pool",
        'src/venus/driver.rs',
        """        if !self.pools.held_by(pool, device) {
            return None;
        }
        let f = d.fns.try_vkTrimCommandPool()?;""",
        """        let f = d.fns.try_vkTrimCommandPool()?;""",
        'a_pool_is_trimmed_only_through_its_own_device',
    ),
    (
        'an extension whose commands are unserved is advertised',
        'src/venus/driver.rs',
        """        offered.retain(|name| served.contains(name));
""",
        """        let _ = served;
""",
        'no_advertised_extension_needs_a_command_this_build_does_not_serve',
    ),
    (
        "a dependency's group counts whether or not the dependency is offered",
        'src/venus/driver.rs',
        """                let live = e.dependent.iter().filter(|(cond, _)| holds(cond, &offered));""",
        """                let live = e.dependent.iter();""",
        'a_dependent_command_withholds_its_extension_only_while_the_dependency_is_offered',
    ),
    (
        'an extension is advertised without what it depends on',
        'src/venus/driver.rs',
        """                !holds(e.depends, &offered)
                    ||""",
        """                false
                    ||""",
        'an_extension_is_withheld_with_what_it_depends_on',
    ),
    (
        'vkCreateDevice enables an extension it never offered',
        'src/venus/driver.rs',
        """        if guest.iter().any(|name| !advertised.contains(&name.as_str())) {""",
        """        if guest.iter().any(|name| !advertised.contains(&name.as_str())) && false {""",
        'a_device_cannot_enable_an_extension_it_was_not_offered',
    ),
    (
        'not-in-reference lines withhold their extensions',
        'src/venus/ledger.rs',
        """        self != Status::NotInReference""",
        """        true""",
        'no_advertised_extension_needs_a_command_this_build_does_not_serve',
    ),
    (
        "vkCmdEndRendering2KHR drops the guest's struct",
        'src/venus/driver.rs',
        """        unsafe { f(cb, ptr(info)) };""",
        """        unsafe { f(cb, ptr(info.filter(|_| false))) };""",
        'end_rendering2_and_depth_clamp_range_hand_the_driver_the_guests_structs',
    ),
    (
        'a user-defined depth clamp with no range reaches the driver',
        'src/venus/context.rs',
        """        if user_defined && args.pDepthClampRange.is_none() {""",
        """        if user_defined && args.pDepthClampRange.is_none() && false {""",
        'end_rendering2_and_depth_clamp_range_hand_the_driver_the_guests_structs',
    ),
    (
        'vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR answers no shapes',
        'src/venus/context.rs',
        """            .enumerate_into(pd, out, |i| i.try_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR());""",
        """            .enumerate_into(pd, out.filter(|_| false), |i| i.try_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR());""",
        'cooperative_matrix_properties_are_counted_then_written',
    ),
    (
        'vkCmdDrawMeshTasksIndirectCountEXT swaps its two offsets',
        'src/venus/driver.rs',
        """        let f = self
            .drawer(cb, BindPoint::Graphics)?
            .try_vkCmdDrawMeshTasksIndirectCountEXT()
            .ok_or(Unrecorded::NoDevice)?;
        // SAFETY: as above.
        unsafe { f(cb, buffer, offset, count_buffer, count_offset, max_draws, stride) };""",
        """        let f = self
            .drawer(cb, BindPoint::Graphics)?
            .try_vkCmdDrawMeshTasksIndirectCountEXT()
            .ok_or(Unrecorded::NoDevice)?;
        // SAFETY: sabotage -- the two offsets swapped.
        unsafe { f(cb, buffer, count_offset, count_buffer, offset, max_draws, stride) };""",
        'the_mesh_task_draws_hand_the_driver_every_argument_in_place',
    ),
    (
        'a host-address union member is accepted even when present',
        'venus-gen/rustgen.py',
        """            return ['if dec.decode_simple_pointer() {',
                    '    dec.set_fatal();',
                    '} else {',
                    '    %s = %s;' % (m, null),
                    '}']""",
        """            return ['if dec.decode_simple_pointer() {',
                    '    let _ = 0;',
                    '} else {',
                    '    %s = %s;' % (m, null),
                    '}']""",
        'a_device_or_host_address_decodes_a_device_address_and_no_host_one',
    ),
    (
        'an array of arrays sizes its rows from a poisoned sibling',
        'venus-gen/rustgen.py',
        """            return (['if dec.poisoned() {', '    return;', '}']""",
        """            return ([]""",
        'a_build_whose_rows_disagree_with_its_infos_poisons_the_stream',
    ),
    (
        'every row of an array of arrays is sized from the first',
        'venus-gen/rustgen.py',
        """        return ('(if %s.is_null() { 0 } else { (*%s.add(row)).%s }) as u64'""",
        """        return ('(if %s.is_null() { 0 } else { (*%s.add(0)).%s }) as u64'""",
        'an_acceleration_structure_build_reproduces_the_wire_row_by_row',
    ),
    (
        'a selected union is decoded by its wire tag, not its selector',
        'venus-gen/rustgen.py',
        """                        out += ['if dec.decode_scalar::<%s>() != tag {' % ty.sty.name,
                                '    dec.set_fatal();',
                                '    return;',
                                '}']""",
        """                        out += ['let tag = dec.decode_scalar::<%s>();' % ty.sty.name]""",
        'a_geometry_whose_union_tag_is_not_its_type_poisons_the_stream',
    ),
    (
        'a build may name its geometries through both arrays',
        'src/venus/context.rs',
        """    info.geometryCount == 0 || info.pGeometries.is_null() != info.ppGeometries.is_null()""",
        """    info.geometryCount == 0 || !info.pGeometries.is_null() || !info.ppGeometries.is_null()""",
        'a_build_whose_geometries_are_in_neither_array_or_both_is_refused',
    ),
    (
        'a build-size query without primitive counts reaches the driver',
        'src/venus/context.rs',
        """        let Some(counts) = args.pMaxPrimitiveCounts() else {""",
        """        let Some(counts) = args.pMaxPrimitiveCounts().or(Some(&[])) else {""",
        'a_build_whose_geometries_are_in_neither_array_or_both_is_refused',
    ),
    (
        'a property write is not held to its query pool',
        'src/venus/driver.rs',
        """        facts.holds(first, n)?;""",
        """        let _ = facts;""",
        'the_acceleration_structure_commands_hand_the_driver_what_the_guest_sent',
    ),
    (
        'a query of any kind may be begun',
        'src/venus/driver.rs',
        """        facts.is_bracketed()?;
        // SAFETY: as above, and a query the pool holds.
        unsafe { (d.vkCmdBeginQuery())""",
        """        // SAFETY: as above, and a query the pool holds.
        unsafe { (d.vkCmdBeginQuery())""",
        'every_query_index_is_held_to_the_pool',
    ),
    (
        'a timestamp may be written into any pool',
        'src/venus/driver.rs',
        """        facts.counts(VkQueryType::VK_QUERY_TYPE_TIMESTAMP)?;
        // SAFETY: as above, and a query the pool holds.
        unsafe { (d.vkCmdWriteTimestamp())""",
        """        // SAFETY: as above, and a query the pool holds.
        unsafe { (d.vkCmdWriteTimestamp())""",
        'every_query_index_is_held_to_the_pool',
    ),
    (
        'a property write need not match its pool',
        'src/venus/driver.rs',
        """_KHR => facts.counts(ty)?,""",
        """_KHR => {}""",
        'every_query_index_is_held_to_the_pool',
    ),
    (
        'a shader group index is not held to the pipeline',
        'src/venus/driver.rs',
        """            Some(end) if end <= groups => Ok(()),""",
        """            Some(_) => Ok(()),""",
        'ray_tracing_groups_are_held_to_the_pipeline',
    ),
    (
        'shader group handles are not held to the room',
        'src/venus/driver.rs',
        """        if (out.len() as u64) < u64::from(count) * u64::from(size) {""",
        """        if (out.len() as u64) < u64::from(size) {""",
        'ray_tracing_groups_are_held_to_the_pipeline',
    ),
    (
        'a destroyed pipeline keeps its ray-tracing record',
        'src/venus/driver.rs',
        """            VkObjectType::VK_OBJECT_TYPE_PIPELINE => {
                self.pipelines.remove(&VkPipeline::from_host(handle));
            }""",
        """            VkObjectType::VK_OBJECT_TYPE_PIPELINE => {}""",
        'ray_tracing_pipelines_are_created_traced_and_forgotten_through_the_handlers',
    ),
    (
        'a ray-tracing group may name any stage',
        'src/venus/driver.rs',
        """            let names_a_stage = |i: u32| i == SHADER_UNUSED || (i as usize) < stages;""",
        """            let names_a_stage = |_: u32| stages < usize::MAX;""",
        'ray_tracing_groups_are_held_to_the_pipeline',
    ),
    (
        'a derivative may name any pipeline of its run as its base',
        'src/venus/context.rs',
        """            base => usize::try_from(base).is_ok_and(|base| base < i),""",
        """            base => usize::try_from(base).is_ok(),""",
        'a_derivative_pipeline_names_only_an_earlier_pipeline_of_its_run_as_its_base',
    ),
    (
        "a derivative's flags2 link is not read",
        'src/venus/context.rs',
        """            Some(two) => two.flags.0,""",
        """            Some(_) => info.flags(),""",
        'a_derivative_pipeline_names_only_an_earlier_pipeline_of_its_run_as_its_base',
    ),
    (
        'a graphics pipeline run is not held to its bases',
        'src/venus/context.rs',
        """    fn vkCreateGraphicsPipelines(&mut self, args: &mut vn_command_vkCreateGraphicsPipelines<'_>) {
        let infos = args.pCreateInfos();
        if !bases_in_run(infos) {""",
        """    fn vkCreateGraphicsPipelines(&mut self, args: &mut vn_command_vkCreateGraphicsPipelines<'_>) {
        let infos = args.pCreateInfos();
        if false && !bases_in_run(infos) {""",
        'a_derivative_pipeline_names_only_an_earlier_pipeline_of_its_run_as_its_base',
    ),
    (
        'a ray-tracing pipeline run is not held to its bases',
        'src/venus/context.rs',
        """        let infos = args.pCreateInfos();
        if !bases_in_run(infos) {
            self.reject(BASE_OUTSIDE_RUN);
            return;
        }
        let ids = args.pPipelines();
        // Read before the shadow is borrowed: see `vkEnumeratePhysicalDevices`.
        let (device, cache, alloc) = (args.device, args.pipelineCache, args.pAllocator);
        let out = args.handle_pPipelines_mut();
        let made = self.driver.create_ray_tracing_pipelines(""",
        """        let infos = args.pCreateInfos();
        let ids = args.pPipelines();
        // Read before the shadow is borrowed: see `vkEnumeratePhysicalDevices`.
        let (device, cache, alloc) = (args.device, args.pipelineCache, args.pAllocator);
        let out = args.handle_pPipelines_mut();
        let made = self.driver.create_ray_tracing_pipelines(""",
        'a_derivative_pipeline_names_only_an_earlier_pipeline_of_its_run_as_its_base',
    ),
    (
        'a draw is recorded with no pipeline bound at its point',
        'src/venus/driver.rs',
        """        let binding = bound.at(point).ok_or(Unrecorded::Unbound)?;
        self.pipelines
            .get(&binding.pipeline)
            .filter(|facts| facts.serial == binding.serial)
            .ok_or(Unrecorded::Destroyed)?;""",
        """        let _ = bound;""",
        'venus::context::tests::a_draw_reaches_the_driver_only_with_a_pipeline_of_its_kind_bound',
    ),
    (
        'a draw is recorded with the pipeline bound at its point destroyed',
        'src/venus/driver.rs',
        """        let binding = bound.at(point).ok_or(Unrecorded::Unbound)?;
        self.pipelines
            .get(&binding.pipeline)
            .filter(|facts| facts.serial == binding.serial)
            .ok_or(Unrecorded::Destroyed)?;""",
        """        let _binding = bound.at(point).ok_or(Unrecorded::Unbound)?;""",
        'venus::context::tests::a_draw_reaches_the_driver_only_with_a_pipeline_of_its_kind_bound',
    ),
    (
        "a draw takes a reused pipeline handle for the pipeline that was bound",
        'src/venus/driver.rs',
        """            .filter(|facts| facts.serial == binding.serial)
            .ok_or(Unrecorded::Destroyed)?;""",
        """            .ok_or(Unrecorded::Destroyed)?;""",
        'venus::context::tests::a_draw_reaches_the_driver_only_with_a_pipeline_of_its_kind_bound',
    ),
    (
        'a trace reaches the driver without asking what is bound',
        'src/venus/driver.rs',
        """            .drawer(cb, BindPoint::RayTracing)?
            .try_vkCmdTraceRaysKHR()""",
        """            .recorder(cb)
            .ok_or(Unrecorded::NoDevice)?
            .try_vkCmdTraceRaysKHR()""",
        'venus::context::tests::ray_tracing_pipelines_are_created_traced_and_forgotten_through_the_handlers',
    ),
    (
        "a pipeline may be bound at another kind's point",
        'src/venus/driver.rs',
        """        if facts.kind.bind_point() != point {""",
        """        if false && facts.kind.bind_point() != point {""",
        'venus::context::tests::a_draw_reaches_the_driver_only_with_a_pipeline_of_its_kind_bound',
    ),
    (
        'a pipeline of another device may be bound',
        'src/venus/driver.rs',
        """            .filter(|p| p.device == device)
            .ok_or(BindRefused::UnknownPipeline)?;""",
        """            .filter(|_| true)
            .ok_or(BindRefused::UnknownPipeline)?;""",
        'venus::context::tests::a_draw_reaches_the_driver_only_with_a_pipeline_of_its_kind_bound',
    ),
    (
        'a served bind marks nothing bound',
        'src/venus/driver.rs',
        """        let child = self.pools.child_mut(cb).expect("the pool record `device_of` just read");
        child.bound = child.bound.with(point, Binding { pipeline, serial: facts.serial });
""",
        """        let child = self.pools.child_mut(cb).expect("the pool record `device_of` just read");
""",
        'venus::context::tests::a_draw_reaches_the_driver_only_with_a_pipeline_of_its_kind_bound',
    ),
    (
        "a begin keeps the last recording's bindings",
        'src/venus/driver.rs',
        """        // success or not: a failed begin leaves nothing a draw may rely on either.
        self.pools.unbind(cb);
""",
        """        // success or not: a failed begin leaves nothing a draw may rely on either.
""",
        'venus::context::tests::a_draw_reaches_the_driver_only_with_a_pipeline_of_its_kind_bound',
    ),
    (
        'a command buffer reset keeps its bindings',
        'src/venus/driver.rs',
        """        self.pools.unbind(cb);
        let d = self.recorder(cb)?;
        // SAFETY: as above.
        Some(unsafe { (d.vkResetCommandBuffer())""",
        """        let d = self.recorder(cb)?;
        // SAFETY: as above.
        Some(unsafe { (d.vkResetCommandBuffer())""",
        'venus::context::tests::a_draw_reaches_the_driver_only_with_a_pipeline_of_its_kind_bound',
    ),
    (
        "a pool reset keeps its buffers' bindings",
        'src/venus/driver.rs',
        """        self.pools.unbind_pool(pool);
""",
        """""",
        'venus::context::tests::a_draw_reaches_the_driver_only_with_a_pipeline_of_its_kind_bound',
    ),
    (
        'a query not ready when asked is dropped, and the guest waits forever for its result',
        'src/vrend/context.rs',
        '        } else if !parked.contains(&h) {\n            parked.push(h);\n        }\n',
        '        }\n',
        'parked_query',
    ),
    (
        'a fence covering a parked query retires before the render thread has answered it',
        'src/vrend/waiter.rs',
        '        if job.owed.queries {\n',
        '        if false {\n',
        'parked_query',
    ),
    (
        'a poll lets the held fences go without answering the queries they wait for',
        'src/vrend/vrend.rs',
        '        pump.serve(|| self.answer_parked(guest));\n',
        '        pump.serve(|| ());\n',
        'parked_query',
    ),
    (
        'a fence naming no context leaves a parked query to a poll that may come before its work',
        'src/vrend/vrend.rs',
        '        if on.is_some() && self.waiter.as_ref().is_some_and(|w| w.pump().subscribed()) {\n',
        '        if self.waiter.as_ref().is_some_and(|w| w.pump().subscribed()) {\n',
        'parked_query',
    ),
    (
        'with nobody to poll, a fence covering a parked query retires with it unanswered',
        'src/vrend/vrend.rs',
        '        self.answer_parked(guest);\n        false\n    }\n',
        '        false\n    }\n',
        'parked_query',
    ),
    (
        'the blitter samples a cube source with the face 2D coordinate',
        'src/vrend/blitter.rs',
        """                        v[4..7].copy_from_slice(&cube_direction(layer % 6, tex[i]));""",
        """                        let _ = cube_direction(layer % 6, tex[i]);""",
        'a_blit_through_the_blitter_lands_at_the_destination_box_s_layers',
    ),
    (
        'the blitter samples every cube source at its first face',
        'src/vrend/blitter.rs',
        """                        v[4..7].copy_from_slice(&cube_direction(layer % 6, tex[i]));""",
        """                        v[4..7].copy_from_slice(&cube_direction(0, tex[i]));""",
        'a_blit_through_the_blitter_lands_at_the_destination_box_s_layers',
    ),
    (
        'a 24-bit depth upload drops its low byte on the way to GL',
        'src/vrend/transfer.rs',
        """        *word = ((z * Z32_MAX).div_ceil(Z24_MAX) as u32).to_ne_bytes();""",
        """        *word = ((z >> 8 << 16) as u32).to_ne_bytes();""",
        'a_24_bit_depth_reads_back_as_written',
    ),
    (
        'a 24-bit depth goes to GL rounded to nearest, which a truncating driver stores one step low',
        'src/vrend/transfer.rs',
        """        *word = ((z * Z32_MAX).div_ceil(Z24_MAX) as u32).to_ne_bytes();""",
        """        *word = (((z * Z32_MAX + Z24_MAX / 2) / Z24_MAX) as u32).to_ne_bytes();""",
        'a_24_bit_depth_reads_back_as_written',
    ),
    (
        'a GLES image both loaded and stored is declared once',
        'src/vrend/shader/glsl/mod.rs',
        """        gles && self.decl.writable && !tex::is_r32_format(self.decl.format)""",
        """        gles && self.decl.writable && !tex::is_r32_format(self.decl.format) && false""",
        'an_image_loaded_and_stored_in_one_shader_is_read_and_written',
    ),
    (
        'a GLES image read beside its store alias is never declared for the read',
        'src/vrend/shader/glsl/inst.rs',
        """                    ctx.images[slot].loaded = true;""",
        """                    let _ = slot;""",
        'an_image_loaded_and_stored_in_one_shader_is_read_and_written',
    ),
    (
        'a recorded resource event is stamped with the sequence of the record before it',
        'src/trace.rs',
        """        let seq = s.seq;
        s.res.push(ev.encode(seq));""",
        """        let seq = s.seq.saturating_sub(1);
        s.res.push(ev.encode(seq));""",
        'a_dump_reads_back_its_records_in_the_c_layout',
    ),
    (
        'a de-indexed draw numbers its vertex IDs from zero',
        'src/vrend/context/select.rs',
        """        key.vs.vertex_id_from_input = stage == Vertex && self.deindexing;""",
        """        key.vs.vertex_id_from_input = false && stage == Vertex && self.deindexing;""",
        'an_indexed_draw_under_transform_feedback_captures_its_indexed_vertices',
    ),
    (
        'a GLES vertex stage leaves the point size to the driver as the C does',
        'src/vrend/context/select.rs',
        """        key.fixed_point_size = host.features.api().is_gles()""",
        """        key.fixed_point_size = false && host.features.api().is_gles()""",
        'a_fixed_point_size_draws_points_that_size',
    ),
    (
        'a GLES vertex stage writes a point size the rasterizer never told it',
        'src/vrend/context.rs',
        """            sub.sysval.point_size = s.point_size;""",
        """            sub.sysval.point_size = 1.0;""",
        'a_fixed_point_size_draws_points_that_size',
    ),
    (
        'only an 8-bit alpha target has its alpha moved into red',
        'src/vrend/formats.rs',
        """    formats.get(format).is_some_and(|e| e.gl.swizzle == Some(ALPHA_FROM_RED))""",
        """    formats.get(format).is_some_and(|e| e.gl.swizzle == Some(ALPHA_FROM_RED))
        && "A8_UNORM" == format.name()""",
        'an_alpha_target_stores_the_fragment_alpha',
    ),
    (
        'a GLES host keeps compressed formats out of the readback mask as the C does',
        'src/vrend/formats.rs',
        """                    let blocks_read = !gles || features.has(Feature::copy_image);""",
        """                    let blocks_read = !gles;""",
        'a_compressed_format_is_advertised_readable_where_its_blocks_read_back',
    ),
    (
        'a GLES compressed readback answers from the pages the guest already holds',
        'src/vrend/transfer.rs',
        """            if !desktop && l.compressed && features.has(Feature::copy_image) {""",
        """            if !desktop && l.compressed && features.has(Feature::copy_image) && false {""",
        'a_compressed_texture_reads_back_its_blocks',
    ),
    (
        'a GLES compressed readback copies from the level origin, not the box',
        'src/vrend/transfer.rs',
        """                    let origin = [b.x, b.y, b.z + d as GLint];""",
        """                    let origin = [0, 0, b.z + d as GLint];""",
        'a_compressed_texture_reads_back_its_blocks',
    ),
    (
        'a GLES 1D image load takes the coordinate the guest wrote',
        'src/vrend/shader/glsl/inst.rs',
        """    let coordinate = if inst.src[0].file == File::Image {""",
        """    let coordinate = if inst.src[0].file == File::Image && false {""",
        'a_1d_image_loads_the_texel_it_names',
    ),
    (
        'a GLES 1D image store takes the coordinate the guest wrote, as the C does',
        'src/vrend/shader/glsl/inst.rs',
        """    } else if inst.dst[0].file == File::Image {""",
        """    } else if inst.dst[0].file == File::Image && false {""",
        'a_1d_image_stores_to_the_texel_it_names',
    ),
    (
        'a load through a GLES store alias is left unordered against the stores',
        'src/vrend/shader/glsl/tex.rs',
        """        if alias {
            ctx.bufs.emit("memoryBarrierImage();\\n");
        }""",
        """        if alias && false {
            ctx.bufs.emit("memoryBarrierImage();\\n");
        }""",
        'loads_and_stores_through_one_image_keep_their_order',
    ),
    (
        'a GLES texture buffer range needs ES 3.2, not GL_EXT_texture_buffer',
        'src/vrend/features.rs',
        """["GL_ARB_texture_buffer_range", "GL_EXT_texture_buffer", "GL_OES_texture_buffer"]""",
        """["GL_ARB_texture_buffer_range"]""",
        'a_texture_buffer_past_the_texel_limit_is_sized_as_the_limit',
    ),
    (
        'a host with one vertex stream keeps transform_feedback3',
        'src/vrend/features.rs',
        """            if streams < 4 {""",
        """            if streams < 1 {""",
        'pipeline_statistics_and_overflow_queries_count_where_the_host_has_them',
    ),
    (
        'a GLES draw restarting at another index is drawn from its own indices',
        'src/vrend/context/draw.rs',
        """                    if draw.primitive_restart
                        && features.api().is_gles()""",
        """                    if draw.primitive_restart
                        && features.api().is_gles() && false""",
        'an_indirect_draw_restarts_at_the_guest_s_restart_index',
    ),
    (
        'the trace ring leaves a sliver smaller than a header at its end',
        'src/trace.rs',
        """        if tail_room >= need && tail_room - need < HEADER {""",
        """        if false && tail_room >= need && tail_room - need < HEADER {""",
        'a_256_byte_ring_stays_walkable',
    ),
    (
        'a memory info query is refused though the caps advertise it',
        'src/vrend/context.rs',
        """            Command::GetMemoryInfo(out) => self.get_memory_info(host, out),""",
        """            Command::GetMemoryInfo(_) => Err(Fault::Unimplemented { cmd: kind, what: "blob resources" }),""",
        'a_memory_info_query_writes_the_host_s_memory_into_the_guest_s_buffer',
    ),
    (
        'a draw counted from a stream-out target draws every vertex its buffer holds',
        'src/vrend/context/draw.rs',
        """                    gl.draw_transform_feedback(mode, tf);""",
        """                    gl.draw_arrays(mode, 0, { let _ = tf; 6 });""",
        'a_draw_counted_from_a_stream_out_target_draws_what_was_captured',
    ),
    (
        'a buffer view of a format with no texture entry is refused',
        'src/vrend/context.rs',
        """        if entry.is_none() && !is_buffer {""",
        """        if entry.is_none() {""",
        'a_buffer_view_of_a_format_the_host_has_no_texture_of_samples',
    ),
    (
        'a render condition is recorded but never applied',
        'src/vrend/context.rs',
        """            gl.begin_conditional_render(id, glmode);""",
        """            let _ = (id, glmode);""",
        'a_render_condition_drops_a_draw_its_query_failed',
    ),
    (
        'an unconditional clear runs under the render condition',
        'src/vrend/context.rs',
        """        let applied = self.applied_render_condition(host).filter(|_| unconditional)?;""",
        """        let applied = self.applied_render_condition(host).filter(|_| !unconditional)?;""",
        'a_render_condition_drops_a_draw_its_query_failed',
    ),
    (
        'a depth format stored shallower than its bits is offered',
        'src/vrend/formats.rs',
        """        return !depth.normalized || held[0] == 0 || held[0] >= depth.bits;""",
        """        return true;""",
        'a_format_is_held_exactly_only_at_a_depth_that_holds_its_values',
    ),
    (
        'an advanced blend equation is never set',
        'src/vrend/context/draw.rs',
        """            gl.blend_equation(advanced_blend_equation(mode));""",
        """            let _ = advanced_blend_equation(mode);""",
        'an_advanced_blend_equation_blends_the_draw',
    ),
    (
        'a blend state carrying an advanced equation is refused',
        'src/vrend/decode.rs',
        """            let rest = if i == 0 { s2 & !(0x1f << 17) } else { s2 };""",
        """            let rest = s2;""",
        'an_advanced_blend_equation_blends_the_draw',
    ),
    (
        'a shader stage the driver has no limits for is offered',
        'src/vrend/features.rs',
        """            if self.have.contains(feature) && gl.get_integer(limit) <= 0 {""",
        """            if self.have.contains(feature) && gl.get_integer(limit) < 0 {""",
        'a_geometry_stage_is_offered_only_where_the_host_draws_through_one',
    ),
    (
        'a submit device-group count is passed to the driver unchecked',
        'src/venus/driver.rs',
        """        let (s, g) = (on.root.get(), this.get());""",
        """        let (s, g) = (on.root.get(), this.get());
        if s.waitSemaphoreCount != u32::MAX {
            return Ok(());
        }""",
        'a_submit_whose_chained_counts_disagree_is_refused_before_the_driver',
    ),
    (
        'a submit timeline signal value count is passed to the driver unchecked',
        'src/venus/driver.rs',
        """        if !covers(
            s.signalSemaphoreCount,
            t.signalSemaphoreValueCount,""",
        """        if false && !covers(
            s.signalSemaphoreCount,
            t.signalSemaphoreValueCount,""",
        'a_submit_whose_chained_counts_disagree_is_refused_before_the_driver',
    ),
    (
        'a host image read out is not held to the room for it',
        'src/venus/driver.rs',
        """        if span > out.len() as u64 {""",
        """        if false && span > out.len() as u64 {""",
        'a_host_copy_that_does_not_fit_is_refused_before_the_driver',
    ),
    (
        'a host image write is not held to the bytes it carried',
        'src/venus/driver.rs',
        """            if span > carried {""",
        """            if false && span > carried {""",
        'a_host_copy_that_does_not_fit_is_refused_before_the_driver',
    ),
    (
        'a host copy region is not held inside the image',
        'src/venus/driver.rs',
        """                .is_some_and(|end| e > 0 && end <= size)""",
        """                .is_some_and(|_| true)""",
        'a_host_copy_that_does_not_fit_is_refused_before_the_driver',
    ),
    (
        'a host copy may name layers the image does not have',
        'src/venus/driver.rs',
        """            || sub.baseArrayLayer.checked_add(layers).is_none_or(|e| e > facts.array_layers)""",
        """            || sub.baseArrayLayer == u32::MAX""",
        'a_host_copy_that_does_not_fit_is_refused_before_the_driver',
    ),
    (
        'a host copy may name a mip level the image does not have',
        'src/venus/driver.rs',
        """        if sub.mipLevel >= facts.mip_levels {""",
        """        if false && sub.mipLevel >= facts.mip_levels {""",
        'a_host_copy_that_does_not_fit_is_refused_before_the_driver',
    ),
    (
        'a host copy may lay rows out shorter than its region',
        'src/venus/driver.rs',
        """        if row_length < extent.width || image_height < extent.height {""",
        """        if false && row_length < extent.width {""",
        'a_host_copy_that_does_not_fit_is_refused_before_the_driver',
    ),
    (
        'constants may be pushed past the device push block',
        'src/venus/driver.rs',
        """    u64::from(offset) + bytes as u64 <= u64::from(limits.maxPushConstantsSize)""",
        """    u64::from(offset) + bytes as u64 <= u64::MAX""",
        'constants_pushed_past_the_push_block_are_refused_before_the_driver',
    ),
    (
        'a push2 size is trusted with no bytes behind it',
        'src/venus/driver.rs',
        """        if carried != info.size as usize {""",
        """        if false && carried != info.size as usize {""",
        'constants_pushed_past_the_push_block_are_refused_before_the_driver',
    ),
    (
        'a null handle reaches the driver where vk.xml requires an object',
        'venus-gen/rustgen.py',
        """                    required = not var.is_optional() and var.can_validate()""",
        """                    required = False""",
        'a_null_handle_where_an_object_is_required_poisons_the_stream',
    ),
    (
        'an enum value the registry does not define reaches the driver',
        'venus-gen/rustgen.py',
        """        if (var.ty.base.category != VkType.ENUM or validity == Gen_INVALID""",
        """        if (True or var.ty.base.category != VkType.ENUM or validity == Gen_INVALID""",
        'an_enum_value_the_registry_does_not_define_poisons_the_stream',
    ),
    (
        'a served command goes unclassified for validation',
        'src/venus/validation.txt',
        '\nvkCmdDraw                                        reviewed\n',
        '\n',
        'every_served_command_is_classified_for_validation',
    ),
    (
        'dynamic state windows may run past the device limit',
        'src/venus/driver.rs',
        """    u64::from(first) + count as u64 <= u64::from(limit)""",
        """    u64::from(first) + count as u64 <= u64::MAX""",
        'viewports_and_scissors_past_the_last_viewport_are_refused_before_the_driver',
    ),
    (
        'scissors are not held to the viewport limit',
        'src/venus/driver.rs',
        """        if !window_fits(limits.maxViewports, first, scissors.len()) {""",
        """        if false && !window_fits(limits.maxViewports, first, scissors.len()) {""",
        'viewports_and_scissors_past_the_last_viewport_are_refused_before_the_driver',
    ),
    (
        'a counted viewport set is not held to the viewport limit',
        'src/venus/driver.rs',
        """        if !window_fits(limits.maxViewports, 0, viewports.len()) {""",
        """        if false && !window_fits(limits.maxViewports, 0, viewports.len()) {""",
        'viewports_and_scissors_past_the_last_viewport_are_refused_before_the_driver',
    ),
    (
        'per-attachment blend state is not held to the attachment limit',
        'src/venus/driver.rs',
        """        if !window_fits(limits.maxColorAttachments, first, values.len()) {""",
        """        if false && !window_fits(limits.maxColorAttachments, first, values.len()) {""",
        'attachment_state_and_vertex_bindings_past_the_last_are_refused_before_the_driver',
    ),
    (
        'vertex bindings are not held to the binding limit',
        'src/venus/driver.rs',
        """        if !window_fits(limits.maxVertexInputBindings, first, count) {""",
        """        if false && !window_fits(limits.maxVertexInputBindings, first, count) {""",
        'attachment_state_and_vertex_bindings_past_the_last_are_refused_before_the_driver',
    ),
    (
        'a color attachment count is not held to the device',
        'src/venus/driver.rs',
        """    if count > facts.limits.maxColorAttachments {""",
        """    if count > u32::MAX {""",
        'color_attachment_counts_past_the_device_do_not_validate',
    ),
    (
        'a vertex binding index is not held to the device',
        'src/venus/driver.rs',
        """        if this.binding >= facts.limits.maxVertexInputBindings {""",
        """        if this.binding >= u32::MAX {""",
        'vertex_input_past_the_device_does_not_validate',
    ),
    (
        'a vertex attribute location is not held to the device',
        'src/venus/driver.rs',
        """        if this.location >= limits.maxVertexInputAttributes {""",
        """        if this.location >= u32::MAX {""",
        'vertex_input_past_the_device_does_not_validate',
    ),
    (
        'a vertex attribute binding is not held to the device',
        'src/venus/driver.rs',
        """        if this.binding >= limits.maxVertexInputBindings {""",
        """        if this.binding >= u32::MAX {""",
        'vertex_input_past_the_device_does_not_validate',
    ),
    (
        'a pipeline layout set count is not held to the device',
        'src/venus/driver.rs',
        """        if this.setLayoutCount > limits.maxBoundDescriptorSets {""",
        """        if this.setLayoutCount > u32::MAX {""",
        'a_pipeline_layout_past_the_device_does_not_validate',
    ),
    (
        'push constant ranges may share a stage',
        'src/venus/driver.rs',
        """            if named == 0 || named & stages != 0 {""",
        """            if named == 0 {""",
        'a_pipeline_layout_past_the_device_does_not_validate',
    ),
    (
        'a push constant range is not held to the push block',
        'src/venus/driver.rs',
        """            if u64::from(r.offset) + u64::from(r.size) > u64::from(limits.maxPushConstantsSize) {""",
        """            if u64::from(r.offset) + u64::from(r.size) > u64::MAX {""",
        'a_pipeline_layout_past_the_device_does_not_validate',
    ),
    (
        'a byte count draw may divide by a zero stride',
        'src/venus/driver.rs',
        '        if stride == 0 {\n            return Err(Unrecorded::Invalid(',
        '        if stride == u32::MAX {\n            return Err(Unrecorded::Invalid(',
        'a_byte_count_draw_with_no_stride_is_refused_before_the_driver',
    ),
    (
        'a sample location count is not held to its grid',
        'src/venus/driver.rs',
        """        if u64::from(this.sampleLocationsCount) != wanted {""",
        """        if u64::from(this.sampleLocationsCount) == u64::MAX {""",
        'sample_locations_the_device_does_not_take_do_not_validate',
    ),
    (
        'a sample location grid is not held to the device',
        'src/venus/driver.rs',
        """        if grid.width > largest.width || grid.height > largest.height {""",
        """        if grid.width > u32::MAX {""",
        'sample_locations_the_device_does_not_take_do_not_validate',
    ),
    (
        'a guest destroy leaves the object record behind',
        'src/venus/driver.rs',
        """        self.forget(VkObjectType(T::OBJECT_TYPE), object.host());""",
        '',
        'every_query_index_is_held_to_the_pool',
    ),
    (
        'a buffer may be bound at an offset past its end',
        'src/venus/driver.rs',
        """        if offset.0 >= len {""",
        """        if offset.0 == u64::MAX {""",
        'buffers_bound_past_their_end_are_refused_before_the_driver',
    ),
    (
        'a buffer range may run past its end',
        'src/venus/driver.rs',
        """            Some(size) if size != VK_WHOLE_SIZE && offset.0.saturating_add(size.0) > len => {""",
        """            Some(size) if size != VK_WHOLE_SIZE && offset.0 > u64::MAX - 1 && size.0 > len => {""",
        'buffers_bound_past_their_end_are_refused_before_the_driver',
    ),
    (
        'a destroyed buffer keeps its size record',
        'src/venus/driver.rs',
        """            VkObjectType::VK_OBJECT_TYPE_BUFFER => {
                self.buffers.remove(&VkBuffer::from_host(handle));
            }""",
        """            VkObjectType::VK_OBJECT_TYPE_BUFFER => {}""",
        'buffers_bound_past_their_end_are_refused_before_the_driver',
    ),
    (
        'a chained struct refusal is dropped by the generated walk',
        'venus-gen/rustgen.py',
        """                    '                    <%s as cs::Validate<_>>::validate(link, &chained)?;' % nt.name,""",
        """                    '                    let _ = <%s as cs::Validate<_>>::validate(link, &chained);' % nt.name,""",
        'a_submit_whose_chained_counts_disagree_is_refused_before_the_driver',
    ),
    (
        'a validated struct skips its chain',
        'src/venus/cs.rs',
        """        T::validate_chain(self, facts)?;
""",
        """""",
        'a_submit_whose_chained_counts_disagree_is_refused_before_the_driver',
    ),
    (
        'a query pool of no queries reaches the driver',
        'src/venus/driver.rs',
        """        if this.queryCount == 0 {""",
        """        if this.queryCount == u32::MAX {""",
        'query_pools_and_timestamps_are_held_to_what_the_device_enabled',
    ),
    (
        'a query pool of any type reaches the driver',
        'src/venus/driver.rs',
        """        if !facts.query_types.contains(&this.queryType) {""",
        """        if facts.query_types.is_empty() {""",
        'query_pools_and_timestamps_are_held_to_what_the_device_enabled',
    ),
    (
        'a present-stage timestamp reaches the driver',
        'src/venus/driver.rs',
        """        if this.timeDomain == VkTimeDomainKHR::VK_TIME_DOMAIN_PRESENT_STAGE_LOCAL_EXT {""",
        """        if this.timeDomain == VkTimeDomainKHR::VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_KHR {""",
        'query_pools_and_timestamps_are_held_to_what_the_device_enabled',
    ),
    (
        'a chained statistics feature is not read',
        'src/venus/driver.rs',
        """    let chained = chained::<VkPhysicalDeviceFeatures2>(info).map(|f| f.get().features);""",
        """    let chained = chained::<VkPhysicalDeviceFeatures2>(info).map(|_| Default::default());""",
        'query_pools_and_timestamps_are_held_to_what_the_device_enabled',
    ),
    (
        'a region may name a color aspect on a depth or planar format',
        'src/venus/driver.rs',
        """        COLOR if depth.is_none() && !stencil && !planar => Ok((whole()?, w, h)),""",
        """        COLOR => Ok((whole()?, w, h)),""",
        'regions_are_held_inside_their_images_and_buffers',
    ),
    (
        'a region may name layers the image does not have',
        'src/venus/driver.rs',
        """            || sub.baseArrayLayer.checked_add(layers).is_none_or(|e| e > facts.array_layers)""",
        """            || sub.baseArrayLayer.checked_add(layers).is_none()""",
        'regions_are_held_inside_their_images_and_buffers',
    ),
    (
        'a region may name a mip level the image does not have',
        'src/venus/driver.rs',
        """        if sub.mipLevel >= facts.mip_levels {""",
        """        if sub.mipLevel == u32::MAX {""",
        'regions_are_held_inside_their_images_and_buffers',
    ),
    (
        'a region may lie outside its image',
        'src/venus/driver.rs',
        """        if !inside(offset.x, extent.width, w) || !inside(offset.y, extent.height, h) {""",
        """        if extent.width == 0 || extent.height == 0 {""",
        'regions_are_held_inside_their_images_and_buffers',
    ),
    (
        'a region may start inside a texel block',
        'src/venus/driver.rs',
        """        if !(offset.x as u32).is_multiple_of(self.block.width)
            || !(offset.y as u32).is_multiple_of(self.block.height)
        {""",
        """        if offset.x < 0 {""",
        'regions_are_held_inside_their_images_and_buffers',
    ),
    (
        'a flat image region may be deeper than one slice',
        'src/venus/driver.rs',
        """        } else if offset.z != 0 || extent.depth != flat_depth {""",
        """        } else if offset.z != 0 || extent.depth == 0 {""",
        'regions_are_held_inside_their_images_and_buffers',
    ),
    (
        'a buffer copy may run past the end of its buffer',
        'src/venus/driver.rs',
        """        if buffer_offset.0.checked_add(span).is_none_or(|end| end > len) {""",
        """        if buffer_offset.0 > len {""",
        'regions_are_held_inside_their_images_and_buffers',
    ),
    (
        'an image copy may name different layer counts',
        'src/venus/driver.rs',
        """        if !s.is_3d() && !d.is_3d() && s.layers != d.layers {""",
        """        if !s.is_3d() && !d.is_3d() && s.layers == u32::MAX {""",
        'regions_are_held_inside_their_images_and_buffers',
    ),
    (
        'a blit corner may lie outside its image',
        'src/venus/driver.rs',
        """        if within(corner.x, w) && within(corner.y, h) && within(corner.z, d) {""",
        """        if within(corner.x, u32::MAX) {""",
        'regions_are_held_inside_their_images_and_buffers',
    ),
    (
        'a blit may sample a compressed image',
        'src/venus/driver.rs',
        """            if p.block.width != 1
                || p.block.height != 1
                || formats::plane_of(p.image.format, 0).is_some()""",
        """            if formats::plane_of(p.image.format, 0).is_some()""",
        'regions_are_held_inside_their_images_and_buffers',
    ),
    (
        'a resolve may name a mode the runtime has no shader for',
        'src/venus/driver.rs',
        """        if !SERVED.contains(&this.resolveMode) || !SERVED.contains(&this.stencilResolveMode) {""",
        """        if this.resolveMode.0 == i32::MAX {""",
        'regions_are_held_inside_their_images_and_buffers',
    ),
    (
        'a copy image region reaches the driver unchecked',
        'src/venus/driver.rs',
        """                    r.extent,
                )
                .map_err(RecordRefused::Invalid)?;
        }
        let d = self.recorder(cb).ok_or(RecordRefused::NoDevice)?;
        // SAFETY: as above; the count is the slice's own length, and every region is held to
        // both images above.
        unsafe {
            (d.vkCmdCopyImage())(""",
        """                    r.extent,
                )
                .ok();
        }
        let d = self.recorder(cb).ok_or(RecordRefused::NoDevice)?;
        // SAFETY: as above; the count is the slice's own length, and every region is held to
        // both images above.
        unsafe {
            (d.vkCmdCopyImage())(""",
        'a_bad_region_stops_the_command_before_the_driver',
    ),
    (
        'an image view may carry the runtime internal flag',
        'src/venus/driver.rs',
        """        if this.flags.0 & !VIEW_FLAGS != 0 {""",
        """        if this.flags.0 & !VIEW_FLAGS == u32::MAX {""",
        'views_clears_and_host_copies_are_held_to_their_image',
    ),
    (
        'an image view may see an image as any type',
        'src/venus/driver.rs',
        """        if !typed {""",
        """        if false && !typed {""",
        'views_clears_and_host_copies_are_held_to_their_image',
    ),
    (
        'an image view may name aspects its image lacks',
        'src/venus/driver.rs',
        """        if r.aspectMask.0 == 0 || r.aspectMask.0 & !has != 0 {""",
        """        if r.aspectMask.0 == 0 {""",
        'views_clears_and_host_copies_are_held_to_their_image',
    ),
    (
        'an image view of 3D slices is held to array layers',
        'src/venus/driver.rs',
        """            (image.depth >> r.baseMipLevel).max(1)""",
        """            image.depth""",
        'views_clears_and_host_copies_are_held_to_their_image',
    ),
    (
        'an image view may hold any number of layers',
        'src/venus/driver.rs',
        """            T::VK_IMAGE_VIEW_TYPE_CUBE => layers == 6,""",
        """            T::VK_IMAGE_VIEW_TYPE_CUBE => layers > 0,""",
        'views_clears_and_host_copies_are_held_to_their_image',
    ),
    (
        'an image view may reinterpret an immutable image',
        'src/venus/driver.rs',
        """            if !flag(VkImageCreateFlagBits::VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT)
                || has & DEPTH_STENCIL != 0""",
        """            if has & DEPTH_STENCIL != 0""",
        'views_clears_and_host_copies_are_held_to_their_image',
    ),
    (
        'an image view may take any texel size',
        'src/venus/driver.rs',
        """            let compatible = if view == of_image {""",
        """            let compatible = if view.width == of_image.width {""",
        'views_clears_and_host_copies_are_held_to_their_image',
    ),
    (
        'a view minimum lod may run past its last level',
        'src/venus/driver.rs',
        """        if !lod.is_finite() || lod < 0.0 || lod > last as f32 {""",
        """        if lod < 0.0 {""",
        'views_clears_and_host_copies_are_held_to_their_image',
    ),
    (
        'a sliced view may name slices the image lacks',
        'src/venus/driver.rs',
        """        range_count(this.sliceOffset, this.sliceCount, depth)""",
        """        range_count(0, 1, depth)""",
        'views_clears_and_host_copies_are_held_to_their_image',
    ),
    (
        'a range may run past its limit',
        'src/venus/driver.rs',
        """    (count > 0 && base.checked_add(count)? <= limit).then_some(count)""",
        """    (count > 0).then_some(count)""",
        'views_clears_and_host_copies_are_held_to_their_image',
    ),
    (
        'a colour clear may name a compressed or planar image',
        'src/venus/driver.rs',
        """                    && formats::texel_block(facts.format)
                        .is_some_and(|b| b.width == 1 && b.height == 1)""",
        """                    && formats::texel_block(facts.format).is_some()""",
        'views_clears_and_host_copies_are_held_to_their_image',
    ),
    (
        'a depth clear may name aspects its image lacks',
        'src/venus/driver.rs',
        """            Clear::DepthStencil => aspect != 0 && aspect & !(has & DEPTH_STENCIL) == 0,""",
        """            Clear::DepthStencil => aspect != 0,""",
        'views_clears_and_host_copies_are_held_to_their_image',
    ),
    (
        'a colour clear range reaches the driver unchecked',
        'src/venus/driver.rs',
        """            facts.clear_range(image, r, Clear::Color).map_err(RecordRefused::Invalid)?;""",
        """            facts.clear_range(image, r, Clear::Color).ok();""",
        'a_bad_region_stops_the_command_before_the_driver',
    ),
    (
        'a disjoint image may be sized or bound whole',
        'src/venus/driver.rs',
        """        if self.facts().shape(image)?.disjoint {""",
        """        if self.facts().shape(image)?.disjoint && false {""",
        'a_disjoint_image_is_sized_and_bound_a_plane_at_a_time',
    ),
    (
        'a disjoint image size may name no plane',
        'src/venus/driver.rs',
        """        if shape.disjoint && chained::<VkImagePlaneMemoryRequirementsInfo>(this).is_none() {""",
        """        if false {""",
        'a_disjoint_image_is_sized_and_bound_a_plane_at_a_time',
    ),
    (
        'a disjoint image bind may name no plane',
        'src/venus/driver.rs',
        """        if shape.disjoint && chained::<VkBindImagePlaneMemoryInfo>(this).is_none() {""",
        """        if false {""",
        'a_disjoint_image_is_sized_and_bound_a_plane_at_a_time',
    ),
    (
        'a plane link may name a plane the format lacks',
        'src/venus/driver.rs',
        """            Some(i) if formats::plane_of(self.format, i).is_some() => Ok(()),""",
        """            Some(_) => Ok(()),""",
        'a_disjoint_image_is_sized_and_bound_a_plane_at_a_time',
    ),
    (
        'a subresource may name an aspect the image lacks',
        'src/venus/driver.rs',
        """        let one_of_its_own = aspect.count_ones() == 1 && aspect & !format_aspects(self.format) == 0;""",
        """        let one_of_its_own = aspect != 0;""",
        'a_disjoint_image_is_sized_and_bound_a_plane_at_a_time',
    ),
    (
        'a subresource may name a level or layer the image lacks',
        'src/venus/driver.rs',
        """        if sub.mipLevel >= self.mip_levels || sub.arrayLayer >= self.array_layers {""",
        """        if sub.mipLevel == u32::MAX {""",
        'a_disjoint_image_is_sized_and_bound_a_plane_at_a_time',
    ),
    (
        'the 1.0 image size forwards a disjoint image',
        'src/venus/context.rs',
        """        if let Err(why) = self.driver.image_is_whole(device, image) {""",
        """        if let Some(why) = None::<&'static str> {""",
        'a_disjoint_image_is_never_sized_or_bound_whole',
    ),
    (
        'the 1.0 image bind forwards a disjoint image',
        'src/venus/context.rs',
        """        if let Err(why) = self.driver.image_is_whole(args.device, args.image) {""",
        """        if let Some(why) = None::<&'static str> {""",
        'a_disjoint_image_is_never_sized_or_bound_whole',
    ),
    (
        'an image may be empty',
        'src/venus/driver.rs',
        """        if e.width == 0 || e.height == 0 || e.depth == 0 {""",
        """        if e.width == u32::MAX {""",
        'an_image_is_made_only_in_a_shape_the_device_has',
    ),
    (
        'a 1D image may have rows',
        'src/venus/driver.rs',
        """            T::VK_IMAGE_TYPE_1D if e.height == 1 && e.depth == 1 && !cube => l.maxImageDimension1D,""",
        """            T::VK_IMAGE_TYPE_1D => l.maxImageDimension1D,""",
        'an_image_is_made_only_in_a_shape_the_device_has',
    ),
    (
        'a 2D image may have slices',
        'src/venus/driver.rs',
        """            T::VK_IMAGE_TYPE_2D if e.depth == 1 && !cube => l.maxImageDimension2D,""",
        """            T::VK_IMAGE_TYPE_2D if !cube => l.maxImageDimension2D,""",
        'an_image_is_made_only_in_a_shape_the_device_has',
    ),
    (
        'a cube may be any shape',
        'src/venus/driver.rs',
        """            T::VK_IMAGE_TYPE_2D if e.depth == 1 && e.width == e.height && this.arrayLayers >= 6 => {""",
        """            T::VK_IMAGE_TYPE_2D if e.depth == 1 => {""",
        'an_image_is_made_only_in_a_shape_the_device_has',
    ),
    (
        'a 3D image may have layers',
        'src/venus/driver.rs',
        """            T::VK_IMAGE_TYPE_3D if this.arrayLayers == 1 && !cube => l.maxImageDimension3D,""",
        """            T::VK_IMAGE_TYPE_3D if !cube => l.maxImageDimension3D,""",
        'an_image_is_made_only_in_a_shape_the_device_has',
    ),
    (
        'an image may be larger than the device makes',
        'src/venus/driver.rs',
        """        if largest > max {""",
        """        if largest > u32::MAX - 1 {""",
        'an_image_is_made_only_in_a_shape_the_device_has',
    ),
    (
        'a 2D image may ask for 2D slices',
        'src/venus/driver.rs',
        """        if as_2d && this.imageType != T::VK_IMAGE_TYPE_3D {""",
        """        if as_2d && false {""",
        'an_image_is_made_only_in_a_shape_the_device_has',
    ),
    (
        'an image may have more layers than the device makes',
        'src/venus/driver.rs',
        """        if this.arrayLayers == 0 || this.arrayLayers > l.maxImageArrayLayers {""",
        """        if this.arrayLayers == 0 {""",
        'an_image_is_made_only_in_a_shape_the_device_has',
    ),
    (
        'an image may have more levels than its extent',
        'src/venus/driver.rs',
        """        if this.mipLevels == 0 || this.mipLevels > u32::BITS - largest.leading_zeros() {""",
        """        if this.mipLevels == 0 {""",
        'an_image_is_made_only_in_a_shape_the_device_has',
    ),
    (
        'an image may have any sample count',
        'src/venus/driver.rs',
        """        if !samples.is_power_of_two() || samples > 64 {""",
        """        if samples == 0 {""",
        'an_image_is_made_only_in_a_shape_the_device_has',
    ),
    (
        'a multisampled image may take any shape',
        'src/venus/driver.rs',
        """        if multisampled
""",
        """        if multisampled && false
""",
        'an_image_is_made_only_in_a_shape_the_device_has',
    ),
    (
        'a modifier image may be made on a device without them',
        'src/venus/driver.rs',
        """            if !facts.formats.takes_modifiers() {""",
        """            if false {""",
        'an_image_is_made_only_in_a_shape_the_device_has',
    ),
    (
        'a modifier image may choose its modifier both ways or neither',
        'src/venus/driver.rs',
        """            if listed == explicit {""",
        """            if listed && explicit {""",
        'an_image_is_made_only_in_a_shape_the_device_has',
    ),
    (
        'an explicit modifier may be one the driver does not offer',
        'src/venus/driver.rs',
        """            None => Err("named a modifier the device does not offer for the format"),""",
        """            None => Ok(()),""",
        'an_image_is_made_only_in_a_shape_the_device_has',
    ),
    (
        'an explicit modifier may lay out any number of planes',
        'src/venus/driver.rs',
        """            Some(n) if n != this.drmFormatModifierPlaneCount => {""",
        """            Some(n) if n == u32::MAX => {""",
        'an_image_is_made_only_in_a_shape_the_device_has',
    ),
    (
        'an image may replay capture data',
        'src/venus/driver.rs',
        """        if on.root.flags.0 & REPLAY != 0 {""",
        """        if on.root.flags.0 & REPLAY == u32::MAX {""",
        'an_image_is_made_only_in_a_shape_the_device_has',
    ),
    (
        'a device image query describes an image unchecked',
        'src/venus/driver.rs',
        """    unsafe { cs::Decoded::vouch(create) }.validate(facts)?;""",
        """    let _ = facts;""",
        'an_image_is_made_only_in_a_shape_the_device_has',
    ),
    (
        'a bind may run past the end of its memory',
        'src/venus/driver.rs',
        """        if offset >= have || size > have - offset {""",
        """        if offset > have {""",
        'a_bind_lies_inside_its_memory',
    ),
    (
        'an image may be bound at any offset',
        'src/venus/driver.rs',
        """        if r.alignment.0 != 0 && !offset.0.is_multiple_of(r.alignment.0) {""",
        """        if r.alignment.0 == u64::MAX {""",
        'a_bind_lies_inside_its_memory',
    ),
    (
        'an image bind is sized as a plane when it is whole',
        'src/venus/driver.rs',
        """                .filter(|_| disjoint)
""",
        """                .filter(|_| disjoint || true)
""",
        'a_bind_lies_inside_its_memory',
    ),
    (
        'the 1.0 image bind forwards past the end of its memory',
        'src/venus/context.rs',
        """        let fits = self.driver.image_bind_fits(
""",
        """        let fits = Ok::<(), &str>(());
        let _ = self.driver.image_bind_fits(
""",
        'a_disjoint_image_is_never_sized_or_bound_whole',
    ),
    (
        'a buffer view may start past the end of its buffer',
        'src/venus/driver.rs',
        """        if offset >= size {""",
        """        if offset > size {""",
        'a_buffer_view_lies_inside_its_buffer_in_a_texel_format',
    ),
    (
        'a buffer view may run past the end of its buffer',
        'src/venus/driver.rs',
        """            || range > size - offset
""",
        """            || range == u64::MAX - 1
""",
        'a_buffer_view_lies_inside_its_buffer_in_a_texel_format',
    ),
    (
        'a buffer view may hold part of a texel',
        'src/venus/driver.rs',
        """            || !range.is_multiple_of(u64::from(block.bytes))
""",
        """            || block.bytes == 0
""",
        'a_buffer_view_lies_inside_its_buffer_in_a_texel_format',
    ),
    (
        'a buffer view may be in a format with no texel buffer',
        'src/venus/driver.rs',
        """        if facts.formats.buffer_features(this.format) & TEXEL == 0 {""",
        """        if facts.formats.buffer_features(this.format) & TEXEL == u32::MAX {""",
        'a_buffer_view_lies_inside_its_buffer_in_a_texel_format',
    ),
    (
        'a buffer view may hold more texels than the device does',
        'src/venus/driver.rs',
        """        if range / u64::from(block.bytes) > u64::from(facts.limits.maxTexelBufferElements) {""",
        """        if range == u64::MAX {""",
        'a_buffer_view_lies_inside_its_buffer_in_a_texel_format',
    ),
    (
        'the modifier list is read without asking for its entries',
        'src/venus/driver.rs',
        """        list.pDrmFormatModifierProperties = all.as_mut_ptr();
""",
        '',
        'the_format_queries_ask_the_real_driver',
    ),
    (
        'an update may run on into a binding of another kind',
        'src/venus/driver.rs',
        """                if b != binding && !first.runs_into(here) {""",
        """                if false {""",
        'a_descriptor_update_is_held_to_its_set',
    ),
    (
        'an update may start past the end of its binding',
        'src/venus/driver.rs',
        """                if e > size {""",
        """                if e > size + 1 {""",
        'a_descriptor_update_is_held_to_its_set',
    ),
    (
        'a variable binding is as big as its layout says',
        'src/venus/driver.rs',
        """            let size = if here.variable() { self.variable } else { here.count };""",
        """            let size = here.count;""",
        'a_descriptor_update_is_held_to_its_set',
    ),
    (
        'a write may leave out the descriptors it writes',
        'src/venus/driver.rs',
        """        if n > 0 && !carried {""",
        """        if false {""",
        'a_descriptor_update_is_held_to_its_set',
    ),
    (
        'a write may put a descriptor where its binding holds another',
        'src/venus/driver.rs',
        """            if !binding.holds(ty) {""",
        """            if false {""",
        'a_descriptor_update_is_held_to_its_set',
    ),
    (
        'an inline block may be written in pieces smaller than a word',
        'src/venus/driver.rs',
        """                if !this.dstArrayElement.is_multiple_of(4) || !n.is_multiple_of(4) {""",
        """                if false {""",
        'a_descriptor_update_is_held_to_its_set',
    ),
    (
        'an inline block may carry another size than it says',
        'src/venus/driver.rs',
        """        if inline && this.dataSize != on.root.descriptorCount {""",
        """        if false {""",
        'a_descriptor_update_is_held_to_its_set',
    ),
    (
        'a structure write may carry fewer structures than it says',
        'src/venus/driver.rs',
        """        if structures && this.accelerationStructureCount != on.root.descriptorCount {""",
        """        if false {""",
        'a_descriptor_update_is_held_to_its_set',
    ),
    (
        'a buffer descriptor may describe any range',
        'src/venus/driver.rs',
        """            let inside =
                offset < size && (range == u64::MAX || (range > 0 && range <= size - offset));
""",
        """            let inside = true;
""",
        'a_descriptor_update_is_held_to_its_set',
    ),
    (
        'a copy may put descriptors into a binding of another kind',
        'src/venus/driver.rs',
        """            if x.ty != y.ty || x.mutable != y.mutable {""",
        """            if false {""",
        'a_descriptor_update_is_held_to_its_set',
    ),
    (
        'variable counts may be given for another number of sets',
        'src/venus/driver.rs',
        """            Some(_) => return Err("gave variable counts for a different number of sets"),""",
        """            Some(_) => &[][..],""",
        'a_descriptor_update_is_held_to_its_set',
    ),
    (
        'a set may be given more variable descriptors than its binding',
        'src/venus/driver.rs',
        """                if variable > room {""",
        """                if false {""",
        'a_descriptor_update_is_held_to_its_set',
    ),
    (
        'a layout may lay out one binding twice',
        'src/venus/driver.rs',
        """            if bindings.insert(b.binding, fact).is_some() {""",
        """            if bindings.insert(b.binding, fact).is_some() && false {""",
        'a_descriptor_update_is_held_to_its_set',
    ),
    (
        'an update may not skip over a binding of no descriptors',
        'src/venus/driver.rs',
        """            if size > 0 || b == binding {""",
        """            if true {""",
        'a_descriptor_update_is_held_to_its_set',
    ),
    (
        'sets may be bound past the last their pipeline layout lays out',
        'src/venus/driver.rs',
        """        if end > self.sets.len() as u64 {""",
        """        if end > u64::MAX - 1 {""",
        'sets_are_bound_and_pushed_through_their_pipeline_layout',
    ),
    (
        'a set may be bound where its pipeline layout lays out another',
        'src/venus/driver.rs',
        """            if *have.layout != **want {""",
        """            if false {""",
        'sets_are_bound_and_pushed_through_their_pipeline_layout',
    ),
    (
        'a bind may carry any number of dynamic offsets',
        'src/venus/driver.rs',
        """        if dynamic != offsets as u64 {""",
        """        if false {""",
        'sets_are_bound_and_pushed_through_their_pipeline_layout',
    ),
    (
        'a set may be pushed into a layout not for pushing',
        'src/venus/driver.rs',
        """        if !layout.push {""",
        """        if false {""",
        'sets_are_bound_and_pushed_through_their_pipeline_layout',
    ),
    (
        'a pipeline layout may name a set layout with no record',
        'src/venus/driver.rs',
        """        PipelineLayoutFacts::of(this, facts.facts.set_layouts)?;
""",
        """""",
        'sets_are_bound_and_pushed_through_their_pipeline_layout',
    ),
    (
        'a submit chained array is trusted by its count alone',
        'src/venus/driver.rs',
        """count == wanted && (wanted == 0 || !array.is_null())""",
        """count == wanted && (wanted == 0 || !array.is_null() || true)""",
        'a_submit_whose_chained_counts_disagree_is_refused_before_the_driver',
    ),
    (
        'a binary semaphore is owed a timeline value',
        'src/venus/driver.rs',
        """facts.semaphores.get(sem).is_none_or(|f| f.kind == SemaphoreKind::Timeline)""",
        """facts.semaphores.get(sem).is_none_or(|_| true)""",
        'a_submit_whose_chained_counts_disagree_is_refused_before_the_driver',
    ),
    (
        'a GLES 3.1 host tells the guest GLSL 310 as the C does',
        'src/vrend/caps.rs',
        """        Api::Gles(v) if v >= 31 => 330,""",
        """        Api::Gles(v) if v >= 31 => 310,""",
        'a_gles_3_1_host_offers_the_guest_glsl_3_30',
    ),
    (
        'a 24-bit depth readback drops the low byte GL gave it',
        'src/vrend/transfer.rs',
        """        *word = (((u * Z24_MAX + Z32_MAX / 2) / Z32_MAX) as u32).to_ne_bytes();""",
        """        *word = ((u >> 16 << 8) as u32).to_ne_bytes();""",
        'a_24_bit_depth_reads_back_as_written',
    ),
]

# Not here, and deliberately: "the ring loop never calls `wait_ring.changed()` after advancing the
# head". A wake lost inside `WaitRing` itself is an entry above, witnessed by a loom model; a call
# site in `run` that never makes the call is not. The loop reads the head from guest memory behind
# std atomics loom cannot see, so no model reaches it, and on hardware the waiter's stuck-log
# timeout doubles as a poll: the missing wake costs latency, not correctness -- the wait ends at
# half a second instead of at microseconds, and prints a line the C's own comment calls a frame
# stutter. A test for that is a stopwatch, and the two arrangements it needs are mutually
# exclusive: the head must advance while the waiter is already asleep, but a waiter that suspends
# before the guest has written is refused outright by the drained-and-short guard, which is correct
# and is itself under test. An entry that can only be caught by winning a race would report a hole
# on a loaded machine and coverage on a quiet one.

# Not here, and deliberately: "a free forgets to credit the ledger". There is no such line to
# break. A charge is a value held by the record of what it paid for, so crediting is that record
# going away -- the edit would have to delete the field, which is a different change. An entry
# that cannot be written because the bug cannot be written is the design working.


def run(cmd, cwd=RS, timeout=None, env=None):
    """Run a command, killing the whole process group if it outstays `timeout`.

    The group, not the child: `cargo test` spawns the test binary, and a sabotage that deadlocks
    leaves that binary wedged forever. Killing only cargo would orphan it, and the next run would
    contend with a process holding the same shared memory.

    A timeout is reported, never raised. It is a legitimate verdict here -- see `main`.
    """
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True, env=env,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.communicate()
        return SimpleNamespace(returncode=124, stdout='', stderr='', timed_out=True)
    return SimpleNamespace(returncode=proc.returncode, stdout=out, stderr=err, timed_out=False)


def command(filt):
    """What runs an entry's witness, as `(argv, env)`: one Kani proof, one loom model, or
    `cargo test` under a filter. `env` is None where the sweep's own environment is used."""
    if filt and filt.startswith('kani:'):
        return ['cargo', 'kani', '--harness', filt[len('kani:'):]], None
    if filt and filt.startswith('loom:'):
        env = dict(os.environ, RUSTFLAGS='--cfg loom', CARGO_TARGET_DIR=str(RS / 'target/loom'))
        return ['cargo', 'test', '--lib', filt[len('loom:'):]], env
    if filt and filt.startswith('doc:'):
        return ['cargo', 'test', '--doc', filt[len('doc:'):]], None
    return ['cargo', 'test'] + ([filt] if filt else []), None


def separate(filt):
    """Whether an entry's witness is one `cargo test` does not run, and so needs its own
    baseline and its own clock."""
    return bool(filt) and filt.startswith(('kani:', 'loom:', 'doc:'))


def main():
    patterns = sys.argv[1:]
    chosen = [s for s in SABOTAGES if not patterns or any(p in s[0] for p in patterns)]
    if not chosen:
        sys.exit('no sabotage matches %r' % patterns)

    dirty = run(['git', 'status', '--porcelain'], cwd=ROOT).stdout.strip()
    if dirty:
        sys.exit('the tree has uncommitted changes; sweep would restore over them:\n' + dirty)

    # Every target, before any building. The per-entry assert below is the guarantee that an edit
    # was made; this is what makes a refactor cost one message naming all of them rather than a
    # build's wait followed by an abort on the first, and another on the next.
    stale = [(n, r) for n, r, old, _, _ in SABOTAGES if old not in (ROOT / r).read_text()]
    if stale:
        sys.exit(
            'sabotage targets no longer in the tree -- fix or retire each:\n'
            + '\n'.join('  %s\n    %s' % (n, r) for n, r in stale)
        )

    started = time.monotonic()
    baseline = run(['cargo', 'test'])
    if baseline.returncode != 0:
        # The run's own words, since a baseline that fails only sometimes is gone by the time
        # anyone reruns it to ask why.
        sys.stderr.write(baseline.stdout + baseline.stderr)
        sys.exit('the tests do not pass before any sabotage; fix that first')
    # Derived from the clean run rather than fixed, so a slow machine is not called a hang and a
    # fast one still catches a wedge quickly. The floor covers a rebuild after each edit.
    budget = max(180.0, (time.monotonic() - started) * 8)
    # A proof or a model is its own baseline: `cargo test` never runs it, so one already failing
    # on the clean tree would read every sabotage aimed at it as caught. It is its own clock too:
    # it can take far longer than the suite, and held to the suite's budget it would be reported
    # as a hang -- caught -- without ever having decided.
    proof_budget = {}
    for filt in sorted({f for *_, f in chosen if separate(f)}):
        began = time.monotonic()
        argv, env = command(filt)
        if run(argv, env=env).returncode != 0:
            sys.exit('%s does not pass before any sabotage; fix that first' % filt)
        proof_budget[filt] = max(180.0, (time.monotonic() - began) * 3)

    holes = []
    for name, rel, old, new, filt in chosen:
        path = ROOT / rel
        original = path.read_text()
        assert old in original, 'sabotage %r no longer matches %s' % (name, rel)
        path.write_text(original.replace(old, new, 1))
        try:
            argv, env = command(filt)
            r = run(argv, timeout=proof_budget.get(filt, budget), env=env)
        finally:
            path.write_text(original)
        if r.timed_out:
            # Caught, and in the loudest way there is. Several of these sabotages delete a
            # deadlock guard, and a missing deadlock guard does not produce a wrong answer -- it
            # produces a wait that never ends. A sweep with no clock of its own hangs here rather
            # than reporting it, which is the instrument failing at exactly the cases it was
            # extended to cover.
            print('RED       %-58s the suite hung: nothing on that path ends on its own' % name)
            continue
        # A sabotaged tree that does not build fails exactly as a catch does, and is one only if
        # the compiler refused the defect rather than the sabotage's own spelling. An error
        # inside the replacement text is the entry being uncompilable -- a refactor renamed what
        # it names, and scored as caught it would read red forever while measuring nothing. An
        # error anywhere else is the type system refusing what the edit broke, which is the
        # catch this project prefers to a test.
        if 'error: could not compile' in r.stderr:
            start = original.index(old)
            first = original.count('\n', 0, start) + 1
            last = first + new.count('\n')
            at = re.findall(r'^\s*--> (\S+?):(\d+):\d+', r.stderr, re.M)
            if any(f == rel and first <= int(n) <= last for f, n in at):
                holes.append(name)
                print('BROKEN    %-58s the sabotage does not compile as written' % name)
            else:
                where = ', '.join(sorted({'%s:%s' % a for a in at})[:2]) or 'the build'
                print('RED       %-58s the build refused it: %s' % (name, where))
            continue
        if r.returncode == 0:
            holes.append(name)
            print('SURVIVED  %s' % name)
            continue
        # Which test noticed, not how many failed. `cargo test` lists the failures only when the
        # run finishes; a panic that aborts the binary ends it early, and then the name is in the
        # panic line instead. Reporting a count from whichever of those happened to be there is
        # how a sweep comes to claim coverage it cannot point at.
        named = re.findall(r"^    (\S+::\S+)$", r.stdout, re.M)
        if not named and filt and filt.startswith('doc:'):
            named = re.findall(r"^    (\S+\.rs - \S+ \(line \d+\))$", r.stdout, re.M)
        if not named and filt and filt.startswith('kani:'):
            named = ['%s: %s' % (filt, d) for d in
                     re.findall(r'Status: FAILURE\n\t - Description: "(.*)"', r.stdout)[:1]]
        if not named:
            named = re.findall(r"^thread '(\S+::\S+)'", r.stdout + r.stderr, re.M)[:1]
        witness = ', '.join(sorted(set(named))[:2]) if named else 'the test binary failed'
        more = len(set(named)) - 2
        print('RED       %-58s %s%s' % (name, witness, ' +%d more' % more if more > 0 else ''))

    print('\n%d of %d caught' % (len(chosen) - len(holes), len(chosen)))
    return 1 if holes else 0


if __name__ == '__main__':
    sys.exit(main())
