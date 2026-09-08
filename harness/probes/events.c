// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva
//
// The `events` group of src/venus/unserved.txt, as a program that names each command.
//
//   vkCreateEvent  vkDestroyEvent  vkCmdSetEvent  vkCmdResetEvent  vkCmdWaitEvents
//   vkCmdSetEvent2  vkCmdResetEvent2  vkCmdWaitEvents2
//
// Why a program and not a workload: the first unserved command poisons its context, so a boot
// discovers exactly one of the eight. This reaches them deliberately, one at a time, and says
// which one it was.
//
// It is self-checking. An event's state is readable from the host side with vkGetEventStatus --
// which this build already serves, against events it has no way to create -- so every command
// below has an observable consequence and none of them is scored by "it did not crash".
//
// Headless on purpose: no surface, no swapchain, no window. It runs over ssh.
//
//   cc -O1 -o events events.c -lvulkan && ./events
//
// Exit 0 = every check passed. Exit 1 = a check failed, and the line above it says which.
// Exit 2 = the device could not be brought up at all (not a verdict on the group).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

static int failures = 0;

// A check whose failure is a fact about one command, not a reason to stop: the point of the
// program is to reach all eight, so a failed check records itself and lets the rest run. The
// exception is a call that leaves us with nothing to test against, which uses `fatal`.
static void check(int ok, const char *what) {
    printf("%-46s %s\n", what, ok ? "ok" : "FAILED");
    if (!ok) {
        failures++;
    }
}

static void fatal(const char *what, VkResult r) {
    fprintf(stderr, "cannot test the group: %s returned %d\n", what, (int)r);
    exit(2);
}

// vkGetEventStatus answers with a state, not an error: VK_EVENT_SET and VK_EVENT_RESET are both
// ordinary. Anything else is the renderer failing to answer, and must not read as either state.
static const char *status_name(VkResult r) {
    switch (r) {
    case VK_EVENT_SET:
        return "SET";
    case VK_EVENT_RESET:
        return "RESET";
    default:
        return "neither (the device did not answer)";
    }
}

static void expect_status(VkDevice dev, VkEvent ev, VkResult want, const char *what) {
    VkResult got = vkGetEventStatus(dev, ev);
    if (got != want) {
        printf("%-46s FAILED (wanted %s, got %s)\n", what, status_name(want), status_name(got));
        failures++;
    } else {
        printf("%-46s ok\n", what);
    }
}

int main(void) {
    VkResult r;

    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "venus-events-probe",
        .apiVersion = VK_API_VERSION_1_3,
    };
    VkInstanceCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
    };
    VkInstance inst;
    if ((r = vkCreateInstance(&ici, NULL, &inst)) != VK_SUCCESS) {
        fatal("vkCreateInstance", r);
    }

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, NULL);
    if (n == 0) {
        fatal("vkEnumeratePhysicalDevices", VK_ERROR_INITIALIZATION_FAILED);
    }
    VkPhysicalDevice *pds = calloc(n, sizeof *pds);
    vkEnumeratePhysicalDevices(inst, &n, pds);
    VkPhysicalDevice pd = pds[0];

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pd, &props);
    printf("device: %s (Vulkan %u.%u)\n\n", props.deviceName,
           VK_API_VERSION_MAJOR(props.apiVersion), VK_API_VERSION_MINOR(props.apiVersion));

    // The `2` commands are core in 1.3. Below that they would need VK_KHR_synchronization2 and
    // its entry points; the guest device here is 1.3+, so the honest thing is to say we skipped
    // rather than to quietly test six of eight and exit 0.
    int have_sync2 = props.apiVersion >= VK_API_VERSION_1_3;

    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, NULL);
    VkQueueFamilyProperties *qs = calloc(qn, sizeof *qs);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qs);
    uint32_t qfam = UINT32_MAX;
    for (uint32_t i = 0; i < qn; i++) {
        if (qs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            qfam = i;
            break;
        }
    }
    if (qfam == UINT32_MAX) {
        fatal("no graphics queue family", VK_ERROR_INITIALIZATION_FAILED);
    }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = qfam,
        .queueCount = 1,
        .pQueuePriorities = &prio,
    };
    VkPhysicalDeviceSynchronization2Features sync2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
        .synchronization2 = VK_TRUE,
    };
    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = have_sync2 ? (void *)&sync2 : NULL,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qci,
    };
    VkDevice dev;
    if ((r = vkCreateDevice(pd, &dci, NULL, &dev)) != VK_SUCCESS) {
        fatal("vkCreateDevice", r);
    }
    VkQueue queue;
    vkGetDeviceQueue(dev, qfam, 0, &queue);

    VkCommandPoolCreateInfo pci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = qfam,
    };
    VkCommandPool pool;
    if ((r = vkCreateCommandPool(dev, &pci, NULL, &pool)) != VK_SUCCESS) {
        fatal("vkCreateCommandPool", r);
    }
    VkCommandBufferAllocateInfo cbai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cb;
    if ((r = vkAllocateCommandBuffers(dev, &cbai, &cb)) != VK_SUCCESS) {
        fatal("vkAllocateCommandBuffers", r);
    }

    // ---------------------------------------------------------------- vkCreateEvent
    //
    // The whole group hangs off this one: with no event to name, the seven below have nothing to
    // be tested against, and the three host-side commands this build already serves are
    // unreachable. So a failure here is fatal rather than counted.
    VkEventCreateInfo eci = {.sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO};
    VkEvent ev;
    if ((r = vkCreateEvent(dev, &eci, NULL, &ev)) != VK_SUCCESS) {
        fprintf(stderr,
                "vkCreateEvent returned %d.\n"
                "If this is the renderer refusing the command, the guest cannot say so: read the\n"
                "worker log for a `[virglrs] refused:` line, which is the only place it is named.\n",
                (int)r);
        exit(2);
    }
    printf("%-46s ok\n", "vkCreateEvent");

    // A fresh event is unsignalled, and this is also the first proof that the handle the renderer
    // gave back resolves to a real event on the host: vkGetEventStatus answers about the object,
    // and a device that does not have it answers VK_ERROR_INITIALIZATION_FAILED instead.
    expect_status(dev, ev, VK_EVENT_RESET, "a new event reads RESET");

    // ---------------------------------------------------------------- the host-side three
    //
    // Already served, and never before reachable. They are the oracle the command-buffer checks
    // below are read through, so they are checked first and on their own.
    check(vkSetEvent(dev, ev) == VK_SUCCESS, "vkSetEvent");
    expect_status(dev, ev, VK_EVENT_SET, "vkSetEvent left it SET");
    check(vkResetEvent(dev, ev) == VK_SUCCESS, "vkResetEvent");
    expect_status(dev, ev, VK_EVENT_RESET, "vkResetEvent left it RESET");

    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cb,
    };

    // ---------------------------------------------------------------- vkCmdSetEvent
    vkResetCommandBuffer(cb, 0);
    vkBeginCommandBuffer(cb, &bi);
    vkCmdSetEvent(cb, ev, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
    vkEndCommandBuffer(cb);
    check(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE) == VK_SUCCESS, "vkCmdSetEvent submitted");
    check(vkQueueWaitIdle(queue) == VK_SUCCESS, "the queue drained it");
    expect_status(dev, ev, VK_EVENT_SET, "vkCmdSetEvent left it SET");

    // ---------------------------------------------------------------- vkCmdResetEvent
    vkResetCommandBuffer(cb, 0);
    vkBeginCommandBuffer(cb, &bi);
    vkCmdResetEvent(cb, ev, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
    vkEndCommandBuffer(cb);
    check(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE) == VK_SUCCESS, "vkCmdResetEvent submitted");
    check(vkQueueWaitIdle(queue) == VK_SUCCESS, "the queue drained it");
    expect_status(dev, ev, VK_EVENT_RESET, "vkCmdResetEvent left it RESET");

    // ---------------------------------------------------------------- vkCmdWaitEvents
    //
    // The wait is satisfied from the host BEFORE the submit, so the queue drains without anyone
    // having to race it. A wait on an event nothing sets is a hang, which would score as a
    // timeout somewhere else and name nothing -- this asks the same question and can answer it.
    check(vkSetEvent(dev, ev) == VK_SUCCESS, "the event is set for the wait");
    vkResetCommandBuffer(cb, 0);
    vkBeginCommandBuffer(cb, &bi);
    VkMemoryBarrier mb = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT,
    };
    vkCmdWaitEvents(cb, 1, &ev, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 1, &mb, 0, NULL, 0, NULL);
    vkEndCommandBuffer(cb);
    check(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE) == VK_SUCCESS, "vkCmdWaitEvents submitted");
    check(vkQueueWaitIdle(queue) == VK_SUCCESS, "the wait was satisfied and the queue drained");

    // ---------------------------------------------------------------- the synchronization2 three
    if (!have_sync2) {
        printf("\nvkCmdSetEvent2 / vkCmdResetEvent2 / vkCmdWaitEvents2 SKIPPED: device is not 1.3\n");
    } else {
        VkMemoryBarrier2 mb2 = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT,
        };
        VkDependencyInfo dep = {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &mb2,
        };

        check(vkResetEvent(dev, ev) == VK_SUCCESS, "the event is reset for vkCmdSetEvent2");
        vkResetCommandBuffer(cb, 0);
        vkBeginCommandBuffer(cb, &bi);
        vkCmdSetEvent2(cb, ev, &dep);
        vkEndCommandBuffer(cb);
        check(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE) == VK_SUCCESS,
              "vkCmdSetEvent2 submitted");
        check(vkQueueWaitIdle(queue) == VK_SUCCESS, "the queue drained it");
        expect_status(dev, ev, VK_EVENT_SET, "vkCmdSetEvent2 left it SET");

        vkResetCommandBuffer(cb, 0);
        vkBeginCommandBuffer(cb, &bi);
        vkCmdResetEvent2(cb, ev, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
        vkEndCommandBuffer(cb);
        check(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE) == VK_SUCCESS,
              "vkCmdResetEvent2 submitted");
        check(vkQueueWaitIdle(queue) == VK_SUCCESS, "the queue drained it");
        expect_status(dev, ev, VK_EVENT_RESET, "vkCmdResetEvent2 left it RESET");

        check(vkSetEvent(dev, ev) == VK_SUCCESS, "the event is set for the wait2");
        vkResetCommandBuffer(cb, 0);
        vkBeginCommandBuffer(cb, &bi);
        vkCmdWaitEvents2(cb, 1, &ev, &dep);
        vkEndCommandBuffer(cb);
        check(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE) == VK_SUCCESS,
              "vkCmdWaitEvents2 submitted");
        check(vkQueueWaitIdle(queue) == VK_SUCCESS, "the wait2 was satisfied and the queue drained");
    }

    // ---------------------------------------------------------------- vkDestroyEvent
    //
    // Nothing observable follows a destroy -- asking about the handle afterwards is undefined, not
    // a check -- so what this scores is that the renderer accepted it and the device is still
    // usable after. A destroy that poisoned the context would fail the drain below.
    vkDestroyEvent(dev, ev, NULL);
    check(vkDeviceWaitIdle(dev) == VK_SUCCESS, "the device is still alive after vkDestroyEvent");

    vkDestroyCommandPool(dev, pool, NULL);
    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(inst, NULL);
    free(pds);
    free(qs);

    printf("\n%s\n", failures ? "FAILED" : "all checks passed");
    return failures ? 1 : 0;
}
