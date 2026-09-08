// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva
//
// The `timestamps` group of src/venus/unserved.txt, as a program that names each command.
//
//   vkCmdWriteTimestamp2  vkGetCalibratedTimestampsKHR
//
// `vkGetCalibratedTimestampsEXT` is not a third command: both spellings encode to venus command
// type 236, so the EXT name a guest calls and the KHR name the ledger carries are one line.
//
// It also measures the 1.0 `vkCmdWriteTimestamp`, which this build *does* serve. That is not a
// check and does not count towards the exit status -- it is here because a synoik session
// reported two timestamps in one submit resolving to the same tick, and the question that
// separates a renderer bug from a driver limitation is whether the host does the same thing with
// no renderer in the way. Run this against KosmicKrisp directly and the answer is a fact about
// the host; run it through venus and the difference, if any, is ours. So the delta is printed and
// not judged: a probe that decided "0 is wrong" would be asserting the host's granularity.
//
// Headless on purpose: no surface, no swapchain, no window. It runs over ssh.
//
//   cc -O1 -o timestamps timestamps.c -lvulkan && ./timestamps
//
// Exit 0 = every check passed. Exit 1 = a check failed, and the line above it says which.
// Exit 2 = the device could not be brought up at all (not a verdict on the group).

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

// Big enough that the fill between two stamps is real work: 205 us of it was reported on this
// host before the behaviour changed, against 7 us for an empty gap.
#define FILL_BYTES (16u * 1024u * 1024u)

static int failures = 0;

static void check(int ok, const char *what) {
    printf("%-52s %s\n", what, ok ? "ok" : "FAILED");
    if (!ok) {
        failures++;
    }
}

static void fatal(const char *what, VkResult r) {
    fprintf(stderr, "cannot test the group: %s returned %d\n", what, (int)r);
    exit(2);
}

static uint32_t mem_type(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) {
            return i;
        }
    }
    return UINT32_MAX;
}

int main(void) {
    VkResult r;

    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "venus-timestamps-probe",
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
    printf("device: %s (Vulkan %u.%u)\n", props.deviceName,
           VK_API_VERSION_MAJOR(props.apiVersion), VK_API_VERSION_MINOR(props.apiVersion));
    // Reported, never asserted on. `timestampPeriod` is the host driver's to choose and this
    // renderer forwards it unchanged, so a change here is a fact about the driver.
    printf("timestampPeriod: %.6f ns per tick\n", props.limits.timestampPeriod);

    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, NULL);
    VkQueueFamilyProperties *qs = calloc(qn, sizeof *qs);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qs);
    uint32_t qfam = UINT32_MAX;
    for (uint32_t i = 0; i < qn; i++) {
        if ((qs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && qs[i].timestampValidBits > 0) {
            qfam = i;
            break;
        }
    }
    if (qfam == UINT32_MAX) {
        fatal("no queue family with timestamp support", VK_ERROR_FEATURE_NOT_PRESENT);
    }
    printf("timestampValidBits: %u (queue family %u)\n\n", qs[qfam].timestampValidBits, qfam);

    // The calibrated-timestamps entry point is an extension either way. Take whichever spelling
    // the device advertises -- they are the same command on the wire.
    uint32_t en = 0;
    vkEnumerateDeviceExtensionProperties(pd, NULL, &en, NULL);
    VkExtensionProperties *exts = calloc(en, sizeof *exts);
    vkEnumerateDeviceExtensionProperties(pd, NULL, &en, exts);
    const char *calib = NULL;
    for (uint32_t i = 0; i < en; i++) {
        if (strcmp(exts[i].extensionName, "VK_KHR_calibrated_timestamps") == 0) {
            calib = "VK_KHR_calibrated_timestamps";
            break;
        }
        if (strcmp(exts[i].extensionName, "VK_EXT_calibrated_timestamps") == 0) {
            calib = "VK_EXT_calibrated_timestamps";
        }
    }
    printf("calibrated timestamps: %s\n\n", calib ? calib : "not advertised");

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
        .pNext = &sync2,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qci,
        .enabledExtensionCount = calib ? 1u : 0u,
        .ppEnabledExtensionNames = calib ? &calib : NULL,
    };
    VkDevice dev;
    if ((r = vkCreateDevice(pd, &dci, NULL, &dev)) != VK_SUCCESS) {
        fatal("vkCreateDevice", r);
    }
    VkQueue queue;
    vkGetDeviceQueue(dev, qfam, 0, &queue);

    VkQueryPoolCreateInfo qpci = {
        .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .queryType = VK_QUERY_TYPE_TIMESTAMP,
        .queryCount = 2,
    };
    VkQueryPool qp;
    if ((r = vkCreateQueryPool(dev, &qpci, NULL, &qp)) != VK_SUCCESS) {
        fatal("vkCreateQueryPool", r);
    }

    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = FILL_BYTES,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VkBuffer fillbuf;
    if ((r = vkCreateBuffer(dev, &bci, NULL, &fillbuf)) != VK_SUCCESS) {
        fatal("vkCreateBuffer", r);
    }
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev, fillbuf, &mr);
    uint32_t type = mem_type(pd, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (type == UINT32_MAX) {
        type = mem_type(pd, mr.memoryTypeBits, 0);
    }
    if (type == UINT32_MAX) {
        fatal("no memory type for the fill buffer", VK_ERROR_INITIALIZATION_FAILED);
    }
    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mr.size,
        .memoryTypeIndex = type,
    };
    VkDeviceMemory fillmem;
    if ((r = vkAllocateMemory(dev, &mai, NULL, &fillmem)) != VK_SUCCESS) {
        fatal("vkAllocateMemory", r);
    }
    vkBindBufferMemory(dev, fillbuf, fillmem, 0);

    VkCommandPoolCreateInfo cpi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = qfam,
    };
    VkCommandPool cmdpool;
    if ((r = vkCreateCommandPool(dev, &cpi, NULL, &cmdpool)) != VK_SUCCESS) {
        fatal("vkCreateCommandPool", r);
    }
    VkCommandBufferAllocateInfo cbai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = cmdpool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cb;
    if ((r = vkAllocateCommandBuffers(dev, &cbai, &cb)) != VK_SUCCESS) {
        fatal("vkAllocateCommandBuffers", r);
    }
    VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    if ((r = vkCreateFence(dev, &fci, NULL, &fence)) != VK_SUCCESS) {
        fatal("vkCreateFence", r);
    }

    // One submit: reset the pool, stamp, fill 16 MiB, stamp. `use2` picks which entry point
    // writes the stamps; everything else about the two runs is identical, which is what makes
    // their deltas comparable.
    uint64_t stamps[2];
    for (int use2 = 0; use2 <= 1; use2++) {
        vkResetCommandBuffer(cb, 0);
        VkCommandBufferBeginInfo cbbi = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        vkBeginCommandBuffer(cb, &cbbi);
        vkCmdResetQueryPool(cb, qp, 0, 2);
        if (use2) {
            vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, qp, 0);
        } else {
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
        }
        vkCmdFillBuffer(cb, fillbuf, 0, FILL_BYTES, 0xa5a5a5a5u);
        if (use2) {
            vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, qp, 1);
        } else {
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
        }
        vkEndCommandBuffer(cb);

        vkResetFences(dev, 1, &fence);
        VkSubmitInfo si = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &cb,
        };
        if ((r = vkQueueSubmit(queue, 1, &si, fence)) != VK_SUCCESS) {
            fatal("vkQueueSubmit", r);
        }
        r = vkWaitForFences(dev, 1, &fence, VK_TRUE, 5ull * 1000 * 1000 * 1000);
        if (r != VK_SUCCESS) {
            // A stamp that never lands is the whole result for this arm; do not read the pool.
            if (use2) {
                check(0, "vkCmdWriteTimestamp2: the submit completed");
            } else {
                printf("%-52s the submit did not complete\n", "vkCmdWriteTimestamp (1.0, served):");
            }
            continue;
        }

        memset(stamps, 0, sizeof stamps);
        r = vkGetQueryPoolResults(dev, qp, 0, 2, sizeof stamps, stamps, sizeof(uint64_t),
                                  VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        uint64_t delta = stamps[1] - stamps[0];
        double ns = (double)delta * (double)props.limits.timestampPeriod;

        if (use2) {
            check(r == VK_SUCCESS, "vkCmdWriteTimestamp2: the queries resolved");
            // A stamp of zero is what an unwritten query reads as, so a nonzero pair is the
            // consequence that says the command reached the device at all.
            check(stamps[0] != 0 || stamps[1] != 0,
                  "vkCmdWriteTimestamp2: the stamps are not both zero");
            printf("  ticks %" PRIu64 " -> %" PRIu64 " (delta %" PRIu64 ", %.3f us)\n", stamps[0],
                   stamps[1], delta, ns / 1000.0);
        } else {
            // Reported, not checked: see the header. Zero here is a fact about how finely the
            // host samples, and this program is not entitled to a verdict on that.
            printf("%-52s %s\n", "vkCmdWriteTimestamp (1.0, served, not a check):",
                   r == VK_SUCCESS ? "resolved" : "did not resolve");
            printf("  ticks %" PRIu64 " -> %" PRIu64 " (delta %" PRIu64 ", %.3f us) over a %u MiB "
                   "fill\n\n",
                   stamps[0], stamps[1], delta, ns / 1000.0, FILL_BYTES / (1024u * 1024u));
        }
    }

    // ---- vkGetCalibratedTimestampsKHR / EXT ---------------------------------------------
    if (!calib) {
        printf("%-52s skipped (extension not advertised)\n", "vkGetCalibratedTimestamps:");
    } else {
        PFN_vkGetCalibratedTimestampsEXT get =
            (PFN_vkGetCalibratedTimestampsEXT)vkGetDeviceProcAddr(dev,
                                                                  "vkGetCalibratedTimestampsEXT");
        if (!get) {
            get = (PFN_vkGetCalibratedTimestampsEXT)vkGetDeviceProcAddr(
                dev, "vkGetCalibratedTimestampsKHR");
        }
        if (!get) {
            check(0, "vkGetCalibratedTimestamps: entry point resolved");
        } else {
            VkCalibratedTimestampInfoEXT info = {
                .sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT,
                .timeDomain = VK_TIME_DOMAIN_DEVICE_EXT,
            };
            uint64_t ts = 0, dev_deviation = 0;
            r = get(dev, 1, &info, &ts, &dev_deviation);
            check(r == VK_SUCCESS, "vkGetCalibratedTimestamps returned VK_SUCCESS");
            // Same trap as every create in this directory: a zeroed reply is shaped like a
            // successful one, so VK_SUCCESS alone says nothing. A device timestamp of zero is
            // what a refusal leaves behind and what a working device essentially never returns.
            check(ts != 0, "the device timestamp it wrote is not zero");
            printf("  device timestamp %" PRIu64 ", max deviation %" PRIu64 "\n", ts,
                   dev_deviation);
        }
    }

    vkDestroyFence(dev, fence, NULL);
    vkDestroyCommandPool(dev, cmdpool, NULL);
    vkDestroyBuffer(dev, fillbuf, NULL);
    vkFreeMemory(dev, fillmem, NULL);
    vkDestroyQueryPool(dev, qp, NULL);
    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(inst, NULL);

    printf("\n%d check(s) failed\n", failures);
    return failures ? 1 : 0;
}
