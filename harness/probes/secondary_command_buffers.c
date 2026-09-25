// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva
//
// The `secondary-command-buffers` group of src/venus/unserved.txt, as a program that names each
// command.
//
//   vkCmdExecuteCommands  vkTrimCommandPool
//
// Both are core: the first in Vulkan 1.0, the second in 1.1. Everything here is a transfer, so
// the secondaries inherit no render pass. Each secondary writes its own region of a zeroed,
// host-visible buffer with its own values:
//
//   A  vkCmdFillBuffer    region 0
//   B  vkCmdUpdateBuffer  region 1, a different word at every index
//   C  vkCmdFillBuffer    region 2
//   D  vkCmdFillBuffer    region 3, recorded only after the trim
//
// and a primary runs A and B in one vkCmdExecuteCommands, with a count of two so a dropped
// second element is visible, and C in a separate call. Region 3 is untouched until the end, so
// a write that lands in the wrong place shows there too.
//
// Then the buffer is zeroed from the host, B is reset with vkResetCommandBuffer and re-recorded
// with new values, C is re-recorded by beginning it again (the implicit reset a
// RESET_COMMAND_BUFFER pool allows), the primary is re-recorded, and everything is resubmitted.
// The renderer journals secondaries and has to drop what a re-recorded one used to hold, so the
// check is that the new values land and A, recorded once, still lands beside them.
//
// vkTrimCommandPool has no observable effect: it only returns unused memory. After spare
// buffers are allocated and freed, the pool is trimmed, and what is scored is that the pool
// still works afterwards -- D is allocated from it, recorded, and run by a new primary beside
// the surviving B. The trim itself is scored by the renderer log only.
//
// Headless on purpose: no surface, no swapchain, no window. It runs over ssh.
//
//   cc -O1 -o secondary_command_buffers secondary_command_buffers.c -lvulkan && ./secondary_command_buffers
//
// Exit 0 = every check passed. Exit 1 = a check failed, and the line above it says which.
// Exit 2 = the device could not be brought up at all (not a verdict on the group).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define REGIONS 4u
#define WORDS 16u
#define REGION_BYTES (WORDS * 4u)
#define BYTES (REGIONS * REGION_BYTES)

#define FILL_A 0xa0a0a0a0u
#define FILL_C 0xc2c2c2c2u
#define FILL_C2 0xc3c3c3c3u
#define FILL_D 0xd4d4d4d4u
#define UPDATE_B 0xb1000000u
#define UPDATE_B2 0xb2000000u

static int failures = 0;

static void check(int ok, const char *what) {
    printf("%-60s %s\n", what, ok ? "ok" : "FAILED");
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

static VkCommandBuffer allocate(VkDevice dev, VkCommandPool pool, VkCommandBufferLevel level) {
    VkCommandBufferAllocateInfo cbai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = pool,
        .level = level,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cb;
    VkResult r = vkAllocateCommandBuffers(dev, &cbai, &cb);
    if (r != VK_SUCCESS) {
        fatal("vkAllocateCommandBuffers", r);
    }
    return cb;
}

// A secondary that runs outside any render pass still has to name its inheritance.
static void begin_secondary(VkCommandBuffer cb) {
    VkCommandBufferInheritanceInfo inh = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO,
    };
    VkCommandBufferBeginInfo cbbi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pInheritanceInfo = &inh,
    };
    VkResult r = vkBeginCommandBuffer(cb, &cbbi);
    if (r != VK_SUCCESS) {
        fatal("vkBeginCommandBuffer (secondary)", r);
    }
}

static void end(VkCommandBuffer cb) {
    VkResult r = vkEndCommandBuffer(cb);
    if (r != VK_SUCCESS) {
        fatal("vkEndCommandBuffer", r);
    }
}

static void record_fill(VkCommandBuffer cb, VkBuffer buf, uint32_t region, uint32_t value) {
    begin_secondary(cb);
    vkCmdFillBuffer(cb, buf, region * REGION_BYTES, REGION_BYTES, value);
    end(cb);
}

static uint32_t update_word(uint32_t base, uint32_t i) {
    return base | (i * 0x0101u + 1u);
}

static void record_update(VkCommandBuffer cb, VkBuffer buf, uint32_t region, uint32_t base) {
    uint32_t words[WORDS];
    for (uint32_t i = 0; i < WORDS; i++) {
        words[i] = update_word(base, i);
    }
    begin_secondary(cb);
    vkCmdUpdateBuffer(cb, buf, region * REGION_BYTES, REGION_BYTES, words);
    end(cb);
}

// The primary runs each group of secondaries with one vkCmdExecuteCommands, then makes the
// transfers visible to the host.
static void record_primary(VkCommandBuffer cb, const VkCommandBuffer *first, uint32_t nfirst,
                           const VkCommandBuffer *second, uint32_t nsecond) {
    VkCommandBufferBeginInfo cbbi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    VkResult r = vkBeginCommandBuffer(cb, &cbbi);
    if (r != VK_SUCCESS) {
        fatal("vkBeginCommandBuffer (primary)", r);
    }
    vkCmdExecuteCommands(cb, nfirst, first);
    if (nsecond) {
        vkCmdExecuteCommands(cb, nsecond, second);
    }
    VkMemoryBarrier host = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
    };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
                         &host, 0, NULL, 0, NULL);
    end(cb);
}

static int submit(VkDevice dev, VkQueue queue, VkFence fence, VkCommandBuffer cb) {
    VkResult r = vkResetFences(dev, 1, &fence);
    if (r != VK_SUCCESS) {
        fatal("vkResetFences", r);
    }
    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cb,
    };
    if ((r = vkQueueSubmit(queue, 1, &si, fence)) != VK_SUCCESS) {
        fatal("vkQueueSubmit", r);
    }
    r = vkWaitForFences(dev, 1, &fence, VK_TRUE, 5ull * 1000 * 1000 * 1000);
    return r == VK_SUCCESS;
}

static uint32_t count_fill(const uint32_t *p, uint32_t region, uint32_t value) {
    uint32_t n = 0;
    for (uint32_t i = 0; i < WORDS; i++) {
        n += p[region * WORDS + i] == value;
    }
    return n;
}

static uint32_t count_update(const uint32_t *p, uint32_t region, uint32_t base) {
    uint32_t n = 0;
    for (uint32_t i = 0; i < WORDS; i++) {
        n += p[region * WORDS + i] == update_word(base, i);
    }
    return n;
}

int main(void) {
    VkResult r;

    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "venus-secondary-command-buffers-probe",
        .apiVersion = VK_API_VERSION_1_1,
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
    if (props.apiVersion < VK_API_VERSION_1_1) {
        fatal("Vulkan 1.1, where vkTrimCommandPool is core", VK_ERROR_FEATURE_NOT_PRESENT);
    }

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
        fatal("no graphics queue", VK_ERROR_FEATURE_NOT_PRESENT);
    }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = qfam,
        .queueCount = 1,
        .pQueuePriorities = &prio,
    };
    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qci,
    };
    VkDevice dev;
    if ((r = vkCreateDevice(pd, &dci, NULL, &dev)) != VK_SUCCESS) {
        fatal("vkCreateDevice", r);
    }
    VkQueue queue;
    vkGetDeviceQueue(dev, qfam, 0, &queue);

    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = BYTES,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VkBuffer buf;
    if ((r = vkCreateBuffer(dev, &bci, NULL, &buf)) != VK_SUCCESS) {
        fatal("vkCreateBuffer", r);
    }
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev, buf, &mr);
    uint32_t type = mem_type(pd, mr.memoryTypeBits,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (type == UINT32_MAX) {
        fatal("no host-visible coherent memory", VK_ERROR_INITIALIZATION_FAILED);
    }
    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mr.size,
        .memoryTypeIndex = type,
    };
    VkDeviceMemory mem;
    if ((r = vkAllocateMemory(dev, &mai, NULL, &mem)) != VK_SUCCESS) {
        fatal("vkAllocateMemory", r);
    }
    vkBindBufferMemory(dev, buf, mem, 0);
    uint32_t *p;
    if ((r = vkMapMemory(dev, mem, 0, VK_WHOLE_SIZE, 0, (void **)&p)) != VK_SUCCESS) {
        fatal("vkMapMemory", r);
    }
    memset(p, 0, BYTES);

    VkCommandPoolCreateInfo cpi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = qfam,
    };
    VkCommandPool pool;
    if ((r = vkCreateCommandPool(dev, &cpi, NULL, &pool)) != VK_SUCCESS) {
        fatal("vkCreateCommandPool", r);
    }
    VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    if ((r = vkCreateFence(dev, &fci, NULL, &fence)) != VK_SUCCESS) {
        fatal("vkCreateFence", r);
    }

    const VkCommandBufferLevel SECONDARY = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
    VkCommandBuffer primary = allocate(dev, pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
    VkCommandBuffer a = allocate(dev, pool, SECONDARY), b = allocate(dev, pool, SECONDARY),
                    c = allocate(dev, pool, SECONDARY);

    // Round 1: A and B in one call, C in another.
    record_fill(a, buf, 0, FILL_A);
    record_update(b, buf, 1, UPDATE_B);
    record_fill(c, buf, 2, FILL_C);
    VkCommandBuffer ab[2] = {a, b};
    record_primary(primary, ab, 2, &c, 1);
    int done = submit(dev, queue, fence, primary);
    check(done, "round 1: the submit completed");
    if (done) {
        uint32_t ra = count_fill(p, 0, FILL_A), rb = count_update(p, 1, UPDATE_B),
                 rc = count_fill(p, 2, FILL_C), rd = count_fill(p, 3, 0);
        check(ra == WORDS, "vkCmdExecuteCommands: first of two, a fill, landed");
        check(rb == WORDS, "vkCmdExecuteCommands: second of two, an update, landed");
        check(rc == WORDS, "vkCmdExecuteCommands: a second call's secondary landed");
        check(rd == WORDS, "vkCmdExecuteCommands: the untouched region stayed zero");
        printf("  A %u/%u, B %u/%u, C %u/%u, untouched %u/%u\n", ra, WORDS, rb, WORDS, rc, WORDS,
               rd, WORDS);
    }

    // Round 2: B reset explicitly, C reset by beginning it again, A left as it was.
    memset(p, 0, BYTES);
    if ((r = vkResetCommandBuffer(b, 0)) != VK_SUCCESS) {
        fatal("vkResetCommandBuffer", r);
    }
    record_update(b, buf, 1, UPDATE_B2);
    record_fill(c, buf, 2, FILL_C2);
    record_primary(primary, ab, 2, &c, 1);
    done = submit(dev, queue, fence, primary);
    check(done, "round 2: the submit completed");
    if (done) {
        uint32_t ra = count_fill(p, 0, FILL_A), rb = count_update(p, 1, UPDATE_B2),
                 rb_old = count_update(p, 1, UPDATE_B), rc = count_fill(p, 2, FILL_C2),
                 rc_old = count_fill(p, 2, FILL_C), rd = count_fill(p, 3, 0);
        check(ra == WORDS, "re-executed: the secondary recorded once still landed");
        check(rb == WORDS, "re-recorded after vkResetCommandBuffer: new values landed");
        check(rc == WORDS, "re-recorded by a second begin: the new value landed");
        check(rd == WORDS, "re-executed: the untouched region stayed zero");
        printf("  A %u/%u, B new %u/%u (old %u), C new %u/%u (old %u), untouched %u/%u\n", ra,
               WORDS, rb, WORDS, rb_old, rc, WORDS, rc_old, rd, WORDS);
    }

    // Trim: free A, C and spares, trim the pool, and show it still serves allocations and
    // recordings. The trim is scored by the renderer log only.
    VkCommandBuffer spares[4];
    VkCommandBufferAllocateInfo spai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = pool,
        .level = SECONDARY,
        .commandBufferCount = 4,
    };
    if ((r = vkAllocateCommandBuffers(dev, &spai, spares)) != VK_SUCCESS) {
        fatal("vkAllocateCommandBuffers (spares)", r);
    }
    for (int i = 0; i < 4; i++) {
        record_fill(spares[i], buf, 3, 0xeeeeeeeeu);
    }
    vkFreeCommandBuffers(dev, pool, 4, spares);
    VkCommandBuffer ac[2] = {a, c};
    vkFreeCommandBuffers(dev, pool, 2, ac);
    vkTrimCommandPool(dev, pool, 0);

    memset(p, 0, BYTES);
    VkCommandBuffer d = allocate(dev, pool, SECONDARY);
    record_fill(d, buf, 3, FILL_D);
    VkCommandBuffer bd[2] = {b, d};
    record_primary(primary, bd, 2, NULL, 0);
    done = submit(dev, queue, fence, primary);
    check(done, "after vkTrimCommandPool: the submit completed");
    if (done) {
        uint32_t ra = count_fill(p, 0, 0), rb = count_update(p, 1, UPDATE_B2),
                 rc = count_fill(p, 2, 0), rd = count_fill(p, 3, FILL_D);
        check(rb == WORDS, "after vkTrimCommandPool: the surviving secondary landed");
        check(rd == WORDS, "after vkTrimCommandPool: a newly allocated secondary landed");
        check(ra == WORDS && rc == WORDS, "after vkTrimCommandPool: freed secondaries stayed out");
        printf("  B %u/%u, D %u/%u, regions 0 and 2 zero %u/%u and %u/%u\n", rb, WORDS, rd, WORDS,
               ra, WORDS, rc, WORDS);
    }

    vkDestroyFence(dev, fence, NULL);
    vkDestroyCommandPool(dev, pool, NULL);
    vkUnmapMemory(dev, mem);
    vkDestroyBuffer(dev, buf, NULL);
    vkFreeMemory(dev, mem, NULL);
    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(inst, NULL);
    free(qs);
    free(pds);

    printf("\n%d check(s) failed\n", failures);
    return failures ? 1 : 0;
}
