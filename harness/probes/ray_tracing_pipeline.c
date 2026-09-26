// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva
//
// The `ray-tracing-pipeline` and `ray-tracing-maintenance1` groups of src/venus/unserved.txt, as a
// program that names their commands.
//
//   vkCreateRayTracingPipelinesKHR
//   vkGetRayTracingShaderGroupHandlesKHR
//   vkGetRayTracingShaderGroupStackSizeKHR
//   vkCmdSetRayTracingPipelineStackSizeKHR
//   vkCmdTraceRaysKHR
//   vkCmdTraceRaysIndirectKHR
//   vkCmdTraceRaysIndirect2KHR                     (VK_KHR_ray_tracing_maintenance1)
//   vkGetRayTracingCaptureReplayShaderGroupHandlesKHR is not named: it needs
//   `rayTracingPipelineShaderGroupHandleCaptureReplay`, which lavapipe does not report.
//
// From VK_KHR_ray_tracing_pipeline, which lavapipe advertises and neither KosmicKrisp nor anv on
// Ice Lake does, so the positive control and the venus run are both on lavapipe. No acceleration
// structure is traced against: the pipeline is two ray generation shaders, each writing one word
// per ray into a storage buffer, `TAG * 1000 + i + 1` for ray `i` of the launch grid, with TAG 1
// for the first group and 2 for the second.
//
// The handles are the load-bearing part. The guest reads them back through the renderer, writes
// one into a shader binding table, and the device picks the shader by the bytes it finds there --
// so a trace that writes the other group's tag, or nothing, is a handle that did not survive.
//
//   - The two groups' handles are written, nonzero and distinct.
//   - Group 0's ray generation shader has a stack size.
//   - vkCmdTraceRaysKHR over 4x3x2 with group 2's handle: every one of 24 words is 2000 + i + 1,
//     and the word past the grid is untouched.
//   - vkCmdTraceRaysIndirectKHR over a 3x2x1 grid read on the device, with group 1's handle.
//   - vkCmdTraceRaysIndirect2KHR over 5x1x1, tables and grid both read on the device, with
//     group 2's handle; only where the device reports `rayTracingPipelineTraceRaysIndirect2`.
//   - vkCmdSetRayTracingPipelineStackSizeKHR is recorded before each trace, the pipeline taking
//     its stack size dynamically; it changes nothing a word can show, so the worker log scores it.
//
// The shaders, compiled with `glslangValidator -V --target-env vulkan1.2 -S rgen -DTAG=<n>u -x`:
//
//   // tag.rgen
//   #version 460
//   #extension GL_EXT_ray_tracing : require
//   layout(set = 0, binding = 0) buffer Out { uint v[]; };
//   void main() {
//       uvec3 id = gl_LaunchIDEXT;
//       uvec3 n = gl_LaunchSizeEXT;
//       uint i = id.x + id.y * n.x + id.z * n.x * n.y;
//       v[i] = TAG * 1000u + i + 1u;
//   }
//
// Headless on purpose: no surface, no swapchain, no window. It runs over ssh.
//
//   cc -O1 -o ray_tracing_pipeline ray_tracing_pipeline.c -lvulkan && ./ray_tracing_pipeline
//
// Pick the device with MESA_VK_DEVICE_SELECT; the probe takes the first one the loader lists.
//
// Exit 0 = every check passed. Exit 1 = a check failed, and the line above it says which.
// Exit 2 = the device could not be brought up at all (not a verdict on the group).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define WORDS 64u
#define SENTINEL 0xa5a5a5a5u

static const uint32_t TAG1_RGEN[] = {
    0x07230203,0x00010500,0x0008000b,0x00000033,0x00000000,0x00020011,0x0000117f,0x0006000a,
    0x5f565053,0x5f52484b,0x5f796172,0x63617274,0x00676e69,0x0006000b,0x00000001,0x4c534c47,
    0x6474732e,0x3035342e,0x00000000,0x0003000e,0x00000000,0x00000001,0x0008000f,0x000014c1,
    0x00000004,0x6e69616d,0x00000000,0x0000000b,0x0000000e,0x00000029,0x00030003,0x00000002,
    0x000001cc,0x00060004,0x455f4c47,0x725f5458,0x745f7961,0x69636172,0x0000676e,0x00040005,
    0x00000004,0x6e69616d,0x00000000,0x00030005,0x00000009,0x00006469,0x00060005,0x0000000b,
    0x4c5f6c67,0x636e7561,0x45444968,0x00005458,0x00030005,0x0000000d,0x0000006e,0x00070005,
    0x0000000e,0x4c5f6c67,0x636e7561,0x7a695368,0x54584565,0x00000000,0x00030005,0x00000011,
    0x00000069,0x00030005,0x00000027,0x0074754f,0x00040006,0x00000027,0x00000000,0x00000076,
    0x00030005,0x00000029,0x00000000,0x00040047,0x0000000b,0x0000000b,0x000014c7,0x00040047,
    0x0000000e,0x0000000b,0x000014c8,0x00040047,0x00000026,0x00000006,0x00000004,0x00030047,
    0x00000027,0x00000002,0x00050048,0x00000027,0x00000000,0x00000023,0x00000000,0x00040047,
    0x00000029,0x00000021,0x00000000,0x00040047,0x00000029,0x00000022,0x00000000,0x00020013,
    0x00000002,0x00030021,0x00000003,0x00000002,0x00040015,0x00000006,0x00000020,0x00000000,
    0x00040017,0x00000007,0x00000006,0x00000003,0x00040020,0x00000008,0x00000007,0x00000007,
    0x00040020,0x0000000a,0x00000001,0x00000007,0x0004003b,0x0000000a,0x0000000b,0x00000001,
    0x0004003b,0x0000000a,0x0000000e,0x00000001,0x00040020,0x00000010,0x00000007,0x00000006,
    0x0004002b,0x00000006,0x00000012,0x00000000,0x0004002b,0x00000006,0x00000015,0x00000001,
    0x0004002b,0x00000006,0x0000001c,0x00000002,0x0003001d,0x00000026,0x00000006,0x0003001e,
    0x00000027,0x00000026,0x00040020,0x00000028,0x0000000c,0x00000027,0x0004003b,0x00000028,
    0x00000029,0x0000000c,0x00040015,0x0000002a,0x00000020,0x00000001,0x0004002b,0x0000002a,
    0x0000002b,0x00000000,0x0004002b,0x00000006,0x0000002d,0x000003e8,0x00040020,0x00000031,
    0x0000000c,0x00000006,0x00050036,0x00000002,0x00000004,0x00000000,0x00000003,0x000200f8,
    0x00000005,0x0004003b,0x00000008,0x00000009,0x00000007,0x0004003b,0x00000008,0x0000000d,
    0x00000007,0x0004003b,0x00000010,0x00000011,0x00000007,0x0004003d,0x00000007,0x0000000c,
    0x0000000b,0x0003003e,0x00000009,0x0000000c,0x0004003d,0x00000007,0x0000000f,0x0000000e,
    0x0003003e,0x0000000d,0x0000000f,0x00050041,0x00000010,0x00000013,0x00000009,0x00000012,
    0x0004003d,0x00000006,0x00000014,0x00000013,0x00050041,0x00000010,0x00000016,0x00000009,
    0x00000015,0x0004003d,0x00000006,0x00000017,0x00000016,0x00050041,0x00000010,0x00000018,
    0x0000000d,0x00000012,0x0004003d,0x00000006,0x00000019,0x00000018,0x00050084,0x00000006,
    0x0000001a,0x00000017,0x00000019,0x00050080,0x00000006,0x0000001b,0x00000014,0x0000001a,
    0x00050041,0x00000010,0x0000001d,0x00000009,0x0000001c,0x0004003d,0x00000006,0x0000001e,
    0x0000001d,0x00050041,0x00000010,0x0000001f,0x0000000d,0x00000012,0x0004003d,0x00000006,
    0x00000020,0x0000001f,0x00050084,0x00000006,0x00000021,0x0000001e,0x00000020,0x00050041,
    0x00000010,0x00000022,0x0000000d,0x00000015,0x0004003d,0x00000006,0x00000023,0x00000022,
    0x00050084,0x00000006,0x00000024,0x00000021,0x00000023,0x00050080,0x00000006,0x00000025,
    0x0000001b,0x00000024,0x0003003e,0x00000011,0x00000025,0x0004003d,0x00000006,0x0000002c,
    0x00000011,0x0004003d,0x00000006,0x0000002e,0x00000011,0x00050080,0x00000006,0x0000002f,
    0x0000002d,0x0000002e,0x00050080,0x00000006,0x00000030,0x0000002f,0x00000015,0x00060041,
    0x00000031,0x00000032,0x00000029,0x0000002b,0x0000002c,0x0003003e,0x00000032,0x00000030,
    0x000100fd,0x00010038
};

static const uint32_t TAG2_RGEN[] = {
    0x07230203,0x00010500,0x0008000b,0x00000033,0x00000000,0x00020011,0x0000117f,0x0006000a,
    0x5f565053,0x5f52484b,0x5f796172,0x63617274,0x00676e69,0x0006000b,0x00000001,0x4c534c47,
    0x6474732e,0x3035342e,0x00000000,0x0003000e,0x00000000,0x00000001,0x0008000f,0x000014c1,
    0x00000004,0x6e69616d,0x00000000,0x0000000b,0x0000000e,0x00000029,0x00030003,0x00000002,
    0x000001cc,0x00060004,0x455f4c47,0x725f5458,0x745f7961,0x69636172,0x0000676e,0x00040005,
    0x00000004,0x6e69616d,0x00000000,0x00030005,0x00000009,0x00006469,0x00060005,0x0000000b,
    0x4c5f6c67,0x636e7561,0x45444968,0x00005458,0x00030005,0x0000000d,0x0000006e,0x00070005,
    0x0000000e,0x4c5f6c67,0x636e7561,0x7a695368,0x54584565,0x00000000,0x00030005,0x00000011,
    0x00000069,0x00030005,0x00000027,0x0074754f,0x00040006,0x00000027,0x00000000,0x00000076,
    0x00030005,0x00000029,0x00000000,0x00040047,0x0000000b,0x0000000b,0x000014c7,0x00040047,
    0x0000000e,0x0000000b,0x000014c8,0x00040047,0x00000026,0x00000006,0x00000004,0x00030047,
    0x00000027,0x00000002,0x00050048,0x00000027,0x00000000,0x00000023,0x00000000,0x00040047,
    0x00000029,0x00000021,0x00000000,0x00040047,0x00000029,0x00000022,0x00000000,0x00020013,
    0x00000002,0x00030021,0x00000003,0x00000002,0x00040015,0x00000006,0x00000020,0x00000000,
    0x00040017,0x00000007,0x00000006,0x00000003,0x00040020,0x00000008,0x00000007,0x00000007,
    0x00040020,0x0000000a,0x00000001,0x00000007,0x0004003b,0x0000000a,0x0000000b,0x00000001,
    0x0004003b,0x0000000a,0x0000000e,0x00000001,0x00040020,0x00000010,0x00000007,0x00000006,
    0x0004002b,0x00000006,0x00000012,0x00000000,0x0004002b,0x00000006,0x00000015,0x00000001,
    0x0004002b,0x00000006,0x0000001c,0x00000002,0x0003001d,0x00000026,0x00000006,0x0003001e,
    0x00000027,0x00000026,0x00040020,0x00000028,0x0000000c,0x00000027,0x0004003b,0x00000028,
    0x00000029,0x0000000c,0x00040015,0x0000002a,0x00000020,0x00000001,0x0004002b,0x0000002a,
    0x0000002b,0x00000000,0x0004002b,0x00000006,0x0000002d,0x000007d0,0x00040020,0x00000031,
    0x0000000c,0x00000006,0x00050036,0x00000002,0x00000004,0x00000000,0x00000003,0x000200f8,
    0x00000005,0x0004003b,0x00000008,0x00000009,0x00000007,0x0004003b,0x00000008,0x0000000d,
    0x00000007,0x0004003b,0x00000010,0x00000011,0x00000007,0x0004003d,0x00000007,0x0000000c,
    0x0000000b,0x0003003e,0x00000009,0x0000000c,0x0004003d,0x00000007,0x0000000f,0x0000000e,
    0x0003003e,0x0000000d,0x0000000f,0x00050041,0x00000010,0x00000013,0x00000009,0x00000012,
    0x0004003d,0x00000006,0x00000014,0x00000013,0x00050041,0x00000010,0x00000016,0x00000009,
    0x00000015,0x0004003d,0x00000006,0x00000017,0x00000016,0x00050041,0x00000010,0x00000018,
    0x0000000d,0x00000012,0x0004003d,0x00000006,0x00000019,0x00000018,0x00050084,0x00000006,
    0x0000001a,0x00000017,0x00000019,0x00050080,0x00000006,0x0000001b,0x00000014,0x0000001a,
    0x00050041,0x00000010,0x0000001d,0x00000009,0x0000001c,0x0004003d,0x00000006,0x0000001e,
    0x0000001d,0x00050041,0x00000010,0x0000001f,0x0000000d,0x00000012,0x0004003d,0x00000006,
    0x00000020,0x0000001f,0x00050084,0x00000006,0x00000021,0x0000001e,0x00000020,0x00050041,
    0x00000010,0x00000022,0x0000000d,0x00000015,0x0004003d,0x00000006,0x00000023,0x00000022,
    0x00050084,0x00000006,0x00000024,0x00000021,0x00000023,0x00050080,0x00000006,0x00000025,
    0x0000001b,0x00000024,0x0003003e,0x00000011,0x00000025,0x0004003d,0x00000006,0x0000002c,
    0x00000011,0x0004003d,0x00000006,0x0000002e,0x00000011,0x00050080,0x00000006,0x0000002f,
    0x0000002d,0x0000002e,0x00050080,0x00000006,0x00000030,0x0000002f,0x00000015,0x00060041,
    0x00000031,0x00000032,0x00000029,0x0000002b,0x0000002c,0x0003003e,0x00000032,0x00000030,
    0x000100fd,0x00010038
};

static int failures = 0;

static void check(int ok, const char *what) {
    printf("%-64s %s\n", what, ok ? "ok" : "FAILED");
    if (!ok) {
        failures++;
    }
}

static void fatal(const char *what, VkResult r) {
    fprintf(stderr, "cannot test the group: %s returned %d\n", what, (int)r);
    exit(2);
}

static int has_ext(VkExtensionProperties *exts, uint32_t n, const char *name) {
    for (uint32_t i = 0; i < n; i++) {
        if (strcmp(exts[i].extensionName, name) == 0) {
            return 1;
        }
    }
    return 0;
}

static VkPhysicalDevice pd;
static VkDevice dev;
static VkQueue queue;
static VkCommandPool cmd_pool;

static PFN_vkVoidFunction need(const char *name) {
    PFN_vkVoidFunction f = vkGetDeviceProcAddr(dev, name);
    if (!f) {
        fprintf(stderr, "cannot test the group: %s did not resolve\n", name);
        exit(2);
    }
    return f;
}

// A host-visible buffer the device can also address, mapped for the whole run.
struct buffer {
    VkBuffer buf;
    VkDeviceMemory mem;
    VkDeviceAddress addr;
    uint8_t *p;
};

static struct buffer buffer(VkDeviceSize size, VkBufferUsageFlags usage) {
    struct buffer b;
    VkResult r;
    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
    };
    if ((r = vkCreateBuffer(dev, &bci, NULL, &b.buf)) != VK_SUCCESS) {
        fatal("vkCreateBuffer", r);
    }
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev, b.buf, &mr);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    VkMemoryPropertyFlags want =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((mr.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) {
            type = i;
            break;
        }
    }
    if (type == UINT32_MAX) {
        fatal("no host-visible coherent memory type", VK_ERROR_INITIALIZATION_FAILED);
    }
    VkMemoryAllocateFlagsInfo flags = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
    };
    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &flags,
        .allocationSize = mr.size,
        .memoryTypeIndex = type,
    };
    if ((r = vkAllocateMemory(dev, &mai, NULL, &b.mem)) != VK_SUCCESS) {
        fatal("vkAllocateMemory", r);
    }
    vkBindBufferMemory(dev, b.buf, b.mem, 0);
    if ((r = vkMapMemory(dev, b.mem, 0, VK_WHOLE_SIZE, 0, (void **)&b.p)) != VK_SUCCESS) {
        fatal("vkMapMemory", r);
    }
    memset(b.p, 0, size);
    VkBufferDeviceAddressInfo bai = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .buffer = b.buf,
    };
    b.addr = vkGetBufferDeviceAddress(dev, &bai);
    return b;
}

static void drop(struct buffer *b) {
    vkUnmapMemory(dev, b->mem);
    vkDestroyBuffer(dev, b->buf, NULL);
    vkFreeMemory(dev, b->mem, NULL);
}

static VkCommandBuffer begin(void) {
    VkCommandBufferAllocateInfo ai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cb;
    vkAllocateCommandBuffers(dev, &ai, &cb);
    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(cb, &bi);
    return cb;
}

static void submit(VkCommandBuffer cb) {
    VkMemoryBarrier mb = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
    };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
                         &mb, 0, NULL, 0, NULL);
    vkEndCommandBuffer(cb);
    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cb,
    };
    VkResult r = vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
    if (r != VK_SUCCESS) {
        fatal("vkQueueSubmit", r);
    }
    if ((r = vkQueueWaitIdle(queue)) != VK_SUCCESS) {
        fatal("vkQueueWaitIdle", r);
    }
    vkFreeCommandBuffers(dev, cmd_pool, 1, &cb);
}

// Whether the first `n` words are `tag`'s rays in order and every word after them is untouched.
static int traced(const uint32_t *words, uint32_t n, uint32_t tag) {
    for (uint32_t i = 0; i < WORDS; i++) {
        uint32_t want = i < n ? tag * 1000u + i + 1u : SENTINEL;
        if (words[i] != want) {
            printf("  word %u is %u, not %u\n", i, words[i], want);
            return 0;
        }
    }
    return 1;
}

static VkDeviceSize align_up(VkDeviceSize v, VkDeviceSize a) {
    return (v + a - 1) / a * a;
}

int main(void) {
    VkResult r;

    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "venus-ray-tracing-pipeline-probe",
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
    pd = pds[0];

    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtp = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR,
    };
    VkPhysicalDeviceProperties2 props = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &rtp,
    };
    vkGetPhysicalDeviceProperties2(pd, &props);
    printf("device: %s (Vulkan %u.%u)\n\n", props.properties.deviceName,
           VK_API_VERSION_MAJOR(props.properties.apiVersion),
           VK_API_VERSION_MINOR(props.properties.apiVersion));

    uint32_t en = 0;
    vkEnumerateDeviceExtensionProperties(pd, NULL, &en, NULL);
    VkExtensionProperties *exts = calloc(en, sizeof *exts);
    vkEnumerateDeviceExtensionProperties(pd, NULL, &en, exts);
    if (!has_ext(exts, en, "VK_KHR_ray_tracing_pipeline")) {
        fatal("VK_KHR_ray_tracing_pipeline", VK_ERROR_EXTENSION_NOT_PRESENT);
    }
    int maintenance1 = has_ext(exts, en, "VK_KHR_ray_tracing_maintenance1");

    VkPhysicalDeviceRayTracingMaintenance1FeaturesKHR m1f = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_MAINTENANCE_1_FEATURES_KHR,
    };
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtf = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
        .pNext = maintenance1 ? &m1f : NULL,
    };
    VkPhysicalDeviceAccelerationStructureFeaturesKHR asf = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
        .pNext = &rtf,
    };
    VkPhysicalDeviceVulkan12Features v12 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &asf,
    };
    VkPhysicalDeviceFeatures2 feats = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &v12,
    };
    vkGetPhysicalDeviceFeatures2(pd, &feats);
    if (!rtf.rayTracingPipeline || !rtf.rayTracingPipelineTraceRaysIndirect ||
        !asf.accelerationStructure || !v12.bufferDeviceAddress) {
        fatal("rayTracingPipeline, indirect traces, accelerationStructure, bufferDeviceAddress",
              VK_ERROR_FEATURE_NOT_PRESENT);
    }
    int indirect2 = maintenance1 && m1f.rayTracingPipelineTraceRaysIndirect2;

    VkPhysicalDeviceRayTracingMaintenance1FeaturesKHR m1f_on = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_MAINTENANCE_1_FEATURES_KHR,
        .rayTracingPipelineTraceRaysIndirect2 = indirect2 ? VK_TRUE : VK_FALSE,
    };
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtf_on = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
        .pNext = maintenance1 ? &m1f_on : NULL,
        .rayTracingPipeline = VK_TRUE,
        .rayTracingPipelineTraceRaysIndirect = VK_TRUE,
    };
    VkPhysicalDeviceAccelerationStructureFeaturesKHR asf_on = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
        .pNext = &rtf_on,
        .accelerationStructure = VK_TRUE,
    };
    VkPhysicalDeviceVulkan12Features v12_on = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &asf_on,
        .bufferDeviceAddress = VK_TRUE,
    };
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = 0,
        .queueCount = 1,
        .pQueuePriorities = &prio,
    };
    const char *want[] = {
        "VK_KHR_ray_tracing_pipeline",
        "VK_KHR_acceleration_structure",
        "VK_KHR_deferred_host_operations",
        "VK_KHR_ray_tracing_maintenance1",
    };
    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &v12_on,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qci,
        .enabledExtensionCount = maintenance1 ? 4 : 3,
        .ppEnabledExtensionNames = want,
    };
    if ((r = vkCreateDevice(pd, &dci, NULL, &dev)) != VK_SUCCESS) {
        fatal("vkCreateDevice", r);
    }
    vkGetDeviceQueue(dev, 0, 0, &queue);
    VkCommandPoolCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = 0,
    };
    vkCreateCommandPool(dev, &cpci, NULL, &cmd_pool);

    PFN_vkCreateRayTracingPipelinesKHR create_pipelines =
        (PFN_vkCreateRayTracingPipelinesKHR)need("vkCreateRayTracingPipelinesKHR");
    PFN_vkGetRayTracingShaderGroupHandlesKHR group_handles =
        (PFN_vkGetRayTracingShaderGroupHandlesKHR)need("vkGetRayTracingShaderGroupHandlesKHR");
    PFN_vkGetRayTracingShaderGroupStackSizeKHR stack_size =
        (PFN_vkGetRayTracingShaderGroupStackSizeKHR)need("vkGetRayTracingShaderGroupStackSizeKHR");
    PFN_vkCmdSetRayTracingPipelineStackSizeKHR set_stack_size =
        (PFN_vkCmdSetRayTracingPipelineStackSizeKHR)need("vkCmdSetRayTracingPipelineStackSizeKHR");
    PFN_vkCmdTraceRaysKHR trace = (PFN_vkCmdTraceRaysKHR)need("vkCmdTraceRaysKHR");
    PFN_vkCmdTraceRaysIndirectKHR trace_indirect =
        (PFN_vkCmdTraceRaysIndirectKHR)need("vkCmdTraceRaysIndirectKHR");
    PFN_vkCmdTraceRaysIndirect2KHR trace_indirect2 =
        indirect2 ? (PFN_vkCmdTraceRaysIndirect2KHR)need("vkCmdTraceRaysIndirect2KHR") : NULL;

    // One storage buffer, written by whichever ray generation shader runs.
    VkDescriptorSetLayoutBinding binding = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
    };
    VkDescriptorSetLayoutCreateInfo dslci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &binding,
    };
    VkDescriptorSetLayout set_layout;
    vkCreateDescriptorSetLayout(dev, &dslci, NULL, &set_layout);
    VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &set_layout,
    };
    VkPipelineLayout layout;
    vkCreatePipelineLayout(dev, &plci, NULL, &layout);

    VkShaderModule modules[2];
    const uint32_t *code[2] = {TAG1_RGEN, TAG2_RGEN};
    size_t sizes[2] = {sizeof TAG1_RGEN, sizeof TAG2_RGEN};
    VkPipelineShaderStageCreateInfo stages[2];
    VkRayTracingShaderGroupCreateInfoKHR groups[2];
    for (int i = 0; i < 2; i++) {
        VkShaderModuleCreateInfo smci = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = sizes[i],
            .pCode = code[i],
        };
        if ((r = vkCreateShaderModule(dev, &smci, NULL, &modules[i])) != VK_SUCCESS) {
            fatal("vkCreateShaderModule", r);
        }
        stages[i] = (VkPipelineShaderStageCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
            .module = modules[i],
            .pName = "main",
        };
        groups[i] = (VkRayTracingShaderGroupCreateInfoKHR){
            .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
            .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
            .generalShader = (uint32_t)i,
            .closestHitShader = VK_SHADER_UNUSED_KHR,
            .anyHitShader = VK_SHADER_UNUSED_KHR,
            .intersectionShader = VK_SHADER_UNUSED_KHR,
        };
    }
    VkDynamicState dynamic = VK_DYNAMIC_STATE_RAY_TRACING_PIPELINE_STACK_SIZE_KHR;
    VkPipelineDynamicStateCreateInfo dsci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 1,
        .pDynamicStates = &dynamic,
    };
    VkRayTracingPipelineCreateInfoKHR rtci = {
        .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
        .stageCount = 2,
        .pStages = stages,
        .groupCount = 2,
        .pGroups = groups,
        .maxPipelineRayRecursionDepth = 1,
        .pDynamicState = &dsci,
        .layout = layout,
    };
    VkPipeline pipeline;
    r = create_pipelines(dev, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &rtci, NULL, &pipeline);
    check(r == VK_SUCCESS, "a pipeline of two ray generation groups is created");
    if (r != VK_SUCCESS) {
        printf("\n%d check(s) failed\n", failures);
        return 1;
    }

    uint32_t handle_size = rtp.shaderGroupHandleSize;
    uint8_t *handles = malloc(2 * handle_size);
    memset(handles, 0, 2 * handle_size);
    r = group_handles(dev, pipeline, 0, 2, 2 * handle_size, handles);
    int nonzero[2] = {0, 0};
    for (uint32_t i = 0; i < 2 * handle_size; i++) {
        nonzero[i / handle_size] |= handles[i] != 0;
    }
    check(r == VK_SUCCESS && nonzero[0] && nonzero[1], "both groups' handles are written");
    check(memcmp(handles, handles + handle_size, handle_size) != 0, "and they differ");
    VkDeviceSize stack = stack_size(dev, pipeline, 0, VK_SHADER_GROUP_SHADER_GENERAL_KHR);
    check(stack > 0, "group 0's ray generation shader has a stack size");

    // The shader binding table: each group's handle at its own base-aligned record.
    VkDeviceSize stride = align_up(handle_size, rtp.shaderGroupHandleAlignment);
    VkDeviceSize base = rtp.shaderGroupBaseAlignment;
    struct buffer sbt = buffer(3 * base, VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR);
    VkDeviceAddress sbt_at = align_up(sbt.addr, base);
    for (int i = 0; i < 2; i++) {
        memcpy(sbt.p + (sbt_at - sbt.addr) + i * base, handles + i * handle_size, handle_size);
    }
    VkStridedDeviceAddressRegionKHR raygen[2] = {
        {.deviceAddress = sbt_at, .stride = stride, .size = stride},
        {.deviceAddress = sbt_at + base, .stride = stride, .size = stride},
    };
    VkStridedDeviceAddressRegionKHR none = {0};

    struct buffer out = buffer(WORDS * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    VkDescriptorPoolCreateInfo dpci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &ps,
    };
    VkDescriptorPool dpool;
    vkCreateDescriptorPool(dev, &dpci, NULL, &dpool);
    VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = dpool,
        .descriptorSetCount = 1,
        .pSetLayouts = &set_layout,
    };
    VkDescriptorSet set;
    vkAllocateDescriptorSets(dev, &dsai, &set);
    VkDescriptorBufferInfo dbi = {out.buf, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet wds = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = set,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &dbi,
    };
    vkUpdateDescriptorSets(dev, 1, &wds, 0, NULL);

    // The grids the indirect traces read, and the maintenance1 form with its tables beside them.
    struct buffer grid = buffer(sizeof(VkTraceRaysIndirectCommand2KHR),
                                VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);

    const uint32_t *words = (const uint32_t *)out.p;
    for (int run = 0; run < 3; run++) {
        if (run == 2 && !indirect2) {
            printf("%-64s %s\n", "vkCmdTraceRaysIndirect2KHR",
                   "skipped: rayTracingPipelineTraceRaysIndirect2");
            break;
        }
        memset(out.p, 0xa5, WORDS * 4);
        VkCommandBuffer cb = begin();
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, layout, 0, 1, &set, 0,
                                NULL);
        set_stack_size(cb, (uint32_t)stack);
        if (run == 0) {
            trace(cb, &raygen[1], &none, &none, &none, 4, 3, 2);
        } else if (run == 1) {
            VkTraceRaysIndirectCommandKHR g = {3, 2, 1};
            memcpy(grid.p, &g, sizeof g);
            trace_indirect(cb, &raygen[0], &none, &none, &none, grid.addr);
        } else {
            VkTraceRaysIndirectCommand2KHR g = {
                .raygenShaderRecordAddress = raygen[1].deviceAddress,
                .raygenShaderRecordSize = raygen[1].size,
                .width = 5,
                .height = 1,
                .depth = 1,
            };
            memcpy(grid.p, &g, sizeof g);
            trace_indirect2(cb, grid.addr);
        }
        submit(cb);
        if (run == 0) {
            check(traced(words, 24, 2), "vkCmdTraceRaysKHR: 4x3x2 rays of group 2");
        } else if (run == 1) {
            check(traced(words, 6, 1), "vkCmdTraceRaysIndirectKHR: 3x2x1 rays of group 1");
        } else {
            check(traced(words, 5, 2), "vkCmdTraceRaysIndirect2KHR: 5x1x1 rays of group 2");
        }
    }

    vkDestroyPipeline(dev, pipeline, NULL);
    for (int i = 0; i < 2; i++) {
        vkDestroyShaderModule(dev, modules[i], NULL);
    }
    vkDestroyDescriptorPool(dev, dpool, NULL);
    vkDestroyPipelineLayout(dev, layout, NULL);
    vkDestroyDescriptorSetLayout(dev, set_layout, NULL);
    drop(&grid);
    drop(&out);
    drop(&sbt);
    free(handles);
    vkDestroyCommandPool(dev, cmd_pool, NULL);
    vkDestroyDevice(dev, NULL);
    free(exts);
    free(pds);
    vkDestroyInstance(inst, NULL);

    printf("\n%d check(s) failed\n", failures);
    return failures ? 1 : 0;
}
