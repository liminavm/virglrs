// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva
//
// The `indexed-and-indirect-draw` group of src/venus/unserved.txt, as a program that names each
// command.
//
//   vkCmdDrawIndexed          vkCmdDrawIndirect          vkCmdDrawIndexedIndirect
//   vkCmdDrawIndirectCount    vkCmdDrawIndexedIndirectCount
//   vkCmdDispatchIndirect     vkCmdDispatchBase          vkCmdBindIndexBuffer2
//
// Core from 1.0 (the plain indexed and indirect ones) to 1.4 (vkCmdBindIndexBuffer2), so every
// Vulkan 1.4 device offers them all.
//
// The draws render points into a 16x8 R32_UINT target cleared to 0. Vertex v lands on pixel
// (v % 16, v / 16) and writes its row + 1, so every row belongs to exactly one command:
//
//   row 0  vkCmdDrawIndexed, indices 0..15 through the 1.0 index bind
//   row 1  vkCmdDrawIndexed again, indices 16..31 through vkCmdBindIndexBuffer2 at an offset
//   row 2  vkCmdDrawIndirect, first vertex 32
//   row 3  vkCmdDrawIndexedIndirect, indices 16..31 with vertex offset 32
//   row 4  vkCmdDrawIndirectCount, first vertex 64
//   row 5  the second record of that count draw, which the count buffer's 1 must leave undrawn
//   row 6  vkCmdDrawIndexedIndirectCount, indices 16..31 with vertex offset 80
//   row 7  the second record of that count draw, likewise undrawn
//
// A dropped command leaves its row clear; a dropped vkCmdBindIndexBuffer2 draws row 1's points
// over row 0 instead; a count taken from maxDrawCount rather than the buffer draws row 5 or 7.
//
// The dispatches write a push-constant tag into a storage buffer, one invocation per element:
// vkCmdDispatchIndirect covers elements 0..7 with tag 1, vkCmdDispatchBase elements 8..15 with
// tag 2, and 16..31 stay 0. A base of zero writes tag 2 over tag 1's elements instead.
//
// The shaders, compiled with `glslangValidator -V --target-env vulkan1.1 -x`:
//
//   // rows.vert
//   #version 450
//   layout(location = 0) flat out uint tag;
//   void main() {
//       uint v = uint(gl_VertexIndex);
//       uint x = v % 16u;
//       uint y = v / 16u;
//       gl_Position = vec4((float(x) + 0.5) / 8.0 - 1.0, (float(y) + 0.5) / 4.0 - 1.0, 0.0, 1.0);
//       gl_PointSize = 1.0;
//       tag = y + 1u;
//   }
//
//   // tag.frag
//   #version 450
//   layout(location = 0) flat in uint tag;
//   layout(location = 0) out uint o;
//   void main() {
//       o = tag;
//   }
//
//   // tag.comp
//   #version 450
//   layout(local_size_x = 1) in;
//   layout(set = 0, binding = 0) writeonly buffer Out { uint o[]; };
//   layout(push_constant) uniform PC { uint tag; } pc;
//   void main() {
//       o[gl_GlobalInvocationID.x] = pc.tag;
//   }
//
// Headless on purpose: no surface, no swapchain, no window. It runs over ssh.
//
//   cc -O1 -o indexed_and_indirect_draw indexed_and_indirect_draw.c -lvulkan
//   ./indexed_and_indirect_draw
//
// Exit 0 = every check passed. Exit 1 = a check failed, and the line above it says which.
// Exit 2 = the device could not be brought up at all (not a verdict on the group).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define W 16u
#define H 8u
#define ELEMS 32u

static const uint32_t VERT[] = {
    0x07230203,0x00010300,0x0008000b,0x00000036,0x00000000,0x00020011,0x00000001,0x0006000b,
    0x00000001,0x4c534c47,0x6474732e,0x3035342e,0x00000000,0x0003000e,0x00000000,0x00000001,
    0x0008000f,0x00000000,0x00000004,0x6e69616d,0x00000000,0x0000000b,0x0000001b,0x00000033,
    0x00030003,0x00000002,0x000001c2,0x00040005,0x00000004,0x6e69616d,0x00000000,0x00030005,
    0x00000008,0x00000076,0x00060005,0x0000000b,0x565f6c67,0x65747265,0x646e4978,0x00007865,
    0x00030005,0x0000000e,0x00000078,0x00030005,0x00000012,0x00000079,0x00060005,0x00000019,
    0x505f6c67,0x65567265,0x78657472,0x00000000,0x00060006,0x00000019,0x00000000,0x505f6c67,
    0x7469736f,0x006e6f69,0x00070006,0x00000019,0x00000001,0x505f6c67,0x746e696f,0x657a6953,
    0x00000000,0x00070006,0x00000019,0x00000002,0x435f6c67,0x4470696c,0x61747369,0x0065636e,
    0x00070006,0x00000019,0x00000003,0x435f6c67,0x446c6c75,0x61747369,0x0065636e,0x00030005,
    0x0000001b,0x00000000,0x00030005,0x00000033,0x00676174,0x00040047,0x0000000b,0x0000000b,
    0x0000002a,0x00030047,0x00000019,0x00000002,0x00050048,0x00000019,0x00000000,0x0000000b,
    0x00000000,0x00050048,0x00000019,0x00000001,0x0000000b,0x00000001,0x00050048,0x00000019,
    0x00000002,0x0000000b,0x00000003,0x00050048,0x00000019,0x00000003,0x0000000b,0x00000004,
    0x00030047,0x00000033,0x0000000e,0x00040047,0x00000033,0x0000001e,0x00000000,0x00020013,
    0x00000002,0x00030021,0x00000003,0x00000002,0x00040015,0x00000006,0x00000020,0x00000000,
    0x00040020,0x00000007,0x00000007,0x00000006,0x00040015,0x00000009,0x00000020,0x00000001,
    0x00040020,0x0000000a,0x00000001,0x00000009,0x0004003b,0x0000000a,0x0000000b,0x00000001,
    0x0004002b,0x00000006,0x00000010,0x00000010,0x00030016,0x00000015,0x00000020,0x00040017,
    0x00000016,0x00000015,0x00000004,0x0004002b,0x00000006,0x00000017,0x00000001,0x0004001c,
    0x00000018,0x00000015,0x00000017,0x0006001e,0x00000019,0x00000016,0x00000015,0x00000018,
    0x00000018,0x00040020,0x0000001a,0x00000003,0x00000019,0x0004003b,0x0000001a,0x0000001b,
    0x00000003,0x0004002b,0x00000009,0x0000001c,0x00000000,0x0004002b,0x00000015,0x0000001f,
    0x3f000000,0x0004002b,0x00000015,0x00000021,0x41000000,0x0004002b,0x00000015,0x00000023,
    0x3f800000,0x0004002b,0x00000015,0x00000028,0x40800000,0x0004002b,0x00000015,0x0000002b,
    0x00000000,0x00040020,0x0000002d,0x00000003,0x00000016,0x0004002b,0x00000009,0x0000002f,
    0x00000001,0x00040020,0x00000030,0x00000003,0x00000015,0x00040020,0x00000032,0x00000003,
    0x00000006,0x0004003b,0x00000032,0x00000033,0x00000003,0x00050036,0x00000002,0x00000004,
    0x00000000,0x00000003,0x000200f8,0x00000005,0x0004003b,0x00000007,0x00000008,0x00000007,
    0x0004003b,0x00000007,0x0000000e,0x00000007,0x0004003b,0x00000007,0x00000012,0x00000007,
    0x0004003d,0x00000009,0x0000000c,0x0000000b,0x0004007c,0x00000006,0x0000000d,0x0000000c,
    0x0003003e,0x00000008,0x0000000d,0x0004003d,0x00000006,0x0000000f,0x00000008,0x00050089,
    0x00000006,0x00000011,0x0000000f,0x00000010,0x0003003e,0x0000000e,0x00000011,0x0004003d,
    0x00000006,0x00000013,0x00000008,0x00050086,0x00000006,0x00000014,0x00000013,0x00000010,
    0x0003003e,0x00000012,0x00000014,0x0004003d,0x00000006,0x0000001d,0x0000000e,0x00040070,
    0x00000015,0x0000001e,0x0000001d,0x00050081,0x00000015,0x00000020,0x0000001e,0x0000001f,
    0x00050088,0x00000015,0x00000022,0x00000020,0x00000021,0x00050083,0x00000015,0x00000024,
    0x00000022,0x00000023,0x0004003d,0x00000006,0x00000025,0x00000012,0x00040070,0x00000015,
    0x00000026,0x00000025,0x00050081,0x00000015,0x00000027,0x00000026,0x0000001f,0x00050088,
    0x00000015,0x00000029,0x00000027,0x00000028,0x00050083,0x00000015,0x0000002a,0x00000029,
    0x00000023,0x00070050,0x00000016,0x0000002c,0x00000024,0x0000002a,0x0000002b,0x00000023,
    0x00050041,0x0000002d,0x0000002e,0x0000001b,0x0000001c,0x0003003e,0x0000002e,0x0000002c,
    0x00050041,0x00000030,0x00000031,0x0000001b,0x0000002f,0x0003003e,0x00000031,0x00000023,
    0x0004003d,0x00000006,0x00000034,0x00000012,0x00050080,0x00000006,0x00000035,0x00000034,
    0x00000017,0x0003003e,0x00000033,0x00000035,0x000100fd,0x00010038
};

static const uint32_t FRAG[] = {
    0x07230203,0x00010300,0x0008000b,0x0000000c,0x00000000,0x00020011,0x00000001,0x0006000b,
    0x00000001,0x4c534c47,0x6474732e,0x3035342e,0x00000000,0x0003000e,0x00000000,0x00000001,
    0x0007000f,0x00000004,0x00000004,0x6e69616d,0x00000000,0x00000008,0x0000000a,0x00030010,
    0x00000004,0x00000007,0x00030003,0x00000002,0x000001c2,0x00040005,0x00000004,0x6e69616d,
    0x00000000,0x00030005,0x00000008,0x0000006f,0x00030005,0x0000000a,0x00676174,0x00040047,
    0x00000008,0x0000001e,0x00000000,0x00030047,0x0000000a,0x0000000e,0x00040047,0x0000000a,
    0x0000001e,0x00000000,0x00020013,0x00000002,0x00030021,0x00000003,0x00000002,0x00040015,
    0x00000006,0x00000020,0x00000000,0x00040020,0x00000007,0x00000003,0x00000006,0x0004003b,
    0x00000007,0x00000008,0x00000003,0x00040020,0x00000009,0x00000001,0x00000006,0x0004003b,
    0x00000009,0x0000000a,0x00000001,0x00050036,0x00000002,0x00000004,0x00000000,0x00000003,
    0x000200f8,0x00000005,0x0004003d,0x00000006,0x0000000b,0x0000000a,0x0003003e,0x00000008,
    0x0000000b,0x000100fd,0x00010038
};

static const uint32_t COMP[] = {
    0x07230203,0x00010300,0x0008000b,0x0000001e,0x00000000,0x00020011,0x00000001,0x0006000b,
    0x00000001,0x4c534c47,0x6474732e,0x3035342e,0x00000000,0x0003000e,0x00000000,0x00000001,
    0x0006000f,0x00000005,0x00000004,0x6e69616d,0x00000000,0x0000000f,0x00060010,0x00000004,
    0x00000011,0x00000001,0x00000001,0x00000001,0x00030003,0x00000002,0x000001c2,0x00040005,
    0x00000004,0x6e69616d,0x00000000,0x00030005,0x00000008,0x0074754f,0x00040006,0x00000008,
    0x00000000,0x0000006f,0x00030005,0x0000000a,0x00000000,0x00080005,0x0000000f,0x475f6c67,
    0x61626f6c,0x766e496c,0x7461636f,0x496e6f69,0x00000044,0x00030005,0x00000014,0x00004350,
    0x00040006,0x00000014,0x00000000,0x00676174,0x00030005,0x00000016,0x00006370,0x00040047,
    0x00000007,0x00000006,0x00000004,0x00030047,0x00000008,0x00000002,0x00040048,0x00000008,
    0x00000000,0x00000019,0x00050048,0x00000008,0x00000000,0x00000023,0x00000000,0x00030047,
    0x0000000a,0x00000019,0x00040047,0x0000000a,0x00000021,0x00000000,0x00040047,0x0000000a,
    0x00000022,0x00000000,0x00040047,0x0000000f,0x0000000b,0x0000001c,0x00030047,0x00000014,
    0x00000002,0x00050048,0x00000014,0x00000000,0x00000023,0x00000000,0x00040047,0x0000001d,
    0x0000000b,0x00000019,0x00020013,0x00000002,0x00030021,0x00000003,0x00000002,0x00040015,
    0x00000006,0x00000020,0x00000000,0x0003001d,0x00000007,0x00000006,0x0003001e,0x00000008,
    0x00000007,0x00040020,0x00000009,0x0000000c,0x00000008,0x0004003b,0x00000009,0x0000000a,
    0x0000000c,0x00040015,0x0000000b,0x00000020,0x00000001,0x0004002b,0x0000000b,0x0000000c,
    0x00000000,0x00040017,0x0000000d,0x00000006,0x00000003,0x00040020,0x0000000e,0x00000001,
    0x0000000d,0x0004003b,0x0000000e,0x0000000f,0x00000001,0x0004002b,0x00000006,0x00000010,
    0x00000000,0x00040020,0x00000011,0x00000001,0x00000006,0x0003001e,0x00000014,0x00000006,
    0x00040020,0x00000015,0x00000009,0x00000014,0x0004003b,0x00000015,0x00000016,0x00000009,
    0x00040020,0x00000017,0x00000009,0x00000006,0x00040020,0x0000001a,0x0000000c,0x00000006,
    0x0004002b,0x00000006,0x0000001c,0x00000001,0x0006002c,0x0000000d,0x0000001d,0x0000001c,
    0x0000001c,0x0000001c,0x00050036,0x00000002,0x00000004,0x00000000,0x00000003,0x000200f8,
    0x00000005,0x00050041,0x00000011,0x00000012,0x0000000f,0x00000010,0x0004003d,0x00000006,
    0x00000013,0x00000012,0x00050041,0x00000017,0x00000018,0x00000016,0x0000000c,0x0004003d,
    0x00000006,0x00000019,0x00000018,0x00060041,0x0000001a,0x0000001b,0x0000000a,0x0000000c,
    0x00000013,0x0003003e,0x0000001b,0x00000019,0x000100fd,0x00010038
};

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

static VkDeviceMemory bind(VkPhysicalDevice pd, VkDevice dev, VkMemoryRequirements mr,
                           VkMemoryPropertyFlags want) {
    uint32_t type = mem_type(pd, mr.memoryTypeBits, want);
    if (type == UINT32_MAX) {
        fatal("no memory type with the wanted properties", VK_ERROR_INITIALIZATION_FAILED);
    }
    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mr.size,
        .memoryTypeIndex = type,
    };
    VkDeviceMemory mem;
    VkResult r = vkAllocateMemory(dev, &mai, NULL, &mem);
    if (r != VK_SUCCESS) {
        fatal("vkAllocateMemory", r);
    }
    return mem;
}

// A host-visible buffer, mapped for the probe to fill or read.
struct mapped {
    VkBuffer buf;
    VkDeviceMemory mem;
    uint32_t *p;
};

static struct mapped mapped(VkPhysicalDevice pd, VkDevice dev, VkDeviceSize size,
                            VkBufferUsageFlags usage) {
    struct mapped m;
    VkResult r;
    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    if ((r = vkCreateBuffer(dev, &bci, NULL, &m.buf)) != VK_SUCCESS) {
        fatal("vkCreateBuffer", r);
    }
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev, m.buf, &mr);
    m.mem = bind(pd, dev, mr,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkBindBufferMemory(dev, m.buf, m.mem, 0);
    if ((r = vkMapMemory(dev, m.mem, 0, VK_WHOLE_SIZE, 0, (void **)&m.p)) != VK_SUCCESS) {
        fatal("vkMapMemory", r);
    }
    memset(m.p, 0, size);
    return m;
}

static void unmapped(VkDevice dev, struct mapped *m) {
    vkUnmapMemory(dev, m->mem);
    vkDestroyBuffer(dev, m->buf, NULL);
    vkFreeMemory(dev, m->mem, NULL);
}

static VkShaderModule module(VkDevice dev, const uint32_t *code, size_t size) {
    VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = size,
        .pCode = code,
    };
    VkShaderModule m;
    VkResult r = vkCreateShaderModule(dev, &smci, NULL, &m);
    if (r != VK_SUCCESS) {
        fatal("vkCreateShaderModule", r);
    }
    return m;
}

static int row_is(const uint32_t *texels, uint32_t row, uint32_t want) {
    for (uint32_t x = 0; x < W; x++) {
        if (texels[row * W + x] != want) {
            return 0;
        }
    }
    return 1;
}

int main(void) {
    VkResult r;

    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "venus-indexed-and-indirect-draw-probe",
        .apiVersion = VK_API_VERSION_1_4,
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
    if (props.apiVersion < VK_API_VERSION_1_4) {
        fatal("Vulkan 1.4, where the whole group is core", VK_ERROR_FEATURE_NOT_PRESENT);
    }

    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, NULL);
    VkQueueFamilyProperties *qs = calloc(qn, sizeof *qs);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qs);
    uint32_t qfam = UINT32_MAX;
    VkQueueFlags both = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
    for (uint32_t i = 0; i < qn; i++) {
        if ((qs[i].queueFlags & both) == both) {
            qfam = i;
            break;
        }
    }
    if (qfam == UINT32_MAX) {
        fatal("no graphics and compute queue", VK_ERROR_FEATURE_NOT_PRESENT);
    }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = qfam,
        .queueCount = 1,
        .pQueuePriorities = &prio,
    };
    VkPhysicalDeviceVulkan14Features f14 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES,
        .maintenance5 = VK_TRUE,
    };
    VkPhysicalDeviceVulkan13Features f13 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext = &f14,
        .dynamicRendering = VK_TRUE,
    };
    VkPhysicalDeviceVulkan12Features f12 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &f13,
        .drawIndirectCount = VK_TRUE,
    };
    VkPhysicalDeviceFeatures2 f2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &f12,
        .features = {.multiDrawIndirect = VK_TRUE},
    };
    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &f2,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qci,
    };
    VkDevice dev;
    if ((r = vkCreateDevice(pd, &dci, NULL, &dev)) != VK_SUCCESS) {
        fatal("vkCreateDevice", r);
    }
    VkQueue queue;
    vkGetDeviceQueue(dev, qfam, 0, &queue);

    // ------------------------------------------------------------------ the draws' objects

    VkImageCreateInfo imci = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_R32_UINT,
        .extent = {W, H, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VkImage image;
    if ((r = vkCreateImage(dev, &imci, NULL, &image)) != VK_SUCCESS) {
        fatal("vkCreateImage", r);
    }
    VkMemoryRequirements imr;
    vkGetImageMemoryRequirements(dev, image, &imr);
    VkDeviceMemory image_mem = bind(pd, dev, imr, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkBindImageMemory(dev, image, image_mem, 0);
    VkImageViewCreateInfo ivci = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = VK_FORMAT_R32_UINT,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    VkImageView view;
    if ((r = vkCreateImageView(dev, &ivci, NULL, &view)) != VK_SUCCESS) {
        fatal("vkCreateImageView", r);
    }

    // Indices 0..15 then 16..31: the 1.0 bind reads the first run, the 1.4 bind the second.
    struct mapped index = mapped(pd, dev, 32 * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    for (uint32_t i = 0; i < 32; i++) {
        index.p[i] = i;
    }

    // The indirect records, each command's pair at its own offset.
    struct mapped indirect = mapped(pd, dev, 256, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
    const VkDeviceSize DRAW_AT = 0, INDEXED_AT = 32, COUNT_AT = 64, INDEXED_COUNT_AT = 128,
                       DISPATCH_AT = 192, COUNT_VALUE_AT = 224;
    VkDrawIndirectCommand *draws = (void *)((char *)indirect.p + DRAW_AT);
    draws[0] = (VkDrawIndirectCommand){16, 1, 32, 0};
    VkDrawIndexedIndirectCommand *indexed = (void *)((char *)indirect.p + INDEXED_AT);
    indexed[0] = (VkDrawIndexedIndirectCommand){16, 1, 0, 32, 0};
    VkDrawIndirectCommand *counted = (void *)((char *)indirect.p + COUNT_AT);
    counted[0] = (VkDrawIndirectCommand){16, 1, 64, 0};
    counted[1] = (VkDrawIndirectCommand){16, 1, 80, 0};
    VkDrawIndexedIndirectCommand *indexed_counted =
        (void *)((char *)indirect.p + INDEXED_COUNT_AT);
    indexed_counted[0] = (VkDrawIndexedIndirectCommand){16, 1, 0, 80, 0};
    indexed_counted[1] = (VkDrawIndexedIndirectCommand){16, 1, 0, 96, 0};
    VkDispatchIndirectCommand *dispatch = (void *)((char *)indirect.p + DISPATCH_AT);
    dispatch[0] = (VkDispatchIndirectCommand){8, 1, 1};
    uint32_t *count_value = (void *)((char *)indirect.p + COUNT_VALUE_AT);
    count_value[0] = 1;

    struct mapped texels =
        mapped(pd, dev, W * H * sizeof(uint32_t), VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    memset(texels.p, 0xee, W * H * sizeof(uint32_t));
    struct mapped elems =
        mapped(pd, dev, ELEMS * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    VkPipelineLayoutCreateInfo gpli = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    VkPipelineLayout glayout;
    if ((r = vkCreatePipelineLayout(dev, &gpli, NULL, &glayout)) != VK_SUCCESS) {
        fatal("vkCreatePipelineLayout (graphics)", r);
    }
    VkShaderModule vs = module(dev, VERT, sizeof VERT), fs = module(dev, FRAG, sizeof FRAG);
    VkPipelineShaderStageCreateInfo stages[2] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vs,
         .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = fs,
         .pName = "main"},
    };
    VkPipelineVertexInputStateCreateInfo vi = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    };
    VkPipelineInputAssemblyStateCreateInfo ia = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
    };
    VkViewport viewport = {0, 0, W, H, 0, 1};
    VkRect2D scissor = {{0, 0}, {W, H}};
    VkPipelineViewportStateCreateInfo vp = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor,
    };
    VkPipelineRasterizationStateCreateInfo rs = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .lineWidth = 1.0f,
    };
    VkPipelineMultisampleStateCreateInfo ms = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };
    VkPipelineColorBlendAttachmentState blend = {.colorWriteMask = VK_COLOR_COMPONENT_R_BIT};
    VkPipelineColorBlendStateCreateInfo cbs = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &blend,
    };
    VkFormat format = VK_FORMAT_R32_UINT;
    VkPipelineRenderingCreateInfo prci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &format,
    };
    VkGraphicsPipelineCreateInfo gpci = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &prci,
        .stageCount = 2,
        .pStages = stages,
        .pVertexInputState = &vi,
        .pInputAssemblyState = &ia,
        .pViewportState = &vp,
        .pRasterizationState = &rs,
        .pMultisampleState = &ms,
        .pColorBlendState = &cbs,
        .layout = glayout,
    };
    VkPipeline gpipe;
    if ((r = vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, NULL, &gpipe)) !=
        VK_SUCCESS) {
        fatal("vkCreateGraphicsPipelines", r);
    }

    // ------------------------------------------------------------ the dispatches' objects

    VkDescriptorSetLayoutBinding binding = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
    };
    VkDescriptorSetLayoutCreateInfo dslci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &binding,
    };
    VkDescriptorSetLayout set_layout;
    if ((r = vkCreateDescriptorSetLayout(dev, &dslci, NULL, &set_layout)) != VK_SUCCESS) {
        fatal("vkCreateDescriptorSetLayout", r);
    }
    VkPushConstantRange range = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)};
    VkPipelineLayoutCreateInfo cpli = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &range,
    };
    VkPipelineLayout clayout;
    if ((r = vkCreatePipelineLayout(dev, &cpli, NULL, &clayout)) != VK_SUCCESS) {
        fatal("vkCreatePipelineLayout (compute)", r);
    }
    VkShaderModule cs = module(dev, COMP, sizeof COMP);
    VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .flags = VK_PIPELINE_CREATE_DISPATCH_BASE_BIT,
        .stage =
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = cs,
                .pName = "main",
            },
        .layout = clayout,
    };
    VkPipeline cpipe;
    if ((r = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &cpipe)) !=
        VK_SUCCESS) {
        fatal("vkCreateComputePipelines", r);
    }
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    VkDescriptorPoolCreateInfo dpci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &ps,
    };
    VkDescriptorPool dpool;
    if ((r = vkCreateDescriptorPool(dev, &dpci, NULL, &dpool)) != VK_SUCCESS) {
        fatal("vkCreateDescriptorPool", r);
    }
    VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = dpool,
        .descriptorSetCount = 1,
        .pSetLayouts = &set_layout,
    };
    VkDescriptorSet set;
    if ((r = vkAllocateDescriptorSets(dev, &dsai, &set)) != VK_SUCCESS) {
        fatal("vkAllocateDescriptorSets", r);
    }
    VkDescriptorBufferInfo elems_info = {elems.buf, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = set,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &elems_info,
    };
    vkUpdateDescriptorSets(dev, 1, &write, 0, NULL);

    // ------------------------------------------------------------------------ recording

    VkCommandPoolCreateInfo cpi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
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
    VkCommandBufferBeginInfo cbbi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(cb, &cbbi);

    VkImageMemoryBarrier to_attachment = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL, 0, NULL, 1,
                         &to_attachment);
    VkRenderingAttachmentInfo att = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = view,
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue = {.color = {.uint32 = {0, 0, 0, 0}}},
    };
    VkRenderingInfo ri = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = {{0, 0}, {W, H}},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &att,
    };
    vkCmdBeginRendering(cb, &ri);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, gpipe);
    // Row 0, through the 1.0 bind.
    vkCmdBindIndexBuffer(cb, index.buf, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cb, 16, 1, 0, 0, 0);
    // Row 1, through the 1.4 bind at the second run. Rows 3 and 6 read this bind too.
    vkCmdBindIndexBuffer2(cb, index.buf, 16 * sizeof(uint32_t), 16 * sizeof(uint32_t),
                          VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cb, 16, 1, 0, 0, 0);
    vkCmdDrawIndirect(cb, indirect.buf, DRAW_AT, 1, sizeof(VkDrawIndirectCommand));
    vkCmdDrawIndexedIndirect(cb, indirect.buf, INDEXED_AT, 1,
                             sizeof(VkDrawIndexedIndirectCommand));
    vkCmdDrawIndirectCount(cb, indirect.buf, COUNT_AT, indirect.buf, COUNT_VALUE_AT, 2,
                           sizeof(VkDrawIndirectCommand));
    vkCmdDrawIndexedIndirectCount(cb, indirect.buf, INDEXED_COUNT_AT, indirect.buf,
                                  COUNT_VALUE_AT, 2, sizeof(VkDrawIndexedIndirectCommand));
    vkCmdEndRendering(cb);

    VkImageMemoryBarrier to_copy = to_attachment;
    to_copy.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    to_copy.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    to_copy.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    to_copy.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &to_copy);
    VkBufferImageCopy copy = {
        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .imageExtent = {W, H, 1},
    };
    vkCmdCopyImageToBuffer(cb, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, texels.buf, 1, &copy);

    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, cpipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, clayout, 0, 1, &set, 0, NULL);
    uint32_t tag = 1;
    vkCmdPushConstants(cb, clayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof tag, &tag);
    vkCmdDispatchIndirect(cb, indirect.buf, DISPATCH_AT);
    // The second dispatch writes other elements, so the two need no barrier between them.
    tag = 2;
    vkCmdPushConstants(cb, clayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof tag, &tag);
    vkCmdDispatchBase(cb, 8, 0, 0, 8, 1, 1);

    VkMemoryBarrier host = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
    };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, NULL, 0, NULL);
    vkEndCommandBuffer(cb);

    VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    if ((r = vkCreateFence(dev, &fci, NULL, &fence)) != VK_SUCCESS) {
        fatal("vkCreateFence", r);
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
    check(r == VK_SUCCESS, "the submit completed");

    if (r == VK_SUCCESS) {
        const uint32_t *t = texels.p;
        check(row_is(t, 0, 1), "vkCmdDrawIndexed: row 0, through the 1.0 index bind");
        check(row_is(t, 1, 2), "vkCmdBindIndexBuffer2: row 1, from the bind's offset");
        check(row_is(t, 2, 3), "vkCmdDrawIndirect: row 2");
        check(row_is(t, 3, 4), "vkCmdDrawIndexedIndirect: row 3");
        check(row_is(t, 4, 5), "vkCmdDrawIndirectCount: row 4");
        check(row_is(t, 5, 0), "vkCmdDrawIndirectCount: its count of 1 left row 5 clear");
        check(row_is(t, 6, 7), "vkCmdDrawIndexedIndirectCount: row 6");
        check(row_is(t, 7, 0), "vkCmdDrawIndexedIndirectCount: its count of 1 left row 7 clear");
        printf("  rows:");
        for (uint32_t y = 0; y < H; y++) {
            printf(" %u", t[y * W]);
        }
        printf("\n");

        uint32_t first = 0, second = 0, beyond = 0;
        for (uint32_t i = 0; i < ELEMS; i++) {
            first += i < 8 && elems.p[i] == 1;
            second += i >= 8 && i < 16 && elems.p[i] == 2;
            beyond += i >= 16 && elems.p[i] == 0;
        }
        check(first == 8, "vkCmdDispatchIndirect: elements 0..7 from the buffer's counts");
        check(second == 8, "vkCmdDispatchBase: elements 8..15, from its base");
        check(beyond == 16, "no dispatch wrote past element 15");
        printf("  elements right: %u/8, %u/8, %u/16 untouched\n", first, second, beyond);
    }

    vkDestroyFence(dev, fence, NULL);
    vkDestroyCommandPool(dev, cmdpool, NULL);
    vkDestroyPipeline(dev, cpipe, NULL);
    vkDestroyPipeline(dev, gpipe, NULL);
    vkDestroyShaderModule(dev, cs, NULL);
    vkDestroyShaderModule(dev, vs, NULL);
    vkDestroyShaderModule(dev, fs, NULL);
    vkDestroyDescriptorPool(dev, dpool, NULL);
    vkDestroyPipelineLayout(dev, clayout, NULL);
    vkDestroyPipelineLayout(dev, glayout, NULL);
    vkDestroyDescriptorSetLayout(dev, set_layout, NULL);
    vkDestroyImageView(dev, view, NULL);
    vkDestroyImage(dev, image, NULL);
    vkFreeMemory(dev, image_mem, NULL);
    unmapped(dev, &index);
    unmapped(dev, &indirect);
    unmapped(dev, &texels);
    unmapped(dev, &elems);
    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(inst, NULL);

    printf("\n%d check(s) failed\n", failures);
    return failures ? 1 : 0;
}
