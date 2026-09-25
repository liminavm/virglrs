// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva
//
// The `core-dynamic-state` group of src/venus/unserved.txt, as a program that names each command.
//
//   vkCmdSetDepthBiasEnable  vkCmdSetDepthBounds  vkCmdSetDeviceMask  vkCmdSetLineStipple
//
// Core from 1.0 (depth bounds) to 1.4 (line stipple). Two of them have a consequence a probe can
// read back, and two do not:
//
//   - vkCmdSetDepthBiasEnable: a triangle at depth 0.5 is drawn twice into fresh D32 images with
//     a large constant bias set, once with the bias switched off and once on. Off reads 0.5; on
//     reads further away. A dropped command leaves the bias at its undefined initial state.
//   - vkCmdSetLineStipple: a Bresenham line across a 16x1 target with pattern 0x0f0f and factor
//     2, which lights pixels 0..7 and leaves 8..15 clear; factor 1 would alternate every four
//     pixels instead. Only where `stippledBresenhamLines` is offered -- anv, not KosmicKrisp --
//     and reported as skipped elsewhere.
//   - vkCmdSetDepthBounds and vkCmdSetDeviceMask are recorded and have no effect this program
//     can see: neither KosmicKrisp nor the anv measured here has `depthBounds`, and a device
//     group of one has one device to mask. The worker log scores them, as it scores everything.
//
// The shaders, compiled with `glslangValidator -V --target-env vulkan1.1 -x`:
//
//   // half.vert
//   #version 450
//   void main() {
//       vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
//       gl_Position = vec4(p * 2.0 - 1.0, 0.5, 1.0);
//   }
//
//   // line.vert
//   #version 450
//   void main() {
//       gl_Position = vec4(gl_VertexIndex == 0 ? -1.0 : 1.0, 0.0, 0.0, 1.0);
//   }
//
//   // ab.frag
//   #version 450
//   layout(location = 0) out uint o;
//   void main() {
//       o = 0xabu;
//   }
//
// Headless on purpose: no surface, no swapchain, no window. It runs over ssh.
//
//   cc -O1 -o core_dynamic_state core_dynamic_state.c -lvulkan && ./core_dynamic_state
//
// Exit 0 = every check passed. Exit 1 = a check failed, and the line above it says which.
// Exit 2 = the device could not be brought up at all (not a verdict on the group).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define DW 4u
#define DH 4u
#define LW 16u

static const uint32_t HALF_VERT[] = {
    0x07230203,0x00010300,0x0008000b,0x0000002b,0x00000000,0x00020011,0x00000001,0x0006000b,
    0x00000001,0x4c534c47,0x6474732e,0x3035342e,0x00000000,0x0003000e,0x00000000,0x00000001,
    0x0007000f,0x00000000,0x00000004,0x6e69616d,0x00000000,0x0000000c,0x0000001d,0x00030003,
    0x00000002,0x000001c2,0x00040005,0x00000004,0x6e69616d,0x00000000,0x00030005,0x00000009,
    0x00000070,0x00060005,0x0000000c,0x565f6c67,0x65747265,0x646e4978,0x00007865,0x00060005,
    0x0000001b,0x505f6c67,0x65567265,0x78657472,0x00000000,0x00060006,0x0000001b,0x00000000,
    0x505f6c67,0x7469736f,0x006e6f69,0x00070006,0x0000001b,0x00000001,0x505f6c67,0x746e696f,
    0x657a6953,0x00000000,0x00070006,0x0000001b,0x00000002,0x435f6c67,0x4470696c,0x61747369,
    0x0065636e,0x00070006,0x0000001b,0x00000003,0x435f6c67,0x446c6c75,0x61747369,0x0065636e,
    0x00030005,0x0000001d,0x00000000,0x00040047,0x0000000c,0x0000000b,0x0000002a,0x00030047,
    0x0000001b,0x00000002,0x00050048,0x0000001b,0x00000000,0x0000000b,0x00000000,0x00050048,
    0x0000001b,0x00000001,0x0000000b,0x00000001,0x00050048,0x0000001b,0x00000002,0x0000000b,
    0x00000003,0x00050048,0x0000001b,0x00000003,0x0000000b,0x00000004,0x00020013,0x00000002,
    0x00030021,0x00000003,0x00000002,0x00030016,0x00000006,0x00000020,0x00040017,0x00000007,
    0x00000006,0x00000002,0x00040020,0x00000008,0x00000007,0x00000007,0x00040015,0x0000000a,
    0x00000020,0x00000001,0x00040020,0x0000000b,0x00000001,0x0000000a,0x0004003b,0x0000000b,
    0x0000000c,0x00000001,0x0004002b,0x0000000a,0x0000000e,0x00000001,0x0004002b,0x0000000a,
    0x00000010,0x00000002,0x00040017,0x00000017,0x00000006,0x00000004,0x00040015,0x00000018,
    0x00000020,0x00000000,0x0004002b,0x00000018,0x00000019,0x00000001,0x0004001c,0x0000001a,
    0x00000006,0x00000019,0x0006001e,0x0000001b,0x00000017,0x00000006,0x0000001a,0x0000001a,
    0x00040020,0x0000001c,0x00000003,0x0000001b,0x0004003b,0x0000001c,0x0000001d,0x00000003,
    0x0004002b,0x0000000a,0x0000001e,0x00000000,0x0004002b,0x00000006,0x00000020,0x40000000,
    0x0004002b,0x00000006,0x00000022,0x3f800000,0x0004002b,0x00000006,0x00000025,0x3f000000,
    0x00040020,0x00000029,0x00000003,0x00000017,0x00050036,0x00000002,0x00000004,0x00000000,
    0x00000003,0x000200f8,0x00000005,0x0004003b,0x00000008,0x00000009,0x00000007,0x0004003d,
    0x0000000a,0x0000000d,0x0000000c,0x000500c4,0x0000000a,0x0000000f,0x0000000d,0x0000000e,
    0x000500c7,0x0000000a,0x00000011,0x0000000f,0x00000010,0x0004006f,0x00000006,0x00000012,
    0x00000011,0x0004003d,0x0000000a,0x00000013,0x0000000c,0x000500c7,0x0000000a,0x00000014,
    0x00000013,0x00000010,0x0004006f,0x00000006,0x00000015,0x00000014,0x00050050,0x00000007,
    0x00000016,0x00000012,0x00000015,0x0003003e,0x00000009,0x00000016,0x0004003d,0x00000007,
    0x0000001f,0x00000009,0x0005008e,0x00000007,0x00000021,0x0000001f,0x00000020,0x00050050,
    0x00000007,0x00000023,0x00000022,0x00000022,0x00050083,0x00000007,0x00000024,0x00000021,
    0x00000023,0x00050051,0x00000006,0x00000026,0x00000024,0x00000000,0x00050051,0x00000006,
    0x00000027,0x00000024,0x00000001,0x00070050,0x00000017,0x00000028,0x00000026,0x00000027,
    0x00000025,0x00000022,0x00050041,0x00000029,0x0000002a,0x0000001d,0x0000001e,0x0003003e,
    0x0000002a,0x00000028,0x000100fd,0x00010038
};

static const uint32_t LINE_VERT[] = {
    0x07230203,0x00010300,0x0008000b,0x0000001c,0x00000000,0x00020011,0x00000001,0x0006000b,
    0x00000001,0x4c534c47,0x6474732e,0x3035342e,0x00000000,0x0003000e,0x00000000,0x00000001,
    0x0007000f,0x00000000,0x00000004,0x6e69616d,0x00000000,0x0000000d,0x00000011,0x00030003,
    0x00000002,0x000001c2,0x00040005,0x00000004,0x6e69616d,0x00000000,0x00060005,0x0000000b,
    0x505f6c67,0x65567265,0x78657472,0x00000000,0x00060006,0x0000000b,0x00000000,0x505f6c67,
    0x7469736f,0x006e6f69,0x00070006,0x0000000b,0x00000001,0x505f6c67,0x746e696f,0x657a6953,
    0x00000000,0x00070006,0x0000000b,0x00000002,0x435f6c67,0x4470696c,0x61747369,0x0065636e,
    0x00070006,0x0000000b,0x00000003,0x435f6c67,0x446c6c75,0x61747369,0x0065636e,0x00030005,
    0x0000000d,0x00000000,0x00060005,0x00000011,0x565f6c67,0x65747265,0x646e4978,0x00007865,
    0x00030047,0x0000000b,0x00000002,0x00050048,0x0000000b,0x00000000,0x0000000b,0x00000000,
    0x00050048,0x0000000b,0x00000001,0x0000000b,0x00000001,0x00050048,0x0000000b,0x00000002,
    0x0000000b,0x00000003,0x00050048,0x0000000b,0x00000003,0x0000000b,0x00000004,0x00040047,
    0x00000011,0x0000000b,0x0000002a,0x00020013,0x00000002,0x00030021,0x00000003,0x00000002,
    0x00030016,0x00000006,0x00000020,0x00040017,0x00000007,0x00000006,0x00000004,0x00040015,
    0x00000008,0x00000020,0x00000000,0x0004002b,0x00000008,0x00000009,0x00000001,0x0004001c,
    0x0000000a,0x00000006,0x00000009,0x0006001e,0x0000000b,0x00000007,0x00000006,0x0000000a,
    0x0000000a,0x00040020,0x0000000c,0x00000003,0x0000000b,0x0004003b,0x0000000c,0x0000000d,
    0x00000003,0x00040015,0x0000000e,0x00000020,0x00000001,0x0004002b,0x0000000e,0x0000000f,
    0x00000000,0x00040020,0x00000010,0x00000001,0x0000000e,0x0004003b,0x00000010,0x00000011,
    0x00000001,0x00020014,0x00000013,0x0004002b,0x00000006,0x00000015,0xbf800000,0x0004002b,
    0x00000006,0x00000016,0x3f800000,0x0004002b,0x00000006,0x00000018,0x00000000,0x00040020,
    0x0000001a,0x00000003,0x00000007,0x00050036,0x00000002,0x00000004,0x00000000,0x00000003,
    0x000200f8,0x00000005,0x0004003d,0x0000000e,0x00000012,0x00000011,0x000500aa,0x00000013,
    0x00000014,0x00000012,0x0000000f,0x000600a9,0x00000006,0x00000017,0x00000014,0x00000015,
    0x00000016,0x00070050,0x00000007,0x00000019,0x00000017,0x00000018,0x00000018,0x00000016,
    0x00050041,0x0000001a,0x0000001b,0x0000000d,0x0000000f,0x0003003e,0x0000001b,0x00000019,
    0x000100fd,0x00010038
};

static const uint32_t AB_FRAG[] = {
    0x07230203,0x00010300,0x0008000b,0x0000000a,0x00000000,0x00020011,0x00000001,0x0006000b,
    0x00000001,0x4c534c47,0x6474732e,0x3035342e,0x00000000,0x0003000e,0x00000000,0x00000001,
    0x0006000f,0x00000004,0x00000004,0x6e69616d,0x00000000,0x00000008,0x00030010,0x00000004,
    0x00000007,0x00030003,0x00000002,0x000001c2,0x00040005,0x00000004,0x6e69616d,0x00000000,
    0x00030005,0x00000008,0x0000006f,0x00040047,0x00000008,0x0000001e,0x00000000,0x00020013,
    0x00000002,0x00030021,0x00000003,0x00000002,0x00040015,0x00000006,0x00000020,0x00000000,
    0x00040020,0x00000007,0x00000003,0x00000006,0x0004003b,0x00000007,0x00000008,0x00000003,
    0x0004002b,0x00000006,0x00000009,0x000000ab,0x00050036,0x00000002,0x00000004,0x00000000,
    0x00000003,0x000200f8,0x00000005,0x0003003e,0x00000008,0x00000009,0x000100fd,0x00010038
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

struct mapped {
    VkBuffer buf;
    VkDeviceMemory mem;
    uint8_t *p;
};

static struct mapped mapped(VkPhysicalDevice pd, VkDevice dev, VkDeviceSize size) {
    struct mapped m;
    VkResult r;
    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
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
    memset(m.p, 0xee, size);
    return m;
}

struct target {
    VkImage image;
    VkImageView view;
    VkDeviceMemory mem;
};

static struct target target(VkPhysicalDevice pd, VkDevice dev, VkFormat format, uint32_t w,
                            uint32_t h, VkImageUsageFlags usage, VkImageAspectFlags aspect) {
    struct target t;
    VkResult r;
    VkImageCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .extent = {w, h, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = usage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    if ((r = vkCreateImage(dev, &ici, NULL, &t.image)) != VK_SUCCESS) {
        fatal("vkCreateImage", r);
    }
    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(dev, t.image, &mr);
    t.mem = bind(pd, dev, mr, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkBindImageMemory(dev, t.image, t.mem, 0);
    VkImageViewCreateInfo vci = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = t.image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format,
        .subresourceRange = {aspect, 0, 1, 0, 1},
    };
    if ((r = vkCreateImageView(dev, &vci, NULL, &t.view)) != VK_SUCCESS) {
        fatal("vkCreateImageView", r);
    }
    return t;
}

static void untarget(VkDevice dev, struct target *t) {
    vkDestroyImageView(dev, t->view, NULL);
    vkDestroyImage(dev, t->image, NULL);
    vkFreeMemory(dev, t->mem, NULL);
}

static void to_layout(VkCommandBuffer cb, VkImage image, VkImageAspectFlags aspect,
                      VkImageLayout from, VkImageLayout to) {
    VkImageMemoryBarrier b = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        .oldLayout = from,
        .newLayout = to,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = {aspect, 0, 1, 0, 1},
    };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &b);
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

// Everything a pipeline here shares; the two differ in topology, attachments and dynamic state.
struct pipeline_desc {
    VkShaderModule vs, fs;
    VkPrimitiveTopology topology;
    uint32_t w, h;
    const VkPipelineRasterizationStateCreateInfo *rs;
    const VkPipelineDepthStencilStateCreateInfo *depth;
    const VkPipelineRenderingCreateInfo *rendering;
    uint32_t color_attachments;
    const VkDynamicState *dynamic;
    uint32_t dynamic_count;
};

static VkPipeline pipeline(VkDevice dev, VkPipelineLayout layout, struct pipeline_desc d) {
    VkPipelineShaderStageCreateInfo stages[2] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = d.vs,
         .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = d.fs,
         .pName = "main"},
    };
    VkPipelineVertexInputStateCreateInfo vi = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    };
    VkPipelineInputAssemblyStateCreateInfo ia = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = d.topology,
    };
    VkViewport viewport = {0, 0, (float)d.w, (float)d.h, 0, 1};
    VkRect2D scissor = {{0, 0}, {d.w, d.h}};
    VkPipelineViewportStateCreateInfo vp = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor,
    };
    VkPipelineMultisampleStateCreateInfo ms = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };
    VkPipelineColorBlendAttachmentState blend = {.colorWriteMask = VK_COLOR_COMPONENT_R_BIT};
    VkPipelineColorBlendStateCreateInfo cbs = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = d.color_attachments,
        .pAttachments = &blend,
    };
    VkPipelineDynamicStateCreateInfo ds = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = d.dynamic_count,
        .pDynamicStates = d.dynamic,
    };
    VkGraphicsPipelineCreateInfo gpci = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = d.rendering,
        .stageCount = d.fs ? 2 : 1,
        .pStages = stages,
        .pVertexInputState = &vi,
        .pInputAssemblyState = &ia,
        .pViewportState = &vp,
        .pRasterizationState = d.rs,
        .pMultisampleState = &ms,
        .pDepthStencilState = d.depth,
        .pColorBlendState = &cbs,
        .pDynamicState = &ds,
        .layout = layout,
    };
    VkPipeline p;
    VkResult r = vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, NULL, &p);
    if (r != VK_SUCCESS) {
        fatal("vkCreateGraphicsPipelines", r);
    }
    return p;
}

int main(void) {
    VkResult r;

    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "venus-core-dynamic-state-probe",
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

    VkPhysicalDeviceVulkan14Features has14 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES,
    };
    VkPhysicalDeviceFeatures2 has = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &has14,
    };
    vkGetPhysicalDeviceFeatures2(pd, &has);
    int stipple = has14.bresenhamLines && has14.stippledBresenhamLines;
    printf("stippled Bresenham lines: %s\n\n", stipple ? "offered" : "not offered");

    VkFormatProperties dfp;
    vkGetPhysicalDeviceFormatProperties(pd, VK_FORMAT_D32_SFLOAT, &dfp);
    VkFormatFeatureFlags dwant =
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT;
    if ((dfp.optimalTilingFeatures & dwant) != dwant) {
        fatal("D32_SFLOAT as a copyable depth attachment", VK_ERROR_FORMAT_NOT_SUPPORTED);
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
    VkPhysicalDeviceVulkan14Features f14 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES,
        .bresenhamLines = stipple ? VK_TRUE : VK_FALSE,
        .stippledBresenhamLines = stipple ? VK_TRUE : VK_FALSE,
    };
    VkPhysicalDeviceVulkan13Features f13 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext = &f14,
        .dynamicRendering = VK_TRUE,
    };
    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &f13,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qci,
    };
    VkDevice dev;
    if ((r = vkCreateDevice(pd, &dci, NULL, &dev)) != VK_SUCCESS) {
        fatal("vkCreateDevice", r);
    }
    VkQueue queue;
    vkGetDeviceQueue(dev, qfam, 0, &queue);

    VkPipelineLayoutCreateInfo pli = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    VkPipelineLayout layout;
    if ((r = vkCreatePipelineLayout(dev, &pli, NULL, &layout)) != VK_SUCCESS) {
        fatal("vkCreatePipelineLayout", r);
    }
    VkShaderModule half = module(dev, HALF_VERT, sizeof HALF_VERT);
    VkShaderModule line = module(dev, LINE_VERT, sizeof LINE_VERT);
    VkShaderModule ab = module(dev, AB_FRAG, sizeof AB_FRAG);

    // The depth pipeline: no colour, depth always written, bias and its switch both dynamic.
    VkPipelineRasterizationStateCreateInfo depth_rs = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .lineWidth = 1.0f,
    };
    VkPipelineDepthStencilStateCreateInfo depth_ds = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_ALWAYS,
    };
    VkPipelineRenderingCreateInfo depth_ri = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .depthAttachmentFormat = VK_FORMAT_D32_SFLOAT,
    };
    VkDynamicState depth_dyn[2] = {VK_DYNAMIC_STATE_DEPTH_BIAS, VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE};
    VkPipeline depth_pipe = pipeline(dev, layout,
                                     (struct pipeline_desc){
                                         .vs = half,
                                         .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                                         .w = DW,
                                         .h = DH,
                                         .rs = &depth_rs,
                                         .depth = &depth_ds,
                                         .rendering = &depth_ri,
                                         .dynamic = depth_dyn,
                                         .dynamic_count = 2,
                                     });

    // The line pipeline, only where stippled Bresenham lines are offered.
    VkPipeline line_pipe = VK_NULL_HANDLE;
    if (stipple) {
        VkPipelineRasterizationLineStateCreateInfo line_state = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_LINE_STATE_CREATE_INFO,
            .lineRasterizationMode = VK_LINE_RASTERIZATION_MODE_BRESENHAM,
            .stippledLineEnable = VK_TRUE,
        };
        VkPipelineRasterizationStateCreateInfo line_rs = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .pNext = &line_state,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_NONE,
            .lineWidth = 1.0f,
        };
        VkFormat color = VK_FORMAT_R32_UINT;
        VkPipelineRenderingCreateInfo line_ri = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
            .colorAttachmentCount = 1,
            .pColorAttachmentFormats = &color,
        };
        VkDynamicState line_dyn = VK_DYNAMIC_STATE_LINE_STIPPLE;
        line_pipe = pipeline(dev, layout,
                             (struct pipeline_desc){
                                 .vs = line,
                                 .fs = ab,
                                 .topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
                                 .w = LW,
                                 .h = 1,
                                 .rs = &line_rs,
                                 .rendering = &line_ri,
                                 .color_attachments = 1,
                                 .dynamic = &line_dyn,
                                 .dynamic_count = 1,
                             });
    }

    const VkImageAspectFlags D = VK_IMAGE_ASPECT_DEPTH_BIT, C = VK_IMAGE_ASPECT_COLOR_BIT;
    struct target unbiased = target(pd, dev, VK_FORMAT_D32_SFLOAT, DW, DH,
                                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, D);
    struct target biased = target(pd, dev, VK_FORMAT_D32_SFLOAT, DW, DH,
                                  VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, D);
    struct target stippled =
        target(pd, dev, VK_FORMAT_R32_UINT, LW, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, C);
    const VkDeviceSize UNBIASED_AT = 0, BIASED_AT = DW * DH * 4, LINE_AT = 2 * DW * DH * 4;
    struct mapped out = mapped(pd, dev, LINE_AT + LW * 4);

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

    // The two with nothing to read back: recorded, and scored by the worker log.
    vkCmdSetDeviceMask(cb, 1);
    vkCmdSetDepthBounds(cb, 0.0f, 1.0f);

    // Depth: off for the first image, on for the second. The bias is 2^20 units of the
    // smallest resolvable difference at 0.5, which is 2^-24 in D32, so it moves the depth to
    // about 0.5625 -- far enough that no rounding reads it as 0.5.
    struct target *depths[2] = {&unbiased, &biased};
    for (int on = 0; on < 2; on++) {
        to_layout(cb, depths[on]->image, D, VK_IMAGE_LAYOUT_UNDEFINED,
                  VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
        VkRenderingAttachmentInfo att = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = depths[on]->view,
            .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue = {.depthStencil = {1.0f, 0}},
        };
        VkRenderingInfo ri = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .renderArea = {{0, 0}, {DW, DH}},
            .layerCount = 1,
            .pDepthAttachment = &att,
        };
        vkCmdBeginRendering(cb, &ri);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, depth_pipe);
        vkCmdSetDepthBias(cb, 1048576.0f, 0.0f, 0.0f);
        vkCmdSetDepthBiasEnable(cb, on ? VK_TRUE : VK_FALSE);
        vkCmdDraw(cb, 3, 1, 0, 0);
        vkCmdEndRendering(cb);
        to_layout(cb, depths[on]->image, D, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        VkBufferImageCopy copy = {
            .bufferOffset = on ? BIASED_AT : UNBIASED_AT,
            .imageSubresource = {D, 0, 0, 1},
            .imageExtent = {DW, DH, 1},
        };
        vkCmdCopyImageToBuffer(cb, depths[on]->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               out.buf, 1, &copy);
    }

    if (stipple) {
        to_layout(cb, stippled.image, C, VK_IMAGE_LAYOUT_UNDEFINED,
                  VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        VkRenderingAttachmentInfo att = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = stippled.view,
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue = {.color = {.uint32 = {0, 0, 0, 0}}},
        };
        VkRenderingInfo ri = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .renderArea = {{0, 0}, {LW, 1}},
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &att,
        };
        vkCmdBeginRendering(cb, &ri);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, line_pipe);
        vkCmdSetLineStipple(cb, 2, 0x0f0f);
        vkCmdDraw(cb, 2, 1, 0, 0);
        vkCmdEndRendering(cb);
        to_layout(cb, stippled.image, C, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        VkBufferImageCopy copy = {
            .bufferOffset = LINE_AT,
            .imageSubresource = {C, 0, 0, 1},
            .imageExtent = {LW, 1, 1},
        };
        vkCmdCopyImageToBuffer(cb, stippled.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               out.buf, 1, &copy);
    }

    VkMemoryBarrier host = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
    };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
                         &host, 0, NULL, 0, NULL);
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
        uint32_t flat = 0, pushed = 0;
        float first_off = 0, first_on = 0;
        for (uint32_t i = 0; i < DW * DH; i++) {
            float off, on;
            memcpy(&off, out.p + UNBIASED_AT + i * 4, 4);
            memcpy(&on, out.p + BIASED_AT + i * 4, 4);
            if (i == 0) {
                first_off = off;
                first_on = on;
            }
            flat += off == 0.5f;
            pushed += on > 0.53f && on < 0.6f;
        }
        check(flat == DW * DH, "vkCmdSetDepthBiasEnable(FALSE): depth stays at 0.5");
        check(pushed == DW * DH, "vkCmdSetDepthBiasEnable(TRUE): depth pushed back");
        printf("  depth off %.6f, on %.6f\n", first_off, first_on);

        if (stipple) {
            uint32_t lit = 0, dark = 0;
            for (uint32_t x = 0; x < LW; x++) {
                uint32_t v;
                memcpy(&v, out.p + LINE_AT + x * 4, 4);
                if (x < 8) {
                    lit += v == 0xab;
                } else {
                    dark += v == 0;
                }
                printf("%s", v == 0xab ? "#" : v == 0 ? "." : "?");
            }
            printf("\n");
            check(lit == 8 && dark == 8, "vkCmdSetLineStipple: pattern 0x0f0f at factor 2");
        } else {
            printf("%-60s %s\n", "vkCmdSetLineStipple: pattern 0x0f0f at factor 2",
                   "skipped (not offered)");
        }
    }

    vkDestroyFence(dev, fence, NULL);
    vkDestroyCommandPool(dev, cmdpool, NULL);
    if (line_pipe) {
        vkDestroyPipeline(dev, line_pipe, NULL);
    }
    vkDestroyPipeline(dev, depth_pipe, NULL);
    vkDestroyShaderModule(dev, half, NULL);
    vkDestroyShaderModule(dev, line, NULL);
    vkDestroyShaderModule(dev, ab, NULL);
    vkDestroyPipelineLayout(dev, layout, NULL);
    untarget(dev, &unbiased);
    untarget(dev, &biased);
    untarget(dev, &stippled);
    vkUnmapMemory(dev, out.mem);
    vkDestroyBuffer(dev, out.buf, NULL);
    vkFreeMemory(dev, out.mem, NULL);
    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(inst, NULL);

    printf("\n%d check(s) failed\n", failures);
    return failures ? 1 : 0;
}
