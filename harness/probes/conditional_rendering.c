// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva
//
// The `conditional-rendering` group of src/venus/unserved.txt, as a program that names each
// command.
//
//   vkCmdBeginConditionalRenderingEXT  vkCmdEndConditionalRenderingEXT
//
// From VK_EXT_conditional_rendering, which KosmicKrisp and anv both advertise. One predicate
// buffer holds 0 at offset 0 and 1 at offset 256. Five renders, each into a fresh R32_UINT image
// cleared to 0, draw one full-screen triangle writing 0xab inside a conditional block:
//
//   case  predicate            flags     expect   a failure there means
//   1     offset 0   (0)      none      clear    the draw ran anyway: begin dropped
//   2     offset 256 (1)      none      drawn    discarded: the offset was dropped, read the 0
//   3     offset 0   (0)      INVERTED  drawn    the flag was dropped
//   4     offset 256 (1)      INVERTED  clear    the flag or the offset was dropped
//   5     offset 0   (0)      none      drawn    see below
//
// Case 5 draws once inside the block, which is discarded, then ends it and draws again: the
// second draw must land. A dropped vkCmdEndConditionalRenderingEXT leaves the predicate in force
// and the image clear. Begin and end sit inside the same dynamic rendering instance each time.
//
// The shaders, compiled with `glslangValidator -V --target-env vulkan1.1 -x`:
//
//   // tri.vert
//   #version 450
//   void main() {
//       vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
//       gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
//   }
//
//   // one.frag
//   #version 450
//   layout(location = 0) out uint a;
//   void main() {
//       a = 0xabu;
//   }
//
// Headless on purpose: no surface, no swapchain, no window. It runs over ssh.
//
//   cc -O1 -o conditional_rendering conditional_rendering.c -lvulkan && ./conditional_rendering
//
// Exit 0 = every check passed. Exit 1 = a check failed, and the line above it says which.
// Exit 2 = the device could not be brought up at all (not a verdict on the group).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define W 8u
#define H 8u
#define VALUE 0xabu
#define CASES 5u
#define ZERO_OFFSET 0u
#define ONE_OFFSET 256u

static const uint32_t VERT[] = {
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
    0x0004002b,0x00000006,0x00000022,0x3f800000,0x0004002b,0x00000006,0x00000025,0x00000000,
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

static const uint32_t FRAG[] = {
    0x07230203,0x00010300,0x0008000b,0x0000000a,0x00000000,0x00020011,0x00000001,0x0006000b,
    0x00000001,0x4c534c47,0x6474732e,0x3035342e,0x00000000,0x0003000e,0x00000000,0x00000001,
    0x0006000f,0x00000004,0x00000004,0x6e69616d,0x00000000,0x00000008,0x00030010,0x00000004,
    0x00000007,0x00030003,0x00000002,0x000001c2,0x00040005,0x00000004,0x6e69616d,0x00000000,
    0x00030005,0x00000008,0x00000061,0x00040047,0x00000008,0x0000001e,0x00000000,0x00020013,
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

// A colour attachment the probe renders into and then copies out.
struct target {
    VkImage image;
    VkImageView view;
    VkDeviceMemory mem;
};

static struct target target(VkPhysicalDevice pd, VkDevice dev) {
    struct target t;
    VkResult r;
    VkImageCreateInfo ici = {
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
        .format = VK_FORMAT_R32_UINT,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    if ((r = vkCreateImageView(dev, &vci, NULL, &t.view)) != VK_SUCCESS) {
        fatal("vkCreateImageView", r);
    }
    return t;
}

// A host-visible buffer, mapped for the probe's whole life.
struct hostbuf {
    VkBuffer buf;
    VkDeviceMemory mem;
    uint32_t *p;
};

static struct hostbuf hostbuf(VkPhysicalDevice pd, VkDevice dev, VkDeviceSize size,
                              VkBufferUsageFlags usage) {
    struct hostbuf b;
    VkResult r;
    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    if ((r = vkCreateBuffer(dev, &bci, NULL, &b.buf)) != VK_SUCCESS) {
        fatal("vkCreateBuffer", r);
    }
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev, b.buf, &mr);
    b.mem = bind(pd, dev, mr,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkBindBufferMemory(dev, b.buf, b.mem, 0);
    if ((r = vkMapMemory(dev, b.mem, 0, VK_WHOLE_SIZE, 0, (void **)&b.p)) != VK_SUCCESS) {
        fatal("vkMapMemory", r);
    }
    return b;
}

static void layout_barrier(VkCommandBuffer cb, VkImage image, VkImageLayout from, VkImageLayout to,
                           VkAccessFlags src_access, VkAccessFlags dst_access,
                           VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage) {
    VkImageMemoryBarrier b = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = src_access,
        .dstAccessMask = dst_access,
        .oldLayout = from,
        .newLayout = to,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    vkCmdPipelineBarrier(cb, src_stage, dst_stage, 0, 0, NULL, 0, NULL, 1, &b);
}

static int has_ext(VkExtensionProperties *exts, uint32_t n, const char *name) {
    for (uint32_t i = 0; i < n; i++) {
        if (strcmp(exts[i].extensionName, name) == 0) {
            return 1;
        }
    }
    return 0;
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

// How many texels of a W*H run hold `want`.
static uint32_t count(const uint32_t *run, uint32_t want) {
    uint32_t n = 0;
    for (uint32_t i = 0; i < W * H; i++) {
        n += run[i] == want;
    }
    return n;
}

int main(void) {
    VkResult r;

    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "venus-conditional-rendering-probe",
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
    if (props.apiVersion < VK_API_VERSION_1_3) {
        fatal("Vulkan 1.3 for dynamic rendering", VK_ERROR_FEATURE_NOT_PRESENT);
    }

    uint32_t en = 0;
    vkEnumerateDeviceExtensionProperties(pd, NULL, &en, NULL);
    VkExtensionProperties *exts = calloc(en, sizeof *exts);
    vkEnumerateDeviceExtensionProperties(pd, NULL, &en, exts);
    const char *ext = "VK_EXT_conditional_rendering";
    if (!has_ext(exts, en, ext)) {
        fatal("VK_EXT_conditional_rendering", VK_ERROR_EXTENSION_NOT_PRESENT);
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
    VkPhysicalDeviceConditionalRenderingFeaturesEXT fcr = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT,
        .conditionalRendering = VK_TRUE,
    };
    VkPhysicalDeviceVulkan13Features f13 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext = &fcr,
        .dynamicRendering = VK_TRUE,
    };
    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &f13,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qci,
        .enabledExtensionCount = 1,
        .ppEnabledExtensionNames = &ext,
    };
    VkDevice dev;
    if ((r = vkCreateDevice(pd, &dci, NULL, &dev)) != VK_SUCCESS) {
        fatal("vkCreateDevice", r);
    }
    VkQueue queue;
    vkGetDeviceQueue(dev, qfam, 0, &queue);

    PFN_vkCmdBeginConditionalRenderingEXT begin_conditional =
        (PFN_vkCmdBeginConditionalRenderingEXT)vkGetDeviceProcAddr(
            dev, "vkCmdBeginConditionalRenderingEXT");
    PFN_vkCmdEndConditionalRenderingEXT end_conditional =
        (PFN_vkCmdEndConditionalRenderingEXT)vkGetDeviceProcAddr(
            dev, "vkCmdEndConditionalRenderingEXT");
    if (!begin_conditional || !end_conditional) {
        fatal("the conditional rendering commands did not resolve", VK_ERROR_EXTENSION_NOT_PRESENT);
    }

    VkPipelineLayoutCreateInfo pli = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    VkPipelineLayout layout;
    if ((r = vkCreatePipelineLayout(dev, &pli, NULL, &layout)) != VK_SUCCESS) {
        fatal("vkCreatePipelineLayout", r);
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
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
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
    VkPipelineColorBlendStateCreateInfo cb_state = {
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
        .pColorBlendState = &cb_state,
        .layout = layout,
    };
    VkPipeline pipeline;
    if ((r = vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, NULL, &pipeline)) !=
        VK_SUCCESS) {
        fatal("vkCreateGraphicsPipelines", r);
    }

    // The predicate: 0 at offset 0 and 1 at offset 256, so a dropped offset reads the wrong one
    // whichever way it goes. Written before the submit, so the fence has nothing to wait for.
    struct hostbuf pred = hostbuf(pd, dev, 2 * ONE_OFFSET,
                                  VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT);
    memset(pred.p, 0, 2 * ONE_OFFSET);
    pred.p[ZERO_OFFSET / 4] = 0;
    pred.p[ONE_OFFSET / 4] = 1;

    struct target t[CASES];
    for (uint32_t k = 0; k < CASES; k++) {
        t[k] = target(pd, dev);
    }
    struct hostbuf out =
        hostbuf(pd, dev, CASES * W * H * sizeof(uint32_t), VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    memset(out.p, 0xee, CASES * W * H * sizeof(uint32_t));

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
    for (uint32_t k = 0; k < CASES; k++) {
        layout_barrier(cb, t[k].image, VK_IMAGE_LAYOUT_UNDEFINED,
                       VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 0,
                       VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
    }

    struct {
        VkDeviceSize offset;
        VkConditionalRenderingFlagsEXT flags;
    } cases[CASES] = {
        {ZERO_OFFSET, 0},
        {ONE_OFFSET, 0},
        {ZERO_OFFSET, VK_CONDITIONAL_RENDERING_INVERTED_BIT_EXT},
        {ONE_OFFSET, VK_CONDITIONAL_RENDERING_INVERTED_BIT_EXT},
        {ZERO_OFFSET, 0},
    };
    for (uint32_t k = 0; k < CASES; k++) {
        VkRenderingAttachmentInfo att = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = t[k].view,
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
        VkConditionalRenderingBeginInfoEXT cbi = {
            .sType = VK_STRUCTURE_TYPE_CONDITIONAL_RENDERING_BEGIN_INFO_EXT,
            .buffer = pred.buf,
            .offset = cases[k].offset,
            .flags = cases[k].flags,
        };
        vkCmdBeginRendering(cb, &ri);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        begin_conditional(cb, &cbi);
        vkCmdDraw(cb, 3, 1, 0, 0);
        end_conditional(cb);
        if (k == CASES - 1) {
            // After the end, unconditional again.
            vkCmdDraw(cb, 3, 1, 0, 0);
        }
        vkCmdEndRendering(cb);
    }

    for (uint32_t k = 0; k < CASES; k++) {
        layout_barrier(cb, t[k].image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                       VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferImageCopy copy = {
            .bufferOffset = k * W * H * sizeof(uint32_t),
            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .imageExtent = {W, H, 1},
        };
        vkCmdCopyImageToBuffer(cb, t[k].image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, out.buf, 1,
                               &copy);
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
        const uint32_t all = W * H;
        const uint32_t *run[CASES];
        for (uint32_t k = 0; k < CASES; k++) {
            run[k] = out.p + k * all;
        }
        check(count(run[0], 0) == all, "vkCmdBeginConditionalRenderingEXT: predicate 0 discards");
        check(count(run[1], VALUE) == all, "offset 256, predicate 1: drawn, the offset honoured");
        check(count(run[2], VALUE) == all, "INVERTED, predicate 0: drawn");
        check(count(run[3], 0) == all, "INVERTED, predicate 1: discarded");
        check(count(run[4], VALUE) == all,
              "vkCmdEndConditionalRenderingEXT: the draw after it lands");
        printf("  first texels %#x, %#x, %#x, %#x, %#x (want 0, 0xab, 0xab, 0, 0xab)\n", run[0][0],
               run[1][0], run[2][0], run[3][0], run[4][0]);
    }

    vkDestroyFence(dev, fence, NULL);
    vkDestroyCommandPool(dev, cmdpool, NULL);
    vkDestroyPipeline(dev, pipeline, NULL);
    vkDestroyShaderModule(dev, vs, NULL);
    vkDestroyShaderModule(dev, fs, NULL);
    vkDestroyPipelineLayout(dev, layout, NULL);
    for (uint32_t k = 0; k < CASES; k++) {
        vkDestroyImageView(dev, t[k].view, NULL);
        vkDestroyImage(dev, t[k].image, NULL);
        vkFreeMemory(dev, t[k].mem, NULL);
    }
    vkUnmapMemory(dev, out.mem);
    vkDestroyBuffer(dev, out.buf, NULL);
    vkFreeMemory(dev, out.mem, NULL);
    vkUnmapMemory(dev, pred.mem);
    vkDestroyBuffer(dev, pred.buf, NULL);
    vkFreeMemory(dev, pred.mem, NULL);
    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(inst, NULL);

    printf("\n%d check(s) failed\n", failures);
    return failures ? 1 : 0;
}
