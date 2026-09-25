// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva
//
// The `depth-bias-control` group of src/venus/unserved.txt, as a program that names its command.
//
//   vkCmdSetDepthBias2EXT
//
// From VK_EXT_depth_bias_control, which anv advertises and KosmicKrisp does not, so the positive
// control and the venus run are both on a Linux host.
//
// A full-viewport triangle at depth 0.5, with the depth test ALWAYS passing and the bias switched
// on in the pipeline, is drawn into fresh depth images cleared to 1.0. Each draw sets its own
// bias with vkCmdSetDepthBias2EXT, and each result is read back:
//
//   - D32_SFLOAT, constant 2^20 and then 2^21. At 0.5 a float format's unit is
//     2^(e-23) = 2^-24, so the depth lands at 0.5625 and then 0.625. A dropped first command
//     leaves the bias undefined; a dropped second leaves 0.5625 where 0.625 belongs.
//   - D16_UNORM, constant 2^-8, with a VkDepthBiasRepresentationInfoEXT asking for
//     VK_DEPTH_BIAS_REPRESENTATION_FLOAT_EXT, whose unit is 1: the constant is a depth, and
//     the depth lands at 0.5 + 2^-8. Without the chained struct the unit is the format's, about
//     2^-16, and the same constant moves the depth by 2^-24, which D16 cannot hold -- it reads
//     0.5. So this is the one read that says the chained struct arrived, not only the command.
//     Where `floatRepresentation` is not offered it is reported as skipped.
//   - D16_UNORM, constant 2^8, the format's representation with `depthBiasExact` set, where it
//     is offered: the unit is then 2^-16 and the depth lands at 0.5 + 2^-8 again. An
//     implementation free to scale may use up to twice that; whether anv does is not something
//     this program knows, so this read shows the exact bit was honoured, not that it was needed.
//
// The representation leastRepresentableValueForceUnormRepresentation is not exercised: anv, the
// only host with the extension, does not offer it.
//
// Tolerances. The D32 answers are exactly representable, and 2^-20 (16 ulps near 0.6) allows
// for a rasteriser that evaluates the plane in slightly different precision without letting any
// expected value be mistaken for another: they are 0.0625 apart. The D16 answers are quantized
// to 1/65535, and are accepted within two steps of it; the alternatives they are told apart from
// are 2^-8 away, some 256 steps.
//
// The shader, compiled with `glslangValidator -V --target-env vulkan1.1 -x`; there is no
// fragment shader, as only depth is written:
//
//   // half.vert
//   #version 450
//   void main() {
//       vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
//       gl_Position = vec4(p * 2.0 - 1.0, 0.5, 1.0);
//   }
//
// Headless on purpose: no surface, no swapchain, no window. It runs over ssh.
//
//   cc -O1 -o depth_bias2 depth_bias2.c -lvulkan && ./depth_bias2
//
// Exit 0 = every check passed. Exit 1 = a check failed, and the line above it says which.
// Exit 2 = the device could not be brought up at all (not a verdict on the group).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define W 4u
#define H 4u
#define RENDERS 4u
#define SLOT (W * H * 4u)

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

static struct target target(VkPhysicalDevice pd, VkDevice dev, VkFormat format) {
    struct target t;
    VkResult r;
    VkImageCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .extent = {W, H, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
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
        .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1},
    };
    if ((r = vkCreateImageView(dev, &vci, NULL, &t.view)) != VK_SUCCESS) {
        fatal("vkCreateImageView", r);
    }
    return t;
}

static void to_layout(VkCommandBuffer cb, VkImage image, VkImageLayout from, VkImageLayout to) {
    VkImageMemoryBarrier b = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        .oldLayout = from,
        .newLayout = to,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1},
    };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &b);
}

static int has_ext(VkExtensionProperties *exts, uint32_t n, const char *name) {
    for (uint32_t i = 0; i < n; i++) {
        if (strcmp(exts[i].extensionName, name) == 0) {
            return 1;
        }
    }
    return 0;
}

static int depth_format_ok(VkPhysicalDevice pd, VkFormat format) {
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(pd, format, &fp);
    VkFormatFeatureFlags want =
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT;
    return (fp.optimalTilingFeatures & want) == want;
}

static VkPipeline pipeline(VkDevice dev, VkPipelineLayout layout, VkShaderModule vs,
                           VkFormat format) {
    VkPipelineShaderStageCreateInfo stage = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT,
        .module = vs,
        .pName = "main",
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
    // The bias is on in the pipeline; its factors are dynamic, and the pipeline's are zero.
    VkPipelineRasterizationStateCreateInfo rs = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .depthBiasEnable = VK_TRUE,
        .lineWidth = 1.0f,
    };
    VkPipelineMultisampleStateCreateInfo ms = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };
    VkPipelineDepthStencilStateCreateInfo depth = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_ALWAYS,
    };
    VkPipelineColorBlendStateCreateInfo cbs = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
    };
    VkDynamicState dynamic = VK_DYNAMIC_STATE_DEPTH_BIAS;
    VkPipelineDynamicStateCreateInfo ds = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 1,
        .pDynamicStates = &dynamic,
    };
    VkPipelineRenderingCreateInfo prci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .depthAttachmentFormat = format,
    };
    VkGraphicsPipelineCreateInfo gpci = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &prci,
        .stageCount = 1,
        .pStages = &stage,
        .pVertexInputState = &vi,
        .pInputAssemblyState = &ia,
        .pViewportState = &vp,
        .pRasterizationState = &rs,
        .pMultisampleState = &ms,
        .pDepthStencilState = &depth,
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

// One draw: its depth format, the bias it sets, and the depth that bias should leave behind.
struct render {
    const char *what;
    VkFormat format;
    float constant;
    int chain; // whether a VkDepthBiasRepresentationInfoEXT goes in the pNext
    VkDepthBiasRepresentationEXT representation;
    VkBool32 exact;
    double want;
    int offered;
};

// How many texels of a slot are within `tolerance` of `want`, and the first one's depth.
static uint32_t count(const uint8_t *slot, VkFormat format, double want, double tolerance,
                      double *first) {
    uint32_t n = 0;
    for (uint32_t i = 0; i < W * H; i++) {
        double v;
        if (format == VK_FORMAT_D32_SFLOAT) {
            float f;
            memcpy(&f, slot + i * 4, 4);
            v = f;
        } else {
            uint16_t u;
            memcpy(&u, slot + i * 2, 2);
            v = u / 65535.0;
        }
        if (i == 0) {
            *first = v;
        }
        n += v >= want - tolerance && v <= want + tolerance;
    }
    return n;
}

int main(void) {
    VkResult r;

    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "venus-depth-bias-control-probe",
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
    if (props.apiVersion < VK_API_VERSION_1_3) {
        fatal("Vulkan 1.3 for dynamic rendering", VK_ERROR_FEATURE_NOT_PRESENT);
    }

    uint32_t en = 0;
    vkEnumerateDeviceExtensionProperties(pd, NULL, &en, NULL);
    VkExtensionProperties *exts = calloc(en, sizeof *exts);
    vkEnumerateDeviceExtensionProperties(pd, NULL, &en, exts);
    const char *ext = "VK_EXT_depth_bias_control";
    if (!has_ext(exts, en, ext)) {
        fatal("VK_EXT_depth_bias_control", VK_ERROR_EXTENSION_NOT_PRESENT);
    }

    VkPhysicalDeviceDepthBiasControlFeaturesEXT has_bias = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_BIAS_CONTROL_FEATURES_EXT,
    };
    VkPhysicalDeviceFeatures2 has = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &has_bias,
    };
    vkGetPhysicalDeviceFeatures2(pd, &has);
    if (!has_bias.depthBiasControl) {
        fatal("depthBiasControl", VK_ERROR_FEATURE_NOT_PRESENT);
    }
    printf("float representation: %s, exact: %s\n\n",
           has_bias.floatRepresentation ? "offered" : "not offered",
           has_bias.depthBiasExact ? "offered" : "not offered");

    if (!depth_format_ok(pd, VK_FORMAT_D32_SFLOAT)) {
        fatal("D32_SFLOAT as a copyable depth attachment", VK_ERROR_FORMAT_NOT_SUPPORTED);
    }
    if (!depth_format_ok(pd, VK_FORMAT_D16_UNORM)) {
        fatal("D16_UNORM as a copyable depth attachment", VK_ERROR_FORMAT_NOT_SUPPORTED);
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
    VkPhysicalDeviceDepthBiasControlFeaturesEXT fbias = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_BIAS_CONTROL_FEATURES_EXT,
        .depthBiasControl = VK_TRUE,
        .floatRepresentation = has_bias.floatRepresentation,
        .depthBiasExact = has_bias.depthBiasExact,
    };
    VkPhysicalDeviceVulkan13Features f13 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext = &fbias,
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

    PFN_vkCmdSetDepthBias2EXT set_depth_bias2 =
        (PFN_vkCmdSetDepthBias2EXT)vkGetDeviceProcAddr(dev, "vkCmdSetDepthBias2EXT");
    if (!set_depth_bias2) {
        fatal("vkCmdSetDepthBias2EXT did not resolve", VK_ERROR_EXTENSION_NOT_PRESENT);
    }

    VkPipelineLayoutCreateInfo pli = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    VkPipelineLayout layout;
    if ((r = vkCreatePipelineLayout(dev, &pli, NULL, &layout)) != VK_SUCCESS) {
        fatal("vkCreatePipelineLayout", r);
    }
    VkShaderModule half = module(dev, HALF_VERT, sizeof HALF_VERT);
    VkPipeline d32_pipe = pipeline(dev, layout, half, VK_FORMAT_D32_SFLOAT);
    VkPipeline d16_pipe = pipeline(dev, layout, half, VK_FORMAT_D16_UNORM);

    const VkDepthBiasRepresentationEXT FORMAT =
        VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORMAT_EXT;
    struct render renders[RENDERS] = {
        {"D32, constant 2^20: depth at 0.5 + 2^-4", VK_FORMAT_D32_SFLOAT, 1048576.0f, 0, FORMAT,
         VK_FALSE, 0.5625, 1},
        {"D32, constant 2^21: depth at 0.5 + 2^-3", VK_FORMAT_D32_SFLOAT, 2097152.0f, 0, FORMAT,
         VK_FALSE, 0.625, 1},
        {"D16, float representation, 2^-8: 0.5 + 2^-8", VK_FORMAT_D16_UNORM, 0.00390625f, 1,
         VK_DEPTH_BIAS_REPRESENTATION_FLOAT_EXT, VK_FALSE, 0.50390625,
         has_bias.floatRepresentation},
        {"D16, exact format representation, 2^8: 0.5 + 2^-8", VK_FORMAT_D16_UNORM, 256.0f, 1,
         FORMAT, VK_TRUE, 0.50390625, has_bias.depthBiasExact},
    };
    struct target t[RENDERS];
    for (uint32_t k = 0; k < RENDERS; k++) {
        t[k] = target(pd, dev, renders[k].format);
    }
    struct mapped out = mapped(pd, dev, RENDERS * SLOT);

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

    for (uint32_t k = 0; k < RENDERS; k++) {
        const struct render *d = &renders[k];
        if (!d->offered) {
            continue;
        }
        to_layout(cb, t[k].image, VK_IMAGE_LAYOUT_UNDEFINED,
                  VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
        VkRenderingAttachmentInfo att = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = t[k].view,
            .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue = {.depthStencil = {1.0f, 0}},
        };
        VkRenderingInfo ri = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .renderArea = {{0, 0}, {W, H}},
            .layerCount = 1,
            .pDepthAttachment = &att,
        };
        vkCmdBeginRendering(cb, &ri);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          d->format == VK_FORMAT_D32_SFLOAT ? d32_pipe : d16_pipe);
        VkDepthBiasRepresentationInfoEXT rep = {
            .sType = VK_STRUCTURE_TYPE_DEPTH_BIAS_REPRESENTATION_INFO_EXT,
            .depthBiasRepresentation = d->representation,
            .depthBiasExact = d->exact,
        };
        VkDepthBiasInfoEXT info = {
            .sType = VK_STRUCTURE_TYPE_DEPTH_BIAS_INFO_EXT,
            .pNext = d->chain ? &rep : NULL,
            .depthBiasConstantFactor = d->constant,
            .depthBiasClamp = 0.0f,
            .depthBiasSlopeFactor = 0.0f,
        };
        set_depth_bias2(cb, &info);
        vkCmdDraw(cb, 3, 1, 0, 0);
        vkCmdEndRendering(cb);
        to_layout(cb, t[k].image, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        VkBufferImageCopy copy = {
            .bufferOffset = k * SLOT,
            .imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1},
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
        for (uint32_t k = 0; k < RENDERS; k++) {
            const struct render *d = &renders[k];
            if (!d->offered) {
                printf("%-60s %s\n", d->what, "skipped (not offered)");
                continue;
            }
            double tolerance = d->format == VK_FORMAT_D32_SFLOAT ? 1.0 / (1 << 20) : 2.0 / 65535;
            double first = 0;
            uint32_t hit = count(out.p + k * SLOT, d->format, d->want, tolerance, &first);
            check(hit == W * H, d->what);
            printf("  %u of %u texels, first at %.8f\n", hit, W * H, first);
        }
    }

    vkDestroyFence(dev, fence, NULL);
    vkDestroyCommandPool(dev, cmdpool, NULL);
    vkDestroyPipeline(dev, d32_pipe, NULL);
    vkDestroyPipeline(dev, d16_pipe, NULL);
    vkDestroyShaderModule(dev, half, NULL);
    vkDestroyPipelineLayout(dev, layout, NULL);
    for (uint32_t k = 0; k < RENDERS; k++) {
        vkDestroyImageView(dev, t[k].view, NULL);
        vkDestroyImage(dev, t[k].image, NULL);
        vkFreeMemory(dev, t[k].mem, NULL);
    }
    vkUnmapMemory(dev, out.mem);
    vkDestroyBuffer(dev, out.buf, NULL);
    vkFreeMemory(dev, out.mem, NULL);
    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(inst, NULL);

    printf("\n%d check(s) failed\n", failures);
    return failures ? 1 : 0;
}
