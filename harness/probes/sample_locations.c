// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva
//
// The `sample-locations` group of src/venus/unserved.txt, as a program that names its command.
//
//   vkCmdSetSampleLocationsEXT
//
// From VK_EXT_sample_locations, which both KosmicKrisp and anv advertise. A white rectangle covers
// [0, 4.5) x [0, 4.5) of an 8x8, 4-sample R8G8B8A8_UNORM target, so the pixels of column 4 and row
// 4 are cut in half by an edge, and pixel (4, 4) in quarters. Which of their samples the rectangle
// covers is decided by where the samples are, and the average resolve turns that into a value:
// 255 for all four, 0 for none, 191 for three, 64 for one.
//
// Each render is its own rendering instance, so the probe needs no variableSampleLocations, and
// sets its locations after the pipeline is bound and before its one draw:
//
//   all four at (1/16, 1/16)         column 4, row 4 and the corner fully covered
//   all four at (15/16, 15/16)       none of them covered
//   three at (1/16, 15/16) and one   column 4 three quarters covered, row 4 one quarter, the
//   at (15/16, 1/16)                 corner not at all
//
// A dropped command leaves the standard pattern, or the previous render's, and a garbled one
// swaps x for y or cuts the count short; each reads back as a different set of values. A fourth
// render, first, uses a pipeline without custom locations: on a device with the standard pattern
// it reads a half on both edges and a quarter at the corner, which proves the edge pixels are
// partial and the resolve averages, so the three renders above are not agreeing by accident.
// The interior pixel (1, 1) and the exterior (6, 6) are every render's own control.
//
// The shaders, compiled with `glslangValidator -V --target-env vulkan1.1 -x`:
//
//   // rect.vert
//   #version 450
//   void main() {
//       vec2 p = vec2((gl_VertexIndex & 1) != 0 ? 0.125 : -1.0,
//                     (gl_VertexIndex & 2) != 0 ? 0.125 : -1.0);
//       gl_Position = vec4(p, 0.0, 1.0);
//   }
//
//   // white.frag
//   #version 450
//   layout(location = 0) out vec4 c;
//   void main() {
//       c = vec4(1.0);
//   }
//
// Headless on purpose: no surface, no swapchain, no window. It runs over ssh.
//
//   cc -O1 -o sample_locations sample_locations.c -lvulkan && ./sample_locations
//
// Exit 0 = every check passed. Exit 1 = a check failed, and the line above it says which.
// Exit 2 = the device could not be brought up at all (not a verdict on the group).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define W 8u
#define H 8u
#define SAMPLES VK_SAMPLE_COUNT_4_BIT
#define RENDERS 4u
#define FORMAT VK_FORMAT_R8G8B8A8_UNORM

static const uint32_t VERT[] = {
    0x07230203,0x00010300,0x0008000b,0x0000002b,0x00000000,0x00020011,0x00000001,0x0006000b,
    0x00000001,0x4c534c47,0x6474732e,0x3035342e,0x00000000,0x0003000e,0x00000000,0x00000001,
    0x0007000f,0x00000000,0x00000004,0x6e69616d,0x00000000,0x0000000c,0x00000022,0x00030003,
    0x00000002,0x000001c2,0x00040005,0x00000004,0x6e69616d,0x00000000,0x00030005,0x00000009,
    0x00000070,0x00060005,0x0000000c,0x565f6c67,0x65747265,0x646e4978,0x00007865,0x00060005,
    0x00000020,0x505f6c67,0x65567265,0x78657472,0x00000000,0x00060006,0x00000020,0x00000000,
    0x505f6c67,0x7469736f,0x006e6f69,0x00070006,0x00000020,0x00000001,0x505f6c67,0x746e696f,
    0x657a6953,0x00000000,0x00070006,0x00000020,0x00000002,0x435f6c67,0x4470696c,0x61747369,
    0x0065636e,0x00070006,0x00000020,0x00000003,0x435f6c67,0x446c6c75,0x61747369,0x0065636e,
    0x00030005,0x00000022,0x00000000,0x00040047,0x0000000c,0x0000000b,0x0000002a,0x00030047,
    0x00000020,0x00000002,0x00050048,0x00000020,0x00000000,0x0000000b,0x00000000,0x00050048,
    0x00000020,0x00000001,0x0000000b,0x00000001,0x00050048,0x00000020,0x00000002,0x0000000b,
    0x00000003,0x00050048,0x00000020,0x00000003,0x0000000b,0x00000004,0x00020013,0x00000002,
    0x00030021,0x00000003,0x00000002,0x00030016,0x00000006,0x00000020,0x00040017,0x00000007,
    0x00000006,0x00000002,0x00040020,0x00000008,0x00000007,0x00000007,0x00040015,0x0000000a,
    0x00000020,0x00000001,0x00040020,0x0000000b,0x00000001,0x0000000a,0x0004003b,0x0000000b,
    0x0000000c,0x00000001,0x0004002b,0x0000000a,0x0000000e,0x00000001,0x0004002b,0x0000000a,
    0x00000010,0x00000000,0x00020014,0x00000011,0x0004002b,0x00000006,0x00000013,0x3e000000,
    0x0004002b,0x00000006,0x00000014,0xbf800000,0x0004002b,0x0000000a,0x00000017,0x00000002,
    0x00040017,0x0000001c,0x00000006,0x00000004,0x00040015,0x0000001d,0x00000020,0x00000000,
    0x0004002b,0x0000001d,0x0000001e,0x00000001,0x0004001c,0x0000001f,0x00000006,0x0000001e,
    0x0006001e,0x00000020,0x0000001c,0x00000006,0x0000001f,0x0000001f,0x00040020,0x00000021,
    0x00000003,0x00000020,0x0004003b,0x00000021,0x00000022,0x00000003,0x0004002b,0x00000006,
    0x00000024,0x00000000,0x0004002b,0x00000006,0x00000025,0x3f800000,0x00040020,0x00000029,
    0x00000003,0x0000001c,0x00050036,0x00000002,0x00000004,0x00000000,0x00000003,0x000200f8,
    0x00000005,0x0004003b,0x00000008,0x00000009,0x00000007,0x0004003d,0x0000000a,0x0000000d,
    0x0000000c,0x000500c7,0x0000000a,0x0000000f,0x0000000d,0x0000000e,0x000500ab,0x00000011,
    0x00000012,0x0000000f,0x00000010,0x000600a9,0x00000006,0x00000015,0x00000012,0x00000013,
    0x00000014,0x0004003d,0x0000000a,0x00000016,0x0000000c,0x000500c7,0x0000000a,0x00000018,
    0x00000016,0x00000017,0x000500ab,0x00000011,0x00000019,0x00000018,0x00000010,0x000600a9,
    0x00000006,0x0000001a,0x00000019,0x00000013,0x00000014,0x00050050,0x00000007,0x0000001b,
    0x00000015,0x0000001a,0x0003003e,0x00000009,0x0000001b,0x0004003d,0x00000007,0x00000023,
    0x00000009,0x00050051,0x00000006,0x00000026,0x00000023,0x00000000,0x00050051,0x00000006,
    0x00000027,0x00000023,0x00000001,0x00070050,0x0000001c,0x00000028,0x00000026,0x00000027,
    0x00000024,0x00000025,0x00050041,0x00000029,0x0000002a,0x00000022,0x00000010,0x0003003e,
    0x0000002a,0x00000028,0x000100fd,0x00010038
};

static const uint32_t FRAG[] = {
    0x07230203,0x00010300,0x0008000b,0x0000000c,0x00000000,0x00020011,0x00000001,0x0006000b,
    0x00000001,0x4c534c47,0x6474732e,0x3035342e,0x00000000,0x0003000e,0x00000000,0x00000001,
    0x0006000f,0x00000004,0x00000004,0x6e69616d,0x00000000,0x00000009,0x00030010,0x00000004,
    0x00000007,0x00030003,0x00000002,0x000001c2,0x00040005,0x00000004,0x6e69616d,0x00000000,
    0x00030005,0x00000009,0x00000063,0x00040047,0x00000009,0x0000001e,0x00000000,0x00020013,
    0x00000002,0x00030021,0x00000003,0x00000002,0x00030016,0x00000006,0x00000020,0x00040017,
    0x00000007,0x00000006,0x00000004,0x00040020,0x00000008,0x00000003,0x00000007,0x0004003b,
    0x00000008,0x00000009,0x00000003,0x0004002b,0x00000006,0x0000000a,0x3f800000,0x0007002c,
    0x00000007,0x0000000b,0x0000000a,0x0000000a,0x0000000a,0x0000000a,0x00050036,0x00000002,
    0x00000004,0x00000000,0x00000003,0x000200f8,0x00000005,0x0003003e,0x00000009,0x0000000b,
    0x000100fd,0x00010038
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

// A colour image with a view: the multisampled target, or the one-sample image it resolves into.
struct target {
    VkImage image;
    VkImageView view;
    VkDeviceMemory mem;
};

static struct target target(VkPhysicalDevice pd, VkDevice dev, VkSampleCountFlagBits samples) {
    struct target t;
    VkResult r;
    VkImageCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = FORMAT,
        .extent = {W, H, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = samples,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                 (samples == VK_SAMPLE_COUNT_1_BIT ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0),
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
        .format = FORMAT,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    if ((r = vkCreateImageView(dev, &vci, NULL, &t.view)) != VK_SUCCESS) {
        fatal("vkCreateImageView", r);
    }
    return t;
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

// The red channel of one resolved texel.
static uint32_t red(const uint8_t *run, uint32_t x, uint32_t y) {
    return run[(y * W + x) * 4];
}

// What a render should resolve to, in quarters of full coverage: the column-4 pixels above the
// corner, the row-4 pixels left of it, and the corner itself.
struct expect {
    const char *what;
    uint32_t column, row, corner;
};

// Whether a resolved value is `quarters` of 255, give or take the resolve's rounding.
static int near(uint32_t got, uint32_t quarters) {
    int want = (int)(quarters * 255 + 2) / 4;
    int d = (int)got - want;
    return d >= -2 && d <= 2;
}

static int scored(const uint8_t *run, struct expect e) {
    int ok = near(red(run, 1, 1), 4) && near(red(run, 6, 6), 0) && near(red(run, 4, 4), e.corner);
    for (uint32_t i = 0; i < 4; i++) {
        ok = ok && near(red(run, 4, i), e.column) && near(red(run, i, 4), e.row);
    }
    printf("  interior %u, exterior %u, column %u %u %u %u, row %u %u %u %u, corner %u\n",
           red(run, 1, 1), red(run, 6, 6), red(run, 4, 0), red(run, 4, 1), red(run, 4, 2),
           red(run, 4, 3), red(run, 0, 4), red(run, 1, 4), red(run, 2, 4), red(run, 3, 4),
           red(run, 4, 4));
    return ok;
}

static VkPipeline pipeline(VkDevice dev, VkPipelineLayout layout, VkShaderModule vs,
                           VkShaderModule fs, int custom) {
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
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
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
    // The locations here are ignored: the dynamic state replaces them. The struct still has to be
    // there for sampleLocationsEnable.
    VkPipelineSampleLocationsStateCreateInfoEXT sl = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SAMPLE_LOCATIONS_STATE_CREATE_INFO_EXT,
        .sampleLocationsEnable = VK_TRUE,
        .sampleLocationsInfo = {.sType = VK_STRUCTURE_TYPE_SAMPLE_LOCATIONS_INFO_EXT},
    };
    VkPipelineMultisampleStateCreateInfo ms = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .pNext = custom ? &sl : NULL,
        .rasterizationSamples = SAMPLES,
    };
    VkPipelineColorBlendAttachmentState blend = {
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };
    VkPipelineColorBlendStateCreateInfo cb_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &blend,
    };
    VkDynamicState dynamic = VK_DYNAMIC_STATE_SAMPLE_LOCATIONS_EXT;
    VkPipelineDynamicStateCreateInfo ds = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 1,
        .pDynamicStates = &dynamic,
    };
    VkFormat format = FORMAT;
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
        .pDynamicState = custom ? &ds : NULL,
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
        .pApplicationName = "venus-sample-locations-probe",
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

    VkPhysicalDeviceSampleLocationsPropertiesEXT sl_props = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLE_LOCATIONS_PROPERTIES_EXT,
    };
    VkPhysicalDeviceProperties2 props2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &sl_props,
    };
    vkGetPhysicalDeviceProperties2(pd, &props2);
    VkPhysicalDeviceProperties props = props2.properties;
    printf("device: %s (Vulkan %u.%u)\n", props.deviceName,
           VK_API_VERSION_MAJOR(props.apiVersion), VK_API_VERSION_MINOR(props.apiVersion));
    if (props.apiVersion < VK_API_VERSION_1_3) {
        fatal("Vulkan 1.3 for dynamic rendering", VK_ERROR_FEATURE_NOT_PRESENT);
    }

    uint32_t en = 0;
    vkEnumerateDeviceExtensionProperties(pd, NULL, &en, NULL);
    VkExtensionProperties *exts = calloc(en, sizeof *exts);
    vkEnumerateDeviceExtensionProperties(pd, NULL, &en, exts);
    const char *ext = "VK_EXT_sample_locations";
    if (!has_ext(exts, en, ext)) {
        fatal("VK_EXT_sample_locations", VK_ERROR_EXTENSION_NOT_PRESENT);
    }
    if (!(sl_props.sampleLocationSampleCounts & SAMPLES)) {
        fatal("4 in sampleLocationSampleCounts", VK_ERROR_FEATURE_NOT_PRESENT);
    }
    if (!(props.limits.framebufferColorSampleCounts & SAMPLES)) {
        fatal("4 in framebufferColorSampleCounts", VK_ERROR_FEATURE_NOT_PRESENT);
    }
    if (sl_props.sampleLocationSubPixelBits < 4 || sl_props.sampleLocationCoordinateRange[0] > 0.0625f ||
        sl_props.sampleLocationCoordinateRange[1] < 0.9375f) {
        fatal("sample locations at 1/16 and 15/16", VK_ERROR_FEATURE_NOT_PRESENT);
    }
    PFN_vkGetPhysicalDeviceMultisamplePropertiesEXT get_ms_props =
        (PFN_vkGetPhysicalDeviceMultisamplePropertiesEXT)vkGetInstanceProcAddr(
            inst, "vkGetPhysicalDeviceMultisamplePropertiesEXT");
    if (!get_ms_props) {
        fatal("vkGetPhysicalDeviceMultisamplePropertiesEXT did not resolve",
              VK_ERROR_EXTENSION_NOT_PRESENT);
    }
    VkMultisamplePropertiesEXT msp = {.sType = VK_STRUCTURE_TYPE_MULTISAMPLE_PROPERTIES_EXT};
    get_ms_props(pd, SAMPLES, &msp);
    if (msp.maxSampleLocationGridSize.width < 1 || msp.maxSampleLocationGridSize.height < 1) {
        fatal("a 1x1 sample location grid at 4 samples", VK_ERROR_FEATURE_NOT_PRESENT);
    }
    printf("variableSampleLocations %u, standardSampleLocations %u (neither is required)\n",
           sl_props.variableSampleLocations, props.limits.standardSampleLocations);
    int standard = props.limits.standardSampleLocations;
    if (!standard) {
        printf("note: no standard sample locations; the baseline render is not scored\n");
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
    VkPhysicalDeviceVulkan13Features f13 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
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

    PFN_vkCmdSetSampleLocationsEXT set_sample_locations =
        (PFN_vkCmdSetSampleLocationsEXT)vkGetDeviceProcAddr(dev, "vkCmdSetSampleLocationsEXT");
    if (!set_sample_locations) {
        fatal("vkCmdSetSampleLocationsEXT did not resolve", VK_ERROR_EXTENSION_NOT_PRESENT);
    }

    VkPipelineLayoutCreateInfo pli = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    VkPipelineLayout layout;
    if ((r = vkCreatePipelineLayout(dev, &pli, NULL, &layout)) != VK_SUCCESS) {
        fatal("vkCreatePipelineLayout", r);
    }
    VkShaderModule vs = module(dev, VERT, sizeof VERT), fs = module(dev, FRAG, sizeof FRAG);
    VkPipeline standard_pipeline = pipeline(dev, layout, vs, fs, 0);
    VkPipeline custom_pipeline = pipeline(dev, layout, vs, fs, 1);

    // Render 0 is the standard-pattern baseline; renders 1-3 set their own locations.
    const float lo = 1.0f / 16, hi = 15.0f / 16;
    VkSampleLocationEXT sets[RENDERS][4] = {
        {{0}},
        {{lo, lo}, {lo, lo}, {lo, lo}, {lo, lo}},
        {{hi, hi}, {hi, hi}, {hi, hi}, {hi, hi}},
        {{lo, hi}, {lo, hi}, {lo, hi}, {hi, lo}},
    };
    struct expect expect[RENDERS] = {
        {"standard pattern: edges half covered, corner a quarter", 2, 2, 1},
        {"all at (1/16, 1/16): edges and corner fully covered", 4, 4, 4},
        {"all at (15/16, 15/16): edges and corner uncovered", 0, 0, 0},
        {"mixed: column three quarters, row one, corner none", 3, 1, 0},
    };

    struct target ms[RENDERS], resolved[RENDERS];
    for (uint32_t k = 0; k < RENDERS; k++) {
        ms[k] = target(pd, dev, SAMPLES);
        resolved[k] = target(pd, dev, VK_SAMPLE_COUNT_1_BIT);
    }

    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = RENDERS * W * H * 4,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VkBuffer out;
    if ((r = vkCreateBuffer(dev, &bci, NULL, &out)) != VK_SUCCESS) {
        fatal("vkCreateBuffer", r);
    }
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev, out, &mr);
    VkDeviceMemory out_mem =
        bind(pd, dev, mr, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkBindBufferMemory(dev, out, out_mem, 0);
    uint8_t *texels;
    if ((r = vkMapMemory(dev, out_mem, 0, VK_WHOLE_SIZE, 0, (void **)&texels)) != VK_SUCCESS) {
        fatal("vkMapMemory", r);
    }
    memset(texels, 0xee, RENDERS * W * H * 4);

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
        layout_barrier(cb, ms[k].image, VK_IMAGE_LAYOUT_UNDEFINED,
                       VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 0,
                       VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
        layout_barrier(cb, resolved[k].image, VK_IMAGE_LAYOUT_UNDEFINED,
                       VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 0,
                       VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
    }

    for (uint32_t k = 0; k < RENDERS; k++) {
        VkRenderingAttachmentInfo att = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = ms[k].view,
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT,
            .resolveImageView = resolved[k].view,
            .resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .clearValue = {.color = {.float32 = {0, 0, 0, 0}}},
        };
        VkRenderingInfo ri = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .renderArea = {{0, 0}, {W, H}},
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &att,
        };
        vkCmdBeginRendering(cb, &ri);
        if (k == 0) {
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, standard_pipeline);
        } else {
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, custom_pipeline);
            VkSampleLocationsInfoEXT info = {
                .sType = VK_STRUCTURE_TYPE_SAMPLE_LOCATIONS_INFO_EXT,
                .sampleLocationsPerPixel = SAMPLES,
                .sampleLocationGridSize = {1, 1},
                .sampleLocationsCount = 4,
                .pSampleLocations = sets[k],
            };
            set_sample_locations(cb, &info);
        }
        vkCmdDraw(cb, 4, 1, 0, 0);
        vkCmdEndRendering(cb);
    }

    for (uint32_t k = 0; k < RENDERS; k++) {
        layout_barrier(cb, resolved[k].image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                       VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferImageCopy copy = {
            .bufferOffset = k * W * H * 4,
            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .imageExtent = {W, H, 1},
        };
        vkCmdCopyImageToBuffer(cb, resolved[k].image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, out, 1,
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
            int ok = scored(texels + k * W * H * 4, expect[k]);
            if (k == 0 && !standard) {
                continue;
            }
            check(ok, expect[k].what);
        }
    }

    vkDestroyFence(dev, fence, NULL);
    vkDestroyCommandPool(dev, cmdpool, NULL);
    vkDestroyPipeline(dev, standard_pipeline, NULL);
    vkDestroyPipeline(dev, custom_pipeline, NULL);
    vkDestroyShaderModule(dev, vs, NULL);
    vkDestroyShaderModule(dev, fs, NULL);
    vkDestroyPipelineLayout(dev, layout, NULL);
    for (uint32_t k = 0; k < RENDERS; k++) {
        vkDestroyImageView(dev, ms[k].view, NULL);
        vkDestroyImage(dev, ms[k].image, NULL);
        vkFreeMemory(dev, ms[k].mem, NULL);
        vkDestroyImageView(dev, resolved[k].view, NULL);
        vkDestroyImage(dev, resolved[k].image, NULL);
        vkFreeMemory(dev, resolved[k].mem, NULL);
    }
    vkUnmapMemory(dev, out_mem);
    vkDestroyBuffer(dev, out, NULL);
    vkFreeMemory(dev, out_mem, NULL);
    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(inst, NULL);

    printf("\n%d check(s) failed\n", failures);
    return failures ? 1 : 0;
}
