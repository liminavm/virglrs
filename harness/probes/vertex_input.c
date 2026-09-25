// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva
//
// The `vertex-input-dynamic-state` group of src/venus/unserved.txt, as a program that names its
// command.
//
//   vkCmdSetVertexInputEXT
//
// From VK_EXT_vertex_input_dynamic_state, which anv advertises and KosmicKrisp does not, so the
// positive control and the venus run are both on a Linux host.
//
// One pipeline, its vertex input left dynamic, draws 16 points into an 8x2 R32_UINT target
// cleared to 0: vertex v lands on pixel (v % 8, v / 8). The vertex shader reads three uint
// attributes and packs them into the texel, a0 | a1 << 8 | a2 << 16, so each attribute has a
// byte lane of its own. The vertex buffer holds words w[i] = i; binding 0 is bound at its start
// and binding 1 at word 32, and neither bind changes between the draws. Only the layout does:
//
//   row 0, vertices 0..7, layout A: 2 bindings, 3 attributes
//     binding 0 stride 4, binding 1 stride 8
//     a0 = R32_UINT, binding 0, offset 0    -> w[v]           = v
//     a1 = R32_UINT, binding 1, offset 4    -> w[32 + 2v + 1] = 33 + 2v
//     a2 = R32_UINT, binding 0, offset 64   -> w[16 + v]      = 16 + v
//
//   row 1, vertices 8..15, layout B: 1 binding, 3 attributes
//     binding 1 stride 16
//     a0 = R32_UINT, offset 0               -> w[32 + 4v]     = 32 + 4v
//     a1 = R16_UINT, offset 8               -> low half of w[32 + 4v + 2] = 34 + 4v
//     a2 = R8_UINT,  offset 12              -> low byte of w[32 + 4v + 3] = 35 + 4v
//
// A dropped second call draws row 1 through layout A; a dropped first call leaves row 0 to the
// driver's undefined default; binding and attribute counts swapped leave an attribute unread; a
// stride, offset, format or binding number lost on the way changes the lane that carries it.
//
// The shaders, compiled with `glslangValidator -V --target-env vulkan1.1 -x`:
//
//   // attrs.vert
//   #version 450
//   layout(location = 0) in uint a0;
//   layout(location = 1) in uint a1;
//   layout(location = 2) in uint a2;
//   layout(location = 0) flat out uint tag;
//   void main() {
//       uint v = uint(gl_VertexIndex);
//       gl_Position = vec4((float(v % 8u) + 0.5) / 4.0 - 1.0, float(v / 8u) - 0.5, 0.0, 1.0);
//       gl_PointSize = 1.0;
//       tag = a0 | (a1 << 8) | (a2 << 16);
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
// Headless on purpose: no surface, no swapchain, no window. It runs over ssh.
//
//   cc -O1 -o vertex_input vertex_input.c -lvulkan && ./vertex_input
//
// Exit 0 = every check passed. Exit 1 = a check failed, and the line above it says which.
// Exit 2 = the device could not be brought up at all (not a verdict on the group).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define W 8u
#define H 2u
#define WORDS 128u
#define BINDING1_WORD 32u

static const uint32_t VERT[] = {
    0x07230203,0x00010300,0x0008000b,0x0000003a,0x00000000,0x00020011,0x00000001,0x0006000b,
    0x00000001,0x4c534c47,0x6474732e,0x3035342e,0x00000000,0x0003000e,0x00000000,0x00000001,
    0x000b000f,0x00000000,0x00000004,0x6e69616d,0x00000000,0x0000000b,0x00000014,0x0000002c,
    0x0000002e,0x00000030,0x00000035,0x00030003,0x00000002,0x000001c2,0x00040005,0x00000004,
    0x6e69616d,0x00000000,0x00030005,0x00000008,0x00000076,0x00060005,0x0000000b,0x565f6c67,
    0x65747265,0x646e4978,0x00007865,0x00060005,0x00000012,0x505f6c67,0x65567265,0x78657472,
    0x00000000,0x00060006,0x00000012,0x00000000,0x505f6c67,0x7469736f,0x006e6f69,0x00070006,
    0x00000012,0x00000001,0x505f6c67,0x746e696f,0x657a6953,0x00000000,0x00070006,0x00000012,
    0x00000002,0x435f6c67,0x4470696c,0x61747369,0x0065636e,0x00070006,0x00000012,0x00000003,
    0x435f6c67,0x446c6c75,0x61747369,0x0065636e,0x00030005,0x00000014,0x00000000,0x00030005,
    0x0000002c,0x00676174,0x00030005,0x0000002e,0x00003061,0x00030005,0x00000030,0x00003161,
    0x00030005,0x00000035,0x00003261,0x00040047,0x0000000b,0x0000000b,0x0000002a,0x00030047,
    0x00000012,0x00000002,0x00050048,0x00000012,0x00000000,0x0000000b,0x00000000,0x00050048,
    0x00000012,0x00000001,0x0000000b,0x00000001,0x00050048,0x00000012,0x00000002,0x0000000b,
    0x00000003,0x00050048,0x00000012,0x00000003,0x0000000b,0x00000004,0x00030047,0x0000002c,
    0x0000000e,0x00040047,0x0000002c,0x0000001e,0x00000000,0x00040047,0x0000002e,0x0000001e,
    0x00000000,0x00040047,0x00000030,0x0000001e,0x00000001,0x00040047,0x00000035,0x0000001e,
    0x00000002,0x00020013,0x00000002,0x00030021,0x00000003,0x00000002,0x00040015,0x00000006,
    0x00000020,0x00000000,0x00040020,0x00000007,0x00000007,0x00000006,0x00040015,0x00000009,
    0x00000020,0x00000001,0x00040020,0x0000000a,0x00000001,0x00000009,0x0004003b,0x0000000a,
    0x0000000b,0x00000001,0x00030016,0x0000000e,0x00000020,0x00040017,0x0000000f,0x0000000e,
    0x00000004,0x0004002b,0x00000006,0x00000010,0x00000001,0x0004001c,0x00000011,0x0000000e,
    0x00000010,0x0006001e,0x00000012,0x0000000f,0x0000000e,0x00000011,0x00000011,0x00040020,
    0x00000013,0x00000003,0x00000012,0x0004003b,0x00000013,0x00000014,0x00000003,0x0004002b,
    0x00000009,0x00000015,0x00000000,0x0004002b,0x00000006,0x00000017,0x00000008,0x0004002b,
    0x0000000e,0x0000001a,0x3f000000,0x0004002b,0x0000000e,0x0000001c,0x40800000,0x0004002b,
    0x0000000e,0x0000001e,0x3f800000,0x0004002b,0x0000000e,0x00000024,0x00000000,0x00040020,
    0x00000026,0x00000003,0x0000000f,0x0004002b,0x00000009,0x00000028,0x00000001,0x00040020,
    0x00000029,0x00000003,0x0000000e,0x00040020,0x0000002b,0x00000003,0x00000006,0x0004003b,
    0x0000002b,0x0000002c,0x00000003,0x00040020,0x0000002d,0x00000001,0x00000006,0x0004003b,
    0x0000002d,0x0000002e,0x00000001,0x0004003b,0x0000002d,0x00000030,0x00000001,0x0004002b,
    0x00000009,0x00000032,0x00000008,0x0004003b,0x0000002d,0x00000035,0x00000001,0x0004002b,
    0x00000009,0x00000037,0x00000010,0x00050036,0x00000002,0x00000004,0x00000000,0x00000003,
    0x000200f8,0x00000005,0x0004003b,0x00000007,0x00000008,0x00000007,0x0004003d,0x00000009,
    0x0000000c,0x0000000b,0x0004007c,0x00000006,0x0000000d,0x0000000c,0x0003003e,0x00000008,
    0x0000000d,0x0004003d,0x00000006,0x00000016,0x00000008,0x00050089,0x00000006,0x00000018,
    0x00000016,0x00000017,0x00040070,0x0000000e,0x00000019,0x00000018,0x00050081,0x0000000e,
    0x0000001b,0x00000019,0x0000001a,0x00050088,0x0000000e,0x0000001d,0x0000001b,0x0000001c,
    0x00050083,0x0000000e,0x0000001f,0x0000001d,0x0000001e,0x0004003d,0x00000006,0x00000020,
    0x00000008,0x00050086,0x00000006,0x00000021,0x00000020,0x00000017,0x00040070,0x0000000e,
    0x00000022,0x00000021,0x00050083,0x0000000e,0x00000023,0x00000022,0x0000001a,0x00070050,
    0x0000000f,0x00000025,0x0000001f,0x00000023,0x00000024,0x0000001e,0x00050041,0x00000026,
    0x00000027,0x00000014,0x00000015,0x0003003e,0x00000027,0x00000025,0x00050041,0x00000029,
    0x0000002a,0x00000014,0x00000028,0x0003003e,0x0000002a,0x0000001e,0x0004003d,0x00000006,
    0x0000002f,0x0000002e,0x0004003d,0x00000006,0x00000031,0x00000030,0x000500c4,0x00000006,
    0x00000033,0x00000031,0x00000032,0x000500c5,0x00000006,0x00000034,0x0000002f,0x00000033,
    0x0004003d,0x00000006,0x00000036,0x00000035,0x000500c4,0x00000006,0x00000038,0x00000036,
    0x00000037,0x000500c5,0x00000006,0x00000039,0x00000034,0x00000038,0x0003003e,0x0000002c,
    0x00000039,0x000100fd,0x00010038
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

// The attribute values vertex v reads, through the layout that draws its row.
static void expected(uint32_t v, uint32_t a[3]) {
    if (v < W) {
        a[0] = v;
        a[1] = BINDING1_WORD + 2 * v + 1;
        a[2] = 16 + v;
    } else {
        a[0] = BINDING1_WORD + 4 * v;
        a[1] = BINDING1_WORD + 4 * v + 2;
        a[2] = BINDING1_WORD + 4 * v + 3;
    }
}

// Whether every texel of a row carries the expected value in one attribute's byte lane.
static int lane_is(const uint32_t *texels, uint32_t row, uint32_t lane) {
    for (uint32_t x = 0; x < W; x++) {
        uint32_t v = row * W + x, a[3];
        expected(v, a);
        if (((texels[v] >> (8 * lane)) & 0xff) != a[lane]) {
            return 0;
        }
    }
    return 1;
}

int main(void) {
    VkResult r;

    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "venus-vertex-input-dynamic-state-probe",
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
    const char *ext = "VK_EXT_vertex_input_dynamic_state";
    if (!has_ext(exts, en, ext)) {
        fatal("VK_EXT_vertex_input_dynamic_state", VK_ERROR_EXTENSION_NOT_PRESENT);
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
    VkPhysicalDeviceVertexInputDynamicStateFeaturesEXT fvi = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_INPUT_DYNAMIC_STATE_FEATURES_EXT,
        .vertexInputDynamicState = VK_TRUE,
    };
    VkPhysicalDeviceVulkan13Features f13 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext = &fvi,
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

    PFN_vkCmdSetVertexInputEXT set_vertex_input =
        (PFN_vkCmdSetVertexInputEXT)vkGetDeviceProcAddr(dev, "vkCmdSetVertexInputEXT");
    if (!set_vertex_input) {
        fatal("vkCmdSetVertexInputEXT did not resolve", VK_ERROR_EXTENSION_NOT_PRESENT);
    }

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

    struct mapped vertices =
        mapped(pd, dev, WORDS * sizeof(uint32_t), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    for (uint32_t i = 0; i < WORDS; i++) {
        vertices.p[i] = i;
    }
    struct mapped texels =
        mapped(pd, dev, W * H * sizeof(uint32_t), VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    memset(texels.p, 0xee, W * H * sizeof(uint32_t));

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
    // Ignored: the vertex input is dynamic, and vkCmdSetVertexInputEXT is its only source.
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
    VkDynamicState dynamic = VK_DYNAMIC_STATE_VERTEX_INPUT_EXT;
    VkPipelineDynamicStateCreateInfo ds = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 1,
        .pDynamicStates = &dynamic,
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
        .pDynamicState = &ds,
        .layout = layout,
    };
    VkPipeline pipeline;
    if ((r = vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, NULL, &pipeline)) !=
        VK_SUCCESS) {
        fatal("vkCreateGraphicsPipelines", r);
    }

    // Layout A: two bindings, three attributes, one of them on the second binding.
    const VkVertexInputBindingDescription2EXT a_bindings[2] = {
        {.sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
         .binding = 0,
         .stride = 4,
         .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
         .divisor = 1},
        {.sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
         .binding = 1,
         .stride = 8,
         .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
         .divisor = 1},
    };
    const VkVertexInputAttributeDescription2EXT a_attrs[3] = {
        {.sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
         .location = 0,
         .binding = 0,
         .format = VK_FORMAT_R32_UINT,
         .offset = 0},
        {.sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
         .location = 1,
         .binding = 1,
         .format = VK_FORMAT_R32_UINT,
         .offset = 4},
        {.sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
         .location = 2,
         .binding = 0,
         .format = VK_FORMAT_R32_UINT,
         .offset = 64},
    };
    // Layout B: one binding, numbered 1, three attributes of three widths.
    const VkVertexInputBindingDescription2EXT b_binding = {
        .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
        .binding = 1,
        .stride = 16,
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
        .divisor = 1,
    };
    const VkVertexInputAttributeDescription2EXT b_attrs[3] = {
        {.sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
         .location = 0,
         .binding = 1,
         .format = VK_FORMAT_R32_UINT,
         .offset = 0},
        {.sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
         .location = 1,
         .binding = 1,
         .format = VK_FORMAT_R16_UINT,
         .offset = 8},
        {.sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
         .location = 2,
         .binding = 1,
         .format = VK_FORMAT_R8_UINT,
         .offset = 12},
    };

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
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    VkBuffer vbufs[2] = {vertices.buf, vertices.buf};
    VkDeviceSize voffsets[2] = {0, BINDING1_WORD * sizeof(uint32_t)};
    vkCmdBindVertexBuffers(cb, 0, 2, vbufs, voffsets);
    set_vertex_input(cb, 2, a_bindings, 3, a_attrs);
    vkCmdDraw(cb, W, 1, 0, 0);
    set_vertex_input(cb, 1, &b_binding, 3, b_attrs);
    vkCmdDraw(cb, W, 1, W, 0);
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
        const uint32_t *t = texels.p;
        check(lane_is(t, 0, 0), "layout A: a0, binding 0 at offset 0");
        check(lane_is(t, 0, 1), "layout A: a1, binding 1 at stride 8, offset 4");
        check(lane_is(t, 0, 2), "layout A: a2, binding 0 at offset 64");
        check(lane_is(t, 1, 0), "layout B: a0, R32_UINT at binding 1, stride 16");
        check(lane_is(t, 1, 1), "layout B: a1, R16_UINT at offset 8");
        check(lane_is(t, 1, 2), "layout B: a2, R8_UINT at offset 12");
        for (uint32_t y = 0; y < H; y++) {
            printf("  row %u:", y);
            for (uint32_t x = 0; x < W; x++) {
                printf(" %06x", t[y * W + x]);
            }
            printf("\n");
        }
    }

    vkDestroyFence(dev, fence, NULL);
    vkDestroyCommandPool(dev, cmdpool, NULL);
    vkDestroyPipeline(dev, pipeline, NULL);
    vkDestroyShaderModule(dev, vs, NULL);
    vkDestroyShaderModule(dev, fs, NULL);
    vkDestroyPipelineLayout(dev, layout, NULL);
    vkDestroyImageView(dev, view, NULL);
    vkDestroyImage(dev, image, NULL);
    vkFreeMemory(dev, image_mem, NULL);
    unmapped(dev, &vertices);
    unmapped(dev, &texels);
    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(inst, NULL);

    printf("\n%d check(s) failed\n", failures);
    return failures ? 1 : 0;
}
