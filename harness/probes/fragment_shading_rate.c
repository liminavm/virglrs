// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva
//
// The `fragment-shading-rate` group of src/venus/unserved.txt, as a program that names each
// command.
//
//   vkGetPhysicalDeviceFragmentShadingRatesKHR    vkCmdSetFragmentShadingRateKHR
//
// From VK_KHR_fragment_shading_rate, which anv advertises and KosmicKrisp does not, so the
// positive control and the venus run are both on a Linux host. anv offers the pipeline rate only
// (no primitive or attachment rate), which is the rate the setter sets.
//
// The query is the two-call enumeration: a count call, a fill call that must agree with it, and a
// short array that must come back VK_INCOMPLETE holding the head of the full list. The list must
// include 1x1 and end with it (the spec orders it largest width first, then largest height), and
// must include the 2x2 and 2x1 the setter asks for.
//
// The setter is scored by counting fragment shader invocations. Three renders, each into its own
// 8x8 R32_UINT target covered by one triangle, at 2x2, then 2x1, then 1x1, with both combiners
// KEEP. The fragment shader takes a ticket from a per-render atomic counter and writes it, so
// every pixel one invocation shaded holds that invocation's ticket: at 1x1 all 64 texels differ,
// at 2x2 each aligned 2x2 block holds one of 16 tickets, at 2x1 each horizontal pair one of 32.
// The counter's final value is the invocation count. A dropped second or third call shades at
// the previous rate; a dropped first leaves the driver's default, 1x1; a width and height swapped
// on the way turns the 2x1 pairs vertical.
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
//   // count.frag
//   #version 450
//   layout(set = 0, binding = 0) buffer Counters { uint n[]; };
//   layout(push_constant) uniform PC { uint slot; } pc;
//   layout(location = 0) out uint o;
//   void main() {
//       o = atomicAdd(n[pc.slot], 1u) + 1u;
//   }
//
// Headless on purpose: no surface, no swapchain, no window. It runs over ssh.
//
//   cc -O1 -o fragment_shading_rate fragment_shading_rate.c -lvulkan && ./fragment_shading_rate
//
// Exit 0 = every check passed. Exit 1 = a check failed, and the line above it says which.
// Exit 2 = the device could not be brought up at all (not a verdict on the group).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define W 8u
#define H 8u
#define RENDERS 3u

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
    0x07230203,0x00010300,0x0008000b,0x0000001b,0x00000000,0x00020011,0x00000001,0x0006000b,
    0x00000001,0x4c534c47,0x6474732e,0x3035342e,0x00000000,0x0003000e,0x00000000,0x00000001,
    0x0006000f,0x00000004,0x00000004,0x6e69616d,0x00000000,0x00000008,0x00030010,0x00000004,
    0x00000007,0x00030003,0x00000002,0x000001c2,0x00040005,0x00000004,0x6e69616d,0x00000000,
    0x00030005,0x00000008,0x0000006f,0x00050005,0x0000000a,0x6e756f43,0x73726574,0x00000000,
    0x00040006,0x0000000a,0x00000000,0x0000006e,0x00030005,0x0000000c,0x00000000,0x00030005,
    0x0000000f,0x00004350,0x00050006,0x0000000f,0x00000000,0x746f6c73,0x00000000,0x00030005,
    0x00000011,0x00006370,0x00040047,0x00000008,0x0000001e,0x00000000,0x00040047,0x00000009,
    0x00000006,0x00000004,0x00030047,0x0000000a,0x00000002,0x00050048,0x0000000a,0x00000000,
    0x00000023,0x00000000,0x00040047,0x0000000c,0x00000021,0x00000000,0x00040047,0x0000000c,
    0x00000022,0x00000000,0x00030047,0x0000000f,0x00000002,0x00050048,0x0000000f,0x00000000,
    0x00000023,0x00000000,0x00020013,0x00000002,0x00030021,0x00000003,0x00000002,0x00040015,
    0x00000006,0x00000020,0x00000000,0x00040020,0x00000007,0x00000003,0x00000006,0x0004003b,
    0x00000007,0x00000008,0x00000003,0x0003001d,0x00000009,0x00000006,0x0003001e,0x0000000a,
    0x00000009,0x00040020,0x0000000b,0x0000000c,0x0000000a,0x0004003b,0x0000000b,0x0000000c,
    0x0000000c,0x00040015,0x0000000d,0x00000020,0x00000001,0x0004002b,0x0000000d,0x0000000e,
    0x00000000,0x0003001e,0x0000000f,0x00000006,0x00040020,0x00000010,0x00000009,0x0000000f,
    0x0004003b,0x00000010,0x00000011,0x00000009,0x00040020,0x00000012,0x00000009,0x00000006,
    0x00040020,0x00000015,0x0000000c,0x00000006,0x0004002b,0x00000006,0x00000017,0x00000001,
    0x0004002b,0x00000006,0x00000018,0x00000000,0x00050036,0x00000002,0x00000004,0x00000000,
    0x00000003,0x000200f8,0x00000005,0x00050041,0x00000012,0x00000013,0x00000011,0x0000000e,
    0x0004003d,0x00000006,0x00000014,0x00000013,0x00060041,0x00000015,0x00000016,0x0000000c,
    0x0000000e,0x00000014,0x000700ea,0x00000006,0x00000019,0x00000016,0x00000017,0x00000018,
    0x00000017,0x00050080,0x00000006,0x0000001a,0x00000019,0x00000017,0x0003003e,0x00000008,
    0x0000001a,0x000100fd,0x00010038
};

// The rates the renders ask for, in recording order.
static const VkExtent2D RATES[RENDERS] = {{2, 2}, {2, 1}, {1, 1}};

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

// An array of `n` shading-rate entries with their sTypes set, for the query to fill.
static VkPhysicalDeviceFragmentShadingRateKHR *rates_array(uint32_t n) {
    VkPhysicalDeviceFragmentShadingRateKHR *a = calloc(n ? n : 1, sizeof *a);
    for (uint32_t i = 0; i < n; i++) {
        a[i].sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_KHR;
    }
    return a;
}

// Whether the list offers a fragment size at one sample.
static int listed(const VkPhysicalDeviceFragmentShadingRateKHR *a, uint32_t n, uint32_t w,
                  uint32_t h) {
    for (uint32_t i = 0; i < n; i++) {
        if (a[i].fragmentSize.width == w && a[i].fragmentSize.height == h &&
            (a[i].sampleCounts & VK_SAMPLE_COUNT_1_BIT)) {
            return 1;
        }
    }
    return 0;
}

// Whether every aligned w x h block of a render holds one ticket, and no two blocks share one.
static int blocks_are(const uint32_t *t, uint32_t w, uint32_t h) {
    for (uint32_t y = 0; y < H; y++) {
        for (uint32_t x = 0; x < W; x++) {
            uint32_t origin = (y - y % h) * W + (x - x % w);
            if (t[y * W + x] != t[origin] || t[origin] == 0) {
                return 0;
            }
            if (origin != y * W + x) {
                continue;
            }
            for (uint32_t o = 0; o < y * W + x; o++) {
                if (o % W % w == 0 && o / W % h == 0 && t[o] == t[origin]) {
                    return 0;
                }
            }
        }
    }
    return 1;
}

int main(void) {
    VkResult r;

    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "venus-fragment-shading-rate-probe",
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
    const char *ext = "VK_KHR_fragment_shading_rate";
    if (!has_ext(exts, en, ext)) {
        fatal("VK_KHR_fragment_shading_rate", VK_ERROR_EXTENSION_NOT_PRESENT);
    }

    VkPhysicalDeviceFragmentShadingRateFeaturesKHR supported = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR,
    };
    VkPhysicalDeviceFeatures2 supported2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &supported,
    };
    vkGetPhysicalDeviceFeatures2(pd, &supported2);
    if (!supported.pipelineFragmentShadingRate) {
        fatal("pipelineFragmentShadingRate", VK_ERROR_FEATURE_NOT_PRESENT);
    }
    if (!supported2.features.fragmentStoresAndAtomics) {
        fatal("fragmentStoresAndAtomics", VK_ERROR_FEATURE_NOT_PRESENT);
    }

    // A physical-device command of a device extension: reached through the instance.
    PFN_vkGetPhysicalDeviceFragmentShadingRatesKHR get_rates =
        (PFN_vkGetPhysicalDeviceFragmentShadingRatesKHR)vkGetInstanceProcAddr(
            inst, "vkGetPhysicalDeviceFragmentShadingRatesKHR");
    if (!get_rates) {
        fatal("vkGetPhysicalDeviceFragmentShadingRatesKHR did not resolve",
              VK_ERROR_EXTENSION_NOT_PRESENT);
    }

    // ------------------------------------------------------------------------ the query

    uint32_t count = 0;
    r = get_rates(pd, &count, NULL);
    check(r == VK_SUCCESS && count > 0, "query: the count call names at least one rate");
    VkPhysicalDeviceFragmentShadingRateKHR *all = rates_array(count);
    uint32_t filled = count;
    r = get_rates(pd, &filled, all);
    check(r == VK_SUCCESS && filled == count, "query: the fill call agrees with the count");
    check(listed(all, filled, 1, 1), "query: 1x1 is listed at one sample");
    check(filled > 0 && all[filled - 1].fragmentSize.width == 1 &&
              all[filled - 1].fragmentSize.height == 1,
          "query: the list ends with 1x1, the smallest");
    check(listed(all, filled, 2, 2) && listed(all, filled, 2, 1),
          "query: 2x2 and 2x1 are listed at one sample");
    printf("  rates:");
    for (uint32_t i = 0; i < filled; i++) {
        printf(" %ux%u/0x%x", all[i].fragmentSize.width, all[i].fragmentSize.height,
               all[i].sampleCounts);
    }
    printf("\n");

    uint32_t short_count = count > 1 ? count - 1 : 0;
    VkPhysicalDeviceFragmentShadingRateKHR *head = rates_array(short_count);
    uint32_t short_filled = short_count;
    r = get_rates(pd, &short_filled, head);
    int same_head = short_filled == short_count;
    for (uint32_t i = 0; same_head && i < short_filled; i++) {
        same_head = head[i].fragmentSize.width == all[i].fragmentSize.width &&
                    head[i].fragmentSize.height == all[i].fragmentSize.height &&
                    head[i].sampleCounts == all[i].sampleCounts;
    }
    check(r == VK_INCOMPLETE && short_filled == short_count,
          "query: a short array is VK_INCOMPLETE, filled to its length");
    check(same_head, "query: the short array holds the full list's head");

    // ------------------------------------------------------------------------ the setter

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
    VkPhysicalDeviceFragmentShadingRateFeaturesKHR ffsr = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR,
        .pipelineFragmentShadingRate = VK_TRUE,
    };
    VkPhysicalDeviceVulkan13Features f13 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext = &ffsr,
        .dynamicRendering = VK_TRUE,
    };
    VkPhysicalDeviceFeatures2 f2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &f13,
        .features = {.fragmentStoresAndAtomics = VK_TRUE},
    };
    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &f2,
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

    PFN_vkCmdSetFragmentShadingRateKHR set_rate =
        (PFN_vkCmdSetFragmentShadingRateKHR)vkGetDeviceProcAddr(dev,
                                                                "vkCmdSetFragmentShadingRateKHR");
    if (!set_rate) {
        fatal("vkCmdSetFragmentShadingRateKHR did not resolve", VK_ERROR_EXTENSION_NOT_PRESENT);
    }

    struct mapped counters =
        mapped(pd, dev, RENDERS * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    VkDescriptorSetLayoutBinding binding = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
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
    VkPushConstantRange range = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(uint32_t)};
    VkPipelineLayoutCreateInfo pli = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &range,
    };
    VkPipelineLayout layout;
    if ((r = vkCreatePipelineLayout(dev, &pli, NULL, &layout)) != VK_SUCCESS) {
        fatal("vkCreatePipelineLayout", r);
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
    VkDescriptorBufferInfo counters_info = {counters.buf, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = set,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &counters_info,
    };
    vkUpdateDescriptorSets(dev, 1, &write, 0, NULL);

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
    VkPipelineColorBlendStateCreateInfo cbs = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &blend,
    };
    VkDynamicState dynamic = VK_DYNAMIC_STATE_FRAGMENT_SHADING_RATE_KHR;
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

    struct target t[RENDERS];
    for (uint32_t k = 0; k < RENDERS; k++) {
        t[k] = target(pd, dev);
    }
    struct mapped texels =
        mapped(pd, dev, RENDERS * W * H * sizeof(uint32_t), VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    memset(texels.p, 0xee, RENDERS * W * H * sizeof(uint32_t));

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
        layout_barrier(cb, t[k].image, VK_IMAGE_LAYOUT_UNDEFINED,
                       VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 0,
                       VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
    }

    const VkFragmentShadingRateCombinerOpKHR keep[2] = {
        VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR,
        VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR,
    };
    for (uint32_t k = 0; k < RENDERS; k++) {
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
        vkCmdBeginRendering(cb, &ri);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &set, 0,
                                NULL);
        vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof k, &k);
        set_rate(cb, &RATES[k], keep);
        vkCmdDraw(cb, 3, 1, 0, 0);
        vkCmdEndRendering(cb);
    }

    for (uint32_t k = 0; k < RENDERS; k++) {
        layout_barrier(cb, t[k].image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                       VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferImageCopy copy = {
            .bufferOffset = k * W * H * sizeof(uint32_t),
            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .imageExtent = {W, H, 1},
        };
        vkCmdCopyImageToBuffer(cb, t[k].image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, texels.buf, 1,
                               &copy);
    }
    VkMemoryBarrier host = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
    };
    vkCmdPipelineBarrier(cb,
                         VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
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
        for (uint32_t k = 0; k < RENDERS; k++) {
            const uint32_t *run = texels.p + k * W * H;
            uint32_t w = RATES[k].width, h = RATES[k].height;
            char what[80];
            snprintf(what, sizeof what, "%ux%u: %u invocations shaded %u pixels", w, h,
                     W * H / (w * h), W * H);
            check(counters.p[k] == W * H / (w * h), what);
            snprintf(what, sizeof what, "%ux%u: each aligned %ux%u block holds one ticket", w, h,
                     w, h);
            check(blocks_are(run, w, h), what);
            printf("  %u invocations; row 0:", counters.p[k]);
            for (uint32_t x = 0; x < W; x++) {
                printf(" %u", run[x]);
            }
            printf("; row 1:");
            for (uint32_t x = 0; x < W; x++) {
                printf(" %u", run[W + x]);
            }
            printf("\n");
        }
    }

    vkDestroyFence(dev, fence, NULL);
    vkDestroyCommandPool(dev, cmdpool, NULL);
    vkDestroyPipeline(dev, pipeline, NULL);
    vkDestroyShaderModule(dev, vs, NULL);
    vkDestroyShaderModule(dev, fs, NULL);
    vkDestroyDescriptorPool(dev, dpool, NULL);
    vkDestroyPipelineLayout(dev, layout, NULL);
    vkDestroyDescriptorSetLayout(dev, set_layout, NULL);
    for (uint32_t k = 0; k < RENDERS; k++) {
        vkDestroyImageView(dev, t[k].view, NULL);
        vkDestroyImage(dev, t[k].image, NULL);
        vkFreeMemory(dev, t[k].mem, NULL);
    }
    unmapped(dev, &counters);
    unmapped(dev, &texels);
    free(all);
    free(head);
    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(inst, NULL);

    printf("\n%d check(s) failed\n", failures);
    return failures ? 1 : 0;
}
