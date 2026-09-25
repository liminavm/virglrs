// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva
//
// The `transform-feedback` group of src/venus/unserved.txt, as a program that names each command.
//
//   vkCmdBindTransformFeedbackBuffersEXT  vkCmdBeginTransformFeedbackEXT
//   vkCmdEndTransformFeedbackEXT  vkCmdBeginQueryIndexedEXT  vkCmdEndQueryIndexedEXT
//   vkCmdDrawIndirectByteCountEXT
//
// From VK_EXT_transform_feedback, which both KosmicKrisp and anv advertise; zink needs it for GL's
// transform feedback. One vertex shader captures a 16-byte record per vertex into xfb buffer 0
// and a tag at byte 4 of a 12-byte record into xfb buffer 1; the rasterizer is discarded. Every
// capture lands in one host-visible buffer, prefilled with a sentinel, at a distinct binding
// offset, so a dropped offset writes where the sentinel should be and a dropped size writes past
// where the capture should stop. Four passes, each its own rendering instance:
//
//   1. both bindings bound, 5 points from firstVertex 10, under stream query 0; the end writes
//      both counters to one counter slot pair.
//   2. rebound identically, begun *from* that counter pair so the capture resumes after the first
//      pass, 3 points from firstVertex 100, under query 1; the end writes a second counter pair.
//   3. vkCmdDrawIndirectByteCountEXT from the second pass's buffer-0 counter, less a counterOffset
//      of 32 bytes, at the 16-byte stride: 6 vertices, 2 instances from firstInstance 3, captured
//      into a fresh region under query 2.
//   4. buffer 0 bound with room for 2 records and 4 points drawn under query 3, which must count
//      2 written and 4 needed, with nothing past the bound size touched.
//
// A dropped begin or end leaves nothing captured; a dropped counter on the end leaves its slot at
// the sentinel, and on the begin restarts the second pass on top of the first; a dropped query
// leaves its answer unavailable.
//
// KosmicKrisp's transform feedback is emulated and fails seven checks on its own, so the positive
// control that counts is anv's. It keeps End's counters in a CPU-side shadow and never writes the
// counter buffer (the two counter checks; resume and the byte-count draw still pass, because they
// read the shadow back); it captures a draw whole or not at all, so pass 4 captures nothing; and
// its query pool has no report layout for the stream query type, so all four query checks fail.
// On that host the query commands and End's counter write are scored by the log only.
//
// The shader, compiled with `glslangValidator -V --target-env vulkan1.1 -x`:
//
//   // capture.vert
//   #version 450
//   layout(xfb_buffer = 0, xfb_offset = 0, xfb_stride = 16, location = 0) out vec4 rec;
//   layout(xfb_buffer = 1, xfb_offset = 4, xfb_stride = 12, location = 1) out float tag;
//   void main() {
//       rec = vec4(float(gl_VertexIndex), float(gl_InstanceIndex), 42.0, 7.0);
//       tag = 1000.0 + float(gl_VertexIndex);
//       gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
//       gl_PointSize = 1.0;
//   }
//
// Headless on purpose: no surface, no swapchain, no window. It runs over ssh.
//
//   cc -O1 -o transform_feedback transform_feedback.c -lvulkan && ./transform_feedback
//
// Exit 0 = every check passed. Exit 1 = a check failed, and the line above it says which.
// Exit 2 = the device could not be brought up at all (not a verdict on the group).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define SENTINEL 0xeeeeeeeeu
#define XFB_SIZE 4096u

// Where each capture lands in the one xfb buffer, in bytes.
#define A_OFFSET 64u    // buffer 0, passes 1 and 2
#define A_SIZE 512u
#define B_OFFSET 1040u  // buffer 1, passes 1 and 2
#define B_SIZE 256u
#define C_OFFSET 2048u  // buffer 0, pass 3
#define C_SIZE 512u
#define D_OFFSET 3072u  // buffer 0, pass 4: room for two records and no more
#define D_SIZE 32u
#define JUNK_OFFSET 3584u  // buffer 1, passes 3 and 4, not checked
#define JUNK_SIZE 512u

#define REC_STRIDE 16u
#define TAG_STRIDE 12u
#define TAG_OFFSET 4u

// Counter slots, in bytes into the counter buffer: the first pass's pair, then the second's.
#define COUNTERS_1 0u
#define COUNTERS_2 16u
#define BYTE_COUNT_OFFSET 32u

static const uint32_t VERT[] = {
    0x07230203,0x00010300,0x0008000b,0x00000028,0x00000000,0x00020011,0x00000001,0x00020011,
    0x00000035,0x0006000b,0x00000001,0x4c534c47,0x6474732e,0x3035342e,0x00000000,0x0003000e,
    0x00000000,0x00000001,0x000a000f,0x00000000,0x00000004,0x6e69616d,0x00000000,0x00000009,
    0x0000000c,0x0000000f,0x00000016,0x00000020,0x00030010,0x00000004,0x0000000b,0x00030003,
    0x00000002,0x000001c2,0x00040005,0x00000004,0x6e69616d,0x00000000,0x00030005,0x00000009,
    0x00636572,0x00060005,0x0000000c,0x565f6c67,0x65747265,0x646e4978,0x00007865,0x00070005,
    0x0000000f,0x495f6c67,0x6174736e,0x4965636e,0x7865646e,0x00000000,0x00030005,0x00000016,
    0x00676174,0x00060005,0x0000001e,0x505f6c67,0x65567265,0x78657472,0x00000000,0x00060006,
    0x0000001e,0x00000000,0x505f6c67,0x7469736f,0x006e6f69,0x00070006,0x0000001e,0x00000001,
    0x505f6c67,0x746e696f,0x657a6953,0x00000000,0x00070006,0x0000001e,0x00000002,0x435f6c67,
    0x4470696c,0x61747369,0x0065636e,0x00070006,0x0000001e,0x00000003,0x435f6c67,0x446c6c75,
    0x61747369,0x0065636e,0x00030005,0x00000020,0x00000000,0x00040047,0x00000009,0x0000001e,
    0x00000000,0x00040047,0x00000009,0x00000023,0x00000000,0x00040047,0x00000009,0x00000024,
    0x00000000,0x00040047,0x00000009,0x00000025,0x00000010,0x00040047,0x0000000c,0x0000000b,
    0x0000002a,0x00040047,0x0000000f,0x0000000b,0x0000002b,0x00040047,0x00000016,0x0000001e,
    0x00000001,0x00040047,0x00000016,0x00000023,0x00000004,0x00040047,0x00000016,0x00000024,
    0x00000001,0x00040047,0x00000016,0x00000025,0x0000000c,0x00030047,0x0000001e,0x00000002,
    0x00050048,0x0000001e,0x00000000,0x0000000b,0x00000000,0x00050048,0x0000001e,0x00000001,
    0x0000000b,0x00000001,0x00050048,0x0000001e,0x00000002,0x0000000b,0x00000003,0x00050048,
    0x0000001e,0x00000003,0x0000000b,0x00000004,0x00040047,0x00000020,0x00000024,0x00000000,
    0x00040047,0x00000020,0x00000025,0x00000010,0x00020013,0x00000002,0x00030021,0x00000003,
    0x00000002,0x00030016,0x00000006,0x00000020,0x00040017,0x00000007,0x00000006,0x00000004,
    0x00040020,0x00000008,0x00000003,0x00000007,0x0004003b,0x00000008,0x00000009,0x00000003,
    0x00040015,0x0000000a,0x00000020,0x00000001,0x00040020,0x0000000b,0x00000001,0x0000000a,
    0x0004003b,0x0000000b,0x0000000c,0x00000001,0x0004003b,0x0000000b,0x0000000f,0x00000001,
    0x0004002b,0x00000006,0x00000012,0x42280000,0x0004002b,0x00000006,0x00000013,0x40e00000,
    0x00040020,0x00000015,0x00000003,0x00000006,0x0004003b,0x00000015,0x00000016,0x00000003,
    0x0004002b,0x00000006,0x00000017,0x447a0000,0x00040015,0x0000001b,0x00000020,0x00000000,
    0x0004002b,0x0000001b,0x0000001c,0x00000001,0x0004001c,0x0000001d,0x00000006,0x0000001c,
    0x0006001e,0x0000001e,0x00000007,0x00000006,0x0000001d,0x0000001d,0x00040020,0x0000001f,
    0x00000003,0x0000001e,0x0004003b,0x0000001f,0x00000020,0x00000003,0x0004002b,0x0000000a,
    0x00000021,0x00000000,0x0004002b,0x00000006,0x00000022,0x00000000,0x0004002b,0x00000006,
    0x00000023,0x3f800000,0x0007002c,0x00000007,0x00000024,0x00000022,0x00000022,0x00000022,
    0x00000023,0x0004002b,0x0000000a,0x00000026,0x00000001,0x00050036,0x00000002,0x00000004,
    0x00000000,0x00000003,0x000200f8,0x00000005,0x0004003d,0x0000000a,0x0000000d,0x0000000c,
    0x0004006f,0x00000006,0x0000000e,0x0000000d,0x0004003d,0x0000000a,0x00000010,0x0000000f,
    0x0004006f,0x00000006,0x00000011,0x00000010,0x00070050,0x00000007,0x00000014,0x0000000e,
    0x00000011,0x00000012,0x00000013,0x0003003e,0x00000009,0x00000014,0x0004003d,0x0000000a,
    0x00000018,0x0000000c,0x0004006f,0x00000006,0x00000019,0x00000018,0x00050081,0x00000006,
    0x0000001a,0x00000017,0x00000019,0x0003003e,0x00000016,0x0000001a,0x00050041,0x00000008,
    0x00000025,0x00000020,0x00000021,0x0003003e,0x00000025,0x00000024,0x00050041,0x00000015,
    0x00000027,0x00000020,0x00000026,0x0003003e,0x00000027,0x00000023,0x000100fd,0x00010038
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

// A host-visible buffer, mapped for the life of the probe.
struct mapped {
    VkBuffer buf;
    VkDeviceMemory mem;
    uint32_t *p;
};

static struct mapped mapped(VkPhysicalDevice pd, VkDevice dev, VkDeviceSize size,
                            VkBufferUsageFlags usage) {
    struct mapped b;
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
    uint32_t type = mem_type(pd, mr.memoryTypeBits,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (type == UINT32_MAX) {
        fatal("no host-visible coherent memory type", VK_ERROR_INITIALIZATION_FAILED);
    }
    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
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
    memset(b.p, 0xee, size);
    return b;
}

static int has_ext(VkExtensionProperties *exts, uint32_t n, const char *name) {
    for (uint32_t i = 0; i < n; i++) {
        if (strcmp(exts[i].extensionName, name) == 0) {
            return 1;
        }
    }
    return 0;
}

static float word_float(const uint32_t *p) {
    float f;
    memcpy(&f, p, sizeof f);
    return f;
}

// Whether `n` 16-byte records from `at` (a byte offset into the xfb buffer) are
// (firstVertex + i, instance, 42, 7).
static int records(const uint32_t *xfb, uint32_t at, uint32_t n, uint32_t first_vertex,
                   uint32_t instance) {
    int ok = 1;
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t *rec = xfb + (at + i * REC_STRIDE) / 4;
        float want[4] = {(float)(first_vertex + i), (float)instance, 42.0f, 7.0f};
        for (int c = 0; c < 4; c++) {
            if (word_float(rec + c) != want[c]) {
                printf("  record %u at byte %u, component %d: %g, want %g\n", i,
                       at + i * REC_STRIDE, c, word_float(rec + c), want[c]);
                ok = 0;
            }
        }
    }
    return ok;
}

// Whether every word in [from, to) bytes still holds the sentinel.
static int untouched(const uint32_t *xfb, uint32_t from, uint32_t to) {
    for (uint32_t b = from; b < to; b += 4) {
        if (xfb[b / 4] != SENTINEL) {
            printf("  byte %u was written: 0x%08x\n", b, xfb[b / 4]);
            return 0;
        }
    }
    return 1;
}

int main(void) {
    VkResult r;

    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "venus-transform-feedback-probe",
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

    VkPhysicalDeviceTransformFeedbackPropertiesEXT xfb_props = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_PROPERTIES_EXT,
    };
    VkPhysicalDeviceProperties2 props2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &xfb_props,
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
    const char *ext = "VK_EXT_transform_feedback";
    if (!has_ext(exts, en, ext)) {
        fatal("VK_EXT_transform_feedback", VK_ERROR_EXTENSION_NOT_PRESENT);
    }

    VkPhysicalDeviceTransformFeedbackFeaturesEXT xfb_have = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT,
    };
    VkPhysicalDeviceFeatures2 have = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &xfb_have,
    };
    vkGetPhysicalDeviceFeatures2(pd, &have);
    if (!xfb_have.transformFeedback) {
        fatal("the transformFeedback feature", VK_ERROR_FEATURE_NOT_PRESENT);
    }
    if (xfb_props.maxTransformFeedbackBuffers < 2) {
        fatal("maxTransformFeedbackBuffers >= 2", VK_ERROR_FEATURE_NOT_PRESENT);
    }
    // Both are properties, not features: a device without them still serves the rest.
    int queries = xfb_props.transformFeedbackQueries;
    int byte_count_draw = xfb_props.transformFeedbackDraw;
    if (!queries) {
        printf("note: transformFeedbackQueries is false; the stream query checks are skipped\n");
    }
    if (!byte_count_draw) {
        printf("note: transformFeedbackDraw is false; the byte-count draw is skipped\n");
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
    VkPhysicalDeviceTransformFeedbackFeaturesEXT fxfb = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT,
        .transformFeedback = VK_TRUE,
    };
    VkPhysicalDeviceVulkan13Features f13 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext = &fxfb,
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

#define RESOLVE(type, name)                                                                        \
    type name = (type)vkGetDeviceProcAddr(dev, #name);                                             \
    if (!name) {                                                                                   \
        fatal(#name " did not resolve", VK_ERROR_EXTENSION_NOT_PRESENT);                           \
    }
    RESOLVE(PFN_vkCmdBindTransformFeedbackBuffersEXT, vkCmdBindTransformFeedbackBuffersEXT)
    RESOLVE(PFN_vkCmdBeginTransformFeedbackEXT, vkCmdBeginTransformFeedbackEXT)
    RESOLVE(PFN_vkCmdEndTransformFeedbackEXT, vkCmdEndTransformFeedbackEXT)
    RESOLVE(PFN_vkCmdBeginQueryIndexedEXT, vkCmdBeginQueryIndexedEXT)
    RESOLVE(PFN_vkCmdEndQueryIndexedEXT, vkCmdEndQueryIndexedEXT)
    RESOLVE(PFN_vkCmdDrawIndirectByteCountEXT, vkCmdDrawIndirectByteCountEXT)
#undef RESOLVE

    VkPipelineLayoutCreateInfo pli = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    VkPipelineLayout layout;
    if ((r = vkCreatePipelineLayout(dev, &pli, NULL, &layout)) != VK_SUCCESS) {
        fatal("vkCreatePipelineLayout", r);
    }
    VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = sizeof VERT,
        .pCode = VERT,
    };
    VkShaderModule vs;
    if ((r = vkCreateShaderModule(dev, &smci, NULL, &vs)) != VK_SUCCESS) {
        fatal("vkCreateShaderModule", r);
    }
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
        .topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
    };
    VkViewport viewport = {0, 0, 1, 1, 0, 1};
    VkRect2D scissor = {{0, 0}, {1, 1}};
    VkPipelineViewportStateCreateInfo vp = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor,
    };
    VkPipelineRasterizationStateCreateInfo rs = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .rasterizerDiscardEnable = VK_TRUE,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .lineWidth = 1.0f,
    };
    VkPipelineMultisampleStateCreateInfo ms = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };
    VkPipelineColorBlendStateCreateInfo cb_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
    };
    VkPipelineRenderingCreateInfo prci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
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
        .pColorBlendState = &cb_state,
        .layout = layout,
    };
    VkPipeline pipeline;
    if ((r = vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, NULL, &pipeline)) !=
        VK_SUCCESS) {
        fatal("vkCreateGraphicsPipelines", r);
    }

    struct mapped xfb = mapped(pd, dev, XFB_SIZE, VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT);
    struct mapped counters =
        mapped(pd, dev, 64,
               VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT |
                   VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);

    VkQueryPoolCreateInfo qpci = {
        .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .queryType = VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT,
        .queryCount = 4,
    };
    VkQueryPool pool = VK_NULL_HANDLE;
    if (queries && (r = vkCreateQueryPool(dev, &qpci, NULL, &pool)) != VK_SUCCESS) {
        fatal("vkCreateQueryPool", r);
    }

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
    if (queries) {
        vkCmdResetQueryPool(cb, pool, 0, 4);
    }

    // No attachments: the rasterizer is discarded, and transform feedback only needs a render
    // pass instance to be active in.
    VkRenderingInfo ri = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = {{0, 0}, {1, 1}},
        .layerCount = 1,
    };
    // The end of one pass writes counters that the next one reads, on the begin or the draw.
    VkMemoryBarrier counter_barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT,
        .dstAccessMask = VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_READ_BIT_EXT |
                         VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
    };

    VkBuffer both[2] = {xfb.buf, xfb.buf};
    VkBuffer counter_bufs[2] = {counters.buf, counters.buf};
    VkDeviceSize ab_offsets[2] = {A_OFFSET, B_OFFSET};
    VkDeviceSize ab_sizes[2] = {A_SIZE, B_SIZE};
    VkDeviceSize counters_1[2] = {COUNTERS_1, COUNTERS_1 + 4};
    VkDeviceSize counters_2[2] = {COUNTERS_2, COUNTERS_2 + 4};

    // Pass 1: five points from firstVertex 10, counters written to the first pair.
    vkCmdBeginRendering(cb, &ri);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdBindTransformFeedbackBuffersEXT(cb, 0, 2, both, ab_offsets, ab_sizes);
    vkCmdBeginTransformFeedbackEXT(cb, 0, 0, NULL, NULL);
    if (queries) {
        vkCmdBeginQueryIndexedEXT(cb, pool, 0, 0, 0);
    }
    vkCmdDraw(cb, 5, 1, 10, 0);
    if (queries) {
        vkCmdEndQueryIndexedEXT(cb, pool, 0, 0);
    }
    vkCmdEndTransformFeedbackEXT(cb, 0, 2, counter_bufs, counters_1);
    vkCmdEndRendering(cb);
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT,
                         VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT |
                             VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                         0, 1, &counter_barrier, 0, NULL, 0, NULL);

    // Pass 2: resumed from the first pair, three points from firstVertex 100.
    vkCmdBeginRendering(cb, &ri);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdBindTransformFeedbackBuffersEXT(cb, 0, 2, both, ab_offsets, ab_sizes);
    vkCmdBeginTransformFeedbackEXT(cb, 0, 2, counter_bufs, counters_1);
    if (queries) {
        vkCmdBeginQueryIndexedEXT(cb, pool, 1, 0, 0);
    }
    vkCmdDraw(cb, 3, 1, 100, 0);
    if (queries) {
        vkCmdEndQueryIndexedEXT(cb, pool, 1, 0);
    }
    vkCmdEndTransformFeedbackEXT(cb, 0, 2, counter_bufs, counters_2);
    vkCmdEndRendering(cb);
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT,
                         VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT |
                             VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                         0, 1, &counter_barrier, 0, NULL, 0, NULL);

    // Pass 3: the vertex count comes from the second pass's buffer-0 counter.
    VkDeviceSize cj_offsets[2] = {C_OFFSET, JUNK_OFFSET};
    VkDeviceSize cj_sizes[2] = {C_SIZE, JUNK_SIZE};
    if (byte_count_draw) {
        vkCmdBeginRendering(cb, &ri);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        vkCmdBindTransformFeedbackBuffersEXT(cb, 0, 2, both, cj_offsets, cj_sizes);
        vkCmdBeginTransformFeedbackEXT(cb, 0, 0, NULL, NULL);
        if (queries) {
            vkCmdBeginQueryIndexedEXT(cb, pool, 2, 0, 0);
        }
        vkCmdDrawIndirectByteCountEXT(cb, 2, 3, counters.buf, COUNTERS_2, BYTE_COUNT_OFFSET,
                                      REC_STRIDE);
        if (queries) {
            vkCmdEndQueryIndexedEXT(cb, pool, 2, 0);
        }
        vkCmdEndTransformFeedbackEXT(cb, 0, 0, NULL, NULL);
        vkCmdEndRendering(cb);
    }

    // Pass 4: four points into room for two.
    VkDeviceSize dj_offsets[2] = {D_OFFSET, JUNK_OFFSET};
    VkDeviceSize dj_sizes[2] = {D_SIZE, JUNK_SIZE};
    vkCmdBeginRendering(cb, &ri);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdBindTransformFeedbackBuffersEXT(cb, 0, 2, both, dj_offsets, dj_sizes);
    vkCmdBeginTransformFeedbackEXT(cb, 0, 0, NULL, NULL);
    if (queries) {
        vkCmdBeginQueryIndexedEXT(cb, pool, 3, 0, 0);
    }
    vkCmdDraw(cb, 4, 1, 200, 0);
    if (queries) {
        vkCmdEndQueryIndexedEXT(cb, pool, 3, 0);
    }
    vkCmdEndTransformFeedbackEXT(cb, 0, 0, NULL, NULL);
    vkCmdEndRendering(cb);

    VkMemoryBarrier host = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFORM_FEEDBACK_WRITE_BIT_EXT |
                         VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
    };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT,
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
        const uint32_t *x = xfb.p;

        // vkCmdBindTransformFeedbackBuffersEXT: both bindings, each at its own offset.
        check(untouched(x, 0, A_OFFSET), "bind: nothing captured below buffer 0's offset");
        check(records(x, A_OFFSET, 5, 10, 0), "pass 1: buffer 0 holds vertices 10-14");
        int tags = 1;
        for (uint32_t i = 0; i < 8; i++) {
            float want = 1000.0f + (float)(i < 5 ? 10 + i : 100 + i - 5);
            float got = word_float(x + (B_OFFSET + i * TAG_STRIDE + TAG_OFFSET) / 4);
            if (got != want) {
                printf("  tag %u: %g, want %g\n", i, got, want);
                tags = 0;
            }
        }
        check(tags, "passes 1-2: buffer 1 holds eight tags at its own offset");

        // vkCmdEndTransformFeedbackEXT: the byte counts written to each counter slot.
        const uint32_t *c = counters.p;
        printf("  counters: pass 1 %u, %u; pass 2 %u, %u\n", c[COUNTERS_1 / 4],
               c[COUNTERS_1 / 4 + 1], c[COUNTERS_2 / 4], c[COUNTERS_2 / 4 + 1]);
        check(c[COUNTERS_1 / 4] == 5 * REC_STRIDE && c[COUNTERS_1 / 4 + 1] == 5 * TAG_STRIDE,
              "end: pass 1 wrote counters of 80 and 60 bytes");
        check(c[COUNTERS_2 / 4] == 8 * REC_STRIDE && c[COUNTERS_2 / 4 + 1] == 8 * TAG_STRIDE,
              "end: pass 2 wrote counters of 128 and 96 bytes");

        // vkCmdBeginTransformFeedbackEXT with counters: the second pass resumed after the first.
        check(records(x, A_OFFSET + 5 * REC_STRIDE, 3, 100, 0) &&
                  untouched(x, A_OFFSET + 8 * REC_STRIDE, A_OFFSET + A_SIZE),
              "begin: pass 2 resumed after vertex 14 with 100-102");

        // vkCmdDrawIndirectByteCountEXT: (128 - 32) / 16 = 6 vertices, instances 3 and 4.
        if (byte_count_draw) {
            check(records(x, C_OFFSET, 6, 0, 3) && records(x, C_OFFSET + 6 * REC_STRIDE, 6, 0, 4) &&
                      untouched(x, C_OFFSET + 12 * REC_STRIDE, C_OFFSET + C_SIZE),
                  "byte-count draw: 6 vertices x instances 3-4 captured");
        }

        // The bound size stops the capture: two records, nothing past them.
        check(untouched(x, D_OFFSET + D_SIZE, JUNK_OFFSET),
              "bind: pass 4 wrote nothing past the bound size");
        check(records(x, D_OFFSET, 2, 200, 0), "pass 4: the two records that fit were captured");

        // vkCmdBeginQueryIndexedEXT / vkCmdEndQueryIndexedEXT: {written, needed} per query.
        if (queries) {
            uint64_t q[4][2];
            memset(q, 0xee, sizeof q);
            r = vkGetQueryPoolResults(dev, pool, 0, 4, sizeof q, q, sizeof q[0],
                                      VK_QUERY_RESULT_64_BIT);
            check(r == VK_SUCCESS, "stream queries: results available");
            printf("  written/needed: %llu/%llu, %llu/%llu, %llu/%llu, %llu/%llu\n",
                   (unsigned long long)q[0][0], (unsigned long long)q[0][1],
                   (unsigned long long)q[1][0], (unsigned long long)q[1][1],
                   (unsigned long long)q[2][0], (unsigned long long)q[2][1],
                   (unsigned long long)q[3][0], (unsigned long long)q[3][1]);
            check(q[0][0] == 5 && q[0][1] == 5, "stream query 0: 5 written of 5");
            check(q[1][0] == 3 && q[1][1] == 3, "stream query 1: 3 written of 3");
            if (byte_count_draw) {
                check(q[2][0] == 12 && q[2][1] == 12, "stream query 2: 12 written of 12");
            }
            check(q[3][0] == 2 && q[3][1] == 4, "stream query 3: 2 written of 4 needed");
        }
    }

    vkDestroyFence(dev, fence, NULL);
    vkDestroyCommandPool(dev, cmdpool, NULL);
    if (queries) {
        vkDestroyQueryPool(dev, pool, NULL);
    }
    vkDestroyPipeline(dev, pipeline, NULL);
    vkDestroyShaderModule(dev, vs, NULL);
    vkDestroyPipelineLayout(dev, layout, NULL);
    vkUnmapMemory(dev, xfb.mem);
    vkDestroyBuffer(dev, xfb.buf, NULL);
    vkFreeMemory(dev, xfb.mem, NULL);
    vkUnmapMemory(dev, counters.mem);
    vkDestroyBuffer(dev, counters.buf, NULL);
    vkFreeMemory(dev, counters.mem, NULL);
    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(inst, NULL);

    printf("\n%d check(s) failed\n", failures);
    return failures ? 1 : 0;
}
