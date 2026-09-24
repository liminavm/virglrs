// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva
//
// The `maintenance6-binding` group of src/venus/unserved.txt, as a program that names each
// command.
//
//   vkCmdPushDescriptorSet2  vkCmdBindDescriptorSets2  vkCmdPushConstants2
//
// All three are core in Vulkan 1.4 and come from VK_KHR_maintenance6 before it. One compute shader
// reads a buffer bound by vkCmdBindDescriptorSets2, adds a vkCmdPushConstants2 value, and writes a
// buffer pushed by vkCmdPushDescriptorSet2. A second dispatch in the same command buffer pushes a
// different output and a different constant, so each command has a consequence of its own: a
// dropped push descriptor leaves an output untouched, a dropped push constant leaves the wrong
// addend, and a dropped bind reads nothing.
//
// The shader, compiled with `glslangValidator -V --target-env vulkan1.1 -x`:
//
//   #version 450
//   layout(local_size_x = 64) in;
//   layout(set = 0, binding = 0) writeonly buffer Out { uint o[]; };
//   layout(set = 1, binding = 0) readonly buffer In { uint i[]; };
//   layout(push_constant) uniform PC { uint add; } pc;
//   void main() {
//       uint x = gl_GlobalInvocationID.x;
//       o[x] = i[x] * 3u + pc.add;
//   }
//
// Headless on purpose: no surface, no swapchain, no window. It runs over ssh.
//
//   cc -O1 -o maintenance6_binding maintenance6_binding.c -lvulkan && ./maintenance6_binding
//
// Exit 0 = every check passed. Exit 1 = a check failed, and the line above it says which.
// Exit 2 = the device could not be brought up at all (not a verdict on the group).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define N 256u
#define FIRST_ADD 0x1000u
#define SECOND_ADD 0x20000u

static const uint32_t SHADER[] = {
    0x07230203,0x00010300,0x0008000b,0x0000002c,0x00000000,0x00020011,0x00000001,0x0006000b,
    0x00000001,0x4c534c47,0x6474732e,0x3035342e,0x00000000,0x0003000e,0x00000000,0x00000001,
    0x0006000f,0x00000005,0x00000004,0x6e69616d,0x00000000,0x0000000b,0x00060010,0x00000004,
    0x00000011,0x00000040,0x00000001,0x00000001,0x00030003,0x00000002,0x000001c2,0x00040005,
    0x00000004,0x6e69616d,0x00000000,0x00030005,0x00000008,0x00000078,0x00080005,0x0000000b,
    0x475f6c67,0x61626f6c,0x766e496c,0x7461636f,0x496e6f69,0x00000044,0x00030005,0x00000011,
    0x0074754f,0x00040006,0x00000011,0x00000000,0x0000006f,0x00030005,0x00000013,0x00000000,
    0x00030005,0x00000018,0x00006e49,0x00040006,0x00000018,0x00000000,0x00000069,0x00030005,
    0x0000001a,0x00000000,0x00030005,0x00000021,0x00004350,0x00040006,0x00000021,0x00000000,
    0x00646461,0x00030005,0x00000023,0x00006370,0x00040047,0x0000000b,0x0000000b,0x0000001c,
    0x00040047,0x00000010,0x00000006,0x00000004,0x00030047,0x00000011,0x00000002,0x00040048,
    0x00000011,0x00000000,0x00000019,0x00050048,0x00000011,0x00000000,0x00000023,0x00000000,
    0x00030047,0x00000013,0x00000019,0x00040047,0x00000013,0x00000021,0x00000000,0x00040047,
    0x00000013,0x00000022,0x00000000,0x00040047,0x00000017,0x00000006,0x00000004,0x00030047,
    0x00000018,0x00000002,0x00040048,0x00000018,0x00000000,0x00000018,0x00050048,0x00000018,
    0x00000000,0x00000023,0x00000000,0x00030047,0x0000001a,0x00000018,0x00040047,0x0000001a,
    0x00000021,0x00000000,0x00040047,0x0000001a,0x00000022,0x00000001,0x00030047,0x00000021,
    0x00000002,0x00050048,0x00000021,0x00000000,0x00000023,0x00000000,0x00040047,0x0000002b,
    0x0000000b,0x00000019,0x00020013,0x00000002,0x00030021,0x00000003,0x00000002,0x00040015,
    0x00000006,0x00000020,0x00000000,0x00040020,0x00000007,0x00000007,0x00000006,0x00040017,
    0x00000009,0x00000006,0x00000003,0x00040020,0x0000000a,0x00000001,0x00000009,0x0004003b,
    0x0000000a,0x0000000b,0x00000001,0x0004002b,0x00000006,0x0000000c,0x00000000,0x00040020,
    0x0000000d,0x00000001,0x00000006,0x0003001d,0x00000010,0x00000006,0x0003001e,0x00000011,
    0x00000010,0x00040020,0x00000012,0x0000000c,0x00000011,0x0004003b,0x00000012,0x00000013,
    0x0000000c,0x00040015,0x00000014,0x00000020,0x00000001,0x0004002b,0x00000014,0x00000015,
    0x00000000,0x0003001d,0x00000017,0x00000006,0x0003001e,0x00000018,0x00000017,0x00040020,
    0x00000019,0x0000000c,0x00000018,0x0004003b,0x00000019,0x0000001a,0x0000000c,0x00040020,
    0x0000001c,0x0000000c,0x00000006,0x0004002b,0x00000006,0x0000001f,0x00000003,0x0003001e,
    0x00000021,0x00000006,0x00040020,0x00000022,0x00000009,0x00000021,0x0004003b,0x00000022,
    0x00000023,0x00000009,0x00040020,0x00000024,0x00000009,0x00000006,0x0004002b,0x00000006,
    0x00000029,0x00000040,0x0004002b,0x00000006,0x0000002a,0x00000001,0x0006002c,0x00000009,
    0x0000002b,0x00000029,0x0000002a,0x0000002a,0x00050036,0x00000002,0x00000004,0x00000000,
    0x00000003,0x000200f8,0x00000005,0x0004003b,0x00000007,0x00000008,0x00000007,0x00050041,
    0x0000000d,0x0000000e,0x0000000b,0x0000000c,0x0004003d,0x00000006,0x0000000f,0x0000000e,
    0x0003003e,0x00000008,0x0000000f,0x0004003d,0x00000006,0x00000016,0x00000008,0x0004003d,
    0x00000006,0x0000001b,0x00000008,0x00060041,0x0000001c,0x0000001d,0x0000001a,0x00000015,
    0x0000001b,0x0004003d,0x00000006,0x0000001e,0x0000001d,0x00050084,0x00000006,0x00000020,
    0x0000001e,0x0000001f,0x00050041,0x00000024,0x00000025,0x00000023,0x00000015,0x0004003d,
    0x00000006,0x00000026,0x00000025,0x00050080,0x00000006,0x00000027,0x00000020,0x00000026,
    0x00060041,0x0000001c,0x00000028,0x00000013,0x00000015,0x00000016,0x0003003e,0x00000028,
    0x00000027,0x000100fd,0x00010038
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

struct mapped {
    VkBuffer buf;
    VkDeviceMemory mem;
    uint32_t *p;
};

static struct mapped storage(VkPhysicalDevice pd, VkDevice dev) {
    struct mapped m;
    VkResult r;
    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = N * sizeof(uint32_t),
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    if ((r = vkCreateBuffer(dev, &bci, NULL, &m.buf)) != VK_SUCCESS) {
        fatal("vkCreateBuffer", r);
    }
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev, m.buf, &mr);
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
    if ((r = vkAllocateMemory(dev, &mai, NULL, &m.mem)) != VK_SUCCESS) {
        fatal("vkAllocateMemory", r);
    }
    vkBindBufferMemory(dev, m.buf, m.mem, 0);
    if ((r = vkMapMemory(dev, m.mem, 0, VK_WHOLE_SIZE, 0, (void **)&m.p)) != VK_SUCCESS) {
        fatal("vkMapMemory", r);
    }
    return m;
}

static int has_ext(VkExtensionProperties *exts, uint32_t n, const char *name) {
    for (uint32_t i = 0; i < n; i++) {
        if (strcmp(exts[i].extensionName, name) == 0) {
            return 1;
        }
    }
    return 0;
}

// The core name on a 1.4 device, the KHR one before it: one command either way.
static PFN_vkVoidFunction proc(VkDevice dev, const char *core, const char *khr) {
    PFN_vkVoidFunction f = vkGetDeviceProcAddr(dev, core);
    return f ? f : vkGetDeviceProcAddr(dev, khr);
}

int main(void) {
    VkResult r;

    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "venus-maintenance6-binding-probe",
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
    int core14 = props.apiVersion >= VK_API_VERSION_1_4;
    printf("device: %s (Vulkan %u.%u)\n", props.deviceName,
           VK_API_VERSION_MAJOR(props.apiVersion), VK_API_VERSION_MINOR(props.apiVersion));

    uint32_t en = 0;
    vkEnumerateDeviceExtensionProperties(pd, NULL, &en, NULL);
    VkExtensionProperties *exts = calloc(en, sizeof *exts);
    vkEnumerateDeviceExtensionProperties(pd, NULL, &en, exts);

    // Before 1.4 the group is two extensions, and without both there is nothing to test.
    const char *names[2] = {"VK_KHR_maintenance6", "VK_KHR_push_descriptor"};
    if (!core14 && !(has_ext(exts, en, names[0]) && has_ext(exts, en, names[1]))) {
        fatal("neither Vulkan 1.4 nor VK_KHR_maintenance6 + VK_KHR_push_descriptor",
              VK_ERROR_FEATURE_NOT_PRESENT);
    }
    printf("reached through: %s\n\n", core14 ? "Vulkan 1.4 core" : "the KHR extensions");

    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, NULL);
    VkQueueFamilyProperties *qs = calloc(qn, sizeof *qs);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qs);
    uint32_t qfam = UINT32_MAX;
    for (uint32_t i = 0; i < qn; i++) {
        if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            qfam = i;
            break;
        }
    }
    if (qfam == UINT32_MAX) {
        fatal("no compute queue", VK_ERROR_FEATURE_NOT_PRESENT);
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
        .maintenance6 = VK_TRUE,
        .pushDescriptor = VK_TRUE,
    };
    VkPhysicalDeviceMaintenance6FeaturesKHR fm6 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_6_FEATURES_KHR,
        .maintenance6 = VK_TRUE,
    };
    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = core14 ? (void *)&f14 : (void *)&fm6,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qci,
        .enabledExtensionCount = core14 ? 0u : 2u,
        .ppEnabledExtensionNames = core14 ? NULL : names,
    };
    VkDevice dev;
    if ((r = vkCreateDevice(pd, &dci, NULL, &dev)) != VK_SUCCESS) {
        fatal("vkCreateDevice", r);
    }
    VkQueue queue;
    vkGetDeviceQueue(dev, qfam, 0, &queue);

    PFN_vkCmdPushDescriptorSet2 push_set = (PFN_vkCmdPushDescriptorSet2)proc(
        dev, "vkCmdPushDescriptorSet2", "vkCmdPushDescriptorSet2KHR");
    PFN_vkCmdBindDescriptorSets2 bind_sets = (PFN_vkCmdBindDescriptorSets2)proc(
        dev, "vkCmdBindDescriptorSets2", "vkCmdBindDescriptorSets2KHR");
    PFN_vkCmdPushConstants2 push_constants =
        (PFN_vkCmdPushConstants2)proc(dev, "vkCmdPushConstants2", "vkCmdPushConstants2KHR");
    if (!push_set || !bind_sets || !push_constants) {
        fatal("an entry point of the group did not resolve", VK_ERROR_EXTENSION_NOT_PRESENT);
    }

    struct mapped in = storage(pd, dev), out1 = storage(pd, dev), out2 = storage(pd, dev);
    for (uint32_t i = 0; i < N; i++) {
        in.p[i] = i + 1;
        out1.p[i] = 0;
        out2.p[i] = 0;
    }

    // Set 0 is pushed, set 1 is allocated and bound.
    VkDescriptorSetLayoutBinding binding = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
    };
    VkDescriptorSetLayoutCreateInfo pushed_li = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT,
        .bindingCount = 1,
        .pBindings = &binding,
    };
    VkDescriptorSetLayoutCreateInfo bound_li = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &binding,
    };
    VkDescriptorSetLayout set_layouts[2];
    if ((r = vkCreateDescriptorSetLayout(dev, &pushed_li, NULL, &set_layouts[0])) != VK_SUCCESS) {
        fatal("vkCreateDescriptorSetLayout (pushed)", r);
    }
    if ((r = vkCreateDescriptorSetLayout(dev, &bound_li, NULL, &set_layouts[1])) != VK_SUCCESS) {
        fatal("vkCreateDescriptorSetLayout (bound)", r);
    }
    VkPushConstantRange range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(uint32_t),
    };
    VkPipelineLayoutCreateInfo pli = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 2,
        .pSetLayouts = set_layouts,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &range,
    };
    VkPipelineLayout layout;
    if ((r = vkCreatePipelineLayout(dev, &pli, NULL, &layout)) != VK_SUCCESS) {
        fatal("vkCreatePipelineLayout", r);
    }

    VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1};
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
        .pSetLayouts = &set_layouts[1],
    };
    VkDescriptorSet bound;
    if ((r = vkAllocateDescriptorSets(dev, &dsai, &bound)) != VK_SUCCESS) {
        fatal("vkAllocateDescriptorSets", r);
    }
    VkDescriptorBufferInfo in_info = {.buffer = in.buf, .offset = 0, .range = VK_WHOLE_SIZE};
    VkWriteDescriptorSet in_write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = bound,
        .dstBinding = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &in_info,
    };
    vkUpdateDescriptorSets(dev, 1, &in_write, 0, NULL);

    VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = sizeof SHADER,
        .pCode = SHADER,
    };
    VkShaderModule module;
    if ((r = vkCreateShaderModule(dev, &smci, NULL, &module)) != VK_SUCCESS) {
        fatal("vkCreateShaderModule", r);
    }
    VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage =
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = module,
                .pName = "main",
            },
        .layout = layout,
    };
    VkPipeline pipeline;
    if ((r = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipeline)) !=
        VK_SUCCESS) {
        fatal("vkCreateComputePipelines", r);
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
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

    VkBindDescriptorSetsInfo bind = {
        .sType = VK_STRUCTURE_TYPE_BIND_DESCRIPTOR_SETS_INFO,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .layout = layout,
        .firstSet = 1,
        .descriptorSetCount = 1,
        .pDescriptorSets = &bound,
    };
    bind_sets(cb, &bind);

    // Two dispatches, each with its own pushed output and its own constant.
    struct mapped *outs[2] = {&out1, &out2};
    uint32_t adds[2] = {FIRST_ADD, SECOND_ADD};
    for (int k = 0; k < 2; k++) {
        VkDescriptorBufferInfo out_info = {.buffer = outs[k]->buf, .range = VK_WHOLE_SIZE};
        VkWriteDescriptorSet out_write = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &out_info,
        };
        VkPushDescriptorSetInfo pushed = {
            .sType = VK_STRUCTURE_TYPE_PUSH_DESCRIPTOR_SET_INFO,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .layout = layout,
            .set = 0,
            .descriptorWriteCount = 1,
            .pDescriptorWrites = &out_write,
        };
        push_set(cb, &pushed);
        VkPushConstantsInfo constant = {
            .sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
            .layout = layout,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(uint32_t),
            .pValues = &adds[k],
        };
        push_constants(cb, &constant);
        vkCmdDispatch(cb, N / 64, 1, 1);
    }
    VkMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
    };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                         1, &barrier, 0, NULL, 0, NULL);
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
        // Counted rather than stopped at the first miss: how much of an output is wrong says
        // which command it was -- all of it untouched is a lost push descriptor, all of it off by
        // the same amount is a lost constant.
        uint32_t right[2] = {0, 0}, untouched[2] = {0, 0};
        for (int k = 0; k < 2; k++) {
            for (uint32_t i = 0; i < N; i++) {
                uint32_t v = outs[k]->p[i];
                right[k] += v == (i + 1) * 3u + adds[k];
                untouched[k] += v == 0;
            }
        }
        check(untouched[0] == 0 && untouched[1] == 0,
              "vkCmdPushDescriptorSet2: both pushed outputs were written");
        check(right[0] == N,
              "first dispatch: bound input, pushed output, first constant");
        check(right[1] == N,
              "second dispatch: re-pushed output and constant took effect");
        check(right[0] == N && right[1] == N,
              "vkCmdBindDescriptorSets2: the input read is the bound set");
        check(right[0] == N && right[1] == N && FIRST_ADD != SECOND_ADD,
              "vkCmdPushConstants2: each dispatch saw its own constant");
        printf("  outputs right: %u/%u and %u/%u; untouched: %u and %u\n", right[0], N, right[1],
               N, untouched[0], untouched[1]);
    }

    vkDestroyFence(dev, fence, NULL);
    vkDestroyCommandPool(dev, cmdpool, NULL);
    vkDestroyPipeline(dev, pipeline, NULL);
    vkDestroyShaderModule(dev, module, NULL);
    vkDestroyDescriptorPool(dev, dpool, NULL);
    vkDestroyPipelineLayout(dev, layout, NULL);
    vkDestroyDescriptorSetLayout(dev, set_layouts[0], NULL);
    vkDestroyDescriptorSetLayout(dev, set_layouts[1], NULL);
    struct mapped *all[3] = {&in, &out1, &out2};
    for (int k = 0; k < 3; k++) {
        vkUnmapMemory(dev, all[k]->mem);
        vkDestroyBuffer(dev, all[k]->buf, NULL);
        vkFreeMemory(dev, all[k]->mem, NULL);
    }
    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(inst, NULL);

    printf("\n%d check(s) failed\n", failures);
    return failures ? 1 : 0;
}
