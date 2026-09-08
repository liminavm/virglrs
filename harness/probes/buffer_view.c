// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva
//
// The `buffer-view` group of src/venus/unserved.txt, as a program that names each command.
//
//   vkCreateBufferView  vkDestroyBufferView
//
// This group is the one with a witness: a stock Fedora 44 desktop refuses `vkCreateBufferView`
// repeatedly while it is merely running, so it is reached by an ordinary session and not only by
// a probe. What that costs is the whole context, and the guest never learns which command did it.
//
// The consequence checked is a read *through* the view. Nothing weaker distinguishes a served
// create from a refused one: the guest allocates its own handle, so a refused create still hands
// back a non-null `VkBufferView` and a `VK_SUCCESS`. Only fetching texels through it on the
// device says whether the host ever made the object. A compute shader reads four texels out of a
// uniform texel buffer and writes them to a storage buffer, and the probe compares them against
// what it put in.
//
// `vkDestroyBufferView` has no observable consequence: destroys return nothing, and the object is
// gone either way. It is called, and scored only by the worker log.
//
// Headless on purpose: no surface, no swapchain, no window. It runs over ssh.
//
//   cc -O1 -o buffer_view buffer_view.c -lvulkan && ./buffer_view
//
// Exit 0 = every check passed. Exit 1 = a check failed, and the line above it says which.
// Exit 2 = the device could not be brought up at all (not a verdict on the group).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define TEXELS 4

static int failures = 0;

static void check(int ok, const char *what) {
    printf("%-46s %s\n", what, ok ? "ok" : "FAILED");
    if (!ok) {
        failures++;
    }
}

static void fatal(const char *what, VkResult r) {
    fprintf(stderr, "cannot test the group: %s returned %d\n", what, (int)r);
    exit(2);
}

// The shader, compiled from:
//
//   #version 450
//   layout(local_size_x = 4) in;
//   layout(set = 0, binding = 0) uniform usamplerBuffer texels;
//   layout(set = 0, binding = 1) buffer Out { uint v[]; } outb;
//   void main() {
//       uint i = gl_GlobalInvocationID.x;
//       outb.v[i] = texelFetch(texels, int(i)).r;
//   }
//
// Embedded rather than compiled at build time: the guest this runs in has no shader compiler,
// and a probe that needs one is a probe that cannot be run where it matters.
static const uint32_t SHADER[] = {
    0x07230203, 0x00010000, 0x0008000b, 0x00000027, 0x00000000, 0x00020011,
    0x00000001, 0x00020011, 0x0000002e, 0x0006000b, 0x00000001, 0x4c534c47,
    0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0006000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000b,
    0x00060010, 0x00000004, 0x00000011, 0x00000004, 0x00000001, 0x00000001,
    0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d,
    0x00000000, 0x00030005, 0x00000008, 0x00000069, 0x00080005, 0x0000000b,
    0x475f6c67, 0x61626f6c, 0x766e496c, 0x7461636f, 0x496e6f69, 0x00000044,
    0x00030005, 0x00000011, 0x0074754f, 0x00040006, 0x00000011, 0x00000000,
    0x00000076, 0x00040005, 0x00000013, 0x6274756f, 0x00000000, 0x00040005,
    0x0000001a, 0x65786574, 0x0000736c, 0x00040047, 0x0000000b, 0x0000000b,
    0x0000001c, 0x00040047, 0x00000010, 0x00000006, 0x00000004, 0x00030047,
    0x00000011, 0x00000003, 0x00050048, 0x00000011, 0x00000000, 0x00000023,
    0x00000000, 0x00040047, 0x00000013, 0x00000021, 0x00000001, 0x00040047,
    0x00000013, 0x00000022, 0x00000000, 0x00040047, 0x0000001a, 0x00000021,
    0x00000000, 0x00040047, 0x0000001a, 0x00000022, 0x00000000, 0x00040047,
    0x00000026, 0x0000000b, 0x00000019, 0x00020013, 0x00000002, 0x00030021,
    0x00000003, 0x00000002, 0x00040015, 0x00000006, 0x00000020, 0x00000000,
    0x00040020, 0x00000007, 0x00000007, 0x00000006, 0x00040017, 0x00000009,
    0x00000006, 0x00000003, 0x00040020, 0x0000000a, 0x00000001, 0x00000009,
    0x0004003b, 0x0000000a, 0x0000000b, 0x00000001, 0x0004002b, 0x00000006,
    0x0000000c, 0x00000000, 0x00040020, 0x0000000d, 0x00000001, 0x00000006,
    0x0003001d, 0x00000010, 0x00000006, 0x0003001e, 0x00000011, 0x00000010,
    0x00040020, 0x00000012, 0x00000002, 0x00000011, 0x0004003b, 0x00000012,
    0x00000013, 0x00000002, 0x00040015, 0x00000014, 0x00000020, 0x00000001,
    0x0004002b, 0x00000014, 0x00000015, 0x00000000, 0x00090019, 0x00000017,
    0x00000006, 0x00000005, 0x00000000, 0x00000000, 0x00000000, 0x00000001,
    0x00000000, 0x0003001b, 0x00000018, 0x00000017, 0x00040020, 0x00000019,
    0x00000000, 0x00000018, 0x0004003b, 0x00000019, 0x0000001a, 0x00000000,
    0x00040017, 0x0000001f, 0x00000006, 0x00000004, 0x00040020, 0x00000022,
    0x00000002, 0x00000006, 0x0004002b, 0x00000006, 0x00000024, 0x00000004,
    0x0004002b, 0x00000006, 0x00000025, 0x00000001, 0x0006002c, 0x00000009,
    0x00000026, 0x00000024, 0x00000025, 0x00000025, 0x00050036, 0x00000002,
    0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0004003b,
    0x00000007, 0x00000008, 0x00000007, 0x00050041, 0x0000000d, 0x0000000e,
    0x0000000b, 0x0000000c, 0x0004003d, 0x00000006, 0x0000000f, 0x0000000e,
    0x0003003e, 0x00000008, 0x0000000f, 0x0004003d, 0x00000006, 0x00000016,
    0x00000008, 0x0004003d, 0x00000018, 0x0000001b, 0x0000001a, 0x0004003d,
    0x00000006, 0x0000001c, 0x00000008, 0x0004007c, 0x00000014, 0x0000001d,
    0x0000001c, 0x00040064, 0x00000017, 0x0000001e, 0x0000001b, 0x0005005f,
    0x0000001f, 0x00000020, 0x0000001e, 0x0000001d, 0x00050051, 0x00000006,
    0x00000021, 0x00000020, 0x00000000, 0x00060041, 0x00000022, 0x00000023,
    0x00000013, 0x00000015, 0x00000016, 0x0003003e, 0x00000023, 0x00000021,
    0x000100fd, 0x00010038,};

// Pick a memory type the host can write and the device can read, and say so rather than guessing.
static uint32_t host_visible_type(VkPhysicalDevice pd, uint32_t bits) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    const VkMemoryPropertyFlags want =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) {
            return i;
        }
    }
    return UINT32_MAX;
}

int main(void) {
    VkResult r;

    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "venus-buffer-view-probe",
        .apiVersion = VK_API_VERSION_1_1,
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

    // A texel buffer of this format has to be sampleable, or the group cannot be tested here at
    // all -- which is a fact about the host, not a verdict on the commands.
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(pd, VK_FORMAT_R32_UINT, &fp);
    if (!(fp.bufferFeatures & VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT)) {
        fatal("R32_UINT uniform texel buffers unsupported", VK_ERROR_FORMAT_NOT_SUPPORTED);
    }

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
        fatal("no compute queue family", VK_ERROR_INITIALIZATION_FAILED);
    }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = qfam,
        .queueCount = 1,
        .pQueuePriorities = &prio,
    };
    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qci,
    };
    VkDevice dev;
    if ((r = vkCreateDevice(pd, &dci, NULL, &dev)) != VK_SUCCESS) {
        fatal("vkCreateDevice", r);
    }
    VkQueue queue;
    vkGetDeviceQueue(dev, qfam, 0, &queue);

    // Two buffers: the texel source the view is taken of, and the storage the shader writes to.
    const VkDeviceSize bytes = TEXELS * sizeof(uint32_t);
    VkBuffer texbuf, outbuf;
    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = bytes,
        .usage = VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    if ((r = vkCreateBuffer(dev, &bci, NULL, &texbuf)) != VK_SUCCESS) {
        fatal("vkCreateBuffer (texel)", r);
    }
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if ((r = vkCreateBuffer(dev, &bci, NULL, &outbuf)) != VK_SUCCESS) {
        fatal("vkCreateBuffer (storage)", r);
    }

    VkMemoryRequirements mr_tex, mr_out;
    vkGetBufferMemoryRequirements(dev, texbuf, &mr_tex);
    vkGetBufferMemoryRequirements(dev, outbuf, &mr_out);
    uint32_t type = host_visible_type(pd, mr_tex.memoryTypeBits & mr_out.memoryTypeBits);
    if (type == UINT32_MAX) {
        fatal("no host-visible memory type for both buffers", VK_ERROR_INITIALIZATION_FAILED);
    }

    VkDeviceMemory mem_tex, mem_out;
    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mr_tex.size,
        .memoryTypeIndex = type,
    };
    if ((r = vkAllocateMemory(dev, &mai, NULL, &mem_tex)) != VK_SUCCESS) {
        fatal("vkAllocateMemory (texel)", r);
    }
    mai.allocationSize = mr_out.size;
    if ((r = vkAllocateMemory(dev, &mai, NULL, &mem_out)) != VK_SUCCESS) {
        fatal("vkAllocateMemory (storage)", r);
    }
    vkBindBufferMemory(dev, texbuf, mem_tex, 0);
    vkBindBufferMemory(dev, outbuf, mem_out, 0);

    // Values chosen so a zeroed output cannot pass: none of them is 0, and they are not equal to
    // their own index either, so neither a zero-fill nor an off-by-one reads as success.
    const uint32_t want[TEXELS] = {0xa1b2c3d4u, 0x11111111u, 0xdeadbeefu, 0x00c0ffeeu};
    void *p = NULL;
    if ((r = vkMapMemory(dev, mem_tex, 0, bytes, 0, &p)) != VK_SUCCESS) {
        fatal("vkMapMemory (texel)", r);
    }
    memcpy(p, want, sizeof want);
    vkUnmapMemory(dev, mem_tex);
    if ((r = vkMapMemory(dev, mem_out, 0, bytes, 0, &p)) != VK_SUCCESS) {
        fatal("vkMapMemory (storage)", r);
    }
    memset(p, 0, bytes);
    vkUnmapMemory(dev, mem_out);

    // ---- vkCreateBufferView -------------------------------------------------------------
    VkBufferViewCreateInfo bvci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
        .buffer = texbuf,
        .format = VK_FORMAT_R32_UINT,
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };
    VkBufferView view = VK_NULL_HANDLE;
    r = vkCreateBufferView(dev, &bvci, NULL, &view);
    check(r == VK_SUCCESS, "vkCreateBufferView returned VK_SUCCESS");
    if (r != VK_SUCCESS) {
        // Nothing below can say anything without a view, and the group is already failed.
        printf("\n%d check(s) failed\n", failures);
        return 1;
    }

    VkDescriptorSetLayoutBinding binds[2] = {
        {.binding = 0,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        {.binding = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
    };
    VkDescriptorSetLayoutCreateInfo dslci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 2,
        .pBindings = binds,
    };
    VkDescriptorSetLayout dsl;
    if ((r = vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl)) != VK_SUCCESS) {
        fatal("vkCreateDescriptorSetLayout", r);
    }

    VkDescriptorPoolSize sizes[2] = {
        {.type = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, .descriptorCount = 1},
        {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1},
    };
    VkDescriptorPoolCreateInfo dpci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 2,
        .pPoolSizes = sizes,
    };
    VkDescriptorPool pool;
    if ((r = vkCreateDescriptorPool(dev, &dpci, NULL, &pool)) != VK_SUCCESS) {
        fatal("vkCreateDescriptorPool", r);
    }
    VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &dsl,
    };
    VkDescriptorSet set;
    if ((r = vkAllocateDescriptorSets(dev, &dsai, &set)) != VK_SUCCESS) {
        fatal("vkAllocateDescriptorSets", r);
    }

    VkDescriptorBufferInfo obi = {.buffer = outbuf, .offset = 0, .range = VK_WHOLE_SIZE};
    VkWriteDescriptorSet writes[2] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = set,
         .dstBinding = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
         .pTexelBufferView = &view},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = set,
         .dstBinding = 1,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &obi},
    };
    vkUpdateDescriptorSets(dev, 2, writes, 0, NULL);

    VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = sizeof SHADER,
        .pCode = SHADER,
    };
    VkShaderModule sm;
    if ((r = vkCreateShaderModule(dev, &smci, NULL, &sm)) != VK_SUCCESS) {
        fatal("vkCreateShaderModule", r);
    }
    VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &dsl,
    };
    VkPipelineLayout layout;
    if ((r = vkCreatePipelineLayout(dev, &plci, NULL, &layout)) != VK_SUCCESS) {
        fatal("vkCreatePipelineLayout", r);
    }
    VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                  .module = sm,
                  .pName = "main"},
        .layout = layout,
    };
    VkPipeline pipe;
    if ((r = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe)) != VK_SUCCESS) {
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
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, NULL);
    vkCmdDispatch(cb, 1, 1, 1);
    vkEndCommandBuffer(cb);

    // A fence, and a finite wait on it. A probe that blocks forever scores as somebody else's
    // timeout and names nothing.
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
    check(r == VK_SUCCESS, "the dispatch reading through the view completed");

    if (r == VK_SUCCESS) {
        uint32_t got[TEXELS];
        if ((r = vkMapMemory(dev, mem_out, 0, bytes, 0, &p)) != VK_SUCCESS) {
            fatal("vkMapMemory (readback)", r);
        }
        memcpy(got, p, sizeof got);
        vkUnmapMemory(dev, mem_out);
        int same = memcmp(got, want, sizeof want) == 0;
        if (!same) {
            printf("  wanted");
            for (int i = 0; i < TEXELS; i++) {
                printf(" %08x", want[i]);
            }
            printf("\n  got   ");
            for (int i = 0; i < TEXELS; i++) {
                printf(" %08x", got[i]);
            }
            printf("\n");
        }
        check(same, "texels fetched through the view are the ones written");
    }

    // ---- vkDestroyBufferView ------------------------------------------------------------
    // No consequence to check: a destroy returns nothing, and a refused one leaves the guest
    // exactly as a served one does. Called so the command is reached at all; the worker log is
    // the only thing that can score it.
    vkDestroyBufferView(dev, view, NULL);
    printf("%-46s called (unobservable; see the log)\n", "vkDestroyBufferView");

    vkDestroyFence(dev, fence, NULL);
    vkDestroyCommandPool(dev, cmdpool, NULL);
    vkDestroyPipeline(dev, pipe, NULL);
    vkDestroyPipelineLayout(dev, layout, NULL);
    vkDestroyShaderModule(dev, sm, NULL);
    vkDestroyDescriptorPool(dev, pool, NULL);
    vkDestroyDescriptorSetLayout(dev, dsl, NULL);
    vkDestroyBuffer(dev, texbuf, NULL);
    vkDestroyBuffer(dev, outbuf, NULL);
    vkFreeMemory(dev, mem_tex, NULL);
    vkFreeMemory(dev, mem_out, NULL);
    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(inst, NULL);

    printf("\n%d check(s) failed\n", failures);
    return failures ? 1 : 0;
}
