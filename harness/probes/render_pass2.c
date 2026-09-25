// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva
//
// The `render-pass2` group of src/venus/unserved.txt, as a program that names each command.
//
//   vkCreateRenderPass2  vkCmdBeginRenderPass2  vkCmdNextSubpass  vkCmdNextSubpass2
//   vkCmdEndRenderPass2
//
// Core in Vulkan 1.2, and vkCmdNextSubpass in 1.0. One render pass, made by vkCreateRenderPass2,
// with three subpasses over three R32_UINT attachments: subpass i writes attachment i and nothing
// else. The pass starts with vkCmdBeginRenderPass2, moves to subpass 1 with the 1.0
// vkCmdNextSubpass and to subpass 2 with vkCmdNextSubpass2, and ends with vkCmdEndRenderPass2,
// whose final layout is the one the copies read from. In each subpass vkCmdClearAttachments
// writes the subpass's number + 1 into its one attachment, so no pipeline is needed.
//
// A dropped subpass advance clears the next value into the previous attachment and leaves its
// own at the pass's clear of 0; a dropped begin or end leaves nothing written at all.
//
// Headless on purpose: no surface, no swapchain, no window. It runs over ssh.
//
//   cc -O1 -o render_pass2 render_pass2.c -lvulkan && ./render_pass2
//
// Exit 0 = every check passed. Exit 1 = a check failed, and the line above it says which.
// Exit 2 = the device could not be brought up at all (not a verdict on the group).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define W 8u
#define H 8u
#define SUBPASSES 3u

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

int main(void) {
    VkResult r;

    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "venus-render-pass2-probe",
        .apiVersion = VK_API_VERSION_1_2,
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
    if (props.apiVersion < VK_API_VERSION_1_2) {
        fatal("Vulkan 1.2, where the group is core", VK_ERROR_FEATURE_NOT_PRESENT);
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

    // ------------------------------------------------------------- the render pass

    VkAttachmentDescription2 atts[SUBPASSES];
    VkAttachmentReference2 refs[SUBPASSES];
    VkSubpassDescription2 subpasses[SUBPASSES];
    for (uint32_t i = 0; i < SUBPASSES; i++) {
        atts[i] = (VkAttachmentDescription2){
            .sType = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2,
            .format = VK_FORMAT_R32_UINT,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        };
        refs[i] = (VkAttachmentReference2){
            .sType = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2,
            .attachment = i,
            .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        };
        subpasses[i] = (VkSubpassDescription2){
            .sType = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2,
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .colorAttachmentCount = 1,
            .pColorAttachments = &refs[i],
        };
    }
    // Each subpass after the first waits for the one before, so the three are ordered even
    // though they touch different attachments; and the copies wait for the whole pass.
    VkSubpassDependency2 deps[SUBPASSES] = {
        {.sType = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2,
         .srcSubpass = 0,
         .dstSubpass = 1,
         .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
         .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT},
        {.sType = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2,
         .srcSubpass = 1,
         .dstSubpass = 2,
         .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
         .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT},
        {.sType = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2,
         .srcSubpass = 2,
         .dstSubpass = VK_SUBPASS_EXTERNAL,
         .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT,
         .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT},
    };
    VkRenderPassCreateInfo2 rpci = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2,
        .attachmentCount = SUBPASSES,
        .pAttachments = atts,
        .subpassCount = SUBPASSES,
        .pSubpasses = subpasses,
        .dependencyCount = SUBPASSES,
        .pDependencies = deps,
    };
    VkRenderPass pass;
    r = vkCreateRenderPass2(dev, &rpci, NULL, &pass);
    check(r == VK_SUCCESS, "vkCreateRenderPass2: the pass was made");
    if (r != VK_SUCCESS) {
        printf("\n%d check(s) failed\n", failures);
        return 1;
    }

    // --------------------------------------------------------- images and framebuffer

    VkImage images[SUBPASSES];
    VkDeviceMemory mems[SUBPASSES];
    VkImageView views[SUBPASSES];
    for (uint32_t i = 0; i < SUBPASSES; i++) {
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
        if ((r = vkCreateImage(dev, &imci, NULL, &images[i])) != VK_SUCCESS) {
            fatal("vkCreateImage", r);
        }
        VkMemoryRequirements mr;
        vkGetImageMemoryRequirements(dev, images[i], &mr);
        mems[i] = bind(pd, dev, mr, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        vkBindImageMemory(dev, images[i], mems[i], 0);
        VkImageViewCreateInfo ivci = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = images[i],
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = VK_FORMAT_R32_UINT,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
        };
        if ((r = vkCreateImageView(dev, &ivci, NULL, &views[i])) != VK_SUCCESS) {
            fatal("vkCreateImageView", r);
        }
    }
    VkFramebufferCreateInfo fbci = {
        .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .renderPass = pass,
        .attachmentCount = SUBPASSES,
        .pAttachments = views,
        .width = W,
        .height = H,
        .layers = 1,
    };
    VkFramebuffer fb;
    if ((r = vkCreateFramebuffer(dev, &fbci, NULL, &fb)) != VK_SUCCESS) {
        fatal("vkCreateFramebuffer", r);
    }

    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = SUBPASSES * W * H * sizeof(uint32_t),
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VkBuffer out;
    if ((r = vkCreateBuffer(dev, &bci, NULL, &out)) != VK_SUCCESS) {
        fatal("vkCreateBuffer", r);
    }
    VkMemoryRequirements bmr;
    vkGetBufferMemoryRequirements(dev, out, &bmr);
    VkDeviceMemory out_mem = bind(
        pd, dev, bmr, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkBindBufferMemory(dev, out, out_mem, 0);
    uint32_t *texels;
    if ((r = vkMapMemory(dev, out_mem, 0, VK_WHOLE_SIZE, 0, (void **)&texels)) != VK_SUCCESS) {
        fatal("vkMapMemory", r);
    }
    memset(texels, 0xee, SUBPASSES * W * H * sizeof(uint32_t));

    // ------------------------------------------------------------------ recording

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

    VkClearValue zero[SUBPASSES] = {{.color = {.uint32 = {0}}},
                                    {.color = {.uint32 = {0}}},
                                    {.color = {.uint32 = {0}}}};
    VkRenderPassBeginInfo rpbi = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = pass,
        .framebuffer = fb,
        .renderArea = {{0, 0}, {W, H}},
        .clearValueCount = SUBPASSES,
        .pClearValues = zero,
    };
    VkSubpassBeginInfo sbi = {
        .sType = VK_STRUCTURE_TYPE_SUBPASS_BEGIN_INFO,
        .contents = VK_SUBPASS_CONTENTS_INLINE,
    };
    VkSubpassEndInfo sei = {.sType = VK_STRUCTURE_TYPE_SUBPASS_END_INFO};
    VkClearRect rect = {{{0, 0}, {W, H}}, 0, 1};

    vkCmdBeginRenderPass2(cb, &rpbi, &sbi);
    for (uint32_t i = 0; i < SUBPASSES; i++) {
        if (i == 1) {
            vkCmdNextSubpass(cb, VK_SUBPASS_CONTENTS_INLINE);
        } else if (i == 2) {
            vkCmdNextSubpass2(cb, &sbi, &sei);
        }
        VkClearAttachment clear = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .colorAttachment = 0,
            .clearValue = {.color = {.uint32 = {i + 1, 0, 0, 0}}},
        };
        vkCmdClearAttachments(cb, 1, &clear, 1, &rect);
    }
    vkCmdEndRenderPass2(cb, &sei);

    for (uint32_t i = 0; i < SUBPASSES; i++) {
        VkBufferImageCopy copy = {
            .bufferOffset = i * W * H * sizeof(uint32_t),
            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .imageExtent = {W, H, 1},
        };
        vkCmdCopyImageToBuffer(cb, images[i], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, out, 1, &copy);
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
        uint32_t right[SUBPASSES] = {0};
        for (uint32_t i = 0; i < SUBPASSES; i++) {
            for (uint32_t t = 0; t < W * H; t++) {
                right[i] += texels[i * W * H + t] == i + 1;
            }
        }
        check(right[0] == W * H, "vkCmdBeginRenderPass2: subpass 0 wrote attachment 0");
        check(right[1] == W * H, "vkCmdNextSubpass: subpass 1 wrote attachment 1");
        check(right[2] == W * H, "vkCmdNextSubpass2: subpass 2 wrote attachment 2");
        check(right[0] + right[1] + right[2] == SUBPASSES * W * H,
              "vkCmdEndRenderPass2: every attachment reached its final layout");
        printf("  attachments hold %u, %u, %u (want 1, 2, 3)\n", texels[0], texels[W * H],
               texels[2 * W * H]);
    }

    vkDestroyFence(dev, fence, NULL);
    vkDestroyCommandPool(dev, cmdpool, NULL);
    vkDestroyFramebuffer(dev, fb, NULL);
    vkDestroyRenderPass(dev, pass, NULL);
    for (uint32_t i = 0; i < SUBPASSES; i++) {
        vkDestroyImageView(dev, views[i], NULL);
        vkDestroyImage(dev, images[i], NULL);
        vkFreeMemory(dev, mems[i], NULL);
    }
    vkUnmapMemory(dev, out_mem);
    vkDestroyBuffer(dev, out, NULL);
    vkFreeMemory(dev, out_mem, NULL);
    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(inst, NULL);

    printf("\n%d check(s) failed\n", failures);
    return failures ? 1 : 0;
}
