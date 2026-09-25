// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva
//
// The `copy-commands2` group of src/venus/unserved.txt, as a program that names each command.
//
//   vkCmdCopyBuffer2  vkCmdCopyBufferToImage2  vkCmdCopyImage2  vkCmdBlitImage2
//   vkCmdCopyImageToBuffer2  vkCmdResolveImage2
//
// Core in Vulkan 1.3. A pattern of 8x8 RGBA8 texels goes through a chain, one command per hop:
//
//   source buffer --CopyBuffer2--> staging buffer --CopyBufferToImage2--> image A
//   image A --CopyImage2--> image B --BlitImage2, mirrored in x--> image C
//   image C --CopyImageToBuffer2--> readback
//
// and, beside it, a 4x multisampled image cleared to one colour --ResolveImage2--> image D,
// read back the same way. The staging buffer is read directly, so the first hop is scored on
// its own; the chain's end is the pattern mirrored, which only a blit that ran, with its
// regions as sent, produces. A dropped hop leaves every later stage at its initial contents.
//
// Headless on purpose: no surface, no swapchain, no window. It runs over ssh.
//
//   cc -O1 -o copy_commands2 copy_commands2.c -lvulkan && ./copy_commands2
//
// Exit 0 = every check passed. Exit 1 = a check failed, and the line above it says which.
// Exit 2 = the device could not be brought up at all (not a verdict on the group).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define W 8u
#define H 8u
#define TEXELS (W * H)
#define BYTES (TEXELS * 4u)

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
    uint32_t *p;
};

static struct mapped mapped(VkPhysicalDevice pd, VkDevice dev, VkDeviceSize size) {
    struct mapped m;
    VkResult r;
    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
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

struct image {
    VkImage image;
    VkDeviceMemory mem;
};

static struct image image(VkPhysicalDevice pd, VkDevice dev, VkSampleCountFlagBits samples) {
    struct image i;
    VkImageCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .extent = {W, H, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = samples,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                 VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VkResult r = vkCreateImage(dev, &ici, NULL, &i.image);
    if (r != VK_SUCCESS) {
        fatal("vkCreateImage", r);
    }
    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(dev, i.image, &mr);
    i.mem = bind(pd, dev, mr, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkBindImageMemory(dev, i.image, i.mem, 0);
    return i;
}

static void unimaged(VkDevice dev, struct image *i) {
    vkDestroyImage(dev, i->image, NULL);
    vkFreeMemory(dev, i->mem, NULL);
}

// Every hop is a transfer, so one full transfer-to-transfer barrier between hops orders them,
// and a layout change rides on it where one is needed.
static void to_layout(VkCommandBuffer cb, VkImage image, VkImageLayout from, VkImageLayout to) {
    VkImageMemoryBarrier b = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        .oldLayout = from,
        .newLayout = to,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                         NULL, 0, NULL, 1, &b);
}

static void transfer_barrier(VkCommandBuffer cb) {
    VkMemoryBarrier b = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
    };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                         &b, 0, NULL, 0, NULL);
}

static uint32_t pattern(uint32_t x, uint32_t y) {
    return 0xff000000u | (y * 16 + x) << 8 | (x * 31 + y * 7);
}

int main(void) {
    VkResult r;

    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "venus-copy-commands2-probe",
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
        fatal("Vulkan 1.3, where the group is core", VK_ERROR_FEATURE_NOT_PRESENT);
    }
    if (!(props.limits.framebufferColorSampleCounts & VK_SAMPLE_COUNT_4_BIT)) {
        fatal("no 4x colour multisampling", VK_ERROR_FEATURE_NOT_PRESENT);
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

    struct mapped source = mapped(pd, dev, BYTES), staging = mapped(pd, dev, BYTES),
                  chain_out = mapped(pd, dev, BYTES), resolve_out = mapped(pd, dev, BYTES);
    for (uint32_t y = 0; y < H; y++) {
        for (uint32_t x = 0; x < W; x++) {
            source.p[y * W + x] = pattern(x, y);
        }
    }
    struct image a = image(pd, dev, VK_SAMPLE_COUNT_1_BIT), b = image(pd, dev, VK_SAMPLE_COUNT_1_BIT),
                 c = image(pd, dev, VK_SAMPLE_COUNT_1_BIT), d = image(pd, dev, VK_SAMPLE_COUNT_1_BIT),
                 ms = image(pd, dev, VK_SAMPLE_COUNT_4_BIT);

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

    const VkImageLayout DST = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    const VkImageLayout SRC = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    const VkImageSubresourceLayers layers = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    struct image *all[5] = {&a, &b, &c, &d, &ms};
    for (int i = 0; i < 5; i++) {
        to_layout(cb, all[i]->image, VK_IMAGE_LAYOUT_UNDEFINED, DST);
    }

    // Hop 1: buffer to buffer.
    VkBufferCopy2 bc = {.sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2, .size = BYTES};
    VkCopyBufferInfo2 cbi = {
        .sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
        .srcBuffer = source.buf,
        .dstBuffer = staging.buf,
        .regionCount = 1,
        .pRegions = &bc,
    };
    vkCmdCopyBuffer2(cb, &cbi);
    transfer_barrier(cb);

    // Hop 2: buffer to image A.
    VkBufferImageCopy2 bic = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
        .imageSubresource = layers,
        .imageExtent = {W, H, 1},
    };
    VkCopyBufferToImageInfo2 b2i = {
        .sType = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2,
        .srcBuffer = staging.buf,
        .dstImage = a.image,
        .dstImageLayout = DST,
        .regionCount = 1,
        .pRegions = &bic,
    };
    vkCmdCopyBufferToImage2(cb, &b2i);
    to_layout(cb, a.image, DST, SRC);

    // Hop 3: image A to image B.
    VkImageCopy2 ic = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_COPY_2,
        .srcSubresource = layers,
        .dstSubresource = layers,
        .extent = {W, H, 1},
    };
    VkCopyImageInfo2 i2i = {
        .sType = VK_STRUCTURE_TYPE_COPY_IMAGE_INFO_2,
        .srcImage = a.image,
        .srcImageLayout = SRC,
        .dstImage = b.image,
        .dstImageLayout = DST,
        .regionCount = 1,
        .pRegions = &ic,
    };
    vkCmdCopyImage2(cb, &i2i);
    to_layout(cb, b.image, DST, SRC);

    // Hop 4: image B to image C, mirrored in x -- the destination's x runs backwards.
    VkImageBlit2 blit = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2,
        .srcSubresource = layers,
        .srcOffsets = {{0, 0, 0}, {W, H, 1}},
        .dstSubresource = layers,
        .dstOffsets = {{W, 0, 0}, {0, H, 1}},
    };
    VkBlitImageInfo2 bli = {
        .sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2,
        .srcImage = b.image,
        .srcImageLayout = SRC,
        .dstImage = c.image,
        .dstImageLayout = DST,
        .regionCount = 1,
        .pRegions = &blit,
        .filter = VK_FILTER_NEAREST,
    };
    vkCmdBlitImage2(cb, &bli);
    to_layout(cb, c.image, DST, SRC);

    // Hop 5: image C back out.
    VkCopyImageToBufferInfo2 i2b = {
        .sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2,
        .srcImage = c.image,
        .srcImageLayout = SRC,
        .dstBuffer = chain_out.buf,
        .regionCount = 1,
        .pRegions = &bic,
    };
    vkCmdCopyImageToBuffer2(cb, &i2b);

    // Beside the chain: a multisampled clear, resolved.
    VkClearColorValue colour = {.float32 = {0.2f, 0.4f, 0.6f, 1.0f}};
    VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cb, ms.image, DST, &colour, 1, &range);
    to_layout(cb, ms.image, DST, SRC);
    VkImageResolve2 res = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_RESOLVE_2,
        .srcSubresource = layers,
        .dstSubresource = layers,
        .extent = {W, H, 1},
    };
    VkResolveImageInfo2 rsi = {
        .sType = VK_STRUCTURE_TYPE_RESOLVE_IMAGE_INFO_2,
        .srcImage = ms.image,
        .srcImageLayout = SRC,
        .dstImage = d.image,
        .dstImageLayout = DST,
        .regionCount = 1,
        .pRegions = &res,
    };
    vkCmdResolveImage2(cb, &rsi);
    to_layout(cb, d.image, DST, SRC);
    VkCopyImageToBufferInfo2 d2b = i2b;
    d2b.srcImage = d.image;
    d2b.dstBuffer = resolve_out.buf;
    vkCmdCopyImageToBuffer2(cb, &d2b);

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
        uint32_t staged = 0, mirrored = 0, straight = 0, resolved = 0;
        for (uint32_t y = 0; y < H; y++) {
            for (uint32_t x = 0; x < W; x++) {
                staged += staging.p[y * W + x] == pattern(x, y);
                mirrored += chain_out.p[y * W + x] == pattern(W - 1 - x, y);
                straight += chain_out.p[y * W + x] == pattern(x, y);
                const uint8_t *px = (const uint8_t *)&resolve_out.p[y * W + x];
                resolved += px[0] == 51 && px[1] == 102 && px[2] == 153 && px[3] == 255;
            }
        }
        check(staged == TEXELS, "vkCmdCopyBuffer2: the staging buffer holds the pattern");
        check(mirrored == TEXELS,
              "buffer->image->image->blit->buffer: the pattern, mirrored in x");
        check(resolved == TEXELS, "vkCmdResolveImage2: the resolved image holds the samples");
        printf("  staged %u/%u, mirrored %u/%u (%u unmirrored), resolved %u/%u\n", staged,
               TEXELS, mirrored, TEXELS, straight, resolved, TEXELS);
    }

    vkDestroyFence(dev, fence, NULL);
    vkDestroyCommandPool(dev, cmdpool, NULL);
    for (int i = 0; i < 5; i++) {
        unimaged(dev, all[i]);
    }
    unmapped(dev, &source);
    unmapped(dev, &staging);
    unmapped(dev, &chain_out);
    unmapped(dev, &resolve_out);
    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(inst, NULL);

    printf("\n%d check(s) failed\n", failures);
    return failures ? 1 : 0;
}
