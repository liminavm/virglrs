// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva
//
// The `image-copy-core` group of src/venus/unserved.txt, as a program that names each command.
//
//   vkCmdClearDepthStencilImage  vkCmdResolveImage  vkCmdUpdateBuffer
//
// All three are core Vulkan 1.0 transfer commands, and each writes something a copy can read
// back:
//
//   - a depth-stencil image is cleared to depth 0.25 and stencil 0x5a, and both aspects are
//     copied out;
//   - a 4x multisampled RGBA8 image is cleared to one colour, so every sample holds it and the
//     resolve's answer is exact, then resolved into a single-sampled image that is copied out;
//   - sixteen bytes are written at offset 8 of a zeroed buffer, which is then read directly.
//
// A dropped clear leaves the depth and stencil at whatever the image held; a dropped resolve
// leaves the single-sampled image at the other colour it was cleared to first; a dropped update
// leaves the buffer zero, and one with its offset and size transposed writes the wrong bytes.
//
// Headless on purpose: no surface, no swapchain, no window. It runs over ssh.
//
//   cc -O1 -o image_copy_core image_copy_core.c -lvulkan && ./image_copy_core
//
// Exit 0 = every check passed. Exit 1 = a check failed, and the line above it says which.
// Exit 2 = the device could not be brought up at all (not a verdict on the group).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define W 8u
#define H 8u
#define DEPTH 0.25f
#define STENCIL 0x5au
#define UPDATE_AT 8u
#define UPDATE_LEN 16u

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

struct image {
    VkImage image;
    VkDeviceMemory mem;
};

static struct image image(VkPhysicalDevice pd, VkDevice dev, VkFormat format,
                          VkSampleCountFlagBits samples, VkImageUsageFlags usage) {
    struct image i;
    VkImageCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .extent = {W, H, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = samples,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = usage,
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

static void barrier(VkCommandBuffer cb, VkImage image, VkImageAspectFlags aspects,
                    VkImageLayout from, VkImageLayout to, VkAccessFlags src, VkAccessFlags dst) {
    VkImageMemoryBarrier b = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = src,
        .dstAccessMask = dst,
        .oldLayout = from,
        .newLayout = to,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = {aspects, 0, 1, 0, 1},
    };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                         NULL, 0, NULL, 1, &b);
}

// The first depth-stencil format this device can clear and copy out of.
static VkFormat depth_stencil_format(VkPhysicalDevice pd) {
    const VkFormat candidates[] = {VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT};
    const VkFormatFeatureFlags want = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                      VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
                                      VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    for (size_t i = 0; i < sizeof candidates / sizeof candidates[0]; i++) {
        VkFormatProperties fp;
        vkGetPhysicalDeviceFormatProperties(pd, candidates[i], &fp);
        if ((fp.optimalTilingFeatures & want) == want) {
            return candidates[i];
        }
    }
    return VK_FORMAT_UNDEFINED;
}

int main(void) {
    VkResult r;

    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "venus-image-copy-core-probe",
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
    printf("device: %s (Vulkan %u.%u)\n", props.deviceName,
           VK_API_VERSION_MAJOR(props.apiVersion), VK_API_VERSION_MINOR(props.apiVersion));

    VkFormat ds = depth_stencil_format(pd);
    if (ds == VK_FORMAT_UNDEFINED) {
        fatal("no clearable, copyable depth-stencil format", VK_ERROR_FORMAT_NOT_SUPPORTED);
    }
    if (!(props.limits.framebufferColorSampleCounts & VK_SAMPLE_COUNT_4_BIT)) {
        fatal("no 4x colour multisampling", VK_ERROR_FEATURE_NOT_PRESENT);
    }
    printf("depth-stencil format: %s\n\n",
           ds == VK_FORMAT_D32_SFLOAT_S8_UINT ? "D32_SFLOAT_S8_UINT" : "D24_UNORM_S8_UINT");

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

    const VkImageUsageFlags xfer = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    struct image depth =
        image(pd, dev, ds, VK_SAMPLE_COUNT_1_BIT,
              xfer | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
    struct image msaa = image(pd, dev, VK_FORMAT_R8G8B8A8_UNORM, VK_SAMPLE_COUNT_4_BIT,
                              xfer | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    struct image single = image(pd, dev, VK_FORMAT_R8G8B8A8_UNORM, VK_SAMPLE_COUNT_1_BIT,
                                xfer | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);

    // Depth as four bytes a texel (D24's are unpacked into the low 24 bits of four), then the
    // stencil as one byte a texel, then the resolved colour, four bytes a texel.
    const VkDeviceSize DEPTH_AT = 0, STENCIL_AT = W * H * 4, COLOR_AT = STENCIL_AT + W * H;
    struct mapped out = mapped(pd, dev, COLOR_AT + W * H * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    memset(out.p, 0xee, COLOR_AT + W * H * 4);
    struct mapped updated = mapped(pd, dev, 64, VK_BUFFER_USAGE_TRANSFER_DST_BIT);

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

    const VkImageAspectFlags DS = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    const VkImageLayout DST = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    const VkImageLayout SRC = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    const VkAccessFlags WRITE = VK_ACCESS_TRANSFER_WRITE_BIT, READ = VK_ACCESS_TRANSFER_READ_BIT;

    // vkCmdClearDepthStencilImage, then both aspects out.
    barrier(cb, depth.image, DS, VK_IMAGE_LAYOUT_UNDEFINED, DST, 0, WRITE);
    VkClearDepthStencilValue value = {DEPTH, STENCIL};
    VkImageSubresourceRange ds_range = {DS, 0, 1, 0, 1};
    vkCmdClearDepthStencilImage(cb, depth.image, DST, &value, 1, &ds_range);
    barrier(cb, depth.image, DS, DST, SRC, WRITE, READ);
    VkBufferImageCopy aspects[2] = {
        {.bufferOffset = DEPTH_AT,
         .imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1},
         .imageExtent = {W, H, 1}},
        {.bufferOffset = STENCIL_AT,
         .imageSubresource = {VK_IMAGE_ASPECT_STENCIL_BIT, 0, 0, 1},
         .imageExtent = {W, H, 1}},
    };
    vkCmdCopyImageToBuffer(cb, depth.image, SRC, out.buf, 2, aspects);

    // vkCmdResolveImage: the destination is cleared to another colour first, so a resolve that
    // never ran is told apart from one that wrote the wrong thing.
    const VkImageAspectFlags C = VK_IMAGE_ASPECT_COLOR_BIT;
    VkImageSubresourceRange c_range = {C, 0, 1, 0, 1};
    VkClearColorValue sampled = {.float32 = {0.2f, 0.4f, 0.6f, 1.0f}};
    VkClearColorValue stale = {.float32 = {1.0f, 0.0f, 1.0f, 1.0f}};
    barrier(cb, msaa.image, C, VK_IMAGE_LAYOUT_UNDEFINED, DST, 0, WRITE);
    vkCmdClearColorImage(cb, msaa.image, DST, &sampled, 1, &c_range);
    barrier(cb, msaa.image, C, DST, SRC, WRITE, READ);
    barrier(cb, single.image, C, VK_IMAGE_LAYOUT_UNDEFINED, DST, 0, WRITE);
    vkCmdClearColorImage(cb, single.image, DST, &stale, 1, &c_range);
    barrier(cb, single.image, C, DST, DST, WRITE, WRITE);
    VkImageResolve resolve = {
        .srcSubresource = {C, 0, 0, 1},
        .dstSubresource = {C, 0, 0, 1},
        .extent = {W, H, 1},
    };
    vkCmdResolveImage(cb, msaa.image, SRC, single.image, DST, 1, &resolve);
    barrier(cb, single.image, C, DST, SRC, WRITE, READ);
    VkBufferImageCopy color = {
        .bufferOffset = COLOR_AT,
        .imageSubresource = {C, 0, 0, 1},
        .imageExtent = {W, H, 1},
    };
    vkCmdCopyImageToBuffer(cb, single.image, SRC, out.buf, 1, &color);

    // vkCmdUpdateBuffer, sixteen bytes into the middle of a zeroed buffer.
    uint8_t bytes[UPDATE_LEN];
    for (uint32_t i = 0; i < UPDATE_LEN; i++) {
        bytes[i] = (uint8_t)(0xa0 + i);
    }
    vkCmdUpdateBuffer(cb, updated.buf, UPDATE_AT, UPDATE_LEN, bytes);

    VkMemoryBarrier host = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = WRITE,
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
        uint32_t depth_right = 0, stencil_right = 0, color_right = 0, color_stale = 0;
        for (uint32_t i = 0; i < W * H; i++) {
            uint32_t raw;
            memcpy(&raw, out.p + DEPTH_AT + i * 4, 4);
            if (ds == VK_FORMAT_D32_SFLOAT_S8_UINT) {
                float d;
                memcpy(&d, &raw, 4);
                depth_right += d == DEPTH;
            } else {
                // 0.25 in 24-bit unorm, with one step either way for the conversion's rounding.
                uint32_t want = (uint32_t)(DEPTH * 0xffffff + 0.5f), got = raw & 0xffffff;
                depth_right += got + 1 >= want && got <= want + 1;
            }
            stencil_right += out.p[STENCIL_AT + i] == STENCIL;
            const uint8_t *px = out.p + COLOR_AT + i * 4;
            color_right += px[0] == 51 && px[1] == 102 && px[2] == 153 && px[3] == 255;
            color_stale += px[0] == 255 && px[1] == 0 && px[2] == 255;
        }
        check(depth_right == W * H, "vkCmdClearDepthStencilImage: every depth is the guest's");
        check(stencil_right == W * H, "vkCmdClearDepthStencilImage: every stencil is the guest's");
        check(color_right == W * H, "vkCmdResolveImage: the resolved image holds the samples");
        printf("  depth %u/%u, stencil %u/%u, resolved %u/%u (%u still the stale colour)\n",
               depth_right, W * H, stencil_right, W * H, color_right, W * H, color_stale);

        uint32_t right = 0, untouched = 0;
        for (uint32_t i = 0; i < 64; i++) {
            int inside = i >= UPDATE_AT && i < UPDATE_AT + UPDATE_LEN;
            if (inside) {
                right += updated.p[i] == bytes[i - UPDATE_AT];
            } else {
                untouched += updated.p[i] == 0;
            }
        }
        check(right == UPDATE_LEN, "vkCmdUpdateBuffer: all sixteen bytes, at offset 8");
        check(untouched == 64 - UPDATE_LEN, "vkCmdUpdateBuffer: nothing outside them");
        printf("  update %u/%u in place, %u/%u around it untouched\n", right, UPDATE_LEN,
               untouched, 64 - UPDATE_LEN);
    }

    vkDestroyFence(dev, fence, NULL);
    vkDestroyCommandPool(dev, cmdpool, NULL);
    unimaged(dev, &depth);
    unimaged(dev, &msaa);
    unimaged(dev, &single);
    unmapped(dev, &out);
    unmapped(dev, &updated);
    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(inst, NULL);

    printf("\n%d check(s) failed\n", failures);
    return failures ? 1 : 0;
}
