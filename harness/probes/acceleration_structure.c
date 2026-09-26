// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva
//
// The `acceleration-structure` group of src/venus/unserved.txt, as a program that names its
// commands.
//
//   vkCreateAccelerationStructureKHR
//   vkDestroyAccelerationStructureKHR
//   vkGetAccelerationStructureBuildSizesKHR
//   vkGetAccelerationStructureDeviceAddressKHR
//   vkGetDeviceAccelerationStructureCompatibilityKHR
//   vkCmdBuildAccelerationStructuresKHR
//   vkCmdBuildAccelerationStructuresIndirectKHR    (when the device allows indirect builds)
//   vkCmdWriteAccelerationStructuresPropertiesKHR
//   vkCmdCopyAccelerationStructureKHR
//   vkCmdCopyAccelerationStructureToMemoryKHR
//   vkCmdCopyMemoryToAccelerationStructureKHR
//
// From VK_KHR_acceleration_structure, which lavapipe advertises and neither KosmicKrisp nor anv on
// Ice Lake does, so the positive control and the venus run are both on lavapipe. No shader reads
// a structure: every consequence here is something the driver writes where the host can read it.
//
//   - The build-size query answers a structure size and a scratch size, both nonzero.
//   - A bottom-level structure over two AABBs is built from `pGeometries`; its device address is
//     nonzero.
//   - Its serialization size, written into a query pool, is more than the serialized header.
//   - Serializing it overwrites the sentinel the buffer was filled with, and writes a header
//     whose serialized size is the one the query wrote. The header's UUIDs are not checked
//     against `driverUUID`, which the spec says they are: lavapipe writes its version string in
//     both, and the compatibility query below is what holds them to the device.
//   - The compatibility query calls that header compatible, and the same header with one byte of
//     its driver UUID flipped incompatible. A zeroed reply reads compatible both times.
//   - Deserializing that into a second structure, and cloning the first into a third, gives two
//     structures whose own serialization sizes equal the first's; and serializing the
//     deserialized one gives back the bytes it was made from.
//   - The same geometry built again from `ppGeometries`, rows of one, serializes to the same
//     size; and where the device allows indirect builds, so does a third build of it with its
//     range read on the device. Lavapipe does not, so there that command is not scored.
//
// Headless on purpose: no surface, no swapchain, no window. It runs over ssh.
//
//   cc -O1 -o acceleration_structure acceleration_structure.c -lvulkan && ./acceleration_structure
//
// Pick the device with MESA_VK_DEVICE_SELECT; the probe takes the first one the loader lists.
//
// Exit 0 = every check passed. Exit 1 = a check failed, and the line above it says which.
// Exit 2 = the device could not be brought up at all (not a verdict on the group).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define SENTINEL 0xa5
#define HEADER (2 * VK_UUID_SIZE + 3 * 8)

static int failures = 0;

static void check(int ok, const char *what) {
    printf("%-64s %s\n", what, ok ? "ok" : "FAILED");
    if (!ok) {
        failures++;
    }
}

static void fatal(const char *what, VkResult r) {
    fprintf(stderr, "cannot test the group: %s returned %d\n", what, (int)r);
    exit(2);
}

static int has_ext(VkExtensionProperties *exts, uint32_t n, const char *name) {
    for (uint32_t i = 0; i < n; i++) {
        if (strcmp(exts[i].extensionName, name) == 0) {
            return 1;
        }
    }
    return 0;
}

static VkPhysicalDevice pd;
static VkDevice dev;
static VkQueue queue;
static VkCommandPool cmd_pool;

static PFN_vkCreateAccelerationStructureKHR create_as;
static PFN_vkDestroyAccelerationStructureKHR destroy_as;
static PFN_vkGetAccelerationStructureBuildSizesKHR build_sizes;
static PFN_vkGetAccelerationStructureDeviceAddressKHR as_address;
static PFN_vkGetDeviceAccelerationStructureCompatibilityKHR compatibility;
static PFN_vkCmdBuildAccelerationStructuresKHR cmd_build;
static PFN_vkCmdBuildAccelerationStructuresIndirectKHR cmd_build_indirect;
static PFN_vkCmdWriteAccelerationStructuresPropertiesKHR cmd_write_properties;
static PFN_vkCmdCopyAccelerationStructureKHR cmd_copy;
static PFN_vkCmdCopyAccelerationStructureToMemoryKHR cmd_copy_to_memory;
static PFN_vkCmdCopyMemoryToAccelerationStructureKHR cmd_copy_from_memory;

static PFN_vkVoidFunction need(const char *name) {
    PFN_vkVoidFunction f = vkGetDeviceProcAddr(dev, name);
    if (!f) {
        fprintf(stderr, "cannot test the group: %s did not resolve\n", name);
        exit(2);
    }
    return f;
}

// A host-visible buffer the device can also address, mapped for the whole run.
struct buffer {
    VkBuffer buf;
    VkDeviceMemory mem;
    VkDeviceAddress addr;
    uint8_t *p;
};

static struct buffer buffer(VkDeviceSize size, VkBufferUsageFlags usage) {
    struct buffer b;
    VkResult r;
    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
    };
    if ((r = vkCreateBuffer(dev, &bci, NULL, &b.buf)) != VK_SUCCESS) {
        fatal("vkCreateBuffer", r);
    }
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev, b.buf, &mr);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    VkMemoryPropertyFlags want =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((mr.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) {
            type = i;
            break;
        }
    }
    if (type == UINT32_MAX) {
        fatal("no host-visible coherent memory type", VK_ERROR_INITIALIZATION_FAILED);
    }
    VkMemoryAllocateFlagsInfo flags = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
    };
    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &flags,
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
    memset(b.p, SENTINEL, size);
    VkBufferDeviceAddressInfo bai = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .buffer = b.buf,
    };
    b.addr = vkGetBufferDeviceAddress(dev, &bai);
    return b;
}

static void drop(struct buffer *b) {
    vkUnmapMemory(dev, b->mem);
    vkDestroyBuffer(dev, b->buf, NULL);
    vkFreeMemory(dev, b->mem, NULL);
}

static VkAccelerationStructureKHR structure(struct buffer *storage, VkDeviceSize size) {
    VkAccelerationStructureCreateInfoKHR ci = {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
        .buffer = storage->buf,
        .size = size,
        .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
    };
    VkAccelerationStructureKHR as;
    VkResult r = create_as(dev, &ci, NULL, &as);
    if (r != VK_SUCCESS) {
        fatal("vkCreateAccelerationStructureKHR", r);
    }
    return as;
}

static VkCommandBuffer begin(void) {
    VkCommandBufferAllocateInfo ai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cb;
    vkAllocateCommandBuffers(dev, &ai, &cb);
    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(cb, &bi);
    return cb;
}

// Everything before it finished and visible to everything after it, the host included.
static void barrier(VkCommandBuffer cb) {
    VkMemoryBarrier mb = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT |
                         VK_ACCESS_HOST_READ_BIT,
    };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
                         &mb, 0, NULL, 0, NULL);
}

static void submit(VkCommandBuffer cb) {
    barrier(cb);
    vkEndCommandBuffer(cb);
    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cb,
    };
    VkResult r = vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
    if (r != VK_SUCCESS) {
        fatal("vkQueueSubmit", r);
    }
    if ((r = vkQueueWaitIdle(queue)) != VK_SUCCESS) {
        fatal("vkQueueWaitIdle", r);
    }
    vkFreeCommandBuffers(dev, cmd_pool, 1, &cb);
}

static VkQueryPool query_pool;

// The serialization size of each of `n` structures, written on the device and read back.
static void serialization_sizes(VkAccelerationStructureKHR *as, uint32_t n, uint64_t *out) {
    VkCommandBuffer cb = begin();
    vkCmdResetQueryPool(cb, query_pool, 0, n);
    cmd_write_properties(cb, n, as, VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR,
                         query_pool, 0);
    submit(cb);
    VkResult r = vkGetQueryPoolResults(dev, query_pool, 0, n, n * sizeof *out, out, sizeof *out,
                                       VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (r != VK_SUCCESS) {
        fatal("vkGetQueryPoolResults", r);
    }
}

static void serialize(VkAccelerationStructureKHR as, struct buffer *into) {
    VkCommandBuffer cb = begin();
    VkCopyAccelerationStructureToMemoryInfoKHR ci = {
        .sType = VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_TO_MEMORY_INFO_KHR,
        .src = as,
        .dst.deviceAddress = into->addr,
        .mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_SERIALIZE_KHR,
    };
    cmd_copy_to_memory(cb, &ci);
    submit(cb);
}

static VkAccelerationStructureCompatibilityKHR compatible(const uint8_t *header) {
    VkAccelerationStructureVersionInfoKHR vi = {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_VERSION_INFO_KHR,
        .pVersionData = header,
    };
    VkAccelerationStructureCompatibilityKHR c = (VkAccelerationStructureCompatibilityKHR)0x7777;
    compatibility(dev, &vi, &c);
    return c;
}

static uint64_t u64_at(const uint8_t *p) {
    uint64_t v;
    memcpy(&v, p, sizeof v);
    return v;
}

int main(void) {
    VkResult r;

    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "venus-acceleration-structure-probe",
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
    pd = pds[0];

    VkPhysicalDeviceAccelerationStructurePropertiesKHR asp = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR,
    };
    VkPhysicalDeviceProperties2 props = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &asp,
    };
    vkGetPhysicalDeviceProperties2(pd, &props);
    printf("device: %s (Vulkan %u.%u)\n\n", props.properties.deviceName,
           VK_API_VERSION_MAJOR(props.properties.apiVersion),
           VK_API_VERSION_MINOR(props.properties.apiVersion));

    uint32_t en = 0;
    vkEnumerateDeviceExtensionProperties(pd, NULL, &en, NULL);
    VkExtensionProperties *exts = calloc(en, sizeof *exts);
    vkEnumerateDeviceExtensionProperties(pd, NULL, &en, exts);
    if (!has_ext(exts, en, "VK_KHR_acceleration_structure")) {
        fatal("VK_KHR_acceleration_structure", VK_ERROR_EXTENSION_NOT_PRESENT);
    }

    VkPhysicalDeviceAccelerationStructureFeaturesKHR asf = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
    };
    VkPhysicalDeviceVulkan12Features v12 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &asf,
    };
    VkPhysicalDeviceFeatures2 feats = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &v12,
    };
    vkGetPhysicalDeviceFeatures2(pd, &feats);
    if (!asf.accelerationStructure || !v12.bufferDeviceAddress) {
        fatal("accelerationStructure and bufferDeviceAddress", VK_ERROR_FEATURE_NOT_PRESENT);
    }
    int indirect = asf.accelerationStructureIndirectBuild;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR asf_on = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
        .accelerationStructure = VK_TRUE,
        .accelerationStructureIndirectBuild = indirect ? VK_TRUE : VK_FALSE,
    };
    VkPhysicalDeviceVulkan12Features v12_on = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &asf_on,
        .bufferDeviceAddress = VK_TRUE,
    };
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = 0,
        .queueCount = 1,
        .pQueuePriorities = &prio,
    };
    const char *want[] = {"VK_KHR_acceleration_structure", "VK_KHR_deferred_host_operations"};
    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &v12_on,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qci,
        .enabledExtensionCount = 2,
        .ppEnabledExtensionNames = want,
    };
    if ((r = vkCreateDevice(pd, &dci, NULL, &dev)) != VK_SUCCESS) {
        fatal("vkCreateDevice", r);
    }
    vkGetDeviceQueue(dev, 0, 0, &queue);
    VkCommandPoolCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = 0,
    };
    vkCreateCommandPool(dev, &cpci, NULL, &cmd_pool);

    create_as = (PFN_vkCreateAccelerationStructureKHR)need("vkCreateAccelerationStructureKHR");
    destroy_as = (PFN_vkDestroyAccelerationStructureKHR)need("vkDestroyAccelerationStructureKHR");
    build_sizes =
        (PFN_vkGetAccelerationStructureBuildSizesKHR)need("vkGetAccelerationStructureBuildSizesKHR");
    as_address = (PFN_vkGetAccelerationStructureDeviceAddressKHR)need(
        "vkGetAccelerationStructureDeviceAddressKHR");
    compatibility = (PFN_vkGetDeviceAccelerationStructureCompatibilityKHR)need(
        "vkGetDeviceAccelerationStructureCompatibilityKHR");
    cmd_build = (PFN_vkCmdBuildAccelerationStructuresKHR)need("vkCmdBuildAccelerationStructuresKHR");
    cmd_build_indirect = (PFN_vkCmdBuildAccelerationStructuresIndirectKHR)need(
        "vkCmdBuildAccelerationStructuresIndirectKHR");
    cmd_write_properties = (PFN_vkCmdWriteAccelerationStructuresPropertiesKHR)need(
        "vkCmdWriteAccelerationStructuresPropertiesKHR");
    cmd_copy = (PFN_vkCmdCopyAccelerationStructureKHR)need("vkCmdCopyAccelerationStructureKHR");
    cmd_copy_to_memory = (PFN_vkCmdCopyAccelerationStructureToMemoryKHR)need(
        "vkCmdCopyAccelerationStructureToMemoryKHR");
    cmd_copy_from_memory = (PFN_vkCmdCopyMemoryToAccelerationStructureKHR)need(
        "vkCmdCopyMemoryToAccelerationStructureKHR");

    VkQueryPoolCreateInfo qpci = {
        .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR,
        .queryCount = 4,
    };
    if ((r = vkCreateQueryPool(dev, &qpci, NULL, &query_pool)) != VK_SUCCESS) {
        fatal("vkCreateQueryPool", r);
    }

    // Two boxes, side by side.
    VkAabbPositionsKHR boxes[2] = {
        {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f},
        {2.0f, 0.0f, 0.0f, 3.0f, 1.0f, 1.0f},
    };
    struct buffer aabbs =
        buffer(sizeof boxes, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
    memcpy(aabbs.p, boxes, sizeof boxes);

    VkAccelerationStructureGeometryKHR geometry = {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .geometryType = VK_GEOMETRY_TYPE_AABBS_KHR,
        .geometry.aabbs =
            {
                .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR,
                .data.deviceAddress = aabbs.addr,
                .stride = sizeof(VkAabbPositionsKHR),
            },
        .flags = VK_GEOMETRY_OPAQUE_BIT_KHR,
    };
    VkAccelerationStructureBuildGeometryInfoKHR info = {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
        .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
        .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
        .geometryCount = 1,
        .pGeometries = &geometry,
    };
    uint32_t primitives = 2;
    VkAccelerationStructureBuildSizesInfoKHR sizes = {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
    };
    build_sizes(dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &info, &primitives, &sizes);
    check(sizes.accelerationStructureSize > 0 && sizes.buildScratchSize > 0,
          "the build-size query answers a structure size and a scratch size");
    if (sizes.accelerationStructureSize == 0) {
        printf("\n%d check(s) failed\n", failures);
        return 1;
    }
    printf("  structure %llu bytes, scratch %llu\n",
           (unsigned long long)sizes.accelerationStructureSize,
           (unsigned long long)sizes.buildScratchSize);

    VkDeviceSize as_size = sizes.accelerationStructureSize;
    VkDeviceSize align = asp.minAccelerationStructureScratchOffsetAlignment;
    VkBufferUsageFlags storage_usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;
    struct buffer storage[5];
    for (int i = 0; i < 5; i++) {
        storage[i] = buffer(as_size, storage_usage);
    }
    struct buffer scratch = buffer(sizes.buildScratchSize + align, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VkDeviceAddress scratch_addr = (scratch.addr + align - 1) / align * align;

    VkAccelerationStructureKHR built = structure(&storage[0], as_size);
    VkAccelerationStructureDeviceAddressInfoKHR ai = {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
        .accelerationStructure = built,
    };
    check(as_address(dev, &ai) != 0, "the structure has a nonzero device address");

    info.dstAccelerationStructure = built;
    info.scratchData.deviceAddress = scratch_addr;
    VkAccelerationStructureBuildRangeInfoKHR range = {.primitiveCount = primitives};
    const VkAccelerationStructureBuildRangeInfoKHR *ranges = &range;
    VkCommandBuffer cb = begin();
    cmd_build(cb, 1, &info, &ranges);
    submit(cb);

    uint64_t size[3] = {0};
    serialization_sizes(&built, 1, size);
    check(size[0] > HEADER, "its serialization size is more than the header");
    printf("  serialization %llu bytes\n", (unsigned long long)size[0]);
    if (size[0] <= HEADER) {
        printf("\n%d check(s) failed\n", failures);
        return 1;
    }

    struct buffer serial[2] = {
        buffer(size[0], storage_usage | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR),
        buffer(size[0], storage_usage),
    };
    serialize(built, &serial[0]);
    const uint8_t *header = serial[0].p;
    int overwritten = 0;
    for (unsigned i = 0; i < 2 * VK_UUID_SIZE; i++) {
        overwritten |= header[i] != SENTINEL;
    }
    check(overwritten, "serializing overwrites the sentinel with a header");
    check(u64_at(header + 2 * VK_UUID_SIZE) == size[0],
          "and the serialized size the query wrote");

    check(compatible(header) == VK_ACCELERATION_STRUCTURE_COMPATIBILITY_COMPATIBLE_KHR,
          "that header is compatible with this device");
    uint8_t flipped[2 * VK_UUID_SIZE];
    memcpy(flipped, header, sizeof flipped);
    flipped[0] ^= 0xff;
    check(compatible(flipped) == VK_ACCELERATION_STRUCTURE_COMPATIBILITY_INCOMPATIBLE_KHR,
          "and one with its driver UUID flipped is not");

    VkAccelerationStructureKHR restored = structure(&storage[1], as_size);
    VkAccelerationStructureKHR cloned = structure(&storage[2], as_size);
    cb = begin();
    VkCopyMemoryToAccelerationStructureInfoKHR from = {
        .sType = VK_STRUCTURE_TYPE_COPY_MEMORY_TO_ACCELERATION_STRUCTURE_INFO_KHR,
        .src.deviceAddress = serial[0].addr,
        .dst = restored,
        .mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_DESERIALIZE_KHR,
    };
    cmd_copy_from_memory(cb, &from);
    VkCopyAccelerationStructureInfoKHR clone = {
        .sType = VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR,
        .src = built,
        .dst = cloned,
        .mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_CLONE_KHR,
    };
    cmd_copy(cb, &clone);
    submit(cb);

    VkAccelerationStructureKHR three[3] = {built, restored, cloned};
    serialization_sizes(three, 3, size);
    check(size[1] == size[0], "the deserialized structure serializes to the same size");
    check(size[2] == size[0], "and so does the clone");
    serialize(restored, &serial[1]);
    check(memcmp(serial[0].p, serial[1].p, size[0]) == 0,
          "serializing the deserialized one gives back the bytes it was made of");

    VkAccelerationStructureKHR by_row_as = structure(&storage[3], as_size);
    const VkAccelerationStructureGeometryKHR *rows = &geometry;
    VkAccelerationStructureBuildGeometryInfoKHR by_row = info;
    by_row.pGeometries = NULL;
    by_row.ppGeometries = &rows;
    by_row.dstAccelerationStructure = by_row_as;
    cb = begin();
    cmd_build(cb, 1, &by_row, &ranges);
    submit(cb);
    VkAccelerationStructureKHR pair[2] = {built, by_row_as};
    serialization_sizes(pair, 2, size);
    check(size[1] == size[0], "a build of the same geometry from ppGeometries serializes the same");

    if (indirect) {
        VkAccelerationStructureKHR again = structure(&storage[4], as_size);
        struct buffer record = buffer(sizeof range, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
        memcpy(record.p, &range, sizeof range);
        by_row.dstAccelerationStructure = again;
        uint32_t stride = sizeof range;
        const uint32_t *bounds = &primitives;
        cb = begin();
        cmd_build_indirect(cb, 1, &by_row, &record.addr, &stride, &bounds);
        submit(cb);
        pair[1] = again;
        serialization_sizes(pair, 2, size);
        check(size[1] == size[0], "and so does an indirect build of it");
        destroy_as(dev, again, NULL);
        drop(&record);
    } else {
        printf("%-64s %s\n", "an indirect build", "skipped: accelerationStructureIndirectBuild");
    }

    destroy_as(dev, by_row_as, NULL);
    destroy_as(dev, cloned, NULL);
    destroy_as(dev, restored, NULL);
    destroy_as(dev, built, NULL);
    for (int i = 0; i < 2; i++) {
        drop(&serial[i]);
    }
    for (int i = 0; i < 5; i++) {
        drop(&storage[i]);
    }
    drop(&scratch);
    drop(&aabbs);
    vkDestroyQueryPool(dev, query_pool, NULL);
    vkDestroyCommandPool(dev, cmd_pool, NULL);
    vkDestroyDevice(dev, NULL);
    free(exts);
    free(pds);
    vkDestroyInstance(inst, NULL);

    printf("\n%d check(s) failed\n", failures);
    return failures ? 1 : 0;
}
