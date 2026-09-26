// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva
//
// The `mesh-shader` group of src/venus/unserved.txt, as a program that names its commands.
//
//   vkCmdDrawMeshTasksEXT
//   vkCmdDrawMeshTasksIndirectEXT
//   vkCmdDrawMeshTasksIndirectCountEXT
//
// From VK_EXT_mesh_shader, which lavapipe advertises and neither KosmicKrisp nor anv on Ice Lake
// does, so the positive control and the venus run are both on lavapipe.
//
// A mesh shader with no task shader draws, for each workgroup, a quad over one pixel column of a
// 4x4 target: workgroup x covers column x. So the columns lit after a draw count the workgroups the
// draw ran, and each command is given a different count. Each draws into its own fresh
// R8G8B8A8_UNORM image cleared to black, read back:
//
//   - vkCmdDrawMeshTasksEXT(2, 1, 1): columns 0 and 1 white, 2 and 3 black.
//   - vkCmdDrawMeshTasksIndirectEXT, one record of (3, 1, 1): columns 0 to 2.
//   - vkCmdDrawMeshTasksIndirectCountEXT over two records, (1, 1, 1) then (4, 1, 1), with a count
//     buffer holding 1 and a maximum of 2: column 0 alone. A count that was not read draws both
//     records, and all four columns.
//
// The shaders, compiled with `glslangValidator -V -x`, --target-env vulkan1.3 for the mesh
// shader and vulkan1.1 for the fragment shader:
//
//   // column.mesh
//   #version 450
//   #extension GL_EXT_mesh_shader : require
//   layout(local_size_x = 1) in;
//   layout(triangles, max_vertices = 4, max_primitives = 2) out;
//   void main() {
//       float x0 = float(gl_WorkGroupID.x) * 0.5 - 1.0;
//       float x1 = x0 + 0.5;
//       SetMeshOutputsEXT(4, 2);
//       gl_MeshVerticesEXT[0].gl_Position = vec4(x0, -1.0, 0.0, 1.0);
//       gl_MeshVerticesEXT[1].gl_Position = vec4(x1, -1.0, 0.0, 1.0);
//       gl_MeshVerticesEXT[2].gl_Position = vec4(x0, 1.0, 0.0, 1.0);
//       gl_MeshVerticesEXT[3].gl_Position = vec4(x1, 1.0, 0.0, 1.0);
//       gl_PrimitiveTriangleIndicesEXT[0] = uvec3(0, 1, 2);
//       gl_PrimitiveTriangleIndicesEXT[1] = uvec3(1, 3, 2);
//   }
//
//   // white.frag
//   #version 450
//   layout(location = 0) out vec4 color;
//   void main() {
//       color = vec4(1.0);
//   }
//
// Headless on purpose: no surface, no swapchain, no window. It runs over ssh.
//
//   cc -O1 -o mesh_shader mesh_shader.c -lvulkan && ./mesh_shader
//
// Pick the device with MESA_VK_DEVICE_SELECT; the probe takes the first one the loader lists.
//
// Exit 0 = every check passed. Exit 1 = a check failed, and the line above it says which.
// Exit 2 = the device could not be brought up at all (not a verdict on the group).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define W 4u
#define H 4u
#define RENDERS 3u
#define SLOT (W * H * 4u)
#define FORMAT VK_FORMAT_R8G8B8A8_UNORM

static const uint32_t COLUMN_MESH[] = {
    0x07230203,0x00010600,0x0008000b,0x00000040,0x00000000,0x00020011,0x000014a3,0x0006000a,
    0x5f565053,0x5f545845,0x6873656d,0x6168735f,0x00726564,0x0006000b,0x00000001,0x4c534c47,
    0x6474732e,0x3035342e,0x00000000,0x0003000e,0x00000000,0x00000001,0x0008000f,0x000014f5,
    0x00000004,0x6e69616d,0x00000000,0x0000000d,0x00000021,0x00000038,0x0006014b,0x00000004,
    0x00000026,0x00000007,0x00000007,0x00000007,0x00040010,0x00000004,0x0000001a,0x00000004,
    0x00040010,0x00000004,0x00001496,0x00000002,0x00030010,0x00000004,0x000014b2,0x00030003,
    0x00000002,0x000001c2,0x00060004,0x455f4c47,0x6d5f5458,0x5f687365,0x64616873,0x00007265,
    0x00040005,0x00000004,0x6e69616d,0x00000000,0x00030005,0x0000000a,0x00003078,0x00060005,
    0x0000000d,0x575f6c67,0x476b726f,0x70756f72,0x00004449,0x00030005,0x00000017,0x00003178,
    0x00070005,0x0000001e,0x4d5f6c67,0x50687365,0x65567265,0x78657472,0x00545845,0x00060006,
    0x0000001e,0x00000000,0x505f6c67,0x7469736f,0x006e6f69,0x00070006,0x0000001e,0x00000001,
    0x505f6c67,0x746e696f,0x657a6953,0x00000000,0x00070006,0x0000001e,0x00000002,0x435f6c67,
    0x4470696c,0x61747369,0x0065636e,0x00070006,0x0000001e,0x00000003,0x435f6c67,0x446c6c75,
    0x61747369,0x0065636e,0x00070005,0x00000021,0x4d5f6c67,0x56687365,0x69747265,0x45736563,
    0x00005458,0x000a0005,0x00000038,0x505f6c67,0x696d6972,0x65766974,0x61697254,0x656c676e,
    0x69646e49,0x45736563,0x00005458,0x00040047,0x0000000d,0x0000000b,0x0000001a,0x00030047,
    0x0000001e,0x00000002,0x00050048,0x0000001e,0x00000000,0x0000000b,0x00000000,0x00050048,
    0x0000001e,0x00000001,0x0000000b,0x00000001,0x00050048,0x0000001e,0x00000002,0x0000000b,
    0x00000003,0x00050048,0x0000001e,0x00000003,0x0000000b,0x00000004,0x00040047,0x00000038,
    0x0000000b,0x000014b0,0x00020013,0x00000002,0x00030021,0x00000003,0x00000002,0x00040015,
    0x00000006,0x00000020,0x00000000,0x0004002b,0x00000006,0x00000007,0x00000001,0x00030016,
    0x00000008,0x00000020,0x00040020,0x00000009,0x00000007,0x00000008,0x00040017,0x0000000b,
    0x00000006,0x00000003,0x00040020,0x0000000c,0x00000001,0x0000000b,0x0004003b,0x0000000c,
    0x0000000d,0x00000001,0x0004002b,0x00000006,0x0000000e,0x00000000,0x00040020,0x0000000f,
    0x00000001,0x00000006,0x0004002b,0x00000008,0x00000013,0x3f000000,0x0004002b,0x00000008,
    0x00000015,0x3f800000,0x0004002b,0x00000006,0x0000001a,0x00000004,0x0004002b,0x00000006,
    0x0000001b,0x00000002,0x00040017,0x0000001c,0x00000008,0x00000004,0x0004001c,0x0000001d,
    0x00000008,0x00000007,0x0006001e,0x0000001e,0x0000001c,0x00000008,0x0000001d,0x0000001d,
    0x0004001c,0x0000001f,0x0000001e,0x0000001a,0x00040020,0x00000020,0x00000003,0x0000001f,
    0x0004003b,0x00000020,0x00000021,0x00000003,0x00040015,0x00000022,0x00000020,0x00000001,
    0x0004002b,0x00000022,0x00000023,0x00000000,0x0004002b,0x00000008,0x00000025,0xbf800000,
    0x0004002b,0x00000008,0x00000026,0x00000000,0x00040020,0x00000028,0x00000003,0x0000001c,
    0x0004002b,0x00000022,0x0000002a,0x00000001,0x0004002b,0x00000022,0x0000002e,0x00000002,
    0x0004002b,0x00000022,0x00000032,0x00000003,0x0004001c,0x00000036,0x0000000b,0x0000001b,
    0x00040020,0x00000037,0x00000003,0x00000036,0x0004003b,0x00000037,0x00000038,0x00000003,
    0x0006002c,0x0000000b,0x00000039,0x0000000e,0x00000007,0x0000001b,0x00040020,0x0000003a,
    0x00000003,0x0000000b,0x0004002b,0x00000006,0x0000003c,0x00000003,0x0006002c,0x0000000b,
    0x0000003d,0x00000007,0x0000003c,0x0000001b,0x0006002c,0x0000000b,0x0000003f,0x00000007,
    0x00000007,0x00000007,0x00050036,0x00000002,0x00000004,0x00000000,0x00000003,0x000200f8,
    0x00000005,0x0004003b,0x00000009,0x0000000a,0x00000007,0x0004003b,0x00000009,0x00000017,
    0x00000007,0x00050041,0x0000000f,0x00000010,0x0000000d,0x0000000e,0x0004003d,0x00000006,
    0x00000011,0x00000010,0x00040070,0x00000008,0x00000012,0x00000011,0x00050085,0x00000008,
    0x00000014,0x00000012,0x00000013,0x00050083,0x00000008,0x00000016,0x00000014,0x00000015,
    0x0003003e,0x0000000a,0x00000016,0x0004003d,0x00000008,0x00000018,0x0000000a,0x00050081,
    0x00000008,0x00000019,0x00000018,0x00000013,0x0003003e,0x00000017,0x00000019,0x000314af,
    0x0000001a,0x0000001b,0x0004003d,0x00000008,0x00000024,0x0000000a,0x00070050,0x0000001c,
    0x00000027,0x00000024,0x00000025,0x00000026,0x00000015,0x00060041,0x00000028,0x00000029,
    0x00000021,0x00000023,0x00000023,0x0003003e,0x00000029,0x00000027,0x0004003d,0x00000008,
    0x0000002b,0x00000017,0x00070050,0x0000001c,0x0000002c,0x0000002b,0x00000025,0x00000026,
    0x00000015,0x00060041,0x00000028,0x0000002d,0x00000021,0x0000002a,0x00000023,0x0003003e,
    0x0000002d,0x0000002c,0x0004003d,0x00000008,0x0000002f,0x0000000a,0x00070050,0x0000001c,
    0x00000030,0x0000002f,0x00000015,0x00000026,0x00000015,0x00060041,0x00000028,0x00000031,
    0x00000021,0x0000002e,0x00000023,0x0003003e,0x00000031,0x00000030,0x0004003d,0x00000008,
    0x00000033,0x00000017,0x00070050,0x0000001c,0x00000034,0x00000033,0x00000015,0x00000026,
    0x00000015,0x00060041,0x00000028,0x00000035,0x00000021,0x00000032,0x00000023,0x0003003e,
    0x00000035,0x00000034,0x00050041,0x0000003a,0x0000003b,0x00000038,0x00000023,0x0003003e,
    0x0000003b,0x00000039,0x00050041,0x0000003a,0x0000003e,0x00000038,0x0000002a,0x0003003e,
    0x0000003e,0x0000003d,0x000100fd,0x00010038
};

static const uint32_t WHITE_FRAG[] = {
    0x07230203,0x00010300,0x0008000b,0x0000000c,0x00000000,0x00020011,0x00000001,0x0006000b,
    0x00000001,0x4c534c47,0x6474732e,0x3035342e,0x00000000,0x0003000e,0x00000000,0x00000001,
    0x0006000f,0x00000004,0x00000004,0x6e69616d,0x00000000,0x00000009,0x00030010,0x00000004,
    0x00000007,0x00030003,0x00000002,0x000001c2,0x00040005,0x00000004,0x6e69616d,0x00000000,
    0x00040005,0x00000009,0x6f6c6f63,0x00000072,0x00040047,0x00000009,0x0000001e,0x00000000,
    0x00020013,0x00000002,0x00030021,0x00000003,0x00000002,0x00030016,0x00000006,0x00000020,
    0x00040017,0x00000007,0x00000006,0x00000004,0x00040020,0x00000008,0x00000003,0x00000007,
    0x0004003b,0x00000008,0x00000009,0x00000003,0x0004002b,0x00000006,0x0000000a,0x3f800000,
    0x0007002c,0x00000007,0x0000000b,0x0000000a,0x0000000a,0x0000000a,0x0000000a,0x00050036,
    0x00000002,0x00000004,0x00000000,0x00000003,0x000200f8,0x00000005,0x0003003e,0x00000009,
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
    memset(m.p, 0xee, size);
    return m;
}

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
        .format = FORMAT,
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
        .format = FORMAT,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    if ((r = vkCreateImageView(dev, &vci, NULL, &t.view)) != VK_SUCCESS) {
        fatal("vkCreateImageView", r);
    }
    return t;
}

static void to_layout(VkCommandBuffer cb, VkImage image, VkImageLayout from, VkImageLayout to) {
    VkImageMemoryBarrier b = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        .oldLayout = from,
        .newLayout = to,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &b);
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

static VkPipeline pipeline(VkDevice dev, VkPipelineLayout layout, VkShaderModule mesh,
                           VkShaderModule frag) {
    VkPipelineShaderStageCreateInfo stages[2] = {
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_MESH_BIT_EXT,
            .module = mesh,
            .pName = "main",
        },
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = frag,
            .pName = "main",
        },
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
    VkPipelineColorBlendAttachmentState blend = {
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };
    VkPipelineColorBlendStateCreateInfo cbs = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &blend,
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
        .pViewportState = &vp,
        .pRasterizationState = &rs,
        .pMultisampleState = &ms,
        .pColorBlendState = &cbs,
        .layout = layout,
    };
    VkPipeline p;
    VkResult r = vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, NULL, &p);
    if (r != VK_SUCCESS) {
        fatal("vkCreateGraphicsPipelines", r);
    }
    return p;
}

// One draw: what it checks, and how many columns it should leave white.
struct render {
    const char *what;
    uint32_t columns;
};

// Whether a slot is white in its first `columns` columns and black past them, and how many
// columns came out wholly white.
static int columns_lit(const uint8_t *slot, uint32_t columns, uint32_t *lit) {
    int ok = 1;
    *lit = 0;
    for (uint32_t x = 0; x < W; x++) {
        int white = 1;
        for (uint32_t y = 0; y < H; y++) {
            const uint8_t *t = slot + (y * W + x) * 4;
            int is_white = t[0] == 0xff && t[1] == 0xff && t[2] == 0xff && t[3] == 0xff;
            int is_black = t[0] == 0 && t[1] == 0 && t[2] == 0 && t[3] == 0xff;
            white &= is_white;
            ok &= x < columns ? is_white : is_black;
        }
        *lit += white;
    }
    return ok;
}

int main(void) {
    VkResult r;

    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "venus-mesh-shader-probe",
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
        fatal("Vulkan 1.3 for dynamic rendering", VK_ERROR_FEATURE_NOT_PRESENT);
    }

    uint32_t en = 0;
    vkEnumerateDeviceExtensionProperties(pd, NULL, &en, NULL);
    VkExtensionProperties *exts = calloc(en, sizeof *exts);
    vkEnumerateDeviceExtensionProperties(pd, NULL, &en, exts);
    const char *ext = "VK_EXT_mesh_shader";
    if (!has_ext(exts, en, ext)) {
        fatal("VK_EXT_mesh_shader", VK_ERROR_EXTENSION_NOT_PRESENT);
    }

    VkPhysicalDeviceMeshShaderFeaturesEXT has_mesh = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,
    };
    VkPhysicalDeviceVulkan12Features has12 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &has_mesh,
    };
    VkPhysicalDeviceFeatures2 has = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &has12,
    };
    vkGetPhysicalDeviceFeatures2(pd, &has);
    if (!has_mesh.meshShader) {
        fatal("meshShader", VK_ERROR_FEATURE_NOT_PRESENT);
    }
    if (!has12.drawIndirectCount) {
        fatal("drawIndirectCount", VK_ERROR_FEATURE_NOT_PRESENT);
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
    VkPhysicalDeviceMeshShaderFeaturesEXT fmesh = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,
        .meshShader = VK_TRUE,
    };
    VkPhysicalDeviceVulkan13Features f13 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext = &fmesh,
        .dynamicRendering = VK_TRUE,
    };
    VkPhysicalDeviceVulkan12Features f12 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &f13,
        .drawIndirectCount = VK_TRUE,
    };
    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &f12,
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

    PFN_vkCmdDrawMeshTasksEXT draw =
        (PFN_vkCmdDrawMeshTasksEXT)vkGetDeviceProcAddr(dev, "vkCmdDrawMeshTasksEXT");
    PFN_vkCmdDrawMeshTasksIndirectEXT draw_indirect =
        (PFN_vkCmdDrawMeshTasksIndirectEXT)vkGetDeviceProcAddr(dev,
                                                               "vkCmdDrawMeshTasksIndirectEXT");
    PFN_vkCmdDrawMeshTasksIndirectCountEXT draw_indirect_count =
        (PFN_vkCmdDrawMeshTasksIndirectCountEXT)vkGetDeviceProcAddr(
            dev, "vkCmdDrawMeshTasksIndirectCountEXT");
    if (!draw || !draw_indirect || !draw_indirect_count) {
        fatal("a mesh draw did not resolve", VK_ERROR_EXTENSION_NOT_PRESENT);
    }

    VkPipelineLayoutCreateInfo pli = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    VkPipelineLayout layout;
    if ((r = vkCreatePipelineLayout(dev, &pli, NULL, &layout)) != VK_SUCCESS) {
        fatal("vkCreatePipelineLayout", r);
    }
    VkShaderModule mesh = module(dev, COLUMN_MESH, sizeof COLUMN_MESH);
    VkShaderModule frag = module(dev, WHITE_FRAG, sizeof WHITE_FRAG);
    VkPipeline pipe = pipeline(dev, layout, mesh, frag);

    // The indirect records: (3, 1, 1) for the indirect draw, then (1, 1, 1) and (4, 1, 1) for the
    // count draw, whose count buffer says 1.
    VkDrawMeshTasksIndirectCommandEXT records[3] = {{3, 1, 1}, {1, 1, 1}, {4, 1, 1}};
    struct mapped args = mapped(pd, dev, sizeof records, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
    memcpy(args.p, records, sizeof records);
    const uint32_t one = 1;
    struct mapped count = mapped(pd, dev, sizeof one, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
    memcpy(count.p, &one, sizeof one);

    struct render renders[RENDERS] = {
        {"vkCmdDrawMeshTasksEXT(2, 1, 1): two columns", 2},
        {"vkCmdDrawMeshTasksIndirectEXT, (3, 1, 1): three columns", 3},
        {"vkCmdDrawMeshTasksIndirectCountEXT, count 1 of 2: one column", 1},
    };
    struct target t[RENDERS];
    for (uint32_t k = 0; k < RENDERS; k++) {
        t[k] = target(pd, dev);
    }
    struct mapped out = mapped(pd, dev, RENDERS * SLOT, VK_BUFFER_USAGE_TRANSFER_DST_BIT);

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

    const uint32_t stride = sizeof(VkDrawMeshTasksIndirectCommandEXT);
    for (uint32_t k = 0; k < RENDERS; k++) {
        to_layout(cb, t[k].image, VK_IMAGE_LAYOUT_UNDEFINED,
                  VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        VkRenderingAttachmentInfo att = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = t[k].view,
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue = {.color = {.float32 = {0.0f, 0.0f, 0.0f, 1.0f}}},
        };
        VkRenderingInfo ri = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .renderArea = {{0, 0}, {W, H}},
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &att,
        };
        vkCmdBeginRendering(cb, &ri);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
        if (k == 0) {
            draw(cb, 2, 1, 1);
        } else if (k == 1) {
            draw_indirect(cb, args.buf, 0, 1, stride);
        } else {
            draw_indirect_count(cb, args.buf, stride, count.buf, 0, 2, stride);
        }
        vkCmdEndRendering(cb);
        to_layout(cb, t[k].image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        VkBufferImageCopy copy = {
            .bufferOffset = k * SLOT,
            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .imageExtent = {W, H, 1},
        };
        vkCmdCopyImageToBuffer(cb, t[k].image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, out.buf, 1,
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
            const struct render *d = &renders[k];
            uint32_t lit = 0;
            check(columns_lit(out.p + k * SLOT, d->columns, &lit), d->what);
            printf("  %u of %u columns white\n", lit, W);
        }
    }

    vkDestroyFence(dev, fence, NULL);
    vkDestroyCommandPool(dev, cmdpool, NULL);
    vkDestroyPipeline(dev, pipe, NULL);
    vkDestroyShaderModule(dev, mesh, NULL);
    vkDestroyShaderModule(dev, frag, NULL);
    vkDestroyPipelineLayout(dev, layout, NULL);
    for (uint32_t k = 0; k < RENDERS; k++) {
        vkDestroyImageView(dev, t[k].view, NULL);
        vkDestroyImage(dev, t[k].image, NULL);
        vkFreeMemory(dev, t[k].mem, NULL);
    }
    struct mapped *bufs[3] = {&out, &args, &count};
    for (uint32_t i = 0; i < 3; i++) {
        vkUnmapMemory(dev, bufs[i]->mem);
        vkDestroyBuffer(dev, bufs[i]->buf, NULL);
        vkFreeMemory(dev, bufs[i]->mem, NULL);
    }
    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(inst, NULL);

    printf("\n%d check(s) failed\n", failures);
    return failures ? 1 : 0;
}
