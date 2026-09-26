// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva
//
// The `cooperative-matrix` group of src/venus/unserved.txt, as a program that names its command.
//
//   vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR
//
// From VK_KHR_cooperative_matrix, which lavapipe advertises and neither KosmicKrisp nor anv on
// Ice Lake does, so the positive control and the venus run are both on lavapipe. The guest's venus
// driver forwards the query to the renderer every time; it keeps no copy.
//
// The command is a query, so its answer is the consequence, and it is checked three ways:
//
//   - the two-call enumeration: a count, then that many shapes, the same count both times;
//   - a call with room for one fewer returns VK_INCOMPLETE, the shorter count, and leaves the
//     slot past it as it was sent -- a renderer that wrote the driver's full count back, or wrote
//     past the room, fails here;
//   - every shape is printed, one line each, so a venus run can be diffed against the host's.
//     A shape the renderer dropped or zeroed shows up in the diff, not in a check.
//
// Headless on purpose: no surface, no swapchain, no window, and no device. It runs over ssh.
//
//   cc -O1 -o cooperative_matrix cooperative_matrix.c -lvulkan && ./cooperative_matrix
//
// Pick the device with MESA_VK_DEVICE_SELECT; the probe takes the first one the loader lists.
//
// Exit 0 = every check passed. Exit 1 = a check failed, and the line above it says which.
// Exit 2 = the device could not be brought up at all (not a verdict on the group).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

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

static int has_ext(VkExtensionProperties *exts, uint32_t n, const char *name) {
    for (uint32_t i = 0; i < n; i++) {
        if (strcmp(exts[i].extensionName, name) == 0) {
            return 1;
        }
    }
    return 0;
}

static VkCooperativeMatrixPropertiesKHR blank(void) {
    VkCooperativeMatrixPropertiesKHR p;
    memset(&p, 0xee, sizeof p);
    p.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
    p.pNext = NULL;
    return p;
}

int main(void) {
    VkResult r;

    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "venus-cooperative-matrix-probe",
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

    uint32_t en = 0;
    vkEnumerateDeviceExtensionProperties(pd, NULL, &en, NULL);
    VkExtensionProperties *exts = calloc(en, sizeof *exts);
    vkEnumerateDeviceExtensionProperties(pd, NULL, &en, exts);
    if (!has_ext(exts, en, "VK_KHR_cooperative_matrix")) {
        fatal("VK_KHR_cooperative_matrix", VK_ERROR_EXTENSION_NOT_PRESENT);
    }

    PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR shapes =
        (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)vkGetInstanceProcAddr(
            inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    if (!shapes) {
        fatal("vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR did not resolve",
              VK_ERROR_EXTENSION_NOT_PRESENT);
    }

    uint32_t count = 0;
    r = shapes(pd, &count, NULL);
    check(r == VK_SUCCESS && count > 0, "the count call answers at least one shape");
    if (count == 0) {
        printf("\n%d check(s) failed\n", failures);
        return 1;
    }

    VkCooperativeMatrixPropertiesKHR *all = calloc(count, sizeof *all);
    for (uint32_t i = 0; i < count; i++) {
        all[i] = blank();
    }
    uint32_t again = count;
    r = shapes(pd, &again, all);
    check(r == VK_SUCCESS && again == count, "the second call writes the same count");
    int written = 1;
    for (uint32_t i = 0; i < count; i++) {
        written &= all[i].sType == VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR &&
                   all[i].MSize != 0xeeeeeeeeu && all[i].MSize != 0 && all[i].KSize != 0;
    }
    check(written, "every shape was written, with sizes");

    // One fewer slot than the driver has, and a sentinel in the slot past the room.
    uint32_t room = count - 1;
    VkCooperativeMatrixPropertiesKHR *short_ = calloc(count, sizeof *short_);
    for (uint32_t i = 0; i < count; i++) {
        short_[i] = blank();
    }
    uint32_t got = room;
    r = shapes(pd, &got, short_);
    check(r == VK_INCOMPLETE && got == room, "one slot short: VK_INCOMPLETE and the shorter count");
    check(short_[room].MSize == 0xeeeeeeeeu, "and the slot past the room is left as it was sent");
    int same = 1;
    for (uint32_t i = 0; i < room; i++) {
        same &= short_[i].MSize == all[i].MSize && short_[i].KSize == all[i].KSize &&
                short_[i].AType == all[i].AType && short_[i].scope == all[i].scope;
    }
    check(same, "the shapes it did write are the first ones");

    printf("\nshapes (M N K A B C Result saturating scope):\n");
    for (uint32_t i = 0; i < count; i++) {
        const VkCooperativeMatrixPropertiesKHR *p = &all[i];
        printf("  %u %u %u %d %d %d %d %u %d\n", p->MSize, p->NSize, p->KSize, p->AType,
               p->BType, p->CType, p->ResultType, p->saturatingAccumulation, p->scope);
    }

    free(short_);
    free(all);
    free(exts);
    free(pds);
    vkDestroyInstance(inst, NULL);

    printf("\n%d check(s) failed\n", failures);
    return failures ? 1 : 0;
}
