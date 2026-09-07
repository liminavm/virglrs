// Spike, third half: does a mach memory entry get along with hv_vm_map?
//
// Two orderings, because the split needs both and they are not the same question.
//
//   forward -- the guest-RAM direction. The VMM hv_vm_maps its anonymous region
//   into the guest, and only then shares it with the renderer. Does the entry
//   still alias the pages the hypervisor has stage-2 mapped, or does the
//   hypervisor's claim on the range change what a later entry sees?
//
//   reverse -- the blob direction, which is what get_map_ptr becomes in a split.
//   The renderer mints storage and hands over an entry; the VMM maps it and then
//   hv_vm_maps *that* address into the guest. Does hv accept a mapping backed by
//   a memory entry from another task at all?
//
// Needs com.apple.security.hypervisor:
//   clang -O1 -Wall -Wextra -framework Hypervisor -o hvmap hvmap.c
//   codesign --entitlements <libkrun's hvf-entitlements.plist> -s - --force hvmap
//
// usage: hvmap forward|reverse
//        hvmap child <service-name> <forward|reverse>

#include <Hypervisor/Hypervisor.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <servers/bootstrap.h>
#include <spawn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#define REGION_SIZE (4u * 1024 * 1024)
#define GUEST_IPA 0x40000000ull

extern char **environ;

typedef struct {
    mach_msg_header_t hdr;
    mach_msg_body_t body;
    mach_msg_port_descriptor_t entry;
    int32_t seed;
    int32_t pad;
} carry_t;

typedef struct {
    mach_msg_header_t hdr;
    int32_t seed;
    int32_t pad;
} plain_t;

static void fill(void *base, size_t len, uint64_t seed) {
    uint64_t *w = base;
    for (size_t i = 0; i < len / sizeof(*w); i++)
        w[i] = seed ^ (uint64_t)i;
}

static long check(const void *base, size_t len, uint64_t seed) {
    const uint64_t *w = base;
    for (size_t i = 0; i < len / sizeof(*w); i++)
        if (w[i] != (seed ^ (uint64_t)i))
            return (long)i;
    return -1;
}

static mach_port_t make_entry(void *base, size_t len) {
    memory_object_size_t size = len;
    mach_port_t entry = MACH_PORT_NULL;
    kern_return_t kr = mach_make_memory_entry_64(
        mach_task_self(), &size, (memory_object_offset_t)(uintptr_t)base,
        MAP_MEM_VM_SHARE | VM_PROT_READ | VM_PROT_WRITE, &entry, MACH_PORT_NULL);
    if (kr != KERN_SUCCESS) {
        fprintf(stderr, "make_memory_entry: %s\n", mach_error_string(kr));
        return MACH_PORT_NULL;
    }
    if (size < len) {
        fprintf(stderr, "short entry: %llu of %zu\n", (unsigned long long)size, len);
        return MACH_PORT_NULL;
    }
    return entry;
}

// --------------------------------------------------------------------- child

static int child_main(const char *service, int reverse) {
    mach_port_t server = MACH_PORT_NULL;
    kern_return_t kr = bootstrap_look_up(bootstrap_port, (char *)service, &server);
    if (kr != KERN_SUCCESS) {
        fprintf(stderr, "renderer: look_up: %s\n", bootstrap_strerror(kr));
        return 2;
    }
    mach_port_t reply = MACH_PORT_NULL;
    if (mach_port_allocate(mach_task_self(), MACH_PORT_RIGHT_RECEIVE, &reply) != KERN_SUCCESS)
        return 2;

    void *addr = NULL;
    uint64_t seed = 0;

    if (reverse) {
        // The renderer mints; the VMM will hv_vm_map what we hand it.
        addr = mmap(NULL, REGION_SIZE, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
        if (addr == MAP_FAILED) {
            perror("renderer: mmap");
            return 2;
        }
        seed = 0xBEEF00;
        fill(addr, REGION_SIZE, seed);
        mach_port_t entry = make_entry(addr, REGION_SIZE);
        if (entry == MACH_PORT_NULL)
            return 2;

        carry_t msg = {0};
        msg.hdr.msgh_bits =
            MACH_MSGH_BITS(MACH_MSG_TYPE_COPY_SEND, MACH_MSG_TYPE_MAKE_SEND_ONCE) |
            MACH_MSGH_BITS_COMPLEX;
        msg.hdr.msgh_size = sizeof(msg);
        msg.hdr.msgh_remote_port = server;
        msg.hdr.msgh_local_port = reply;
        msg.hdr.msgh_id = 1;
        msg.body.msgh_descriptor_count = 1;
        msg.entry.name = entry;
        msg.entry.disposition = MACH_MSG_TYPE_COPY_SEND;
        msg.entry.type = MACH_MSG_PORT_DESCRIPTOR;
        msg.seed = (int32_t)seed;
        if (mach_msg(&msg.hdr, MACH_SEND_MSG, sizeof(msg), 0, MACH_PORT_NULL,
                     MACH_MSG_TIMEOUT_NONE, MACH_PORT_NULL) != KERN_SUCCESS)
            return 2;
        // Wait for the VMM to map it and hv_vm_map it before touching the pages
        // again -- otherwise the second write below races its first check.
        union { plain_t m; char pad[sizeof(plain_t) + MAX_TRAILER_SIZE]; } gb;
        memset(&gb, 0, sizeof(gb));
        kr = mach_msg(&gb.m.hdr, MACH_RCV_MSG | MACH_RCV_TIMEOUT, 0, sizeof(gb), reply,
                      20000, MACH_PORT_NULL);
        if (kr != KERN_SUCCESS) {
            fprintf(stderr, "renderer: recv go-ahead: %s\n", mach_error_string(kr));
            return 2;
        }
        if (gb.m.seed == 0) {
            fprintf(stderr, "renderer: the VMM could not use our storage\n");
            return 5;
        }
        fprintf(stderr, "renderer: minted and handed over, the VMM has it in the guest\n");
    } else {
        // The VMM shares guest RAM it has already hv_vm_mapped.
        carry_t req = {0};
        req.hdr.msgh_bits = MACH_MSGH_BITS(MACH_MSG_TYPE_COPY_SEND, MACH_MSG_TYPE_MAKE_SEND_ONCE);
        req.hdr.msgh_size = sizeof(req);
        req.hdr.msgh_remote_port = server;
        req.hdr.msgh_local_port = reply;
        req.hdr.msgh_id = 1;
        if (mach_msg(&req.hdr, MACH_SEND_MSG, sizeof(req), 0, MACH_PORT_NULL,
                     MACH_MSG_TIMEOUT_NONE, MACH_PORT_NULL) != KERN_SUCCESS)
            return 2;

        union { carry_t m; char pad[sizeof(carry_t) + MAX_TRAILER_SIZE]; } rb;
        memset(&rb, 0, sizeof(rb));
        kr = mach_msg(&rb.m.hdr, MACH_RCV_MSG | MACH_RCV_TIMEOUT, 0, sizeof(rb), reply,
                      15000, MACH_PORT_NULL);
        if (kr != KERN_SUCCESS) {
            fprintf(stderr, "renderer: recv entry: %s\n", mach_error_string(kr));
            return 2;
        }
        seed = (uint64_t)(uint32_t)rb.m.seed;
        mach_vm_address_t a = 0;
        kr = mach_vm_map(mach_task_self(), &a, REGION_SIZE, 0, VM_FLAGS_ANYWHERE,
                         rb.m.entry.name, 0, FALSE, VM_PROT_READ | VM_PROT_WRITE,
                         VM_PROT_READ | VM_PROT_WRITE, VM_INHERIT_NONE);
        if (kr != KERN_SUCCESS) {
            fprintf(stderr, "renderer: mach_vm_map of hv-mapped guest RAM: %s\n",
                    mach_error_string(kr));
            return 3;
        }
        addr = (void *)a;
        if (check(addr, REGION_SIZE, seed) >= 0) {
            fprintf(stderr, "renderer: FAIL guest RAM pattern not visible\n");
            return 4;
        }
        fprintf(stderr, "renderer: sees hv-mapped guest RAM at %p\n", addr);
    }

    // Both directions: write, let the other side check, then read what it wrote.
    fill(addr, REGION_SIZE, seed + 0x1111);

    carry_t note = {0};
    note.hdr.msgh_bits = MACH_MSGH_BITS(MACH_MSG_TYPE_COPY_SEND, MACH_MSG_TYPE_MAKE_SEND_ONCE);
    note.hdr.msgh_size = sizeof(note);
    note.hdr.msgh_remote_port = server;
    note.hdr.msgh_local_port = reply;
    note.hdr.msgh_id = 2;
    if (mach_msg(&note.hdr, MACH_SEND_MSG, sizeof(note), 0, MACH_PORT_NULL,
                 MACH_MSG_TIMEOUT_NONE, MACH_PORT_NULL) != KERN_SUCCESS)
        return 2;

    union { plain_t m; char pad[sizeof(plain_t) + MAX_TRAILER_SIZE]; } ab;
    memset(&ab, 0, sizeof(ab));
    kr = mach_msg(&ab.m.hdr, MACH_RCV_MSG | MACH_RCV_TIMEOUT, 0, sizeof(ab), reply,
                  15000, MACH_PORT_NULL);
    if (kr != KERN_SUCCESS)
        return 2;
    if (ab.m.seed == 0) {
        fprintf(stderr, "renderer: the VMM did not see our write\n");
        return 5;
    }
    if (check(addr, REGION_SIZE, (uint64_t)(uint32_t)ab.m.seed) >= 0) {
        fprintf(stderr, "renderer: FAIL the VMM's later write is not visible\n");
        return 6;
    }
    fprintf(stderr, "renderer: coherent both ways alongside the hypervisor\n");
    return 0;
}

// -------------------------------------------------------------------- parent

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (argc >= 4 && strcmp(argv[1], "child") == 0)
        return child_main(argv[2], strcmp(argv[3], "reverse") == 0);

    int reverse = (argc >= 2 && strcmp(argv[1], "reverse") == 0);
    printf("== %s: hv_vm_map %s the memory entry ==\n",
           reverse ? "reverse (blob path)" : "forward (guest-RAM path)",
           reverse ? "after" : "before");

    hv_return_t hv = hv_vm_create(NULL);
    if (hv != HV_SUCCESS) {
        printf("vmm: hv_vm_create failed (0x%x) -- is the binary signed with "
               "com.apple.security.hypervisor?\n", hv);
        return 1;
    }
    printf("vmm: VM created\n");

    void *base = NULL;
    uint64_t seed = 0xA5A500;
    mach_port_t entry = MACH_PORT_NULL;

    if (!reverse) {
        base = mmap(NULL, REGION_SIZE, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
        if (base == MAP_FAILED) {
            perror("vmm: mmap");
            return 1;
        }
        fill(base, REGION_SIZE, seed);
        hv = hv_vm_map(base, GUEST_IPA, REGION_SIZE,
                       HV_MEMORY_READ | HV_MEMORY_WRITE | HV_MEMORY_EXEC);
        if (hv != HV_SUCCESS) {
            printf("vmm: hv_vm_map: 0x%x\n", hv);
            return 1;
        }
        printf("vmm: guest RAM hv_vm_mapped at ipa %#llx, now making the entry\n", GUEST_IPA);
        entry = make_entry(base, REGION_SIZE);
        if (entry == MACH_PORT_NULL) {
            printf("RESULT: an entry cannot be made over hv-mapped memory\n");
            return 1;
        }
        printf("vmm: entry port 0x%x made over hv-mapped guest RAM\n", entry);
    }

    char name[128];
    snprintf(name, sizeof(name), "eti.noronha.hvmap.%d", getpid());
    mach_port_t service = MACH_PORT_NULL;
    kern_return_t kr = bootstrap_check_in(bootstrap_port, name, &service);
    if (kr != KERN_SUCCESS) {
        printf("vmm: bootstrap_check_in: %s\n", bootstrap_strerror(kr));
        return 1;
    }
    char *child_argv[] = {argv[0], "child", name, reverse ? "reverse" : "forward", NULL};
    pid_t pid = 0;
    int err = posix_spawn(&pid, argv[0], NULL, NULL, child_argv, environ);
    if (err != 0) {
        printf("vmm: posix_spawn: %s\n", strerror(err));
        return 1;
    }

    int ok = 0;
    for (int i = 0; i < 2; i++) {
        union { carry_t m; char pad[sizeof(carry_t) + MAX_TRAILER_SIZE]; } qb;
        memset(&qb, 0, sizeof(qb));
        kr = mach_msg(&qb.m.hdr, MACH_RCV_MSG | MACH_RCV_TIMEOUT, 0, sizeof(qb), service,
                      20000, MACH_PORT_NULL);
        if (kr != KERN_SUCCESS) {
            printf("vmm: recv: %s\n", mach_error_string(kr));
            return 1;
        }
        mach_port_t rp = qb.m.hdr.msgh_remote_port;

        if (qb.m.hdr.msgh_id == 1) {
            if (reverse) {
                // Map the renderer's storage, then hand it to the guest.
                mach_vm_address_t a = 0;
                kr = mach_vm_map(mach_task_self(), &a, REGION_SIZE, 0, VM_FLAGS_ANYWHERE,
                                 qb.m.entry.name, 0, FALSE, VM_PROT_READ | VM_PROT_WRITE,
                                 VM_PROT_READ | VM_PROT_WRITE, VM_INHERIT_NONE);
                if (kr != KERN_SUCCESS) {
                    printf("vmm: mach_vm_map of the renderer's storage: %s\n",
                           mach_error_string(kr));
                    return 1;
                }
                base = (void *)a;
                seed = (uint64_t)(uint32_t)qb.m.seed;
                if (check(base, REGION_SIZE, seed) >= 0) {
                    printf("vmm: the renderer's bytes are not visible\n");
                    return 1;
                }
                printf("vmm: mapped the renderer's storage at %p\n", base);
                hv = hv_vm_map(base, GUEST_IPA, REGION_SIZE,
                               HV_MEMORY_READ | HV_MEMORY_WRITE);
                if (hv != HV_SUCCESS) {
                    printf("vmm: hv_vm_map of memory-entry-backed storage: 0x%x\n", hv);
                    printf("RESULT: hv REFUSES a memory-entry mapping -- get_map_ptr "
                           "cannot publish renderer storage this way\n");
                    return 1;
                }
                printf("vmm: hv_vm_map accepted it at ipa %#llx\n", GUEST_IPA);

                plain_t rep = {0};
                rep.hdr.msgh_bits = MACH_MSGH_BITS(MACH_MSG_TYPE_MOVE_SEND_ONCE, 0);
                rep.hdr.msgh_size = sizeof(rep);
                rep.hdr.msgh_remote_port = rp;
                rep.seed = 1;
                mach_msg(&rep.hdr, MACH_SEND_MSG, sizeof(rep), 0, MACH_PORT_NULL,
                         MACH_MSG_TIMEOUT_NONE, MACH_PORT_NULL);
                continue;
            }
            carry_t rep = {0};
            rep.hdr.msgh_bits =
                MACH_MSGH_BITS(MACH_MSG_TYPE_MOVE_SEND_ONCE, 0) | MACH_MSGH_BITS_COMPLEX;
            rep.hdr.msgh_size = sizeof(rep);
            rep.hdr.msgh_remote_port = rp;
            rep.body.msgh_descriptor_count = 1;
            rep.entry.name = entry;
            rep.entry.disposition = MACH_MSG_TYPE_COPY_SEND;
            rep.entry.type = MACH_MSG_PORT_DESCRIPTOR;
            rep.seed = (int32_t)seed;
            if (mach_msg(&rep.hdr, MACH_SEND_MSG, sizeof(rep), 0, MACH_PORT_NULL,
                         MACH_MSG_TIMEOUT_NONE, MACH_PORT_NULL) != KERN_SUCCESS) {
                printf("vmm: send entry failed\n");
                return 1;
            }
        } else {
            long bad = check(base, REGION_SIZE, seed + 0x1111);
            plain_t rep = {0};
            rep.hdr.msgh_bits = MACH_MSGH_BITS(MACH_MSG_TYPE_MOVE_SEND_ONCE, 0);
            rep.hdr.msgh_size = sizeof(rep);
            rep.hdr.msgh_remote_port = rp;
            if (bad >= 0) {
                printf("vmm: FAIL the renderer's write is not visible (word %ld)\n", bad);
                rep.seed = 0;
            } else {
                printf("vmm: sees the renderer's write\n");
                fill(base, REGION_SIZE, seed + 0x2222);
                rep.seed = (int32_t)(seed + 0x2222);
                ok = 1;
            }
            mach_msg(&rep.hdr, MACH_SEND_MSG, sizeof(rep), 0, MACH_PORT_NULL,
                     MACH_MSG_TIMEOUT_NONE, MACH_PORT_NULL);
        }
    }

    int status = 0;
    waitpid(pid, &status, 0);
    int code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    printf("vmm: renderer exited %d\n", code);
    hv_vm_destroy();
    if (ok && code == 0) {
        printf("RESULT: VIABLE alongside hv_vm_map\n");
        return 0;
    }
    printf("RESULT: NOT VIABLE\n");
    return 1;
}
