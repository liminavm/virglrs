// SPDX-License-Identifier: MIT
// Copyright © 2026 Gustavo Noronha Silva
// Spike: can a mach memory entry over the VMM's anonymous guest RAM be mapped
// coherently into a second process, with no file descriptor anywhere?
//
// Two questions, both empirical:
//
//   1. mach_make_memory_entry_64 over MAP_ANON memory -- does the entry reference
//      the pages, or a copy of them? vm-memory's MmapRegion (which is what
//      GuestMemoryMmap::from_ranges gives libkrun) allocates MAP_PRIVATE|MAP_ANON,
//      which is the cell that could plausibly copy-on-write instead of alias.
//      Proven by writing from BOTH sides after the map and requiring each to see
//      the other -- a one-way check passes on a copy taken at map time.
//
//   2. How the port reaches the child without a descriptor. mach ports do not
//      travel over SCM_RIGHTS, so the transport lane is part of the answer.
//      Tries bootstrap_check_in first, then bootstrap_register.
//
// usage: machmem [private|shared] [vmshare|plain]
//        machmem child <service-name>

#include <errno.h>
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

// Overridable so the entry can be tried at guest-RAM scale, not just a token region.
static unsigned long REGION_SIZE = 2ul * 1024 * 1024;
static void region_size_from_env(void) {
    const char *mb = getenv("MACHMEM_MB");
    if (mb)
        REGION_SIZE = strtoul(mb, NULL, 10) * 1024ul * 1024ul;
}
#define MSG_HELLO 1
#define MSG_WROTE 2

extern char **environ;

// A request carries no payload; the reply carries the memory entry.
typedef struct {
    mach_msg_header_t hdr;
    int32_t id;
    int32_t pad;
} request_t;

typedef struct {
    mach_msg_header_t hdr;
    mach_msg_body_t body;
    mach_msg_port_descriptor_t entry;
    int32_t seed;
    int32_t pad;
} reply_with_port_t;

typedef struct {
    mach_msg_header_t hdr;
    int32_t seed;
    int32_t pad;
} reply_plain_t;

static size_t stride_words(void) {
    return getenv("MACHMEM_SPARSE") ? (1ul << 20) / sizeof(uint64_t) : 1;
}

static void fill(void *base, size_t len, uint64_t seed) {
    uint64_t *w = base;
    size_t st = stride_words();
    for (size_t i = 0; i < len / sizeof(*w); i += st)
        w[i] = seed ^ (uint64_t)i;
}

// Returns the index of the first mismatching word, or -1 if all match.
static long check(const void *base, size_t len, uint64_t seed) {
    const uint64_t *w = base;
    size_t st = stride_words();
    for (size_t i = 0; i < len / sizeof(*w); i += st)
        if (w[i] != (seed ^ (uint64_t)i))
            return (long)i;
    return -1;
}

// ------------------------------------------------------------------- child

static int child_main(const char *service) {
    region_size_from_env();
    mach_port_t server = MACH_PORT_NULL;
    kern_return_t kr = bootstrap_look_up(bootstrap_port, (char *)service, &server);
    if (kr != KERN_SUCCESS) {
        fprintf(stderr, "child: bootstrap_look_up(%s): %s\n", service,
                bootstrap_strerror(kr));
        return 2;
    }
    fprintf(stderr, "child: looked up the service, no descriptor involved\n");

    mach_port_t reply = MACH_PORT_NULL;
    kr = mach_port_allocate(mach_task_self(), MACH_PORT_RIGHT_RECEIVE, &reply);
    if (kr != KERN_SUCCESS) {
        fprintf(stderr, "child: mach_port_allocate: %s\n", mach_error_string(kr));
        return 2;
    }

    request_t req = {0};
    req.hdr.msgh_bits = MACH_MSGH_BITS(MACH_MSG_TYPE_COPY_SEND, MACH_MSG_TYPE_MAKE_SEND_ONCE);
    req.hdr.msgh_size = sizeof(req);
    req.hdr.msgh_remote_port = server;
    req.hdr.msgh_local_port = reply;
    req.hdr.msgh_id = MSG_HELLO;
    req.id = MSG_HELLO;

    union { reply_with_port_t m; char pad[sizeof(reply_with_port_t) + MAX_TRAILER_SIZE]; } rb;
    memset(&rb, 0, sizeof(rb));
    reply_with_port_t *repp = &rb.m;
    kr = mach_msg(&req.hdr, MACH_SEND_MSG, sizeof(req), 0, MACH_PORT_NULL,
                  MACH_MSG_TIMEOUT_NONE, MACH_PORT_NULL);
    if (kr != KERN_SUCCESS) {
        fprintf(stderr, "child: send hello: %s\n", mach_error_string(kr));
        return 2;
    }
    kr = mach_msg(&repp->hdr, MACH_RCV_MSG | MACH_RCV_TIMEOUT, 0, sizeof(rb), reply,
                  10000, MACH_PORT_NULL);
    if (kr != KERN_SUCCESS) {
        fprintf(stderr, "child: recv entry: %s\n", mach_error_string(kr));
        return 2;
    }

    mach_port_t entry = repp->entry.name;
    uint64_t seed_a = (uint64_t)(uint32_t)repp->seed;
    fprintf(stderr, "child: received memory entry port 0x%x\n", entry);

    mach_vm_address_t addr = 0;
    kr = mach_vm_map(mach_task_self(), &addr, REGION_SIZE, 0, VM_FLAGS_ANYWHERE,
                     entry, 0, FALSE, VM_PROT_READ | VM_PROT_WRITE,
                     VM_PROT_READ | VM_PROT_WRITE, VM_INHERIT_NONE);
    if (kr != KERN_SUCCESS) {
        fprintf(stderr, "child: mach_vm_map: %s (0x%x)\n", mach_error_string(kr), kr);
        return 3;
    }
    fprintf(stderr, "child: mapped at 0x%llx\n", (unsigned long long)addr);

    long bad = check((void *)addr, REGION_SIZE, seed_a);
    if (bad >= 0) {
        fprintf(stderr, "child: FAIL parent's pattern A not visible (word %ld)\n", bad);
        return 4;
    }
    fprintf(stderr, "child: sees the parent's pattern A\n");

    // Write B, ask the parent whether it sees it.
    uint64_t seed_b = seed_a + 0x1111;
    fill((void *)addr, REGION_SIZE, seed_b);

    request_t req2 = {0};
    req2.hdr.msgh_bits = MACH_MSGH_BITS(MACH_MSG_TYPE_COPY_SEND, MACH_MSG_TYPE_MAKE_SEND_ONCE);
    req2.hdr.msgh_size = sizeof(req2);
    req2.hdr.msgh_remote_port = server;
    req2.hdr.msgh_local_port = reply;
    req2.hdr.msgh_id = MSG_WROTE;
    req2.id = MSG_WROTE;
    kr = mach_msg(&req2.hdr, MACH_SEND_MSG, sizeof(req2), 0, MACH_PORT_NULL,
                  MACH_MSG_TIMEOUT_NONE, MACH_PORT_NULL);
    if (kr != KERN_SUCCESS) {
        fprintf(stderr, "child: send wrote: %s\n", mach_error_string(kr));
        return 2;
    }

    union { reply_plain_t m; char pad[sizeof(reply_plain_t) + MAX_TRAILER_SIZE]; } ab;
    memset(&ab, 0, sizeof(ab));
    reply_plain_t *rep2 = &ab.m;
    kr = mach_msg(&rep2->hdr, MACH_RCV_MSG | MACH_RCV_TIMEOUT, 0, sizeof(ab), reply,
                  10000, MACH_PORT_NULL);
    if (kr != KERN_SUCCESS) {
        fprintf(stderr, "child: recv ack: %s\n", mach_error_string(kr));
        return 2;
    }
    if (rep2->seed == 0) {
        fprintf(stderr, "child: parent did not see pattern B\n");
        return 5;
    }

    // The parent has since written C; a copy taken at map time would not show it.
    uint64_t seed_c = (uint64_t)(uint32_t)rep2->seed;
    bad = check((void *)addr, REGION_SIZE, seed_c);
    if (bad >= 0) {
        fprintf(stderr, "child: FAIL parent's later pattern C not visible (word %ld)\n", bad);
        return 6;
    }
    fprintf(stderr, "child: sees the parent's later pattern C -- coherent both ways\n");
    return 0;
}

// ------------------------------------------------------------------ parent

// Try both lanes; report which one the machine actually allows.
//
// check_in hands back a receive right of its own -- the service port is *its*
// port, not one we allocated. Listening on our own port instead is a hang, not
// an error, because the child's message goes somewhere real.
static kern_return_t publish(const char *name, mach_port_t *service, const char **lane) {
    mach_port_t checked = MACH_PORT_NULL;
    kern_return_t kr = bootstrap_check_in(bootstrap_port, (char *)name, &checked);
    if (kr == KERN_SUCCESS) {
        *service = checked;
        *lane = "bootstrap_check_in";
        return kr;
    }
    fprintf(stderr, "parent: bootstrap_check_in: %s\n", bootstrap_strerror(kr));

    kr = mach_port_allocate(mach_task_self(), MACH_PORT_RIGHT_RECEIVE, service);
    if (kr != KERN_SUCCESS)
        return kr;
    kr = mach_port_insert_right(mach_task_self(), *service, *service, MACH_MSG_TYPE_MAKE_SEND);
    if (kr != KERN_SUCCESS)
        return kr;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    kr = bootstrap_register(bootstrap_port, (char *)name, *service);
#pragma clang diagnostic pop
    if (kr == KERN_SUCCESS) {
        *lane = "bootstrap_register";
        return kr;
    }
    fprintf(stderr, "parent: bootstrap_register: %s\n", bootstrap_strerror(kr));
    return kr;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    region_size_from_env();
    if (argc >= 3 && strcmp(argv[1], "child") == 0)
        return child_main(argv[2]);

    int map_shared = (argc >= 2 && strcmp(argv[1], "shared") == 0);
    int vm_share = !(argc >= 3 && strcmp(argv[2], "plain") == 0);

    int flags = MAP_ANON | (map_shared ? MAP_SHARED : MAP_PRIVATE);
    printf("== mmap %s | MAP_ANON, entry perms %s ==\n",
           map_shared ? "MAP_SHARED" : "MAP_PRIVATE",
           vm_share ? "RW|MAP_MEM_VM_SHARE" : "RW");

    void *base = mmap(NULL, REGION_SIZE, PROT_READ | PROT_WRITE, flags, -1, 0);
    if (base == MAP_FAILED) {
        perror("parent: mmap");
        return 1;
    }
    uint64_t seed_a = 0xA5A5A5A5;
    fill(base, REGION_SIZE, seed_a);
    printf("parent: %lu bytes of \"guest RAM\" at %p, pattern A written\n", (unsigned long)REGION_SIZE, base);

    memory_object_size_t size = REGION_SIZE;
    mach_port_t entry = MACH_PORT_NULL;
    vm_prot_t perm = VM_PROT_READ | VM_PROT_WRITE;
    if (vm_share)
        perm |= MAP_MEM_VM_SHARE;
    kern_return_t kr = mach_make_memory_entry_64(mach_task_self(), &size,
                                                 (memory_object_offset_t)(uintptr_t)base,
                                                 perm, &entry, MACH_PORT_NULL);
    if (kr != KERN_SUCCESS) {
        printf("parent: mach_make_memory_entry_64: %s (0x%x)  => NOT VIABLE\n",
               mach_error_string(kr), kr);
        return 1;
    }
    printf("parent: entry port 0x%x covers %llu bytes (asked %lu)\n", entry,
           (unsigned long long)size, (unsigned long)REGION_SIZE);
    if (size < REGION_SIZE) {
        printf("parent: SHORT ENTRY -- the kernel would only share %llu bytes\n",
               (unsigned long long)size);
        return 1;
    }

    mach_port_t service = MACH_PORT_NULL;
    char name[128];
    snprintf(name, sizeof(name), "eti.noronha.machmem.%d", getpid());
    const char *lane = NULL;
    if (publish(name, &service, &lane) != KERN_SUCCESS) {
        printf("parent: NO BOOTSTRAP LANE -- the port cannot reach a spawned child this way\n");
        return 1;
    }
    printf("parent: published as %s via %s\n", name, lane);

    char *child_argv[] = {argv[0], "child", name, NULL};
    pid_t pid = 0;
    int err = posix_spawn(&pid, argv[0], NULL, NULL, child_argv, environ);
    if (err != 0) {
        printf("parent: posix_spawn: %s\n", strerror(err));
        return 1;
    }
    printf("parent: spawned child %d\n", pid);

    int served = 0;
    uint64_t seed_c = seed_a + 0x2222;
    for (int i = 0; i < 2; i++) {
        union { request_t m; char pad[sizeof(request_t) + MAX_TRAILER_SIZE]; } qb;
        memset(&qb, 0, sizeof(qb));
        request_t *req = &qb.m;
        kr = mach_msg(&req->hdr, MACH_RCV_MSG | MACH_RCV_TIMEOUT, 0, sizeof(qb), service,
                      15000, MACH_PORT_NULL);
        if (kr != KERN_SUCCESS) {
            printf("parent: recv: %s\n", mach_error_string(kr));
            return 1;
        }
        mach_port_t rp = req->hdr.msgh_remote_port;

        if (req->hdr.msgh_id == MSG_HELLO) {
            reply_with_port_t rep = {0};
            rep.hdr.msgh_bits = MACH_MSGH_BITS(MACH_MSG_TYPE_MOVE_SEND_ONCE, 0) |
                                MACH_MSGH_BITS_COMPLEX;
            rep.hdr.msgh_size = sizeof(rep);
            rep.hdr.msgh_remote_port = rp;
            rep.hdr.msgh_local_port = MACH_PORT_NULL;
            rep.body.msgh_descriptor_count = 1;
            rep.entry.name = entry;
            rep.entry.disposition = MACH_MSG_TYPE_COPY_SEND;
            rep.entry.type = MACH_MSG_PORT_DESCRIPTOR;
            rep.seed = (int32_t)seed_a;
            kr = mach_msg(&rep.hdr, MACH_SEND_MSG, sizeof(rep), 0, MACH_PORT_NULL,
                          MACH_MSG_TIMEOUT_NONE, MACH_PORT_NULL);
            if (kr != KERN_SUCCESS) {
                printf("parent: send entry: %s\n", mach_error_string(kr));
                return 1;
            }
            printf("parent: handed the entry over, still no descriptor\n");
        } else {
            long bad = check(base, REGION_SIZE, seed_a + 0x1111);
            reply_plain_t rep = {0};
            rep.hdr.msgh_bits = MACH_MSGH_BITS(MACH_MSG_TYPE_MOVE_SEND_ONCE, 0);
            rep.hdr.msgh_size = sizeof(rep);
            rep.hdr.msgh_remote_port = rp;
            if (bad >= 0) {
                printf("parent: FAIL child's pattern B not visible (word %ld) "
                       "=> the entry was a copy\n", bad);
                rep.seed = 0;
            } else {
                printf("parent: sees the child's pattern B\n");
                fill(base, REGION_SIZE, seed_c);
                rep.seed = (int32_t)seed_c;
                served = 1;
            }
            kr = mach_msg(&rep.hdr, MACH_SEND_MSG, sizeof(rep), 0, MACH_PORT_NULL,
                          MACH_MSG_TIMEOUT_NONE, MACH_PORT_NULL);
            if (kr != KERN_SUCCESS) {
                printf("parent: send ack: %s\n", mach_error_string(kr));
                return 1;
            }
        }
    }

    int status = 0;
    waitpid(pid, &status, 0);
    int code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    printf("parent: child exited %d\n", code);
    if (served && code == 0) {
        printf("RESULT: VIABLE -- coherent both ways, mach only\n");
        return 0;
    }
    printf("RESULT: NOT VIABLE for this combination\n");
    return 1;
}
