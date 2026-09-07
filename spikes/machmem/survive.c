// Spike, second half: does memory minted *inside* the renderer outlive the
// renderer being killed, if the VMM holds a memory entry for it?
//
// This is the cheapest half of crash recovery. A host-visible blob today is
// HostShm/Storage owned by the renderer; if the pages die with the process, then
// after a crash every guest mapping published through hv_vm_map dangles and there
// is nothing to recover to. If they survive on the strength of a port the VMM
// holds, the guest keeps its mapping and its bytes across a renderer crash, and
// only the Vulkan objects need rebuilding.
//
// The child mints, fills, hands over an entry, and then SIGKILLs itself with no
// cleanup of any kind -- as close to an abort() in the renderer as a spike gets.
//
// usage: survive
//        survive child <service-name>

#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <servers/bootstrap.h>
#include <signal.h>
#include <spawn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#define REGION_SIZE (4u * 1024 * 1024)
#define SEED_MINTED 0xC0FFEEu

extern char **environ;

typedef struct {
    mach_msg_header_t hdr;
    mach_msg_body_t body;
    mach_msg_port_descriptor_t entry;
    int32_t seed;
    int32_t pad;
} handover_t;

typedef struct {
    mach_msg_header_t hdr;
    int32_t ok;
    int32_t pad;
} ack_t;

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

// ------------------------------------------------------- child ("renderer")

static int child_main(const char *service) {
    mach_port_t server = MACH_PORT_NULL;
    kern_return_t kr = bootstrap_look_up(bootstrap_port, (char *)service, &server);
    if (kr != KERN_SUCCESS) {
        fprintf(stderr, "renderer: bootstrap_look_up: %s\n", bootstrap_strerror(kr));
        return 2;
    }

    void *base = mmap(NULL, REGION_SIZE, PROT_READ | PROT_WRITE,
                      MAP_ANON | MAP_PRIVATE, -1, 0);
    if (base == MAP_FAILED) {
        perror("renderer: mmap");
        return 2;
    }
    fill(base, REGION_SIZE, SEED_MINTED);
    fprintf(stderr, "renderer: minted %u bytes at %p and filled them\n", REGION_SIZE, base);

    memory_object_size_t size = REGION_SIZE;
    mach_port_t entry = MACH_PORT_NULL;
    kr = mach_make_memory_entry_64(mach_task_self(), &size,
                                   (memory_object_offset_t)(uintptr_t)base,
                                   MAP_MEM_VM_SHARE | VM_PROT_READ | VM_PROT_WRITE,
                                   &entry, MACH_PORT_NULL);
    if (kr != KERN_SUCCESS) {
        fprintf(stderr, "renderer: make_memory_entry: %s\n", mach_error_string(kr));
        return 2;
    }

    mach_port_t reply = MACH_PORT_NULL;
    kr = mach_port_allocate(mach_task_self(), MACH_PORT_RIGHT_RECEIVE, &reply);
    if (kr != KERN_SUCCESS)
        return 2;

    handover_t msg = {0};
    msg.hdr.msgh_bits = MACH_MSGH_BITS(MACH_MSG_TYPE_COPY_SEND, MACH_MSG_TYPE_MAKE_SEND_ONCE) |
                        MACH_MSGH_BITS_COMPLEX;
    msg.hdr.msgh_size = sizeof(msg);
    msg.hdr.msgh_remote_port = server;
    msg.hdr.msgh_local_port = reply;
    msg.hdr.msgh_id = 1;
    msg.body.msgh_descriptor_count = 1;
    msg.entry.name = entry;
    msg.entry.disposition = MACH_MSG_TYPE_COPY_SEND;
    msg.entry.type = MACH_MSG_PORT_DESCRIPTOR;
    msg.seed = (int32_t)SEED_MINTED;
    kr = mach_msg(&msg.hdr, MACH_SEND_MSG, sizeof(msg), 0, MACH_PORT_NULL,
                  MACH_MSG_TIMEOUT_NONE, MACH_PORT_NULL);
    if (kr != KERN_SUCCESS) {
        fprintf(stderr, "renderer: send handover: %s\n", mach_error_string(kr));
        return 2;
    }

    union { ack_t m; char pad[sizeof(ack_t) + MAX_TRAILER_SIZE]; } ab;
    memset(&ab, 0, sizeof(ab));
    kr = mach_msg(&ab.m.hdr, MACH_RCV_MSG | MACH_RCV_TIMEOUT, 0, sizeof(ab), reply,
                  10000, MACH_PORT_NULL);
    if (kr != KERN_SUCCESS) {
        fprintf(stderr, "renderer: recv ack: %s\n", mach_error_string(kr));
        return 2;
    }

    fprintf(stderr, "renderer: dying now, with no cleanup at all\n");
    kill(getpid(), SIGKILL);
    return 99; // unreachable
}

// ------------------------------------------------------------ parent ("VMM")

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (argc >= 3 && strcmp(argv[1], "child") == 0)
        return child_main(argv[2]);

    char name[128];
    snprintf(name, sizeof(name), "eti.noronha.survive.%d", getpid());
    mach_port_t service = MACH_PORT_NULL;
    kern_return_t kr = bootstrap_check_in(bootstrap_port, name, &service);
    if (kr != KERN_SUCCESS) {
        printf("vmm: bootstrap_check_in: %s\n", bootstrap_strerror(kr));
        return 1;
    }

    char *child_argv[] = {argv[0], "child", name, NULL};
    pid_t pid = 0;
    int err = posix_spawn(&pid, argv[0], NULL, NULL, child_argv, environ);
    if (err != 0) {
        printf("vmm: posix_spawn: %s\n", strerror(err));
        return 1;
    }
    printf("vmm: spawned renderer %d\n", pid);

    union { handover_t m; char pad[sizeof(handover_t) + MAX_TRAILER_SIZE]; } hb;
    memset(&hb, 0, sizeof(hb));
    kr = mach_msg(&hb.m.hdr, MACH_RCV_MSG | MACH_RCV_TIMEOUT, 0, sizeof(hb), service,
                  15000, MACH_PORT_NULL);
    if (kr != KERN_SUCCESS) {
        printf("vmm: recv handover: %s\n", mach_error_string(kr));
        return 1;
    }
    mach_port_t entry = hb.m.entry.name;
    uint64_t seed = (uint64_t)(uint32_t)hb.m.seed;
    printf("vmm: holding entry port 0x%x for the renderer's storage\n", entry);

    mach_vm_address_t addr = 0;
    kr = mach_vm_map(mach_task_self(), &addr, REGION_SIZE, 0, VM_FLAGS_ANYWHERE,
                     entry, 0, FALSE, VM_PROT_READ | VM_PROT_WRITE,
                     VM_PROT_READ | VM_PROT_WRITE, VM_INHERIT_NONE);
    if (kr != KERN_SUCCESS) {
        printf("vmm: mach_vm_map: %s\n", mach_error_string(kr));
        return 1;
    }
    if (check((void *)addr, REGION_SIZE, seed) >= 0) {
        printf("vmm: the renderer's bytes were not visible before the crash\n");
        return 1;
    }
    printf("vmm: mapped it at 0x%llx, bytes are there\n", (unsigned long long)addr);

    ack_t ack = {0};
    ack.hdr.msgh_bits = MACH_MSGH_BITS(MACH_MSG_TYPE_MOVE_SEND_ONCE, 0);
    ack.hdr.msgh_size = sizeof(ack);
    ack.hdr.msgh_remote_port = hb.m.hdr.msgh_remote_port;
    ack.ok = 1;
    kr = mach_msg(&ack.hdr, MACH_SEND_MSG, sizeof(ack), 0, MACH_PORT_NULL,
                  MACH_MSG_TIMEOUT_NONE, MACH_PORT_NULL);
    if (kr != KERN_SUCCESS) {
        printf("vmm: send ack: %s\n", mach_error_string(kr));
        return 1;
    }

    int status = 0;
    waitpid(pid, &status, 0);
    if (WIFSIGNALED(status))
        printf("vmm: renderer died on signal %d\n", WTERMSIG(status));
    else
        printf("vmm: renderer exited %d (expected a signal)\n", WEXITSTATUS(status));

    long bad = check((void *)addr, REGION_SIZE, seed);
    if (bad >= 0) {
        printf("vmm: FAIL the bytes went with the process (word %ld)\n", bad);
        printf("RESULT: storage does NOT survive -- it must be minted VMM-side\n");
        return 1;
    }
    printf("vmm: the renderer's bytes are still readable after its death\n");

    fill((void *)addr, REGION_SIZE, seed + 7);
    if (check((void *)addr, REGION_SIZE, seed + 7) >= 0) {
        printf("vmm: FAIL the pages are no longer writable\n");
        return 1;
    }
    printf("vmm: and still writable\n");
    printf("RESULT: storage SURVIVES the renderer -- a held entry is enough\n");
    return 0;
}
