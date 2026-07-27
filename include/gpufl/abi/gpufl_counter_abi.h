#ifndef GPUFL_COUNTER_ABI_H
#define GPUFL_COUNTER_ABI_H

/*
 * Counter registry ABI.
 *
 * Exists because `gpufl` is a STATIC library, linked separately into
 * gpufl_inject.dll, the Python extension, and any host application. Each copy
 * would hold its own registry, so a Python server under `gpufl trace` would
 * tick one and the injected evaluator would read another - the counter would
 * read as Missing forever, which is exactly the case counters were added for.
 *
 * A small shared runtime owns the registry and everyone binds to it. Only C
 * types cross the boundary: no std:: types, no atomics passed by address, no
 * layout that depends on the compiler that built either side.
 *
 * Versioning: `abi_version` gates compatibility and `struct_size` allows
 * appending members. A consumer must check both before calling anything, and
 * must not assume a member exists because its own header declares it.
 */

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  define GPUFL_COUNTER_EXPORT __declspec(dllexport)
#else
#  define GPUFL_COUNTER_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define GPUFL_COUNTER_ABI_VERSION 1u

/* Opaque. Stable for the life of the process once returned. */
typedef void* gpufl_counter_handle;

typedef struct gpufl_counter_provider_v1 {
    uint32_t abi_version;
    uint32_t struct_size;

    /*
     * Find or create the counter named [name, name + name_length).
     * Returns NULL when the name is invalid or the counter limit is reached.
     * The same name always returns the same handle, including across modules
     * and from several threads at once.
     */
    gpufl_counter_handle (*register_counter)(const char* name, size_t name_length);

    /*
     * Add to a counter. The caller validates its own input; this ignores a
     * NULL handle. Values wrap, and readers take unsigned deltas, which stay
     * correct across a wrap.
     */
    void (*add)(gpufl_counter_handle handle, uint64_t value);

    /* Raw value, including anything added before the current session. */
    uint64_t (*load)(gpufl_counter_handle handle);

    /* Value accrued since the current session's baseline. */
    uint64_t (*load_since_baseline)(gpufl_counter_handle handle);

    /*
     * Baseline every counter at its current value. Called when a runtime
     * initialises, so ticks from a previous session - or from while no runtime
     * was active - are not counted as this one's.
     */
    void (*begin_session)(void);
    void (*end_session)(void);
    int  (*session_active)(void);

    /*
     * Find WITHOUT creating. Returns NULL when the application has not
     * registered this counter.
     *
     * A rule naming a counter must not bring it into existence: doing so makes
     * "the application never registered it" indistinguishable from "registered
     * but never ticked", and those need different answers - the first is a
     * config mistake worth reporting, the second is a workload that is simply
     * idle.
     */
    gpufl_counter_handle (*lookup)(const char* name, size_t name_length);
} gpufl_counter_provider_v1;

/*
 * Entry point the shared runtime exports. Never returns NULL from a runtime
 * that loaded successfully.
 */
GPUFL_COUNTER_EXPORT const gpufl_counter_provider_v1*
gpufl_get_counter_provider_v1(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* GPUFL_COUNTER_ABI_H */
