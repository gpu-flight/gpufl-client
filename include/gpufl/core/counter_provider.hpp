#pragma once

#include "gpufl/abi/gpufl_counter_abi.h"

namespace gpufl::detail {

/**
 * @brief Binds this module to the process-wide counter runtime.
 *
 * Every module holding a copy of the static `gpufl` library calls this and
 * ends up on the same registry, which is the whole point: without it a Python
 * target ticks the extension's registry while the injected evaluator reads the
 * injection DLL's, and the counter reads as Missing forever.
 *
 * Loaded explicitly by absolute path rather than through an import table.
 * Injection arrives via CUDA_INJECTION64_PATH, so the driver loads
 * gpufl_inject.dll into a process whose DLL search path has no reason to
 * include our bin directory - an import entry would fail there. Resolving the
 * path from THIS module's own location works wherever the module was loaded
 * from.
 *
 * Whoever binds first loads the library; everyone after gets the same instance,
 * which is what makes ordering between Python import and CUDA init irrelevant.
 */
class CounterProvider {
public:
    /** @brief The provider, or nullptr if the runtime could not be loaded. */
    static const gpufl_counter_provider_v1* get();

    /**
     * @brief True when counters are shared across modules.
     *
     * False means the shared runtime was not found and this module fell back to
     * its own in-process registry. That is correct for an embedded host, which
     * has exactly one copy of gpufl, and wrong under injection, where the
     * target and the evaluator are different modules. Callers that care - the
     * rule evaluator - must refuse custom counter rules when this is false and
     * more than one module is in play.
     */
    static bool isShared();

    /** @brief Test seam: forget the binding so the next get() resolves again. */
    static void resetForTesting();
};

/**
 * @brief The provider in use: the shared runtime, or this module's own.
 *
 * Never null, so callers do not each need a fallback branch.
 */
const gpufl_counter_provider_v1* ActiveCounterProvider();

}  // namespace gpufl::detail
