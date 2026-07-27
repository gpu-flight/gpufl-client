// The shared counter runtime. One instance per process, so a target that ticks
// a counter and an injected evaluator that reads it are talking about the same
// slot - see gpufl_counter_abi.h for why that cannot be assumed otherwise.
//
// Deliberately thin: it owns CounterRegistry and wraps it in C. Keeping the
// logic in the registry means the embedded fallback path and this one cannot
// drift apart in behaviour.

#include "gpufl/abi/gpufl_counter_abi.h"

#include <string>

#include "gpufl/core/counter_registry.hpp"

namespace {

using gpufl::detail::CounterRegistry;

// Handles are slot ids biased by one, cast to a pointer, rather than addresses
// into the registry. An id survives anything the container does, and NULL then
// falls out naturally as "invalid" without needing a sentinel address.
gpufl_counter_handle ToHandle(const CounterRegistry::SlotId slot) {
    if (slot == CounterRegistry::kInvalidSlot) return nullptr;
    return reinterpret_cast<gpufl_counter_handle>(
        static_cast<uintptr_t>(slot) + 1u);
}

bool FromHandle(const gpufl_counter_handle handle, CounterRegistry::SlotId* out) {
    if (handle == nullptr) return false;
    *out = static_cast<CounterRegistry::SlotId>(
        reinterpret_cast<uintptr_t>(handle) - 1u);
    return true;
}

gpufl_counter_handle RegisterCounter(const char* name, const size_t name_length) {
    if (name == nullptr) return nullptr;
    return ToHandle(CounterRegistry::instance().registerCounter(
        std::string(name, name_length)));
}

void Add(const gpufl_counter_handle handle, const uint64_t value) {
    CounterRegistry::SlotId slot;
    if (!FromHandle(handle, &slot)) return;
    CounterRegistry::instance().addRaw(slot, value);
}

uint64_t Load(const gpufl_counter_handle handle) {
    CounterRegistry::SlotId slot;
    if (!FromHandle(handle, &slot)) return 0;
    return CounterRegistry::instance().rawValue(slot);
}

uint64_t LoadSinceBaseline(const gpufl_counter_handle handle) {
    CounterRegistry::SlotId slot;
    if (!FromHandle(handle, &slot)) return 0;
    return CounterRegistry::instance().valueSinceBaseline(slot);
}

void BeginSession() { CounterRegistry::instance().beginSession(); }
void EndSession() { CounterRegistry::instance().endSession(); }
int SessionActive() { return CounterRegistry::instance().sessionActive() ? 1 : 0; }

const gpufl_counter_provider_v1 kProvider = {
    GPUFL_COUNTER_ABI_VERSION,
    sizeof(gpufl_counter_provider_v1),
    &RegisterCounter,
    &Add,
    &Load,
    &LoadSinceBaseline,
    &BeginSession,
    &EndSession,
    &SessionActive,
};

}  // namespace

extern "C" GPUFL_COUNTER_EXPORT const gpufl_counter_provider_v1*
gpufl_get_counter_provider_v1(void) {
    return &kProvider;
}
