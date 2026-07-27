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

// A handle IS the slot's address. Not a slot id: an id has to be
// bounds-checked against the container, checking it means locking the
// container, and that would put every ticking thread on one mutex - distorting
// the throughput a rule exists to measure. Slots are never freed and a deque
// never relocates them, so the address stays good for the life of the process.
using Slot = CounterRegistry::Slot;

Slot* AsSlot(const gpufl_counter_handle h) { return static_cast<Slot*>(h); }

gpufl_counter_handle RegisterCounter(const char* name, const size_t name_length) {
    if (name == nullptr) return nullptr;
    CounterRegistry& reg = CounterRegistry::instance();
    return reg.slotFor(reg.registerCounter(std::string(name, name_length)));
}

gpufl_counter_handle Lookup(const char* name, const size_t name_length) {
    if (name == nullptr) return nullptr;
    CounterRegistry& reg = CounterRegistry::instance();
    return reg.slotFor(reg.findCounter(std::string(name, name_length)));
}

// Lock-free from here down: one relaxed atomic each.
void Add(const gpufl_counter_handle handle, const uint64_t value) {
    CounterRegistry::addRaw(AsSlot(handle), value);
}

uint64_t Load(const gpufl_counter_handle handle) {
    return CounterRegistry::rawValue(AsSlot(handle));
}

uint64_t LoadSinceBaseline(const gpufl_counter_handle handle) {
    return CounterRegistry::valueSinceBaseline(AsSlot(handle));
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
    &Lookup,
};

}  // namespace

extern "C" GPUFL_COUNTER_EXPORT const gpufl_counter_provider_v1*
gpufl_get_counter_provider_v1(void) {
    return &kProvider;
}
