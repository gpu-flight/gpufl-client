#include "gpufl.hpp"

#include "gpufl/abi/gpufl_counter_abi.h"
#include "gpufl/core/counter_provider.hpp"
#include "gpufl/core/counter_registry.hpp"

namespace gpufl {
namespace {

// Fallback provider, used when no shared runtime is present. Correct for an
// embedded host, which holds the only copy of gpufl in the process. Under
// injection it is NOT correct - the target and the evaluator are separate
// modules and would each get their own registry - which is why
// CounterProvider::isShared() exists for the evaluator to gate on.
//
// Implemented as a provider rather than a separate branch in add() so both
// paths go through one code shape and cannot drift.
using detail::CounterRegistry;

gpufl_counter_handle LocalRegister(const char* name, size_t len) {
    if (name == nullptr) return nullptr;
    const auto slot = CounterRegistry::instance().registerCounter(std::string(name, len));
    if (slot == CounterRegistry::kInvalidSlot) return nullptr;
    return reinterpret_cast<gpufl_counter_handle>(static_cast<uintptr_t>(slot) + 1u);
}

bool LocalSlot(gpufl_counter_handle h, CounterRegistry::SlotId* out) {
    if (h == nullptr) return false;
    *out = static_cast<CounterRegistry::SlotId>(reinterpret_cast<uintptr_t>(h) - 1u);
    return true;
}

void LocalAdd(gpufl_counter_handle h, uint64_t v) {
    CounterRegistry::SlotId slot;
    if (LocalSlot(h, &slot)) CounterRegistry::instance().addRaw(slot, v);
}

uint64_t LocalLoad(gpufl_counter_handle h) {
    CounterRegistry::SlotId slot;
    return LocalSlot(h, &slot) ? CounterRegistry::instance().rawValue(slot) : 0;
}

uint64_t LocalLoadSinceBaseline(gpufl_counter_handle h) {
    CounterRegistry::SlotId slot;
    return LocalSlot(h, &slot) ? CounterRegistry::instance().valueSinceBaseline(slot) : 0;
}

gpufl_counter_handle LocalLookup(const char* name, size_t len) {
    if (name == nullptr) return nullptr;
    const auto slot = CounterRegistry::instance().findCounter(std::string(name, len));
    if (slot == CounterRegistry::kInvalidSlot) return nullptr;
    return reinterpret_cast<gpufl_counter_handle>(static_cast<uintptr_t>(slot) + 1u);
}

void LocalBeginSession() { CounterRegistry::instance().beginSession(); }
void LocalEndSession() { CounterRegistry::instance().endSession(); }
int LocalSessionActive() { return CounterRegistry::instance().sessionActive() ? 1 : 0; }

const gpufl_counter_provider_v1 kLocalProvider = {
    GPUFL_COUNTER_ABI_VERSION,
    sizeof(gpufl_counter_provider_v1),
    &LocalRegister, &LocalAdd, &LocalLoad, &LocalLoadSinceBaseline,
    &LocalBeginSession, &LocalEndSession, &LocalSessionActive,
    &LocalLookup,
};

}  // namespace

namespace detail {

const gpufl_counter_provider_v1* ActiveCounterProvider() {
    if (const auto* shared = CounterProvider::get()) return shared;
    return &kLocalProvider;
}

}  // namespace detail

void Counter::add(const int64_t n) const {
    if (handle_ == nullptr || n <= 0 || n > kMaxAddPerCall) return;
    static_cast<const gpufl_counter_provider_v1*>(provider_)
        ->add(handle_, static_cast<uint64_t>(n));
}

Counter counter(const std::string& name) {
    const gpufl_counter_provider_v1* provider = detail::ActiveCounterProvider();
    gpufl_counter_handle handle =
        provider->register_counter(name.c_str(), name.size());
    if (handle == nullptr) return Counter{};
    return Counter{provider, handle};
}

void tick(const std::string& name, const int64_t n) {
    // Deliberately does the lookup every call. Documented as not for tight
    // loops; anyone who cares holds a Counter instead.
    counter(name).add(n);
}

}  // namespace gpufl
