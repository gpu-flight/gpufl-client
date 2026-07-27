#include "gpufl.hpp"
#include "gpufl/core/counter_registry.hpp"

namespace gpufl {

Counter counter(const std::string& name) {
    auto& registry = detail::CounterRegistry::instance();
    const auto slot = registry.registerCounter(name);
    if (slot == detail::CounterRegistry::kInvalidSlot) return Counter{};
    // The address, not the id: add() must not touch the container, or it would
    // race a concurrent registration.
    return Counter{registry.valueSlot(slot)};
}

void tick(const std::string& name, int64_t n) {
    // Deliberately does the lookup every call. Documented as not for tight
    // loops; anyone who cares holds a Counter instead.
    counter(name).add(n);
}

}  // namespace gpufl
