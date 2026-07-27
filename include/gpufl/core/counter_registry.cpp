#include "gpufl/core/counter_registry.hpp"

#include <algorithm>

#include "gpufl/core/debug_logger.hpp"

namespace gpufl::detail {

CounterRegistry& CounterRegistry::instance() {
    // Function-local static: the table has to outlive every runtime, and this
    // is the only storage duration that guarantees it without an init order
    // dependency on whoever registers first.
    static CounterRegistry registry;
    return registry;
}

bool CounterRegistry::nameIsValid(const std::string& name) {
    if (name.empty() || name.size() > kMaxNameLength) return false;
    // Explicit ASCII, not std::isalnum: that one answers according to the
    // current locale, and the accepted set here is part of a wire contract
    // rather than something a host program's locale gets to widen.
    return std::all_of(name.begin(), name.end(), [](const char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
               (c >= '0' && c <= '9') || c == '_' || c == '.' || c == '-';
    });
}

std::string CounterRegistry::forLog(const std::string& name) {
    // A rejected name is attacker- or bug-supplied and can be arbitrarily long.
    // Show enough to identify it, never enough to bloat the log.
    constexpr size_t kMaxLogged = 32;
    if (name.size() <= kMaxLogged) return name;
    return name.substr(0, kMaxLogged) + "...(" + std::to_string(name.size()) + " chars)";
}

CounterRegistry::SlotId CounterRegistry::findCounter(const std::string& name) const {
    std::lock_guard lk(mu_);
    const auto it = byName_.find(name);
    return it == byName_.end() ? kInvalidSlot : it->second;
}

CounterRegistry::SlotId CounterRegistry::registerCounter(const std::string& name) {
    if (!nameIsValid(name)) {
        std::lock_guard lk(mu_);
        if (!loggedInvalidName_) {
            loggedInvalidName_ = true;
            GFL_LOG_ERROR("[CounterRegistry] rejected counter name '", forLog(name),
                          "': must be 1-", kMaxNameLength,
                          " characters of [A-Za-z0-9._-] "
                          "(further name rejections suppressed)");
        }
        return kInvalidSlot;
    }

    std::lock_guard lk(mu_);
    // Same name from several threads resolves to one slot; the map lookup
    // under the lock is what makes that true.
    if (const auto it = byName_.find(name); it != byName_.end()) return it->second;

    if (slots_.size() >= kMaxCounters) {
        if (!loggedLimitReached_) {
            loggedLimitReached_ = true;
            GFL_LOG_ERROR("[CounterRegistry] counter limit reached (", kMaxCounters,
                          "); '", forLog(name),
                          "' not registered (further rejections suppressed)");
        }
        return kInvalidSlot;
    }

    const auto slot = static_cast<SlotId>(slots_.size());
    slots_.emplace_back();
    slots_.back().name = name;
    // Registered mid-generation: baseline at the current value so ticks from a
    // previous session, or from before init(), do not land in this one.
    slots_.back().baseline = slots_.back().value.load(std::memory_order_relaxed);
    byName_.emplace(name, slot);
    return slot;
}

std::atomic<uint64_t>* CounterRegistry::valueSlot(SlotId slot) {
    std::lock_guard lk(mu_);
    if (slot >= slots_.size()) return nullptr;
    // Safe to hand out and keep: a deque never relocates existing elements, and
    // slots are never erased outside resetForTesting().
    return &slots_[slot].value;
}

void CounterRegistry::addRaw(SlotId slot, uint64_t value) {
    std::lock_guard lk(mu_);
    if (slot >= slots_.size()) return;
    slots_[slot].value.fetch_add(value, std::memory_order_relaxed);
}

uint64_t CounterRegistry::rawValue(SlotId slot) const {
    std::lock_guard lk(mu_);
    if (slot >= slots_.size()) return 0;
    return slots_[slot].value.load(std::memory_order_relaxed);
}

uint64_t CounterRegistry::valueSinceBaseline(SlotId slot) const {
    std::lock_guard lk(mu_);
    if (slot >= slots_.size()) return 0;
    // Unsigned subtraction, deliberately: correct across a wrap, where a signed
    // difference would go negative.
    return slots_[slot].value.load(std::memory_order_relaxed) - slots_[slot].baseline;
}

const std::string& CounterRegistry::name(SlotId slot) const {
    static const std::string kEmpty;
    std::lock_guard lk(mu_);
    if (slot >= slots_.size()) return kEmpty;
    return slots_[slot].name;
}

size_t CounterRegistry::counterCount() const {
    std::lock_guard lk(mu_);
    return slots_.size();
}

void CounterRegistry::beginSession() {
    std::lock_guard lk(mu_);
    for (auto& slot : slots_) {
        slot.baseline = slot.value.load(std::memory_order_relaxed);
    }
    sessionActive_.store(true, std::memory_order_release);
}

void CounterRegistry::endSession() {
    // Values stay. A handle that outlives this keeps pointing at a live slot;
    // it is the baseline taken by the next beginSession that stops those adds
    // from being counted twice.
    sessionActive_.store(false, std::memory_order_release);
}

void CounterRegistry::resetForTesting() {
    std::lock_guard lk(mu_);
    slots_.clear();
    byName_.clear();
    sessionActive_.store(false, std::memory_order_release);
    loggedInvalidName_ = false;
    loggedLimitReached_ = false;
}

}  // namespace gpufl::detail
