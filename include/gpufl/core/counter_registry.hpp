#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>

namespace gpufl::detail {

/**
 * @brief Named counters the application increments, read as rates.
 *
 * Exists so a rule can watch something only the application knows - tokens,
 * steps, requests - without the application computing a rate or reaching for a
 * scope. A scope would cost two locked batch pushes and a wire row per
 * iteration, which rules it out of a decode loop; this costs one relaxed atomic
 * add, and unlike a scope it can carry a count, so a step that produced eight
 * tokens says so.
 *
 * ## Slots outlive the runtime, deliberately
 *
 * A handle can be stored in a static, or held across a shutdown()/init() cycle
 * by an embedded host. A generation number alone would not make that safe: it
 * cannot stop shutdown() freeing state while another thread is inside add().
 * So the slot itself is never freed. init()/shutdown() only change which
 * runtime reads it, and nothing add() touches can disappear underneath it.
 *
 * That also removes any need to rebind a stale handle. There is nothing to
 * rebind to - the slot a handle points at is still the right one - so the hot
 * path carries no generation check either.
 *
 * The cost is a value that survives sessions, so each runtime records a
 * baseline at init() and reports only what accrued since. Adds made while no
 * runtime is active land in the slot and are excluded from the next session
 * rather than counted twice.
 */
class CounterRegistry {
public:
    /** @brief Index into the permanent slot table. kInvalidSlot on rejection. */
    using SlotId = uint32_t;
    static constexpr SlotId kInvalidSlot = 0xFFFFFFFFu;

    /** Bounds. A permanent table is exactly what must not grow without limit. */
    static constexpr size_t kMaxCounters = 256;
    static constexpr size_t kMaxNameLength = 96;
    /**
     * Largest single add accepted.
     *
     * Not an overflow guard - the counter is 64-bit and rates are unsigned
     * deltas, so a wrap is both unreachable in practice and handled if it
     * happened. This catches a caller passing something that is not a count at
     * all: a pointer, an uninitialised value, a negative cast to unsigned. One
     * of those silently makes every rate meaningless, which is worse than being
     * told the value was refused.
     */
    static constexpr int64_t kMaxAddPerCall = 1LL << 40;   // ~1.1e12

    static CounterRegistry& instance();

    /**
     * @brief Find or create the slot for @p name.
     *
     * Rejects a name that is empty, over kMaxNameLength, contains anything
     * outside [A-Za-z0-9._-], or would exceed kMaxCounters. Registering the
     * same name twice returns the same slot, including from several threads at
     * once.
     */
    SlotId registerCounter(const std::string& name);

    /**
     * @brief Stable address of a slot's value, or nullptr for kInvalidSlot.
     *
     * Handed to the caller once, at registration, so add() never touches the
     * container. Reading slots_.size() or indexing it without the mutex would
     * race a concurrent registration even though a deque keeps references to
     * existing elements valid.
     */
    std::atomic<uint64_t>* valueSlot(SlotId slot);

    /** @brief Raw slot value, including anything added before this runtime. */
    uint64_t rawValue(SlotId slot) const;

    /** @brief Value accrued since the current runtime took its baseline. */
    uint64_t valueSinceBaseline(SlotId slot) const;

    const std::string& name(SlotId slot) const;
    size_t counterCount() const;

    /**
     * @brief Start a session, baselining every slot at its current value.
     *
     * What keeps a permanent slot from leaking one session's ticks into the
     * next. Anything added while no session was active is excluded here.
     */
    void beginSession();
    /** @brief End the current session. Slots keep their values on purpose. */
    void endSession();
    /** @brief Whether a session is currently active. */
    bool sessionActive() const { return sessionActive_.load(std::memory_order_acquire); }

    /** @brief Test seam. Drops every slot, which a real process never does. */
    void resetForTesting();

private:
    CounterRegistry() = default;

    struct Slot {
        std::atomic<uint64_t> value{0};
        uint64_t baseline = 0;   // guarded by mu_
        std::string name;        // guarded by mu_, never changes once set
    };

    static bool nameIsValid(const std::string& name);
    /** @brief Truncate an untrusted name so a rejection cannot bloat the log. */
    static std::string forLog(const std::string& name);

    // Guards registration and the name/baseline fields. Never taken by add().
    mutable std::mutex mu_;
    // Deque, not vector: a handle holds a Slot* and a vector would invalidate
    // it on growth. Slots are never erased.
    std::deque<Slot> slots_;
    std::unordered_map<std::string, SlotId> byName_;
    // A flag, not a counter. Nothing needs to tell one session from another -
    // baselines already separate them - and a monotonic epoch that no reader
    // consults is a thing that drifts out of date without anyone noticing.
    std::atomic<bool> sessionActive_{false};
    // Rejections are logged once each. A name that fails validation usually
    // fails on every call, and tick() in a decode loop would otherwise write a
    // line per iteration.
    bool loggedInvalidName_ = false;
    bool loggedLimitReached_ = false;
};

}  // namespace gpufl::detail
