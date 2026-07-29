#pragma once

#include <atomic>
#include <cstdint>
#include <string>

struct gpufl_counter_provider_v1;

namespace gpufl::detail {

/**
 * @brief Routes NVTX Counters extension samples into the counter registry.
 *
 * Lets an application drive a conditional deep window through the STANDARD
 * NVIDIA API instead of gpufl's own: `nvtxCounterRegister` + `nvtxCounterSample*`
 * reach us as direct calls into the injection library (the launcher already
 * owns NVTX_INJECTION64_PATH), so nothing goes through a CUPTI activity
 * buffer and the per-sample cost stays one relaxed atomic.
 *
 * Deliberately free of NVTX types so it is testable without an injected
 * process; inject_entry.cpp does the NVTX-shaped translation.
 */
class NvtxCounterBridge {
public:
    /** How the application says its samples should be read. */
    enum class ValueType {
        /// No counter semantics were attached. NOT assumed to be anything.
        Unspecified,
        Absolute,
        Delta,
        DeltaSinceStart,
    };

    /** Why a registration was refused, for the capability report. */
    enum class RegisterStatus {
        Accepted,
        /// Value type absent or not Delta. Reading an absolute series as
        /// deltas would sum a monotonic curve into nonsense, so it is refused
        /// rather than guessed.
        UnsupportedValueType,
        /// An application-chosen (static) ID. Unique only WITHIN its domain,
        /// so honouring it needs a (domain, id) table this build does not
        /// have; two domains reusing one id would share a slot and silently
        /// add their rates together.
        StaticIdUnsupported,
        /// The name could not be turned into a metric name, or two DIFFERENT
        /// NVTX names canonicalise to the same one - merging those would
        /// silently add two unrelated workloads into a single rate.
        BadName,
        /// The registry is full, or this bridge's table is.
        LimitReached,
    };

    struct RegisterResult {
        /// Non-zero only when Accepted. Encodes the table index, so a sample
        /// resolves in constant time with no lookup and no lock.
        uint64_t       id = 0;
        RegisterStatus status = RegisterStatus::BadName;
        /// The `custom.<name>_rate` a rule would be written against. Reported
        /// so the link between the NVTX name and the CLI name is visible
        /// rather than something the user has to infer.
        std::string    metric;
    };

    /** Base of the IDs this bridge hands out. Matches NVTX_COUNTER_ID_DYNAMIC_START. */
    static constexpr uint64_t kDynamicIdBase = static_cast<uint64_t>(1) << 32;
    /** Bound on tracked counters. The registry has its own, lower, limit. */
    static constexpr size_t kMaxTracked = 64;

    static NvtxCounterBridge& instance();

    /**
     * @brief Bind an NVTX counter to a registry slot.
     *
     * @param domain_name Domain name, or empty. Prefixed onto the counter name
     *        so two domains using the same counter name stay distinct.
     * @param requested_id The application's counterId; 0 means "assign one".
     */
    RegisterResult registerCounter(const std::string& domain_name,
                                   const std::string& counter_name,
                                   uint64_t requested_id, ValueType type);

    /**
     * @brief A delta sample. Ignores anything this bridge did not hand out.
     *
     * Lock-free: the entry holds the provider that issued its handle, taken
     * once at registration, so the sample path never resolves the provider -
     * resolving it takes a mutex, and a mutex per sample in a decode loop
     * changes the throughput the rule is measuring.
     *
     * Negative deltas are counted and dropped: the registry is an unsigned
     * monotonic accumulator, and wrapping one backwards would turn the next
     * rate into an astronomically large number rather than a small one.
     */
    void sampleDelta(uint64_t id, int64_t value);

    /**
     * @brief A sample that carries no value.
     *
     * `zero` and `unchanged` both mean the delta is 0, which for a rate is
     * exactly "do nothing" - NOT an event. `unavailable` means the application
     * failed to read its own counter: the true delta is UNKNOWN, not zero, and
     * a rate computed over the gap would sag below any threshold and fire the
     * rule on a workload that never slowed down. It is recorded per counter so
     * the metric layer can discard the affected window.
     */
    void sampleNoValue(uint64_t id, uint8_t reason);

    /**
     * @brief Failed reads recorded for the counter behind @p canonical_name.
     *
     * Polled by MetricSource at bucket close: when this advances, the current
     * rate window contains a gap of unknown size and is discarded rather than
     * evaluated. 0 for names this bridge does not track, so non-NVTX custom
     * counters are unaffected.
     */
    uint64_t unavailableCountFor(const std::string& canonical_name) const;

    /**
     * @brief One session's data-quality tallies, each with ONE meaning.
     *
     * Kept apart on purpose: "registration was refused" is a configuration
     * problem, "an id we never issued" is an application bug, "unavailable"
     * is a read the application itself failed, and a negative delta is a
     * value that cannot be applied. Folding any two together produces a
     * number nobody can act on.
     */
    struct QualitySnapshot {
        uint64_t registration_rejected = 0;
        uint64_t unknown_id_samples = 0;
        uint64_t unavailable_samples = 0;
        uint64_t negative_delta_samples = 0;
        /**
         * Valid samples this session: accepted deltas (a zero delta is a real
         * observation) plus ZERO/UNCHANGED no-value samples. The denominator
         * that makes an all-zero failure row mean something - "0 failures out
         * of 12,000 samples" is a clean bill; "0 out of 0" is a session where
         * nothing was watched, and without this field the two are the same
         * row.
         */
        uint64_t samples_observed = 0;

        /// Failures only, on purpose: samples_observed is the denominator,
        /// not a problem.
        bool any() const {
            return registration_rejected != 0 || unknown_id_samples != 0 ||
                   unavailable_samples != 0 || negative_delta_samples != 0;
        }
    };

    /**
     * @brief This session's tallies, advancing the session baseline.
     *
     * The tallies are process-lifetime (like the counter slots), and an
     * embedded host re-initialises in one process - exporting the raw values
     * would re-report session one's problems as session two's. Each call
     * returns what accrued since the previous call; the FIRST call returns
     * everything since process start, which is where pre-init NVTX
     * registrations belong: they happened during this run's startup and no
     * other session can report them.
     *
     * Single consumer: the session-summary emit at shutdown.
     */
    QualitySnapshot takeSessionSnapshot();

    /** @brief Process-lifetime registrations refused, any reason. */
    uint64_t registrationRejected() const;
    /** @brief Samples whose id this bridge never issued. */
    uint64_t unknownIdSamples() const;
    /** @brief Samples the application itself could not read, all counters. */
    uint64_t unavailableSamples() const;
    /** @brief Samples dropped for carrying a negative delta. */
    uint64_t negativeSamples() const;
    /** @brief Counters bound so far. */
    size_t   trackedCount() const;

    /** @brief Test seam. A real process never drops these bindings. */
    void resetForTesting();

    /**
     * @brief NVTX name to registry name: `<domain>.<counter>`, canonicalised.
     *
     * Counter names reach a rule as `custom.<name>_rate`, whose charset is
     * [A-Za-z0-9._-]; NVTX names are free-form. Everything outside the charset
     * becomes '_'. That is not injective - and the domain joins with '.', so
     * ("a", "b.c") and ("a.b", "c") also meet - which is why the entry keeps
     * the ORIGINAL pair and a collision is refused at registration instead of
     * silently merging two counters into one slot.
     */
    static std::string canonicalName(const std::string& domain_name,
                                     const std::string& counter_name);

private:
    NvtxCounterBridge() = default;

    /// The actual registration; the public wrapper counts refusals in one
    /// place so no refusal path can forget to.
    RegisterResult registerCounterInner(const std::string& domain_name,
                                        const std::string& counter_name,
                                        uint64_t requested_id, ValueType type);

    struct Entry {
        /// The provider that issued `handle`. A handle is a pointer into the
        /// issuing provider's registry, so the pair must never be split.
        const gpufl_counter_provider_v1* provider = nullptr;
        void*       handle = nullptr;    // gpufl_counter_handle
        std::string name;                // canonical registry name
        /// The NVTX names as the application wrote them. What tells an
        /// idempotent re-registration apart from a canonicalisation collision.
        std::string domain_original;
        std::string counter_original;
        /// Failed reads (SAMPLE_UNAVAILABLE) for this counter.
        std::atomic<uint64_t> unavailable{0};
    };

    Entry entries_[kMaxTracked];
    /**
     * Published with release AFTER an entry's fields are written, and read
     * with acquire on the sample path. The mutex only serialises writers; a
     * plain size_t here was a data race with the lock-free readers, and the
     * race window is exactly a counter's first samples.
     */
    std::atomic<size_t> count_{0};
};

}  // namespace gpufl::detail
