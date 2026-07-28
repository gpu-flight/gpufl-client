#include "gpufl/core/nvtx_counters.hpp"

#include <atomic>
#include <mutex>

#include "gpufl/core/counter_provider.hpp"
#include "gpufl/core/counter_registry.hpp"
#include "gpufl/core/debug_logger.hpp"

namespace gpufl::detail {
namespace {

// NVTX_COUNTER_SAMPLE_* from nvToolsExtCounters.h. Duplicated as plain
// constants so this file stays free of NVTX headers and can be unit-tested
// without them; the values are part of a released ABI and pinned by a test.
constexpr uint8_t kSampleZero        = 0;
constexpr uint8_t kSampleUnchanged   = 1;
constexpr uint8_t kSampleUnavailable = 2;

// Serialises registration only. The sample path never takes it.
std::mutex g_mu;

std::atomic<uint64_t> g_registration_rejected{0};
std::atomic<uint64_t> g_unknown_id_samples{0};
std::atomic<uint64_t> g_unavailable_samples{0};
std::atomic<uint64_t> g_negative_samples{0};
std::atomic<uint64_t> g_samples_observed{0};

// Where the previous session's report ended. Guarded by g_mu; the snapshot is
// taken once per session at shutdown, never on a hot path.
uint64_t g_base_registration_rejected = 0;
uint64_t g_base_unknown_id = 0;
uint64_t g_base_unavailable = 0;
uint64_t g_base_negative = 0;
uint64_t g_base_observed = 0;

bool CharsetOk(const char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-';
}

// Logged once each: a rejected registration usually repeats, and a sample the
// bridge does not know about can arrive on every iteration of a decode loop.
bool g_logged_static_id = false;
bool g_logged_value_type = false;
bool g_logged_bad_name = false;
bool g_logged_collision = false;
bool g_logged_limit = false;
std::atomic<bool> g_logged_unknown_id{false};
std::atomic<bool> g_logged_negative{false};

}  // namespace

NvtxCounterBridge& NvtxCounterBridge::instance() {
    static auto* bridge = new NvtxCounterBridge();
    return *bridge;
}

std::string NvtxCounterBridge::canonicalName(const std::string& domain_name,
                                             const std::string& counter_name) {
    std::string joined;
    if (!domain_name.empty()) {
        joined = domain_name;
        joined += '.';
    }
    joined += counter_name;

    std::string out;
    out.reserve(joined.size());
    for (const char c : joined) out.push_back(CharsetOk(c) ? c : '_');

    // A name of nothing but separators would produce a metric like
    // `custom.___rate`, which names no counter anybody could have meant.
    bool has_alnum = false;
    for (const char c : out) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9')) {
            has_alnum = true;
            break;
        }
    }
    if (!has_alnum) return {};
    if (out.size() > CounterRegistry::kMaxNameLength) return {};
    return out;
}

NvtxCounterBridge::RegisterResult NvtxCounterBridge::registerCounter(
    const std::string& domain_name, const std::string& counter_name,
    const uint64_t requested_id, const ValueType type) {
    RegisterResult result = registerCounterInner(domain_name, counter_name,
                                                 requested_id, type);
    if (result.status != RegisterStatus::Accepted) {
        g_registration_rejected.fetch_add(1, std::memory_order_relaxed);
    }
    return result;
}

NvtxCounterBridge::RegisterResult NvtxCounterBridge::registerCounterInner(
    const std::string& domain_name, const std::string& counter_name,
    const uint64_t requested_id, const ValueType type) {
    RegisterResult result;

    // An application-chosen ID is unique only inside its domain, so binding it
    // to a slot needs a (domain, id) table. Without one, two domains that both
    // use id 123 would share a slot and add their rates together - a wrong
    // number that looks like a real one. Refused until that table exists.
    if (requested_id != 0) {
        result.status = RegisterStatus::StaticIdUnsupported;
        std::lock_guard lk(g_mu);
        if (!g_logged_static_id) {
            g_logged_static_id = true;
            GFL_LOG_WARN("[NvtxCounters] counter '", counter_name,
                         "' uses an application-assigned id; gpufl only "
                         "tracks tool-assigned ids (pass "
                         "NVTX_COUNTER_ID_NONE). This counter is not "
                         "available to rules.");
        }
        return result;
    }

    // The only shape the registry can represent. It accumulates unsigned, so
    // an absolute series read as deltas would sum a monotonic curve, and a
    // counter with no semantics at all is not enough to tell the two apart.
    if (type != ValueType::Delta) {
        result.status = RegisterStatus::UnsupportedValueType;
        std::lock_guard lk(g_mu);
        if (!g_logged_value_type) {
            g_logged_value_type = true;
            GFL_LOG_WARN("[NvtxCounters] counter '", counter_name,
                         "' is not a DELTA counter (attach "
                         "nvtxSemanticsCounter_t with "
                         "NVTX_COUNTER_FLAG_VALUETYPE_DELTA). Absolute and "
                         "unspecified counters are not converted, because "
                         "reading one as deltas produces a plausible wrong "
                         "rate rather than an obvious failure.");
        }
        return result;
    }

    const std::string name = canonicalName(domain_name, counter_name);
    if (name.empty()) {
        result.status = RegisterStatus::BadName;
        std::lock_guard lk(g_mu);
        if (!g_logged_bad_name) {
            g_logged_bad_name = true;
            GFL_LOG_WARN("[NvtxCounters] counter name has nothing usable in "
                         "it after canonicalisation; rules address counters "
                         "as custom.<name>_rate over [A-Za-z0-9._-]");
        }
        return result;
    }

    std::lock_guard lk(g_mu);
    const size_t count = count_.load(std::memory_order_relaxed);

    // Canonicalisation maps every out-of-charset byte to '_' and joins the
    // domain with '.', so it is not injective: "a b" and "a/b" meet at "a_b",
    // and ("a", "b.c") meets ("a.b", "c"). The ORIGINAL pair decides which
    // case this is: the same pair again is an idempotent re-registration; a
    // different pair is two counters, and merging them would silently add two
    // unrelated workloads into a single rate.
    for (size_t i = 0; i < count; ++i) {
        if (entries_[i].name != name) continue;
        if (entries_[i].domain_original == domain_name &&
            entries_[i].counter_original == counter_name) {
            result.id = kDynamicIdBase + i;
            result.status = RegisterStatus::Accepted;
            result.metric = "custom." + name + "_rate";
            return result;
        }
        result.status = RegisterStatus::BadName;
        if (!g_logged_collision) {
            g_logged_collision = true;
            GFL_LOG_WARN("[NvtxCounters] counter '", counter_name,
                         "' in domain '", domain_name, "' canonicalises to '",
                         name, "', which already belongs to counter '",
                         entries_[i].counter_original, "' in domain '",
                         entries_[i].domain_original,
                         "'. Refused rather than merged; rename one of them.");
        }
        return result;
    }

    if (count >= kMaxTracked) {
        result.status = RegisterStatus::LimitReached;
        if (!g_logged_limit) {
            g_logged_limit = true;
            GFL_LOG_WARN("[NvtxCounters] tracking limit (", kMaxTracked,
                         ") reached; '", name, "' is not available to rules");
        }
        return result;
    }

    // Through the provider: the evaluator reads whatever
    // ActiveCounterProvider() resolves, and registering into this module's own
    // registry instead leaves the counter invisible to it wherever a shared
    // runtime is bound - proven by mutation on the 3090, where exactly this
    // bypass turned a firing rule into custom_metric_never_registered.
    const gpufl_counter_provider_v1* provider = ActiveCounterProvider();
    gpufl_counter_handle handle =
        provider->register_counter(name.c_str(), name.size());
    if (handle == nullptr) {
        result.status = RegisterStatus::LimitReached;
        if (!g_logged_limit) {
            g_logged_limit = true;
            GFL_LOG_WARN("[NvtxCounters] the counter registry refused '", name,
                         "'; it is not available to rules");
        }
        return result;
    }

    // Fields first, count last with release: the sample path reads count_
    // with acquire and no lock, so the store below is what publishes the
    // entry. The provider travels with the handle it issued - a handle is a
    // pointer into that provider's registry, and the pair must never split.
    entries_[count].provider = provider;
    entries_[count].handle = handle;
    entries_[count].name = name;
    entries_[count].domain_original = domain_name;
    entries_[count].counter_original = counter_name;
    entries_[count].unavailable.store(0, std::memory_order_relaxed);
    count_.store(count + 1, std::memory_order_release);

    result.id = kDynamicIdBase + count;
    result.status = RegisterStatus::Accepted;
    result.metric = "custom." + name + "_rate";
    // Printed because nothing else connects the NVTX name the application
    // wrote to the metric name a rule has to be written against.
    GFL_LOG_INFO("[NvtxCounters] counter '", counter_name, "' -> rule metric ",
                 result.metric);
    return result;
}

void NvtxCounterBridge::sampleDelta(const uint64_t id, const int64_t value) {
    if (id < kDynamicIdBase) {
        g_unknown_id_samples.fetch_add(1, std::memory_order_relaxed);
        if (!g_logged_unknown_id.exchange(true, std::memory_order_relaxed)) {
            GFL_LOG_WARN("[NvtxCounters] sample for a counter this build did "
                         "not assign an id to; it is not reaching any rule");
        }
        return;
    }
    const uint64_t index = id - kDynamicIdBase;
    if (index >= count_.load(std::memory_order_acquire)) {
        g_unknown_id_samples.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (value < 0) {
        // The registry accumulates unsigned. Wrapping it backwards turns the
        // very next rate into an enormous number instead of a small one, so a
        // negative delta is dropped and counted rather than applied.
        g_negative_samples.fetch_add(1, std::memory_order_relaxed);
        if (!g_logged_negative.exchange(true, std::memory_order_relaxed)) {
            GFL_LOG_WARN("[NvtxCounters] negative delta dropped; a DELTA "
                         "counter feeding a rate must not go backwards");
        }
        return;
    }
    g_samples_observed.fetch_add(1, std::memory_order_relaxed);
    if (value == 0) return;   // a real observation of "no traffic"; nothing to add

    const Entry& entry = entries_[index];
    entry.provider->add(entry.handle, static_cast<uint64_t>(value));
}

void NvtxCounterBridge::sampleNoValue(const uint64_t id, const uint8_t reason) {
    if (id < kDynamicIdBase ||
        id - kDynamicIdBase >= count_.load(std::memory_order_acquire)) {
        g_unknown_id_samples.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    switch (reason) {
        case kSampleZero:
        case kSampleUnchanged:
            // Both say the delta is zero. For a rate that is precisely "do
            // nothing" - adding 1 here would invent traffic out of a sample
            // that exists to report the absence of it. Still a valid
            // OBSERVATION, so it counts toward the denominator.
            g_samples_observed.fetch_add(1, std::memory_order_relaxed);
            return;
        case kSampleUnavailable:
        default:
            // The application could not read its own counter, so the true
            // delta over this stretch is unknown - NOT zero. Recorded per
            // counter; MetricSource discards the rate window it lands in,
            // because a rate computed over the gap sags below any threshold
            // and fires the rule on a workload that never slowed down.
            entries_[id - kDynamicIdBase].unavailable.fetch_add(
                1, std::memory_order_relaxed);
            g_unavailable_samples.fetch_add(1, std::memory_order_relaxed);
            return;
    }
}

uint64_t NvtxCounterBridge::unavailableCountFor(
    const std::string& canonical_name) const {
    // Lock-free on purpose: names are immutable once their entry is published,
    // and count_ is acquire-loaded, so this is safe against a concurrent
    // registration of a LATER entry.
    const size_t count = count_.load(std::memory_order_acquire);
    for (size_t i = 0; i < count; ++i) {
        if (entries_[i].name == canonical_name) {
            return entries_[i].unavailable.load(std::memory_order_relaxed);
        }
    }
    return 0;
}

NvtxCounterBridge::QualitySnapshot NvtxCounterBridge::takeSessionSnapshot() {
    std::lock_guard lk(g_mu);
    QualitySnapshot snap;
    const uint64_t rr = g_registration_rejected.load(std::memory_order_relaxed);
    const uint64_t ui = g_unknown_id_samples.load(std::memory_order_relaxed);
    const uint64_t ua = g_unavailable_samples.load(std::memory_order_relaxed);
    const uint64_t ng = g_negative_samples.load(std::memory_order_relaxed);
    const uint64_t ob = g_samples_observed.load(std::memory_order_relaxed);
    snap.registration_rejected = rr - g_base_registration_rejected;
    snap.unknown_id_samples = ui - g_base_unknown_id;
    snap.unavailable_samples = ua - g_base_unavailable;
    snap.negative_delta_samples = ng - g_base_negative;
    snap.samples_observed = ob - g_base_observed;
    g_base_registration_rejected = rr;
    g_base_unknown_id = ui;
    g_base_unavailable = ua;
    g_base_negative = ng;
    g_base_observed = ob;
    return snap;
}

uint64_t NvtxCounterBridge::registrationRejected() const {
    return g_registration_rejected.load(std::memory_order_relaxed);
}

uint64_t NvtxCounterBridge::unknownIdSamples() const {
    return g_unknown_id_samples.load(std::memory_order_relaxed);
}

uint64_t NvtxCounterBridge::unavailableSamples() const {
    return g_unavailable_samples.load(std::memory_order_relaxed);
}

uint64_t NvtxCounterBridge::negativeSamples() const {
    return g_negative_samples.load(std::memory_order_relaxed);
}

size_t NvtxCounterBridge::trackedCount() const {
    return count_.load(std::memory_order_acquire);
}

void NvtxCounterBridge::resetForTesting() {
    std::lock_guard lk(g_mu);
    const size_t count = count_.load(std::memory_order_relaxed);
    for (size_t i = 0; i < count; ++i) {
        entries_[i].provider = nullptr;
        entries_[i].handle = nullptr;
        entries_[i].name.clear();
        entries_[i].domain_original.clear();
        entries_[i].counter_original.clear();
        entries_[i].unavailable.store(0, std::memory_order_relaxed);
    }
    count_.store(0, std::memory_order_release);
    g_registration_rejected.store(0, std::memory_order_relaxed);
    g_unknown_id_samples.store(0, std::memory_order_relaxed);
    g_unavailable_samples.store(0, std::memory_order_relaxed);
    g_negative_samples.store(0, std::memory_order_relaxed);
    g_base_registration_rejected = 0;
    g_base_unknown_id = 0;
    g_base_unavailable = 0;
    g_base_negative = 0;
    g_samples_observed.store(0, std::memory_order_relaxed);
    g_base_observed = 0;
    g_logged_static_id = false;
    g_logged_value_type = false;
    g_logged_bad_name = false;
    g_logged_collision = false;
    g_logged_limit = false;
    g_logged_unknown_id.store(false, std::memory_order_relaxed);
    g_logged_negative.store(false, std::memory_order_relaxed);
}

}  // namespace gpufl::detail
