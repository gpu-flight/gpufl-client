#include "gpufl/core/metric_id.hpp"

#include <algorithm>
#include <cstdio>
#include <limits>

#include "gpufl/core/counter_registry.hpp"

namespace gpufl::detail {
namespace {

constexpr const char* kCustomPrefix = "custom.";
constexpr const char* kRateSuffix   = "_rate";

bool startsWith(const std::string& s, const char* prefix) {
    const size_t n = std::char_traits<char>::length(prefix);
    return s.size() >= n && s.compare(0, n, prefix) == 0;
}

bool endsWith(const std::string& s, const char* suffix) {
    const size_t n = std::char_traits<char>::length(suffix);
    return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

/**
 * Same charset the counter registry enforces. Shared on purpose: a name the
 * registry would refuse must not parse into a rule that then waits forever for
 * a counter that can never exist.
 */
bool customNameCharsetOk(const std::string& name) {
    for (const char c : name) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                        (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-';
        if (!ok) return false;
    }
    return true;
}

/** Parse `gpu[N].<field>`. Returns false if the shape does not match at all. */
bool parseDeviceMetric(const std::string& text, MetricParseResult* out) {
    if (!startsWith(text, "gpu[")) return false;

    const size_t close = text.find(']');
    if (close == std::string::npos || close == 4 || text.size() <= close + 1 ||
        text[close + 1] != '.') {
        out->error = MetricParseError::MalformedDeviceIndex;
        return true;
    }

    int64_t index = 0;
    for (size_t i = 4; i < close; ++i) {
        const char c = text[i];
        if (c < '0' || c > '9') {
            out->error = MetricParseError::MalformedDeviceIndex;
            return true;
        }
        index = index * 10 + (c - '0');
        if (index > 4095) {   // no plausible host, and keeps the parse bounded
            out->error = MetricParseError::MalformedDeviceIndex;
            return true;
        }
    }

    const std::string field = text.substr(close + 2);
    if (field == "util_pct") {
        out->id.kind = MetricKind::GpuUtilPct;
    } else if (field == "power_mw") {
        out->id.kind = MetricKind::GpuPowerMw;
    } else if (field == "sm_clock_mhz") {
        out->id.kind = MetricKind::GpuSmClockMhz;
    } else {
        out->error = MetricParseError::UnknownBuiltinMetric;
        return true;
    }

    out->id.device_index = static_cast<int>(index);
    // Canonicalised so gpu[00] and gpu[0] cannot hash to two different rule ids.
    char buf[64];
    std::snprintf(buf, sizeof(buf), "gpu[%d].%s", out->id.device_index, field.c_str());
    out->id.canonical = buf;
    return true;
}

}  // namespace

MetricShape MetricId::shape() const {
    switch (kind) {
        case MetricKind::KernelLaunchRate:
        case MetricKind::CustomRate:
            return MetricShape::Rate;
        case MetricKind::RecentKernelMs:
            return MetricShape::Percentile;
        case MetricKind::GpuUtilPct:
        case MetricKind::GpuPowerMw:
        case MetricKind::GpuSmClockMhz:
            break;
    }
    return MetricShape::Gauge;
}

const char* toString(MetricKind k) {
    switch (k) {
        case MetricKind::GpuUtilPct:       return "gpu_util_pct";
        case MetricKind::GpuPowerMw:       return "gpu_power_mw";
        case MetricKind::GpuSmClockMhz:    return "gpu_sm_clock_mhz";
        case MetricKind::KernelLaunchRate: return "kernel_launch_rate";
        case MetricKind::RecentKernelMs:   return "recent_kernel_ms";
        case MetricKind::CustomRate:       return "custom_rate";
    }
    return "unknown";
}

const char* toString(MetricParseError e) {
    switch (e) {
        case MetricParseError::None:                  return "ok";
        case MetricParseError::Empty:                 return "empty_metric_name";
        case MetricParseError::UnknownBuiltinMetric:  return "unknown_builtin_metric";
        case MetricParseError::MissingCustomPrefix:   return "missing_custom_prefix";
        case MetricParseError::MalformedDeviceIndex:  return "malformed_device_index";
        case MetricParseError::MalformedCustomMetric: return "malformed_custom_metric";
        case MetricParseError::CustomNameTooLong:     return "custom_name_too_long";
        case MetricParseError::CustomNameCharset:     return "custom_name_charset";
    }
    return "unknown";
}

MetricParseResult parseMetric(const std::string& text) {
    MetricParseResult out;

    if (text.empty()) {
        out.error = MetricParseError::Empty;
        return out;
    }

    if (parseDeviceMetric(text, &out)) return out;

    if (text == "kernel_launch_rate") {
        out.id.kind = MetricKind::KernelLaunchRate;
        out.id.canonical = text;
        return out;
    }
    if (text == "recent_kernel_ms") {
        out.id.kind = MetricKind::RecentKernelMs;
        out.id.canonical = text;
        return out;
    }

    if (startsWith(text, kCustomPrefix)) {
        const std::string body =
            text.substr(std::char_traits<char>::length(kCustomPrefix));
        if (!endsWith(body, kRateSuffix)) {
            out.error = MetricParseError::MalformedCustomMetric;
            return out;
        }
        const std::string name =
            body.substr(0, body.size() - std::char_traits<char>::length(kRateSuffix));
        if (name.empty()) {
            out.error = MetricParseError::MalformedCustomMetric;
            return out;
        }
        if (name.size() > CounterRegistry::kMaxNameLength) {
            out.error = MetricParseError::CustomNameTooLong;
            return out;
        }
        if (!customNameCharsetOk(name)) {
            out.error = MetricParseError::CustomNameCharset;
            return out;
        }
        out.id.kind = MetricKind::CustomRate;
        out.id.custom_name = name;
        out.id.canonical = std::string(kCustomPrefix) + name + kRateSuffix;
        return out;
    }

    // Anything left is either a misspelled built-in or a custom counter written
    // without its prefix. Both are config errors, and both are caught here
    // rather than at shutdown, which is the entire reason the prefix exists.
    out.error = MetricParseError::MissingCustomPrefix;
    return out;
}

int64_t MetricWindowConfig::bucketIntervalMs() const {
    if (rate_window_ms <= 0) return 10;
    return std::max<int64_t>(10, std::min<int64_t>(100, rate_window_ms / 10));
}

int64_t MetricWindowConfig::bucketCount() const {
    const int64_t bucket = bucketIntervalMs();
    if (rate_window_ms <= 0) return 1;
    return std::max<int64_t>(1, (rate_window_ms + bucket - 1) / bucket);
}

const char* toString(ConfigError e) {
    switch (e) {
        case ConfigError::None:                  return "ok";
        case ConfigError::RateWindowNotPositive: return "rate_window_not_positive";
        case ConfigError::RateWindowTooLarge:    return "rate_window_too_large";
        case ConfigError::SustainedNegative:     return "sustained_negative";
        case ConfigError::StaleAfterNotPositive: return "stale_after_not_positive";
        case ConfigError::StaleBeforeEvidence:   return "stale_before_evidence";
    }
    return "unknown";
}

ConfigError validate(const MetricWindowConfig& cfg) {
    if (cfg.rate_window_ms <= 0) return ConfigError::RateWindowNotPositive;
    if (cfg.rate_window_ms > MetricWindowConfig::kMaxRateWindowMs) {
        return ConfigError::RateWindowTooLarge;
    }
    if (cfg.sustained_ms < 0) return ConfigError::SustainedNegative;
    if (cfg.stale_after_ms <= 0) return ConfigError::StaleAfterNotPositive;

    // Overflow-safe: the sum is the point of the check, so it must not be the
    // thing that breaks it.
    constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
    int64_t need = cfg.rate_window_ms;
    if (cfg.sustained_ms > kMax - need) return ConfigError::StaleBeforeEvidence;
    need += cfg.sustained_ms;
    const int64_t bucket = cfg.bucketIntervalMs();
    if (bucket > kMax - need) return ConfigError::StaleBeforeEvidence;
    need += bucket;

    if (cfg.stale_after_ms < need) return ConfigError::StaleBeforeEvidence;
    return ConfigError::None;
}

std::string explain(const MetricWindowConfig& cfg, const ConfigError e) {
    char buf[320];
    switch (e) {
        case ConfigError::None:
            return "ok";
        case ConfigError::RateWindowNotPositive:
            std::snprintf(buf, sizeof(buf),
                          "rate window must be > 0 (got %lldms)",
                          static_cast<long long>(cfg.rate_window_ms));
            break;
        case ConfigError::RateWindowTooLarge:
            std::snprintf(buf, sizeof(buf),
                          "rate window %lldms exceeds the %lldms limit",
                          static_cast<long long>(cfg.rate_window_ms),
                          static_cast<long long>(MetricWindowConfig::kMaxRateWindowMs));
            break;
        case ConfigError::SustainedNegative:
            std::snprintf(buf, sizeof(buf),
                          "sustained must be >= 0 (got %lldms)",
                          static_cast<long long>(cfg.sustained_ms));
            break;
        case ConfigError::StaleAfterNotPositive:
            std::snprintf(buf, sizeof(buf),
                          "stale-after must be > 0 (got %lldms)",
                          static_cast<long long>(cfg.stale_after_ms));
            break;
        case ConfigError::StaleBeforeEvidence:
            // Spell out the arithmetic: the fields are individually reasonable
            // and the reader has no way to see why the combination cannot fire.
            std::snprintf(
                buf, sizeof(buf),
                "stale-after %lldms is too short to ever fire: needs >= rate "
                "window %lld + sustained %lld + bucket %lld = %lldms",
                static_cast<long long>(cfg.stale_after_ms),
                static_cast<long long>(cfg.rate_window_ms),
                static_cast<long long>(cfg.sustained_ms),
                static_cast<long long>(cfg.bucketIntervalMs()),
                static_cast<long long>(cfg.rate_window_ms + cfg.sustained_ms +
                                       cfg.bucketIntervalMs()));
            break;
    }
    return buf;
}

}  // namespace gpufl::detail
