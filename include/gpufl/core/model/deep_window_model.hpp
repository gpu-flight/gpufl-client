#pragma once

#include "gpufl/core/events.hpp"
#include "gpufl/core/model/serializable.hpp"

namespace gpufl::model {

/**
 * JSON serializer for DeepWindowEvent. Emitted to the Scope channel: a
 * deep window is a named region with start/end timestamps, so it belongs
 * with scope_event_batch and nvtx_marker_event rather than in the
 * per-device event stream.
 */
struct DeepWindowModel final : IJsonSerializable {
    explicit DeepWindowModel(const DeepWindowEvent& e) : e_(e) {}
    std::string buildJson() const override;
    Channel channel() const override { return Channel::Scope; }

   private:
    const DeepWindowEvent& e_;
};

/**
 * JSON serializer for the conditional-rule summary. Same channel as the window
 * itself: the two are read together, and a summary that arrived on a different
 * channel could be ingested after the windows it explains.
 */
// Lives beside the rule-summary model because its consumer is the same
// conditional-window feature; the event itself is counter data quality.
struct CounterDataQualitySummaryModel final : IJsonSerializable {
    explicit CounterDataQualitySummaryModel(const CounterDataQualitySummaryEvent& e)
        : e_(e) {}
    std::string buildJson() const override;
    // Scope channel, like the rule summary it is read next to.
    Channel channel() const override { return Channel::Scope; }

   private:
    const CounterDataQualitySummaryEvent& e_;
};

struct DeepWindowRuleSummaryModel final : IJsonSerializable {
    explicit DeepWindowRuleSummaryModel(const DeepWindowRuleSummaryEvent& e)
        : e_(e) {}
    std::string buildJson() const override;
    Channel channel() const override { return Channel::Scope; }

   private:
    const DeepWindowRuleSummaryEvent& e_;
};

}  // namespace gpufl::model
