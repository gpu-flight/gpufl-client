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
struct DeepWindowRuleSummaryModel final : IJsonSerializable {
    explicit DeepWindowRuleSummaryModel(const DeepWindowRuleSummaryEvent& e)
        : e_(e) {}
    std::string buildJson() const override;
    Channel channel() const override { return Channel::Scope; }

   private:
    const DeepWindowRuleSummaryEvent& e_;
};

}  // namespace gpufl::model
