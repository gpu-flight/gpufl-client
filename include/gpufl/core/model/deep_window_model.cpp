#include "gpufl/core/model/deep_window_model.hpp"

#include <sstream>

#include "gpufl/core/model/model_utils.hpp"

namespace gpufl::model {

std::string DeepWindowModel::buildJson() const {
    std::ostringstream engines;
    engines << '[';
    for (size_t i = 0; i < e_.engines.size(); ++i) {
        if (i) engines << ',';
        engines << '"' << jsonEscape(e_.engines[i]) << '"';
    }
    engines << ']';

    // Omitted entirely when nothing triggered the window, so a manual window
    // does not carry an all-zero rule that reads like a real one.
    std::ostringstream trigger;
    if (e_.trigger.present) {
        trigger << ",\"trigger\":{"
                << "\"rule_id\":\""      << jsonEscape(e_.trigger.rule_id) << "\""
                << ",\"metric\":\""      << jsonEscape(e_.trigger.metric)  << "\""
                << ",\"op\":\""          << jsonEscape(e_.trigger.op)      << "\""
                << ",\"threshold\":"     << e_.trigger.threshold
                << ",\"rearm_threshold\":" << e_.trigger.rearm_threshold
                << ",\"observed\":"      << e_.trigger.observed
                << ",\"rate_window_ms\":" << e_.trigger.rate_window_ms
                << ",\"sustained_ms\":"  << e_.trigger.sustained_ms
                << ",\"first_true_ns\":" << e_.trigger.first_true_ns
                << ",\"fired_ns\":"      << e_.trigger.fired_ns
                << "}";
    }

    std::ostringstream oss;
    oss << "{\"type\":\"deep_window_event\""
        << ",\"pid\":"                     << e_.pid
        << ",\"app\":\""                   << jsonEscape(e_.app)          << "\""
        << ",\"session_id\":\""            << jsonEscape(e_.session_id)   << "\""
        << ",\"name\":\""                  << jsonEscape(e_.name)         << "\""
        << ",\"close_reason\":\""          << jsonEscape(e_.close_reason) << "\""
        << ",\"engines\":"                 << engines.str()
        << ",\"start_ns\":"                << e_.start_ns
        << ",\"end_ns\":"                  << e_.end_ns
        << ",\"duration_ns\":"             << e_.duration_ns
        << ",\"launches_covered\":"        << e_.launches_covered
        << ",\"requested_duration_ms\":"   << e_.requested_duration_ms
        << ",\"requested_max_launches\":"  << e_.requested_max_launches
        << trigger.str()
        << "}";
    return oss.str();
}

std::string DeepWindowRuleSummaryModel::buildJson() const {
    std::ostringstream oss;
    oss << "{\"type\":\"deep_window_rule_summary\""
        << ",\"pid\":"              << e_.pid
        << ",\"app\":\""            << jsonEscape(e_.app)          << "\""
        << ",\"session_id\":\""     << jsonEscape(e_.session_id)   << "\""
        << ",\"rule_id\":\""        << jsonEscape(e_.rule_id)      << "\""
        << ",\"expression\":\""     << jsonEscape(e_.expression)   << "\""
        << ",\"state\":\""          << jsonEscape(e_.state)        << "\""
        << ",\"outcome\":\""        << jsonEscape(e_.outcome)      << "\""
        << ",\"reason\":\""         << jsonEscape(e_.reason)       << "\""
        << ",\"metric_state\":\""   << jsonEscape(e_.metric_state) << "\""
        << ",\"samples_seen\":"     << e_.samples_seen
        << ",\"windows_opened\":"   << e_.windows_opened
        << ",\"truncated_samples\":" << e_.truncated_samples
        << ",\"state_sequence\":"   << e_.state_sequence
        << ",\"emitted_ns\":"       << e_.emitted_ns;
    // Written only when there is one. A null would have to be distinguished
    // from 0 downstream, and 0 is a legitimate reading for every metric here.
    if (e_.has_last_value) {
        oss << ",\"last_value\":"       << e_.last_value
            << ",\"last_observed_ns\":" << e_.last_observed_ns;
    }
    oss << "}";
    return oss.str();
}

}  // namespace gpufl::model
