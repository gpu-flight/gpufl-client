#include "gpufl/core/dictionary_manager.hpp"

#include <sstream>

#include "gpufl/core/logger/logger.hpp"
#include "gpufl/core/model/model_utils.hpp"
#include "gpufl/core/model/serializable.hpp"

namespace gpufl {
namespace {

struct DictLine final : IJsonSerializable {
    std::string json;
    explicit DictLine(std::string j) : json(std::move(j)) {}
    std::string buildJson() const override { return json; }
    Channel channel() const override { return Channel::All; }
};

void appendDict(std::ostringstream& oss, const char* key,
                const std::unordered_map<std::string, uint32_t>& dirty,
                bool& firstField) {
    if (dirty.empty()) return;
    if (!firstField) oss << ',';
    firstField = false;
    oss << '"' << key << "\":{";
    bool first = true;
    for (const auto& [name, id] : dirty) {
        if (!first) oss << ',';
        first = false;
        oss << '"' << id << "\":\"" << model::jsonEscape(name) << '"';
    }
    oss << '}';
}

}  // namespace

void DictionaryManager::flushDictionary(Logger& logger,
                                         const std::string& session_id) {
    std::unordered_map<std::string, uint32_t> dk, ds, df, dm, dsf;
    {
        std::lock_guard lk(mu_);
        if (dirty_kernels_.empty() && dirty_scope_names_.empty() &&
            dirty_functions_.empty() && dirty_metrics_.empty() &&
            dirty_source_files_.empty()) {
            return;
        }
        dk  = std::move(dirty_kernels_);
        ds  = std::move(dirty_scope_names_);
        df  = std::move(dirty_functions_);
        dm  = std::move(dirty_metrics_);
        dsf = std::move(dirty_source_files_);
    }

    std::ostringstream oss;
    oss << "{\"version\":1,\"type\":\"dictionary_update\",\"session_id\":\""
        << model::jsonEscape(session_id) << '"';

    bool firstField = false;
    appendDict(oss, "kernel_dict",      dk,  firstField);
    appendDict(oss, "scope_name_dict",  ds,  firstField);
    appendDict(oss, "function_dict",    df,  firstField);
    appendDict(oss, "metric_dict",      dm,  firstField);
    appendDict(oss, "source_file_dict", dsf, firstField);

    oss << '}';
    logger.write(DictLine{oss.str()});
}

}  // namespace gpufl
