#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

namespace gpufl {

class Logger;

class DictionaryManager {
   public:
    uint32_t internKernel(const std::string& name) {
        std::lock_guard lk(mu_);
        if (const auto it = kernel_dict_.find(name); it != kernel_dict_.end())
            return it->second;
        const uint32_t id = next_kernel_id_++;
        kernel_dict_[name] = id;
        dirty_kernels_[name] = id;
        return id;
    }

    uint32_t internScopeName(const std::string& name) {
        std::lock_guard lk(mu_);
        if (const auto it = scope_name_dict_.find(name);
            it != scope_name_dict_.end())
            return it->second;
        const uint32_t id = next_scope_name_id_++;
        scope_name_dict_[name] = id;
        dirty_scope_names_[name] = id;
        return id;
    }

    uint32_t internFunction(const std::string& name) {
        std::lock_guard lk(mu_);
        if (const auto it = function_dict_.find(name);
            it != function_dict_.end())
            return it->second;
        const uint32_t id = next_function_id_++;
        function_dict_[name] = id;
        dirty_functions_[name] = id;
        return id;
    }

    uint32_t internMetric(const std::string& name) {
        if (name.empty()) return 0;
        std::lock_guard lk(mu_);
        if (const auto it = metric_dict_.find(name); it != metric_dict_.end())
            return it->second;
        const uint32_t id = next_metric_id_++;
        metric_dict_[name] = id;
        dirty_metrics_[name] = id;
        return id;
    }

    // Emits a dictionary_update JSON line to Channel::All for any new entries
    // accumulated since the last call.  No-op if nothing is dirty.
    void flushDictionary(Logger& logger, const std::string& session_id);

    void reset() {
        std::lock_guard lk(mu_);
        kernel_dict_.clear();
        dirty_kernels_.clear();
        next_kernel_id_ = 1;
        scope_name_dict_.clear();
        dirty_scope_names_.clear();
        next_scope_name_id_ = 1;
        function_dict_.clear();
        dirty_functions_.clear();
        next_function_id_ = 1;
        metric_dict_.clear();
        dirty_metrics_.clear();
        next_metric_id_ = 1;
    }

   private:
    std::mutex mu_;

    std::unordered_map<std::string, uint32_t> kernel_dict_;
    std::unordered_map<std::string, uint32_t> dirty_kernels_;
    uint32_t next_kernel_id_ = 1;

    std::unordered_map<std::string, uint32_t> scope_name_dict_;
    std::unordered_map<std::string, uint32_t> dirty_scope_names_;
    uint32_t next_scope_name_id_ = 1;

    std::unordered_map<std::string, uint32_t> function_dict_;
    std::unordered_map<std::string, uint32_t> dirty_functions_;
    uint32_t next_function_id_ = 1;

    std::unordered_map<std::string, uint32_t> metric_dict_;
    std::unordered_map<std::string, uint32_t> dirty_metrics_;
    uint32_t next_metric_id_ = 1;
};

}  // namespace gpufl
