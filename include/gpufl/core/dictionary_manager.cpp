#include "gpufl/core/dictionary_manager.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#ifndef _WIN32
#include <unistd.h>
#endif

#include "gpufl/core/debug_logger.hpp"
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

uint32_t DictionaryManager::internSourceFile(const std::string& path) {
    if (path.empty()) return 0;
    std::lock_guard lk(mu_);
    if (const auto it = source_file_dict_.find(path);
        it != source_file_dict_.end())
        return it->second;
    const uint32_t id = next_source_file_id_++;
    source_file_dict_[path] = id;
    dirty_source_files_[path] = id;

    // Read file content eagerly; failures are silently ignored (file may not
    // be accessible on the machine running the backend ingestor).
    std::ifstream f(path);
    if (f.is_open()) {
        std::vector<std::string> lines;
        std::string line;
        while (std::getline(f, line)) lines.push_back(line);
        if (!lines.empty()) pending_source_content_[id] = std::move(lines);
    }
    return id;
}

void DictionaryManager::flushSourceContent(Logger& logger,
                                            const std::string& session_id) {
    std::unordered_map<uint32_t, std::vector<std::string>> pending;
    {
        std::lock_guard lk(mu_);
        if (pending_source_content_.empty()) return;
        pending = std::move(pending_source_content_);
    }
    for (auto& [file_id, lines] : pending) {
        std::ostringstream oss;
        oss << "{\"version\":1,\"type\":\"source_file_content\",\"session_id\":\""
            << model::jsonEscape(session_id) << "\",\"source_file_id\":" << file_id
            << ",\"lines\":[";
        bool first = true;
        for (const auto& ln : lines) {
            if (!first) oss << ',';
            first = false;
            oss << '"' << model::jsonEscape(ln) << '"';
        }
        oss << "]}";
        logger.write(DictLine{oss.str()});
    }
}

void DictionaryManager::enqueueDisassembly(uint64_t crc, const uint8_t* data,
                                            size_t size) {
    std::lock_guard lk(mu_);
    if (pending_disasm_cubins_.count(crc)) return;
    pending_disasm_cubins_[crc].assign(data, data + size);
}

void DictionaryManager::flushDisassembly(Logger& logger,
                                          const std::string& session_id) {
#ifdef _WIN32
    (void)logger; (void)session_id;
    return;  // nvdisasm subprocess not supported on Windows
#else
    std::unordered_map<uint64_t, std::vector<uint8_t>> pending;
    {
        std::lock_guard lk(mu_);
        if (pending_disasm_cubins_.empty()) return;
        pending = std::move(pending_disasm_cubins_);
    }

    for (auto& [crc, bytes] : pending) {
        // Write cubin to a temp file
        char tmpPath[] = "/tmp/gpufl_cubin_XXXXXX";
        int fd = mkstemp(tmpPath);
        if (fd < 0) {
            GFL_LOG_ERROR("[flushDisassembly] mkstemp failed");
            continue;
        }
        {
            ssize_t written = ::write(fd, bytes.data(), bytes.size());
            ::close(fd);
            if (written < 0) {
                ::unlink(tmpPath);
                continue;
            }
        }

        // Run nvdisasm --print-code
        char cmd[640];
        std::snprintf(cmd, sizeof(cmd),
                      "/usr/local/cuda/bin/nvdisasm --print-code %s 2>/dev/null",
                      tmpPath);
        FILE* pipe = ::popen(cmd, "r");
        if (!pipe) {
            ::unlink(tmpPath);
            GFL_LOG_ERROR("[flushDisassembly] popen failed — nvdisasm unavailable?");
            continue;
        }

        // Parse nvdisasm output.
        // Function labels appear as bare labels at column 0: "funcName:"
        // (not starting with '.' or whitespace).
        // Instructions appear with leading whitespace: "/*XXXX*/   INSTR ;"
        std::unordered_map<std::string,
                           std::vector<std::pair<uint32_t, std::string>>>
            funcEntries;
        std::string currentFunc;
        char lineBuf[2048];
        while (std::fgets(lineBuf, sizeof(lineBuf), pipe)) {
            std::string raw = lineBuf;
            // Strip trailing newline/CR
            while (!raw.empty() &&
                   (raw.back() == '\n' || raw.back() == '\r'))
                raw.pop_back();
            if (raw.empty()) continue;

            // Function label: starts at column 0, doesn't start with '.' or
            // whitespace, ends with ':'. e.g. "_Z11vectorScalePiii:"
            if (!std::isspace((unsigned char)raw[0]) && raw[0] != '.' &&
                raw.back() == ':') {
                currentFunc = raw.substr(0, raw.size() - 1);
                if (!funcEntries.count(currentFunc))
                    funcEntries[currentFunc] = {};
                continue;
            }

            // Instruction line: leading whitespace then "/*XXXX*/   INSTR ;"
            if (currentFunc.empty()) continue;
            const size_t start = raw.find_first_not_of(" \t");
            if (start == std::string::npos) continue;
            const std::string s = raw.substr(start);

            if (s.rfind("/*", 0) != 0) continue;
            const size_t end = s.find("*/");
            if (end == std::string::npos) continue;
            const std::string hexStr = s.substr(2, end - 2);
            uint32_t pc = 0;
            try {
                pc = static_cast<uint32_t>(std::stoul(hexStr, nullptr, 16));
            } catch (...) {
                continue;
            }
            size_t insStart = s.find_first_not_of(" \t", end + 2);
            if (insStart == std::string::npos) continue;
            std::string ins = s.substr(insStart);
            // Remove trailing " ;"
            if (!ins.empty() && ins.back() == ';') ins.pop_back();
            while (!ins.empty() && ins.back() == ' ') ins.pop_back();
            funcEntries[currentFunc].emplace_back(pc, std::move(ins));
        }
        GFL_LOG_DEBUG("[flushDisassembly] parsed ", funcEntries.size(),
                      " functions from cubin crc=", crc);
        ::pclose(pipe);
        ::unlink(tmpPath);

        // Emit one cubin_disassembly JSON message per function
        for (auto& [funcName, entries] : funcEntries) {
            if (entries.empty()) continue;
            std::ostringstream oss;
            oss << "{\"version\":1,\"type\":\"cubin_disassembly\",\"session_id\":\""
                << model::jsonEscape(session_id) << "\",\"cubin_crc\":" << crc
                << ",\"function_name\":\"" << model::jsonEscape(funcName)
                << "\",\"entries\":[";
            bool first = true;
            for (auto& [pc, sass] : entries) {
                if (!first) oss << ',';
                first = false;
                oss << "{\"pc\":" << pc << ",\"sass\":\""
                    << model::jsonEscape(sass) << "\"}";
            }
            oss << "]}";
            logger.write(DictLine{oss.str()});
        }
    }
#endif  // !_WIN32
}

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
