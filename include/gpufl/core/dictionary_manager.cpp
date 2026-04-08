#include "gpufl/core/dictionary_manager.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#ifdef _WIN32
#include <io.h>       // _open, _write, _close, _unlink, _mktemp_s
#include <fcntl.h>    // _O_CREAT, _O_WRONLY, _O_BINARY
#include <windows.h>  // GetTempPathA
#else
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
    std::unordered_map<uint64_t, std::vector<uint8_t>> pending;
    {
        std::lock_guard lk(mu_);
        if (pending_disasm_cubins_.empty()) return;
        pending = std::move(pending_disasm_cubins_);
    }

    for (auto& [crc, bytes] : pending) {
        // Write cubin to a temp file
        std::string tmpPathStr;
#ifdef _WIN32
        char tmpDir[MAX_PATH];
        DWORD len = GetTempPathA(MAX_PATH, tmpDir);
        if (len == 0 || len >= MAX_PATH) {
            GFL_LOG_ERROR("[flushDisassembly] GetTempPathA failed");
            continue;
        }
        char tmpTemplate[MAX_PATH];
        std::snprintf(tmpTemplate, MAX_PATH, "%sgpufl_cubin_XXXXXX", tmpDir);
        if (_mktemp_s(tmpTemplate, std::strlen(tmpTemplate) + 1) != 0) {
            GFL_LOG_ERROR("[flushDisassembly] _mktemp_s failed");
            continue;
        }
        int fd = _open(tmpTemplate, _O_CREAT | _O_WRONLY | _O_BINARY, 0600);
        if (fd < 0) {
            GFL_LOG_ERROR("[flushDisassembly] _open failed");
            continue;
        }
        {
            int written = _write(fd, bytes.data(), static_cast<unsigned>(bytes.size()));
            _close(fd);
            if (written < 0) {
                _unlink(tmpTemplate);
                continue;
            }
        }
        tmpPathStr = tmpTemplate;
#else
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
        tmpPathStr = tmpPath;
#endif

        // Detect platform and select disassembler
        // AMD code objects are ELF with e_machine == EM_AMDGPU (0xE0)
        bool isAmd = (bytes.size() >= 18 &&
                      bytes[0] == 0x7f && bytes[1] == 'E' &&
                      bytes[2] == 'L' && bytes[3] == 'F' &&
                      (bytes[18] == 0xE0 && bytes[19] == 0x00));

        char cmd[640];
        if (isAmd) {
#ifdef _WIN32
            // ROCm on Windows: not typical, skip AMD disassembly
            std::remove(tmpPathStr.c_str());
            continue;
#else
            // Discover llvm-objdump path
            const char* rocmPath = std::getenv("ROCM_PATH");
            if (rocmPath && rocmPath[0]) {
                std::snprintf(cmd, sizeof(cmd),
                              "%s/llvm/bin/llvm-objdump -d %s 2>/dev/null",
                              rocmPath, tmpPathStr.c_str());
            } else {
                std::snprintf(cmd, sizeof(cmd),
                              "/opt/rocm/llvm/bin/llvm-objdump -d %s 2>/dev/null",
                              tmpPathStr.c_str());
            }
#endif
        } else {
#ifdef _WIN32
            // Discover nvdisasm.exe via CUDA_PATH env var
            const char* cudaPath = std::getenv("CUDA_PATH");
            if (cudaPath && cudaPath[0]) {
                std::snprintf(cmd, sizeof(cmd),
                              "\"%s\\bin\\nvdisasm.exe\" --print-code \"%s\" 2>NUL",
                              cudaPath, tmpPathStr.c_str());
            } else {
                // Fallback: try nvdisasm on PATH
                std::snprintf(cmd, sizeof(cmd),
                              "nvdisasm.exe --print-code \"%s\" 2>NUL",
                              tmpPathStr.c_str());
            }
#else
            std::snprintf(cmd, sizeof(cmd),
                          "/usr/local/cuda/bin/nvdisasm --print-code %s 2>/dev/null",
                          tmpPathStr.c_str());
#endif
        }

#ifdef _WIN32
        FILE* pipe = _popen(cmd, "r");
#else
        FILE* pipe = ::popen(cmd, "r");
#endif
        if (!pipe) {
            std::remove(tmpPathStr.c_str());
            GFL_LOG_ERROR("[flushDisassembly] popen failed — disassembler unavailable?");
            continue;
        }

        std::unordered_map<std::string,
                           std::vector<std::pair<uint32_t, std::string>>>
            funcEntries;
        std::string currentFunc;
        uint64_t currentFuncBase = 0;
        char lineBuf[2048];

        while (std::fgets(lineBuf, sizeof(lineBuf), pipe)) {
            std::string raw = lineBuf;
            while (!raw.empty() &&
                   (raw.back() == '\n' || raw.back() == '\r'))
                raw.pop_back();
            if (raw.empty()) continue;

            if (isAmd) {
                // AMD llvm-objdump format:
                // Function: "0000000000001F00 <_Z15vectorAddKernelPKiS0_Pii>:"
                // Instruction: "\ts_clause 0x1  // 000000001F00: BF850001"
                if (raw.rfind("Disassembly of", 0) == 0) continue;

                // Function label: contains '<' and '>' and ends with ':'
                auto langle = raw.find('<');
                auto rangle = raw.find('>');
                if (langle != std::string::npos && rangle != std::string::npos &&
                    rangle > langle && raw.back() == ':') {
                    currentFunc = raw.substr(langle + 1, rangle - langle - 1);
                    // Parse base address from the leading hex
                    std::string addrStr = raw.substr(0, raw.find_first_of(" \t"));
                    try { currentFuncBase = std::stoull(addrStr, nullptr, 16); }
                    catch (...) { currentFuncBase = 0; }
                    if (!funcEntries.count(currentFunc))
                        funcEntries[currentFunc] = {};
                    continue;
                }

                // Comment-only lines ("; %bb.0:")
                size_t firstNonWs = raw.find_first_not_of(" \t");
                if (firstNonWs != std::string::npos && raw[firstNonWs] == ';')
                    continue;

                if (currentFunc.empty()) continue;

                // Instruction line: has "// XXXXXXXXXXXX:" comment
                size_t commentPos = raw.find("// ");
                if (commentPos == std::string::npos) continue;
                std::string afterComment = raw.substr(commentPos + 3);
                size_t colonPos = afterComment.find(':');
                if (colonPos == std::string::npos) continue;

                uint64_t absPC = 0;
                try { absPC = std::stoull(afterComment.substr(0, colonPos), nullptr, 16); }
                catch (...) { continue; }
                uint32_t pc = static_cast<uint32_t>(absPC - currentFuncBase);

                // Instruction text: before the "//" comment, trimmed
                std::string ins = raw.substr(0, commentPos);
                size_t insStart = ins.find_first_not_of(" \t");
                if (insStart == std::string::npos) continue;
                ins = ins.substr(insStart);
                while (!ins.empty() && (ins.back() == ' ' || ins.back() == '\t'))
                    ins.pop_back();
                funcEntries[currentFunc].emplace_back(pc, std::move(ins));
            } else {
                // NVIDIA nvdisasm format:
                // Function: "_Z11vectorScalePiii:"
                // Instruction: "/*XXXX*/   INSTR ;"
                if (!std::isspace((unsigned char)raw[0]) && raw[0] != '.' &&
                    raw.back() == ':') {
                    currentFunc = raw.substr(0, raw.size() - 1);
                    if (!funcEntries.count(currentFunc))
                        funcEntries[currentFunc] = {};
                    continue;
                }

                if (currentFunc.empty()) continue;
                const size_t start = raw.find_first_not_of(" \t");
                if (start == std::string::npos) continue;
                const std::string s = raw.substr(start);

                if (s.rfind("/*", 0) != 0) continue;
                const size_t end = s.find("*/");
                if (end == std::string::npos) continue;
                const std::string hexStr = s.substr(2, end - 2);
                uint32_t pc = 0;
                try { pc = static_cast<uint32_t>(std::stoul(hexStr, nullptr, 16)); }
                catch (...) { continue; }
                size_t insStart = s.find_first_not_of(" \t", end + 2);
                if (insStart == std::string::npos) continue;
                std::string ins = s.substr(insStart);
                if (!ins.empty() && ins.back() == ';') ins.pop_back();
                while (!ins.empty() && ins.back() == ' ') ins.pop_back();
                funcEntries[currentFunc].emplace_back(pc, std::move(ins));
            }
        }
        GFL_LOG_DEBUG("[flushDisassembly] parsed ", funcEntries.size(),
                      " functions from ", isAmd ? "code object" : "cubin",
                      " crc=", crc);
#ifdef _WIN32
        _pclose(pipe);
#else
        ::pclose(pipe);
#endif
        std::remove(tmpPathStr.c_str());

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
