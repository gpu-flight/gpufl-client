#include "gpufl/core/dictionary_manager.hpp"

#include "gpufl/core/env_vars.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "gpufl/core/sass_compressor.hpp"

#ifdef _WIN32
#include <io.h>       // _open, _write, _close, _unlink, _mktemp_s
#include <fcntl.h>    // _O_CREAT, _O_WRONLY, _O_BINARY
#include <windows.h>  // GetTempPathA
#else
#include <fcntl.h>     // O_WRONLY (redirect child stderr -> /dev/null)
#include <spawn.h>     // posix_spawn - fork-safe subprocess (NOT popen)
#include <sys/wait.h>  // waitpid
#include <unistd.h>    // pipe, close, dup2, STDOUT_FILENO
#endif

#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/logger/logger.hpp"
#include "gpufl/core/model/model_utils.hpp"
#include "gpufl/core/model/serializable.hpp"

#ifndef _WIN32
// posix_spawn needs the current environment for the disassembler child.
extern "C" char **environ;
#endif

namespace gpufl {
namespace {

/// Dictionary updates go to ALL channels because every channel's data
/// references dictionary IDs (function_id, metric_id, scope_name_id).
struct DictLine final : IJsonSerializable {
    std::string json;
    explicit DictLine(std::string j) : json(std::move(j)) {}
    std::string buildJson() const override { return json; }
    Channel channel() const override { return Channel::All; }
};

/// Disassembly and source content go to Device channel only.
/// These are large payloads that don't need to be tripled across all logs.
struct DeviceLine final : IJsonSerializable {
    std::string json;
    explicit DeviceLine(std::string j) : json(std::move(j)) {}
    std::string buildJson() const override { return json; }
    Channel channel() const override { return Channel::Device; }
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

#ifndef _WIN32
// Launch argv[0] with stdout connected to a pipe we read and stderr sent
// to /dev/null, via posix_spawn - deliberately NOT popen/system.
//
// popen() forks and runs /bin/sh. A plain fork() in a multithreaded
// process clones only the calling thread but inherits every lock in its
// current state; if another thread holds the glibc malloc arena lock at
// the instant of the fork, the child deadlocks the first time it needs
// that lock during pre-exec setup, never exec's, and the parent then
// blocks forever reading the pipe. flushDisassembly runs on the collector
// thread while application threads are busy launching kernels, so a fork
// there frequently coincides with a held malloc lock - that is the
// Deep-mode hang. posix_spawn runs only async-signal-safe code in the
// child before exec (vfork/CLONE_VFORK semantics), so it cannot deadlock
// on an inherited lock.
//
// Returns a readable FILE* (caller fcloses) and sets outPid for reaping,
// or nullptr on failure.
FILE *spawnReadPipe(char *const argv[], pid_t &outPid) {
    int pipefd[2];
    if (::pipe(pipefd) != 0) return nullptr;

    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    // child stdout -> pipe write end; child stderr -> /dev/null
    posix_spawn_file_actions_adddup2(&fa, pipefd[1], STDOUT_FILENO);
    posix_spawn_file_actions_addopen(&fa, STDERR_FILENO, "/dev/null",
                                     O_WRONLY, 0);
    // close the inherited pipe fds in the child (write end is dup'd to 1)
    posix_spawn_file_actions_addclose(&fa, pipefd[0]);
    posix_spawn_file_actions_addclose(&fa, pipefd[1]);

    pid_t pid = -1;
    const int rc =
        posix_spawn(&pid, argv[0], &fa, nullptr, argv, environ);
    posix_spawn_file_actions_destroy(&fa);
    ::close(pipefd[1]);  // parent keeps only the read end
    if (rc != 0) {
        ::close(pipefd[0]);
        return nullptr;
    }
    outPid = pid;
    return ::fdopen(pipefd[0], "r");
}
#endif  // !_WIN32

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

    // Read file content eagerly when source collection is enabled.
    // When disabled, we still intern the path (needed for function keys
    // and source_file_id in profile samples) but skip reading the actual
    // source code from disk - users who don't want their source code
    // sent to the backend can set enable_source_collection = false.
    if (enable_source_collection) {
        std::ifstream f(path);
        if (f.is_open()) {
            std::vector<std::string> lines;
            std::string line;
            while (std::getline(f, line)) lines.push_back(line);
            if (!lines.empty()) pending_source_content_[id] = std::move(lines);
        }
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
        logger.write(DeviceLine{oss.str()});
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

        // Launch the disassembler. POSIX uses posix_spawn (see
        // spawnReadPipe) instead of popen - popen's fork() can deadlock on
        // an inherited malloc lock in this multithreaded process, which is
        // the Deep-mode hang. We exec the tool directly with an argv vector
        // (no shell), so paths/args need no quoting and there's no /bin/sh.
        FILE* pipe = nullptr;
#ifndef _WIN32
        pid_t childPid = -1;
#endif
        if (isAmd) {
#ifdef _WIN32
            // ROCm on Windows: not typical, skip AMD disassembly
            std::remove(tmpPathStr.c_str());
            continue;
#else
            // llvm-objdump -d -l (-l adds DWARF source lines for correlation)
            const char* rocmPath = std::getenv("ROCM_PATH");
            std::string objdump = (rocmPath && rocmPath[0])
                ? std::string(rocmPath) + "/llvm/bin/llvm-objdump"
                : std::string("/opt/rocm/llvm/bin/llvm-objdump");
            std::vector<std::string> args = {objdump, "-d", "-l", tmpPathStr};
            std::vector<char*> argv;
            argv.reserve(args.size() + 1);
            for (auto& a : args) argv.push_back(a.data());
            argv.push_back(nullptr);
            pipe = spawnReadPipe(argv.data(), childPid);
#endif
        } else {
#ifdef _WIN32
            // Discover nvdisasm.exe via CUDA_PATH env var.
            // Wrap entire command in outer quotes for cmd.exe /c - needed
            // when both the executable path and arguments contain spaces
            // (e.g., "C:\Program Files\...").
            char cmd[640];
            const char* cudaPath = std::getenv(gpufl::env::kCudaPath);
            if (cudaPath && cudaPath[0]) {
                std::snprintf(cmd, sizeof(cmd),
                              "\"\"%s\\bin\\nvdisasm.exe\" --print-code \"%s\"\"",
                              cudaPath, tmpPathStr.c_str());
            } else {
                std::snprintf(cmd, sizeof(cmd),
                              "\"nvdisasm.exe --print-code \"%s\"\"",
                              tmpPathStr.c_str());
            }
            pipe = _popen(cmd, "r");
#else
            std::vector<std::string> args = {
                "/usr/local/cuda/bin/nvdisasm", "--print-code", tmpPathStr};
            std::vector<char*> argv;
            argv.reserve(args.size() + 1);
            for (auto& a : args) argv.push_back(a.data());
            argv.push_back(nullptr);
            pipe = spawnReadPipe(argv.data(), childPid);
#endif
        }

        if (!pipe) {
            std::remove(tmpPathStr.c_str());
            GFL_LOG_ERROR("[flushDisassembly] failed to launch disassembler "
                          "- nvdisasm / llvm-objdump unavailable?");
            continue;
        }

        // Per-instruction entry with optional DWARF source info (AMD -l flag).
        struct DisasmEntry {
            uint32_t pc;
            std::string sass;
            std::string source_file;   // empty if no DWARF
            uint32_t source_line = 0;
        };

        std::unordered_map<std::string, std::vector<DisasmEntry>> funcEntries;
        std::string currentFunc;
        uint64_t currentFuncBase = 0;
        // Tracked DWARF source location (AMD -l output)
        std::string currentSourceFile;
        uint32_t currentSourceLine = 0;
        char lineBuf[2048];
        while (std::fgets(lineBuf, sizeof(lineBuf), pipe)) {
            std::string raw = lineBuf;
            while (!raw.empty() &&
                   (raw.back() == '\n' || raw.back() == '\r'))
                raw.pop_back();
            if (raw.empty()) continue;

            if (isAmd) {
                // AMD llvm-objdump -d -l format:
                // Source line: "; /path/to/source.hip:42"
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
                    // Reset source tracking for new function
                    currentSourceFile.clear();
                    currentSourceLine = 0;
                    continue;
                }

                // Comment/annotation lines starting with ';'
                size_t firstNonWs = raw.find_first_not_of(" \t");
                if (firstNonWs != std::string::npos && raw[firstNonWs] == ';') {
                    // DWARF source line annotation from -l flag:
                    // "; /path/to/source.hip:42"
                    std::string comment = raw.substr(firstNonWs + 1);
                    size_t cs = comment.find_first_not_of(" \t");
                    if (cs != std::string::npos && comment[cs] == '/') {
                        // Absolute path - find the last ':' for line number
                        size_t lastColon = comment.rfind(':');
                        if (lastColon != std::string::npos && lastColon > cs) {
                            std::string path = comment.substr(cs, lastColon - cs);
                            try {
                                currentSourceLine = static_cast<uint32_t>(
                                    std::stoul(comment.substr(lastColon + 1)));
                                currentSourceFile = path;
                            } catch (...) {}
                        }
                    }
                    continue;
                }

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
                funcEntries[currentFunc].push_back(
                    {pc, std::move(ins), currentSourceFile, currentSourceLine});
            } else {
                // NVIDIA nvdisasm format:
                // Function: "_Z11vectorScalePiii:"
                // Instruction: "/*XXXX*/   INSTR ;"
                if (!std::isspace(static_cast<unsigned char>(raw[0])) && raw[0] != '.' &&
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
                funcEntries[currentFunc].push_back({pc, std::move(ins), {}, 0});
            }
        }
        GFL_LOG_DEBUG("[flushDisassembly] parsed ", funcEntries.size(),
                      " functions from ", isAmd ? "code object" : "cubin",
                      " crc=", crc);
#ifdef _WIN32
        _pclose(pipe);
#else
        // We spawned with posix_spawn (not popen), so close the FILE*
        // ourselves and reap the child explicitly to avoid a zombie.
        ::fclose(pipe);
        if (childPid > 0) {
            int status = 0;
            ::waitpid(childPid, &status, 0);
        }
#endif
        std::remove(tmpPathStr.c_str());

        // Emit one cubin_disassembly JSON message per function.
        // SassCompressor detects runs of structurally identical instructions
        // (same opcode + registers, differing only in immediates) and merges
        // them into a single entry with "count": N.
        for (auto& [funcName, entries] : funcEntries) {
            if (entries.empty()) continue;

            // Build (pc, sass) pairs for compression
            std::vector<std::pair<uint32_t, std::string>> pairs;
            pairs.reserve(entries.size());
            for (const auto& e : entries)
                pairs.emplace_back(e.pc, e.sass);

            auto compressed = SassCompressor::compress(pairs);
            std::ostringstream oss;
            oss << "{\"version\":1,\"type\":\"cubin_disassembly\",\"session_id\":\""
                << model::jsonEscape(session_id) << "\",\"cubin_crc\":" << crc
                << ",\"function_name\":\"" << model::jsonEscape(funcName)
                << "\",\"entries\":[";
            bool first = true;
            for (auto& [pc, sass, count] : compressed) {
                if (!first) oss << ',';
                first = false;
                oss << "{\"pc\":" << pc << ",\"sass\":\""
                    << model::jsonEscape(sass) << "\"";
                if (count > 1) {
                    oss << ",\"count\":" << count;
                }
                oss << '}';
            }
            oss << "]}";
            logger.write(DeviceLine{oss.str()});
        }

        // For AMD code objects with DWARF debug info, emit source file content
        // and a profile_sample_batch that maps pc_offset → source_file:line.
        // This enables the same source-correlated ISA view as NVIDIA/CUPTI.
        if (isAmd) {
            // Collect unique source files and intern them
            std::unordered_map<std::string, uint32_t> sourceFileIds;
            for (const auto& [funcName, entries] : funcEntries) {
                for (const auto& e : entries) {
                    if (e.source_file.empty() || e.source_line == 0) continue;
                    if (!sourceFileIds.count(e.source_file)) {
                        sourceFileIds[e.source_file] =
                            internSourceFile(e.source_file);
                    }
                }
            }

            if (!sourceFileIds.empty()) {
                // Intern the metric and all function keys first, then flush
                // the dictionary and source content so the backend has
                // all IDs resolved BEFORE we emit profile_sample_batch.
                const uint32_t mapMetricId = internMetric("isa_inst_present");

                // Pre-intern all function keys
                for (const auto& [funcName, entries] : funcEntries) {
                    bool hasSource = false;
                    std::string primarySourceFile;
                    std::unordered_map<std::string, uint32_t> fileCounts;
                    for (const auto& e : entries) {
                        if (!e.source_file.empty() && e.source_line > 0) {
                            hasSource = true;
                            ++fileCounts[e.source_file];
                        }
                    }
                    if (!hasSource) continue;
                    uint32_t bestCount = 0;
                    for (const auto& [path, cnt] : fileCounts) {
                        bool isSys = (path.find("/include/hip/") != std::string::npos ||
                                      path.find("/opt/rocm") == 0);
                        bool bestIsSys = (!primarySourceFile.empty() &&
                            (primarySourceFile.find("/include/hip/") != std::string::npos ||
                             primarySourceFile.find("/opt/rocm") == 0));
                        if (primarySourceFile.empty() ||
                            (!isSys && bestIsSys) ||
                            (isSys == bestIsSys && cnt > bestCount)) {
                            primarySourceFile = path;
                            bestCount = cnt;
                        }
                    }
                    internFunction(funcName + "@" + primarySourceFile);
                }

                // Flush dictionary + source content NOW so IDs appear
                // in the log before the profile_sample_batch records.
                flushDictionary(logger, session_id);
                flushSourceContent(logger, session_id);

                const auto now_ns = std::chrono::duration_cast<
                    std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count();

                for (const auto& [funcName, entries] : funcEntries) {
                    // Check if this function has any source mapping
                    bool hasSource = false;
                    for (const auto& e : entries) {
                        if (!e.source_file.empty() && e.source_line > 0) {
                            hasSource = true;
                            break;
                        }
                    }
                    if (!hasSource) continue;

                    // Determine the primary source file for the function key.
                    // Pick the most frequent source file, preferring user code
                    // over system headers (e.g., HIP runtime).
                    std::unordered_map<std::string, uint32_t> fileCounts;
                    for (const auto& e : entries) {
                        if (!e.source_file.empty() && e.source_line > 0)
                            ++fileCounts[e.source_file];
                    }
                    std::string primarySourceFile;
                    uint32_t bestCount = 0;
                    for (const auto& [path, cnt] : fileCounts) {
                        // Prefer non-system paths (skip /opt/rocm, /include/hip)
                        bool isSys = (path.find("/include/hip/") != std::string::npos ||
                                      path.find("/opt/rocm") == 0);
                        bool bestIsSys = (!primarySourceFile.empty() &&
                            (primarySourceFile.find("/include/hip/") != std::string::npos ||
                             primarySourceFile.find("/opt/rocm") == 0));
                        if (primarySourceFile.empty() ||
                            (!isSys && bestIsSys) ||
                            (isSys == bestIsSys && cnt > bestCount)) {
                            primarySourceFile = path;
                            bestCount = cnt;
                        }
                    }
                    const std::string funcKey = funcName + "@" + primarySourceFile;
                    const uint32_t functionId = internFunction(funcKey);

                    // Build profile_sample_batch JSON
                    std::ostringstream poss;
                    poss << "{\"version\":1,\"type\":\"profile_sample_batch\""
                         << ",\"session_id\":\"" << model::jsonEscape(session_id)
                         << "\",\"batch_id\":" << (++disasm_batch_id_)
                         << ",\"base_time_ns\":" << now_ns
                         << ",\"columns\":[\"dt_ns\",\"corr_id\",\"device_id\","
                            "\"function_id\",\"pc_offset\",\"metric_id\","
                            "\"metric_value\",\"stall_reason\",\"sample_kind\","
                            "\"scope_name_id\",\"source_file_id\","
                            "\"source_line\"]"
                         << ",\"rows\":[";
                    bool pfirst = true;
                    for (const auto& e : entries) {
                        if (e.source_file.empty() || e.source_line == 0) continue;
                        // Skip AMD padding instructions from source mapping
                        if (e.sass.compare(0, 10, "s_code_end") == 0 ||
                            e.sass.compare(0, 5, "s_nop") == 0) continue;
                        auto it = sourceFileIds.find(e.source_file);
                        if (it == sourceFileIds.end()) continue;
                        if (!pfirst) poss << ',';
                        pfirst = false;
                        poss << "[0,0,0," << functionId << ',' << e.pc << ','
                             << mapMetricId << ",1,0,2,0,"
                             << it->second << ',' << e.source_line << ']';
                    }
                    poss << "]}";
                    if (!pfirst) {
                        logger.write(DictLine{poss.str()});
                    }
                }
            }
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
