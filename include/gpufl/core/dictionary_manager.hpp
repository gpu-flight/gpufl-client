#pragma once

#include <cstdint>
#include <mutex>
#include <unordered_set>
#include <string>
#include <unordered_map>
#include <vector>

#include "gpufl/core/source_capture_policy.hpp"

namespace gpufl {

class Logger;
class DictionaryManager;

/**
 * Per-segment view of the process-wide dictionary registry.
 *
 * Numeric IDs remain stable for the full run, but every segment must emit its
 * own copy of each mapping before a batch in that segment references it. This
 * object records only the IDs already emitted into one SegmentContext; it
 * never consumes another segment's dirty state.
 */
class SegmentDictionaryEmitter {
   public:
    void flush(DictionaryManager& registry, Logger& logger,
               const std::string& session_id);

   private:
    friend class DictionaryManager;
    std::mutex mu_;
    std::unordered_set<uint32_t> kernels_;
    std::unordered_set<uint32_t> scope_names_;
    std::unordered_set<uint32_t> functions_;
    std::unordered_set<uint32_t> function_symbols_;
    std::unordered_set<uint32_t> metrics_;
    std::unordered_set<uint32_t> source_files_;
};

class DictionaryManager {
   public:
    /// Configure the root/type/file-kind/budget policy used before any
    /// profiler-discovered path is opened.
    void configureSourceCapture(bool enabled,
                                const SourceCaptureSettings& settings);

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

    // funcSymbol = the BARE mangled symbol (no @source), recorded once per id in a parallel
    // dictionary so the backend can compute funcKey = md5(symbol) = md5(functions.name) and the
    // SASS/source view can join samples -> disassembly by a stable key. Empty = not supplied.
    uint32_t internFunction(const std::string& name, const std::string& funcSymbol = "") {
        std::lock_guard lk(mu_);
        uint32_t id;
        if (const auto it = function_dict_.find(name); it != function_dict_.end()) {
            id = it->second;
        } else {
            id = next_function_id_++;
            function_dict_[name] = id;
            dirty_functions_[name] = id;
        }
        if (!funcSymbol.empty() && !function_symbol_dict_.count(id)) {
            function_symbol_dict_[id] = funcSymbol;
            dirty_function_symbols_[id] = funcSymbol;
        }
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

    uint32_t internSourceFile(const std::string& path);
    std::string sourceFileName(uint32_t source_file_id);

    // Emits a dictionary_update JSON line to Channel::All for any new entries
    // accumulated since the last call.  No-op if nothing is dirty.
    void flushDictionary(Logger& logger, const std::string& session_id);

    /**
     * Emit every mapping not yet present in this segment.
     *
     * Unlike flushDictionary(), this does not consume the process-global dirty
     * maps. Two retiring/live segments can therefore flush concurrently
     * without stealing mappings from one another.
     */
    void flushDictionaryForSegment(SegmentDictionaryEmitter& emitter,
                                   Logger& logger,
                                   const std::string& session_id);

    // Emits one source_file_content JSON line per newly-seen source file.
    // No-op if no new source files since last call.
    void flushSourceContent(Logger& logger, const std::string& session_id);

    // Enqueues a cubin (by CRC) for SASS disassembly on the next flush.
    // No-op if the CRC was already enqueued.
    void enqueueDisassembly(uint64_t crc, const uint8_t* data, size_t size);

    // Runs nvdisasm on each pending cubin and emits one cubin_disassembly
    // JSON line per function.  No-op if nothing is pending.
    void flushDisassembly(Logger& logger, const std::string& session_id);

    // Keep the line tables of AMD code objects disassembled from now on, for
    // lookupAmdIsaSource. Only AMD PC sampling needs them; reset() drops
    // them and turns retention off.
    void retainAmdIsaLines();

    // The source line an AMD ISA row shows for an instruction, from the
    // llvm-objdump line table flushDisassembly parsed. NotParsed until the
    // code object's disassembly has run; Found or NoLine afterwards. On
    // Found, function_source is the file the function's ISA rows are keyed
    // by, and source_file/source_line the instruction's own line.
    enum class IsaSourceLookup { NotParsed, NoLine, Found };
    struct IsaSourceLine {
        std::string function_source;
        std::string source_file;
        uint32_t source_line = 0;
    };
    IsaSourceLookup lookupAmdIsaSource(uint64_t crc,
                                       const std::string& function,
                                       uint32_t pc_offset,
                                       IsaSourceLine* out);

    // Runs cuobjdump -ptx on the running executable and emits one
    // ptx_disassembly JSON line per kernel (joined to kernels/SASS by the same
    // mangled function name). PTX lives in the fatbin, not the SASS cubins we
    // capture, so this reads the executable directly; that dump runs once per
    // session. Every call then emits the PTX modules loaded at run time (see
    // enqueueLoadedPtx).
    void flushPtx(Logger& logger, const std::string& session_id);

    // Keeps a copy of a PTX module handed to the driver at run time
    // (cuModuleLoadData and friends): JIT-compiled code such as cuda-oxide
    // (Rust) kernels has no fatbin PTX for cuobjdump to find. `limit` bounds
    // the read; binaries (ELF cubins, fatbins) and other text are ignored.
    // Returns true when the image was PTX and is now queued.
    bool enqueueLoadedPtx(const void* image, size_t limit);

    // Emits one ptx_disassembly line per kernel of the PTX modules queued by
    // enqueueLoadedPtx(). A kernel already emitted (by cuobjdump or an
    // earlier module) is not emitted again. No-op if nothing is queued.
    void flushLoadedPtx(Logger& logger, const std::string& session_id);

    // Length of the PTX text at `image` (up to its NUL or `limit`), or 0 when
    // the bytes are not PTX: a module that starts with `.version` after
    // optional whitespace, `//` comment lines and `/* */` comments.
    static size_t loadedPtxLength(const void* image, size_t limit);

    void reset() {
        // The PTX state belongs to the session too: a new session dumps the
        // executable again and keeps the PTX of every module it loads, seen
        // in an earlier session or not. Same lock order as the PTX flushes.
        std::scoped_lock lk(ptx_flush_mu_, mu_);
        kernel_dict_.clear();
        dirty_kernels_.clear();
        next_kernel_id_ = 1;
        scope_name_dict_.clear();
        dirty_scope_names_.clear();
        next_scope_name_id_ = 1;
        function_dict_.clear();
        dirty_functions_.clear();
        function_symbol_dict_.clear();
        dirty_function_symbols_.clear();
        next_function_id_ = 1;
        metric_dict_.clear();
        dirty_metrics_.clear();
        next_metric_id_ = 1;
        source_file_dict_.clear();
        source_file_names_.clear();
        dirty_source_files_.clear();
        next_source_file_id_ = 1;
        pending_source_content_.clear();
        source_capture_.reset();
        pending_disasm_cubins_.clear();
        pending_loaded_ptx_.clear();
        pending_loaded_ptx_bytes_ = 0;
        seen_loaded_ptx_.clear();
        emitted_ptx_names_.clear();
        ptx_emitted_ = false;
        retain_amd_isa_lines_ = false;
        amd_isa_lines_.clear();
    }

   private:
    // Runs cuobjdump -ptx on the running executable and emits its kernels.
    void dumpExecutablePtx(Logger& logger, const std::string& session_id);

    // Emits and clears the queued run-time PTX modules. ptx_flush_mu_ held.
    void drainLoadedPtx(Logger& logger, const std::string& session_id);

    // Splits PTX text into kernels and emits one ptx_disassembly line per
    // kernel whose name is not in `skip`, recording each emitted name in
    // emitted_ptx_names_ (`skip` may be that set). Returns the number emitted.
    int emitPtxText(Logger& logger, const std::string& session_id,
                    const std::string& ptx,
                    const std::unordered_set<std::string>& skip);

    // Serializes the PTX flushes, which run from more than one flush path.
    std::mutex ptx_flush_mu_;

    std::mutex mu_;

    std::unordered_map<std::string, uint32_t> kernel_dict_;
    std::unordered_map<std::string, uint32_t> dirty_kernels_;
    uint32_t next_kernel_id_ = 1;

    std::unordered_map<std::string, uint32_t> scope_name_dict_;
    std::unordered_map<std::string, uint32_t> dirty_scope_names_;
    uint32_t next_scope_name_id_ = 1;

    std::unordered_map<std::string, uint32_t> function_dict_;
    std::unordered_map<std::string, uint32_t> dirty_functions_;
    // id -> BARE mangled symbol, parallel to function_dict_ (which holds demangled "name@source").
    std::unordered_map<uint32_t, std::string> function_symbol_dict_;
    std::unordered_map<uint32_t, std::string> dirty_function_symbols_;
    uint32_t next_function_id_ = 1;

    std::unordered_map<std::string, uint32_t> metric_dict_;
    std::unordered_map<std::string, uint32_t> dirty_metrics_;
    uint32_t next_metric_id_ = 1;

    std::unordered_map<std::string, uint32_t> source_file_dict_;
    // file_id -> normalized logical path emitted to product-facing logs.
    // source_file_dict_ keeps the local lookup key but is never serialized.
    std::unordered_map<uint32_t, std::string> source_file_names_;
    std::unordered_map<std::string, uint32_t> dirty_source_files_;
    uint32_t next_source_file_id_ = 1;

    // file_id → lines (populated once per new source file, flushed via flushSourceContent)
    std::unordered_map<uint32_t, std::vector<std::string>> pending_source_content_;
    detail::SourceCapturePolicy source_capture_;

    // cubin_crc → raw bytes (populated once per cubin, flushed via flushDisassembly)
    std::unordered_map<uint64_t, std::vector<uint8_t>> pending_disasm_cubins_;

    // AMD code object crc -> its ISA rows' source lines, while retained.
    // A crc is present once its disassembly ran.
    struct AmdIsaLine {
        uint32_t pc = 0;
        uint32_t file = 0;  // index into AmdIsaCodeObject::files
        uint32_t line = 0;
    };
    struct AmdIsaFunctionLines {
        uint32_t primary_file = 0;
        std::vector<AmdIsaLine> lines;  // by pc
    };
    struct AmdIsaCodeObject {
        std::vector<std::string> files;
        std::unordered_map<std::string, AmdIsaFunctionLines> functions;
    };
    bool retain_amd_isa_lines_ = false;
    std::unordered_map<uint64_t, AmdIsaCodeObject> amd_isa_lines_;

    // PTX is dumped from the executable once per session (see flushPtx).
    bool ptx_emitted_ = false;

    // PTX modules loaded at run time (see enqueueLoadedPtx), their total size,
    // the hashes of modules already queued, and the kernels already emitted
    // by either path.
    std::vector<std::string> pending_loaded_ptx_;
    size_t pending_loaded_ptx_bytes_ = 0;
    std::unordered_set<size_t> seen_loaded_ptx_;
    std::unordered_set<std::string> emitted_ptx_names_;

    // Batch ID counter for profile_sample_batch records emitted from
    // AMD DWARF source mapping (avoids collision with monitor batch IDs
    // by starting at a high offset).
    uint64_t disasm_batch_id_ = 1000000;
};

}  // namespace gpufl
