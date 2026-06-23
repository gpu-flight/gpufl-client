import os
import sys
from contextlib import contextmanager

# Import-order guard: gpufl and PyTorch each bundle a CUPTI version.
# If gpufl is imported before torch, two incompatible CUPTI DLLs end up
# loaded and conflict during profiling (crash in cubin callback).
# Detect torch already being imported and warn if we loaded before it.
if os.name == 'nt' and 'torch' not in sys.modules:
    # torch not yet imported - emit a one-time advisory.  We don't raise
    # here because headless / CPU-only code should still work.
    import warnings
    warnings.warn(
        "[gpufl] Import order advisory: 'import torch' should come before "
        "'import gpufl' to avoid a CUPTI version conflict. "
        "When gpufl loads first on Windows, CUDA 13+ CUPTI (bundled with gpufl) "
        "initialises before PyTorch's own CUPTI, which can crash on the first "
        "CUDA kernel launch under profiling. "
        "Reorder your imports: torch → gpufl.",
        ImportWarning,
        stacklevel=2,
    )
    del warnings

# 1. Windows DLL Handling - ensure CUDA and CUPTI DLLs are findable.
# os.add_dll_directory() alone is insufficient for some Python builds;
# we also prepend to PATH as a belt-and-suspenders approach.
if os.name == 'nt':
    cuda_path = os.environ.get('CUDA_PATH')
    if cuda_path:
        _dll_dirs = [
            os.path.join(cuda_path, 'bin'),
            # CUDA 13+: runtime DLLs (cudart, cublas, curand, ...) moved
            # under bin/x64/. Keep bin/ above it for older toolkits.
            os.path.join(cuda_path, 'bin', 'x64'),
            os.path.join(cuda_path, 'extras', 'CUPTI', 'lib64'),
        ]
        # CUPTI transitively depends on zlib.dll, which CUDA does NOT ship
        # but Nsight tools do. Add their bin dirs as a fallback so imports
        # work out of the box on a typical dev box.
        import glob as _glob
        for nsight_glob in [
            r'C:\Program Files\NVIDIA Corporation\Nsight Compute *\host\windows-desktop-win7-x64',
            r'C:\Program Files\NVIDIA Corporation\Nsight Systems *\host-windows-x64',
        ]:
            for p in _glob.glob(nsight_glob):
                if os.path.isfile(os.path.join(p, 'zlib.dll')):
                    _dll_dirs.append(p)
                    break  # one per glob is enough
        for d in _dll_dirs:
            if os.path.isdir(d):
                try:
                    os.add_dll_directory(d)
                except (AttributeError, OSError):
                    pass
                # Also add to PATH for Python extension module loading
                if d not in os.environ.get('PATH', ''):
                    os.environ['PATH'] = d + os.pathsep + os.environ.get('PATH', '')


# 1b. Linux: preload the MATCHING PerfWorks libraries before the C extension.
#
# gpufl's _gpufl_client.so links libnvperf_host.so / libnvperf_target.so
# DIRECTLY (DT_NEEDED - see the NVPERF block in CMakeLists.txt), so the
# dynamic loader resolves them the instant the extension is imported,
# long before any of our initialize() code runs. In a PyTorch venv the
# recommended import order is `torch` THEN `gpufl`; by the time gpufl
# imports, torch has already loaded the pip `nvidia-cuXX` libcupti.so -
# but NOT nvperf (torch doesn't touch PerfWorks at import). So our NEEDED
# libnvperf_host.so has nothing resident to dedupe against, and the
# loader finds the SYSTEM copy via ldconfig (/usr/local/cuda/.../
# libnvperf_host.so). The result is venv CUPTI + system PerfWorks - a
# split install that SEGFAULTS in NVPW_CUDA_LoadDriver the first time PC
# sampling (PcSampling) or the Profiler API (SassMetrics / RangeProfiler
# / Deep) runs. It "sometimes works" only when something happened to make
# the matching nvperf resident first.
#
# `LD_LIBRARY_PATH=<venv>/nvidia/cuXX/lib` fixes it by winning that first
# resolution. We do the same automatically and deterministically: find
# the directory of the libcupti that's already mapped into the process
# (torch loaded it; nvperf ships in the SAME wheel dir, so versions are
# guaranteed to match) and preload its sibling PerfWorks libs with
# RTLD_GLOBAL, in dependency order (target before host) so soname-dedup
# binds our NEEDED entries to the matching copies. Best-effort and
# silent; set GPUFL_DEBUG_PRELOAD=1 to trace what was preloaded.
def _preload_matching_perfworks():
    if not sys.platform.startswith('linux'):
        return
    import ctypes
    import glob as _glob

    # RTLD_LAZY|RTLD_GLOBAL - same flags the C++ side uses. GLOBAL so the
    # symbols/soname win subsequent NEEDED resolutions; LAZY so we don't
    # force-bind symbols that depend on libs not resident yet at import.
    _mode = os.RTLD_LAZY | os.RTLD_GLOBAL

    _debug = os.environ.get('GPUFL_DEBUG_PRELOAD', '').strip().lower() in (
        '1', 'true', 'yes', 'on')

    def _dbg(msg):
        if _debug:
            print(f"[gpufl] perfworks-preload: {msg}", file=sys.stderr)

    candidate_dirs = []
    # (1) Strongest signal: the directory of a libcupti already mapped
    #     into this process. nvperf MUST match that CUPTI's CUDA version,
    #     and it ships in the same wheel directory.
    try:
        with open('/proc/self/maps', 'r') as _maps:
            for line in _maps:
                slash = line.find('/')
                if slash == -1:
                    continue
                path = line[slash:].rstrip()
                if os.path.basename(path).startswith('libcupti.so'):
                    d = os.path.dirname(path)
                    if d and d not in candidate_dirs:
                        candidate_dirs.append(d)
    except OSError:
        pass

    # (2) Lower-priority fallback: nvidia CUDA wheels under site-packages.
    #     Appended AFTER the libcupti dirs (which are preferred because
    #     they're version-matched), so it covers (a) gpufl imported before
    #     torch / pure-gpufl processes, and (b) torch bundling its libcupti
    #     in a dir that has no sibling nvperf (e.g. torch/lib) - the loop
    #     below skips dirs without the PerfWorks set and falls through here.
    roots = []
    try:
        import importlib.util
        spec = importlib.util.find_spec('nvidia')
        if spec and spec.submodule_search_locations:
            roots.extend(list(spec.submodule_search_locations))
    except Exception:
        pass
    try:
        _site = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        roots.append(os.path.join(_site, 'nvidia'))
    except Exception:
        pass
    for root in roots:
        for d in _glob.glob(os.path.join(root, '*', 'lib')):
            if d not in candidate_dirs:
                candidate_dirs.append(d)

    # A heavy pipeline (torch + torchvision + other CUDA extensions) can
    # have MORE THAN ONE libcupti resident - e.g. the pip venv copy AND a
    # system /usr/local/cuda copy. /proc/self/maps order is by address, not
    # load order, so "first seen" is arbitrary and could point at the
    # system dir while gpufl actually binds the venv CUPTI (→ we'd preload
    # the wrong, mismatched nvperf). gpufl is a pip wheel and binds the pip
    # CUPTI, so prefer site-packages dirs. Stable sort keeps relative order
    # within each group.
    candidate_dirs.sort(key=lambda p: 0 if 'site-packages' in p else 1)

    _dbg(f"candidate dirs: {candidate_dirs}")

    # Preload from the FIRST directory that actually has the PerfWorks
    # set. Loading a second CUDA version's nvperf would soname-collide
    # with the first (identical SONAME, different ABI), so stop after one.
    for d in candidate_dirs:
        if not _glob.glob(os.path.join(d, 'libnvperf_host.so*')):
            continue
        loaded_any = False
        for pattern in ('libnvperf_target.so*', 'libpcsamplingutil.so*',
                        'libnvperf_host.so*'):
            for so in sorted(_glob.glob(os.path.join(d, pattern))):
                try:
                    ctypes.CDLL(so, mode=_mode)
                    loaded_any = True
                    _dbg(f"loaded {so}")
                except OSError as exc:
                    _dbg(f"skip {so} ({exc})")
        if loaded_any:
            break


_preload_matching_perfworks()


# 2. Import C++ Core Bindings
try:
    from ._gpufl_client import (
        Scope as _CScope, init, shutdown, system_start, system_stop,
        BackendKind, InitOptions, ProfilingEngine,
        upload_logs as _c_upload_logs, UploadOptions, UploadResult,
    )
except ImportError as e:
    # We catch ImportError specifically to handle missing libcuda.so.1 or DLLs
    import sys
    print(f"[WARNING] Failed to import _gpufl_client extension: {e}", file=sys.stderr)
    print(f"[WARNING] Using fallback stub implementation (No GPU Mode)", file=sys.stderr)

    # --- FIX START ---
    # The previous code forced a crash in CI/CD. We removed it so
    # verify_pipeline.py can pass even without a GPU.

    # For local dev AND CI, keep a safe fallback
    def init(*args, **kwargs):
        print("[GPUFL] Warning: init() called in stub mode (No GPU detected).", file=sys.stderr)
        return False

    def shutdown():
        return None

    def system_start(name="system"):
        return None

    def system_stop(name="system"):
        return None

    class BackendKind:
        Auto = "Auto"
        Nvidia = "Nvidia"
        Amd = "Amd"
        None_ = "None"

    class ProfilingEngine:
        # Six canonical names, no aliases - mirrors the C++ enum.
        Monitor       = "Monitor"
        Trace         = "Trace"
        PcSampling    = "PcSampling"
        SassMetrics   = "SassMetrics"
        PmSampling    = "PmSampling"
        RangeProfiler = "RangeProfiler"
        RangeProfilerKernelReplay = "RangeProfilerKernelReplay"
        Deep          = "Deep"

    class InitOptions:
        def __init__(self):
            self.app_name = "gpufl"
            self.log_path = ""
            self.continuous_system_sampling = False
            self.system_sample_rate_ms = 0
            self.kernel_sample_rate_ms = 0
            self.backend = BackendKind.Auto
            self.enable_debug_output = False
            self.enable_stack_trace = False
            self.enable_source_collection = True
            self.flush_logs_always = False
            self.profiling_engine = ProfilingEngine.Monitor
            self.pm_sampling_interval_us = 100
            self.pm_sampling_max_samples = 4096
            self.pm_sampling_preset = "overview"
            self.pm_sampling_metrics = []
            self.pm_sampling_scope_only = True
            self.config_file = ""
            self.enabled = True  # mirror C++ InitOptions::enabled

    class _CScope:
        # Accept kwargs (repeat/warmup added in 1.0.3) so the no-GPU
        # fallback matches the real pybind11 binding's signature.
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass

    # Stub UploadResult - mirror the C++ binding's field set so calling
    # code can introspect even in no-GPU mode without AttributeError.
    class UploadResult:
        def __init__(self):
            self.success = False
            self.files_processed = 0
            self.files_skipped_by_cursor = 0
            self.events_uploaded = 0
            self.bytes_uploaded = 0
            self.elapsed_ms = 0
            self.warnings = ["gpufl C++ extension not loaded - upload is a no-op in stub mode"]

    class UploadOptions:
        def __init__(self):
            self.log_path = ""
            self.backend_url = ""
            self.api_key = ""
            self.api_path = ""
            self.total_timeout_ms = 5 * 60 * 1000
            self.connect_timeout_ms = 10 * 1000
            self.read_timeout_ms = 30 * 1000
            self.max_retries = 1
            self.retry_delay_ms = 1000
            self.cursor_filename = ".gpufl-upload-cursor.json"
            self.report_progress = True
            self.session_id_filter = ""
            self.all_sessions = False
            self.force = False

    def _c_upload_logs(*args, **kwargs):  # type: ignore[no-redef]
        print("[GPUFL] upload_logs called in stub mode - returning empty result.",
              file=sys.stderr)
        return UploadResult()
    # --- FIX END ---

except Exception as e:
    # Catch other unexpected errors (like syntax errors in the C++ extension)
    import sys
    print(f"[FATAL] Unexpected error importing _gpufl_client: {e}", file=sys.stderr)
    raise e

__version__ = "1.2.1"

# ── Backend upload ────────────────────────────────────────────────────────────
#
# Upload happens post-shutdown via the deferred path implemented in
# C++ at include/gpufl/upload/upload_logs.cpp. The historical live-
# streaming sink (HttpLogSink) was removed; nothing in this Python
# wrapper opens HTTP connections during the session. Call
# `gpufl.upload_logs(...)` after `gpufl.shutdown()`, or wrap the whole
# thing in `with gpufl.session(backend_url=..., api_key=...):` to have
# it run automatically. The legacy `remote_upload=True` kwarg on init()
# is gone - passing it raises TypeError (see the check inside init()).
#
# (Historical note: a `config_name` kwarg used to trigger a remote
# named-config fetch from the backend before init completed. That
# feature was removed alongside the backend's ConfigController and
# the SPA's ProfilingConfigPage. All configuration now flows through
# InitOptions, env vars, and the local config file.)

# ── Session / log-path bookkeeping ──────────────────────────────────────────
#
# Tracked purely so `clean_logs()` can (a) refuse to delete the logs of a
# session that's still being written in THIS process, and (b) default to the
# last init()'s log location when called with no arguments.
_session_active = False
_last_log_path = None
_last_app_name = None

# ── Disable flag ─────────────────────────────────────────────────────────────
#
# When set, every public entry point (init, shutdown, system_start/stop,
# upload_logs, Scope) becomes a no-op. Set by either:
#   * gpufl.init(..., enabled=False)
#   * the GPUFL_DISABLED env var (1/true/yes/on)
# Env wins - it lets users force-off without editing code (handy for CI,
# debugging, or comparing runs with vs. without instrumentation).
_disabled = False


def _env_disabled():
    """True if the GPUFL_DISABLED env var is set to a truthy value."""
    return os.environ.get('GPUFL_DISABLED', '').strip().lower() in (
        '1', 'true', 'yes', 'on')


class _NoopUploadResult:
    """Stand-in returned by :func:`upload_logs` when gpufl is disabled.

    The real :class:`UploadResult` is the C++ binding from pybind11 and
    cannot be default-constructed or mutated from Python - its fields
    are all ``def_readonly`` and there's no exposed ``py::init<>()``.
    Trying to do ``r = UploadResult(); r.success = True`` from the
    disabled-mode no-op path therefore crashes in production (in stub
    mode the pure-Python class above masks the bug - same name, very
    different shape).

    This class mirrors the binding's public field set so existing caller
    code - ``r.success``, ``r.events_uploaded``, ``isinstance(...)``-free
    duck typing - keeps working unchanged. If you add a field to the
    C++ ``UploadResult``, add it here too (and to the stub-mode class
    above).
    """
    def __init__(self):
        self.success = True
        self.files_processed = 0
        self.files_skipped_by_cursor = 0
        self.events_uploaded = 0
        self.bytes_uploaded = 0
        self.elapsed_ms = 0
        self.warnings = ["gpufl is disabled - upload_logs is a no-op"]
        self.spool_ids = []

    def __repr__(self):
        return ("<UploadResult success=True events=0 bytes=0 files=0 "
                "warnings=1 spool_ids=0 elapsed_ms=0 (disabled)>")

def _apply_eager_module_loading(profiling_engine):
    """Optionally set CUDA_MODULE_LOADING=EAGER - opt-in only.

    EAGER is now OPT-IN. By default gpufl leaves CUDA on its normal LAZY
    module loading. The per-architecture SASS exclusion gate
    (GPUFL_SASS_EXCLUDE_ARCHS, read by the C++ SASS engine) is the default
    mechanism for the lazy-patching deadlock: it disables SASS only on
    architectures confirmed to hang (e.g. RTX 3090 / sm_86) instead of
    paying EAGER's whole-process startup + memory cost on every machine.

    EAGER stays available as an alternative per-run workaround. CUPTI's SASS
    instrumentation + LAZY module loading can deadlock under concurrent
    kernel launches (PyTorch): each kernel's module is finalized AND
    SASS-patched on its first launch, and many such first-launches racing
    across threads invert CUPTI/driver locks. EAGER finalizes every module
    up front while the process is quiescent, closing that window.

    Overrides:
      * GPUFL_EAGER_MODULE_LOADING=1|true|yes|on → force EAGER for this run.
      * Unset / anything else                    → leave CUDA on LAZY (default).
      * An existing CUDA_MODULE_LOADING (set by the user) is always honored.

    Must run before the CUDA context is created. In the recommended order
    (`import torch` → `import gpufl` → `gpufl.init()` → train) the context
    isn't created until the first CUDA op in the loop, so setting it at the
    top of init() takes effect.
    """
    # Opt-in only: act solely when GPUFL_EAGER_MODULE_LOADING is truthy.
    # (profiling_engine is kept for call-site compatibility / future use.)
    knob = os.environ.get('GPUFL_EAGER_MODULE_LOADING', '').strip().lower()
    if knob not in ('1', 'true', 'yes', 'on'):
        return  # default: leave CUDA on LAZY module loading

    if os.environ.get('CUDA_MODULE_LOADING'):
        return  # respect a value the user set themselves

    # If CUDA is already up, the env is read-at-context-creation, so setting
    # it now is a no-op - warn instead of silently doing nothing. Only peek
    # at torch if it's already imported (never import it here).
    _torch = sys.modules.get('torch')
    if _torch is not None:
        try:
            if _torch.cuda.is_initialized():
                import warnings
                warnings.warn(
                    "[gpufl] CUDA is already initialized, so setting "
                    "CUDA_MODULE_LOADING=EAGER now has no effect. SASS/Deep "
                    "profiling can deadlock under CUDA's default lazy module "
                    "loading; set CUDA_MODULE_LOADING=EAGER on the command "
                    "line (before the process starts) to avoid it.",
                    RuntimeWarning, stacklevel=3)
                return
        except Exception:
            pass

    os.environ['CUDA_MODULE_LOADING'] = 'EAGER'


# Wrap the C++ init to apply multi-pass env tagging and to reject removed
# kwargs (remote_upload / sampling_auto_start / backend_url / api_key).
_original_init = init

def init(*args,
         enabled=True, analysis_id=None, pass_index=None, pass_count=None,
         **kwargs):
    """Initialize GPUFlight.

    Configuration precedence (low → high). Each layer may override the
    previous; your explicit field sets on this call always win:

      1. InitOptions defaults (built-in).
      2. Local config file (config_file=...).
      3. Env vars (GPUFL_PROFILING_ENGINE / GPUFL_CONFIG_FILE).
      4. The kwargs you pass to this function.

    Backend credentials are NOT init() arguments - pass them to
    ``gpufl.upload_logs(backend_url=..., api_key=...)`` or
    ``with gpufl.session(backend_url=..., api_key=...):`` instead.

    Args:
        enabled: When False, init becomes a no-op - no daemon spawn, no
            NVML probe, no log files, no atexit handler. Subsequent
            gpufl calls (Scope, shutdown, upload_logs, system_start/stop)
            also no-op for the rest of the process. Useful for toggling
            instrumentation on/off without removing the call. The
            ``GPUFL_DISABLED`` env var (set to ``1``/``true``/``yes``/
            ``on``) forces the same behavior regardless of this kwarg -
            that way you can disable gpufl for a one-off run without
            editing code: ``GPUFL_DISABLED=1 python train.py``.
        analysis_id: Tag this run as one pass of a multi-pass "analysis
            group" so the backend merges it with its sibling passes. Run
            the SAME targeted workload once per engine with a shared
            ``analysis_id`` and a distinct ``pass_index``; the dashboard's
            Analysis Group view then shows the merged cross-engine result
            (timing from the Trace pass, stalls from PcSampling, SASS rows
            from SassMetrics). Sets ``GPUFL_ANALYSIS_ID`` for the C++ core
            (an explicit value wins over a pre-set env var). None → an
            ordinary single run that is not grouped.

            Note: CUPTI allows only ONE profiling engine per process, so a
            multi-engine deep-dive is several runs (one engine each),
            NOT one process - bound each run to the hot region (e.g. break
            the loop after a few iterations, or load a checkpoint) so a
            long job isn't re-run end to end.
        pass_index: 0-based position of this run within the analysis group.
            Only meaningful alongside ``analysis_id``. Sets
            ``GPUFL_PASS_INDEX``.
        pass_count: Total passes planned for the analysis group (lets the
            backend flag a missing/failed pass). Only meaningful alongside
            ``analysis_id``. Sets ``GPUFL_PASS_COUNT``.
        **kwargs: All other InitOptions fields passed to C++ init.

    Returns:
        bool: True on successful init, False when stub-mode, disabled,
        or init otherwise failed. Callers may branch on this:
        ``if not gpufl.init(...): ...``.
    """
    # ── disable check ───────────────────────────────────────────────────
    # Env var wins over the kwarg - it's the "force off without editing
    # code" knob. Set early so every downstream gpufl call sees it.
    global _disabled
    if _env_disabled() or not enabled:
        _disabled = True
        return False
    # Re-enable on a successful enabled-init in case a prior call in
    # this process disabled it. Lets the same interpreter session run
    # disabled-then-enabled tests / notebooks without restarting.
    _disabled = False

    # Removed in v1.2: reject the old kwargs with a clear migration
    # message instead of a cryptic "unexpected keyword argument" from
    # the C++ binding.
    if 'remote_upload' in kwargs:
        raise TypeError(
            "init(remote_upload=...) was removed in v1.2. Live HTTP "
            "streaming is gone; upload happens after the session ends. "
            "Use `with gpufl.session(backend_url=..., api_key=...):` or "
            "call gpufl.upload_logs(...) after gpufl.shutdown().")
    if 'sampling_auto_start' in kwargs:
        raise TypeError(
            "init(sampling_auto_start=...) was removed in v1.2; use "
            "'continuous_system_sampling' instead (False = sample only "
            "inside GFL_SCOPE / between systemStart/stop; True = sample "
            "continuously from init to shutdown).")
    if 'backend_url' in kwargs or 'api_key' in kwargs:
        raise TypeError(
            "init(backend_url=...) / init(api_key=...) were removed in v1.2 - "
            "backend credentials live on the upload path now. Pass them to "
            "gpufl.upload_logs(backend_url=..., api_key=...) or "
            "`with gpufl.session(backend_url=..., api_key=...):` instead.")

    # EAGER module loading is OPT-IN (default: CUDA's normal LAZY). The
    # per-architecture SASS exclusion gate (GPUFL_SASS_EXCLUDE_ARCHS) is the
    # default guard for the CUPTI lazy-patching deadlock. Set
    # GPUFL_EAGER_MODULE_LOADING=1 to force EAGER for this run instead - must
    # happen before the training loop creates the CUDA context, hence here.
    _apply_eager_module_loading(kwargs.get('profiling_engine'))

    # ── Multi-pass analysis grouping ──────────
    # Let an embedded job self-tag as one pass of an analysis group without
    # going through the launcher's --passes driver. gpufl::init() reads
    # GPUFL_ANALYSIS_ID / _PASS_INDEX / _PASS_COUNT straight from the
    # environment (see include/gpufl/core/gpufl.cpp + env_vars.hpp); writing
    # them via os.environ here makes them visible to that getenv at C++ init
    # in this same process. An explicit kwarg overwrites a pre-set env var
    # (the launcher path), so embedded and launcher use don't both apply.
    # pass_index/pass_count are only honored by the core when analysis_id is
    # set, so we gate them the same way.
    if analysis_id is not None:
        os.environ['GPUFL_ANALYSIS_ID'] = str(analysis_id)
        if pass_index is not None:
            os.environ['GPUFL_PASS_INDEX'] = str(int(pass_index))
        if pass_count is not None:
            os.environ['GPUFL_PASS_COUNT'] = str(int(pass_count))

    # Remember where this session writes its logs so clean_logs() can
    # default to it later, and guard against wiping an active session.
    global _session_active, _last_log_path, _last_app_name
    _last_app_name = kwargs.get('app_name', args[0] if args else None)
    _last_log_path = kwargs.get('log_path', args[1] if len(args) > 1 else None)

    result = _original_init(*args, **kwargs)
    if result:
        _session_active = True
    return result


# Wrap shutdown so the active-session guard in clean_logs() clears once the
# logs are flushed and closed.
_original_shutdown = shutdown
_original_system_start = system_start
_original_system_stop = system_stop

def shutdown():
    """Stop the runtime, flush and close all log files.

    No-op when gpufl is disabled (init was called with ``enabled=False``
    or the ``GPUFL_DISABLED`` env var was set).
    """
    global _session_active
    if _disabled:
        return None
    try:
        return _original_shutdown()
    finally:
        _session_active = False


def system_start(name="system"):
    """Start a system-monitoring scope. No-op when gpufl is disabled."""
    if _disabled:
        return None
    return _original_system_start(name)


def system_stop(name="system"):
    """Stop a system-monitoring scope. No-op when gpufl is disabled."""
    if _disabled:
        return None
    return _original_system_stop(name)


# ── upload_logs: deferred bulk upload to the backend ────────────────────────
#
# Thin Python wrapper around the C++ uploadLogs() so callers can use
# env-var fallbacks (GPUFL_BACKEND_URL / GPUFL_API_KEY) and default
# log_path to the last session's log location. Always runs synchronously
# in the caller's thread; the C++ side releases the GIL during HTTP so
# other Python threads keep running.
def upload_logs(*, log_path=None, backend_url=None, api_key=None,
                api_path="", total_timeout_ms=5 * 60 * 1000,
                connect_timeout_ms=10 * 1000, read_timeout_ms=30 * 1000,
                max_retries=1, retry_delay_ms=1000,
                cursor_filename=".gpufl-upload-cursor.json",
                report_progress=True,
                session_id=None, all_sessions=False, force=False):
    """Upload a session's NDJSON logs to the GPUFlight backend.

    Call this AFTER `gpufl.shutdown()` returns. The local NDJSON files
    are not deleted - successful uploads only update an internal cursor
    file (`.gpufl-upload-cursor.json` in the log dir) so subsequent
    invocations don't accidentally re-upload completed sessions.

    Session selection (mutually exclusive):
        - default: upload only the LATEST session present in the files
          (the most recent `job_start.ts_ns`).
        - session_id=<uuid>: upload only that specific session.
        - all_sessions=True: upload every session found in the dir.

    Args:
        log_path: Same shape as InitOptions.log_path - both the
            directory AND filename-prefix. Defaults to the last
            init()'s log_path. Raises ValueError if neither is set.
        backend_url: Backend base URL. Env: GPUFL_BACKEND_URL.
        api_key: Bearer token. Env: GPUFL_API_KEY.
        api_path: Override for reverse-proxy mounts. Defaults to /api/v1.
        total_timeout_ms: Hard wall budget for the whole upload.
        connect_timeout_ms / read_timeout_ms: Per-POST timeouts.
        max_retries: Transient-failure retries per POST. Default 1.
        retry_delay_ms: Delay before the retry.
        cursor_filename: Override the cursor filename in the log dir.
        report_progress: Periodic progress log lines on stderr.
        session_id: Upload only this session_id. Mutually exclusive
            with all_sessions.
        all_sessions: Upload every session found in the directory.
            Sessions already in the cursor are skipped silently unless
            force=True.
        force: Re-upload sessions even if the cursor says they've
            already been shipped.

    Returns:
        UploadResult with .success / .events_uploaded / .warnings / etc.
        Never raises on network errors - inspect .success.

    When gpufl is disabled, returns an empty upload-result stand-in
    (``_NoopUploadResult``) with the same field shape as the real
    :class:`UploadResult` - ``success=True``, ``events_uploaded=0``, a
    single warning explaining why. No network calls, no disk I/O. Safe
    to call even when ``gpufl.init()`` was never called this process.
    """
    if _disabled:
        return _NoopUploadResult()
    if not log_path:
        log_path = _last_log_path
    if not log_path:
        raise ValueError(
            "upload_logs: log_path is required (no init() called yet, "
            "or init() was called without log_path). Pass it explicitly.")
    if not backend_url:
        backend_url = os.environ.get('GPUFL_BACKEND_URL', '')
    if not api_key:
        api_key = os.environ.get('GPUFL_API_KEY', '')
    if session_id and all_sessions:
        raise ValueError(
            "upload_logs: session_id and all_sessions are mutually "
            "exclusive - pass only one.")
    return _c_upload_logs(
        log_path=log_path,
        backend_url=backend_url,
        api_key=api_key,
        api_path=api_path,
        total_timeout_ms=total_timeout_ms,
        connect_timeout_ms=connect_timeout_ms,
        read_timeout_ms=read_timeout_ms,
        max_retries=max_retries,
        retry_delay_ms=retry_delay_ms,
        cursor_filename=cursor_filename,
        report_progress=report_progress,
        session_id_filter=session_id or "",
        all_sessions=bool(all_sessions),
        force=bool(force),
    )


# ── Session: top-level context manager around init() / shutdown() ────────────
#
# Wraps init() + shutdown() + optional upload_logs() so callers don't
# have to remember the orchestration. The deferred-upload model means
# nothing reaches the dashboard during the workload - uploading at the
# end is what makes the session visible. session() handles that:
#
#   with gpufl.session(app_name="my_app",
#                      backend_url="https://api.gpuflight.com",
#                      api_key="gpfl_xxxxx"):
#       # ... train ...
#   # On exit: shutdown() runs, then upload_logs() if creds were set.
#
# Without backend_url + api_key (or the matching env vars), the session
# stays fully offline - shutdown happens but no upload is attempted.
# All kwargs forward verbatim to init().
@contextmanager
def session(*args, **kwargs):
    """Run a GPUFlight session as a context manager.

    Equivalent to::

        gpufl.init(*args, **kwargs)
        try:
            yield
        finally:
            gpufl.shutdown()
            if creds were set:
                gpufl.upload_logs(...)

    Yields:
        The value returned by `init()` - truthy on success, False in
        stub / no-GPU mode. Lets callers do
        ``with gpufl.session(...) as ok: if not ok: ...``.
    """
    # Capture credentials for the post-shutdown upload BEFORE init()
    # mutates kwargs (it pops some of them while resolving env fallbacks).
    # Pop (not get) the upload creds: they belong to the upload step, and
    # init() rejects backend_url / api_key / api_path now.
    upload_backend_url = (kwargs.pop('backend_url', None)
                          or os.environ.get('GPUFL_BACKEND_URL', ''))
    upload_api_key    = (kwargs.pop('api_key', None)
                          or os.environ.get('GPUFL_API_KEY', ''))
    upload_api_path   = kwargs.pop('api_path', '')

    result = init(*args, **kwargs)
    try:
        yield result
    finally:
        # shutdown() is idempotent - safe even when init() failed and
        # no real session was started.
        try:
            shutdown()
        except Exception:
            # Don't let a shutdown bug suppress the upload - the local
            # NDJSON files are the source of truth and they're already
            # on disk. Log and continue.
            import traceback
            traceback.print_exc()

        # Deferred upload only if creds are present AND init succeeded.
        # Skipping when init failed avoids uploading empty / partial
        # logs from a stub-mode run that didn't actually capture anything.
        if result and upload_backend_url and upload_api_key:
            try:
                upload_logs(
                    backend_url=upload_backend_url,
                    api_key=upload_api_key,
                    api_path=upload_api_path,
                )
            except Exception as e:
                # upload_logs is supposed to never raise on network
                # errors, but a bad input could (e.g. missing log_path).
                # Print and swallow - the with-block must not raise
                # from cleanup.
                print(f"[gpufl.session] upload_logs failed: {e}",
                      file=sys.stderr)


# ── Scope: context manager + benchmark-iteration helper ─────────────────────
#
# Thin Python wrapper over the C++ `_CScope`. Backward compatible: used as a
# `with` block it behaves exactly as before. New in 1.0.3 it is also
# *iterable* when constructed with `repeat=N`, which removes the
# `with Scope(...): for _ in range(N):` boilerplate and - via `warmup=K` -
# lets you exclude cold-start iterations (JIT compile, cold caches) from the
# measured/profiled window without hand-writing two loops.
class Scope:
    """A logical profiling scope.

    Two usage forms:

    1. Context manager (unchanged) - bracket an arbitrary block::

           with gpufl.Scope("inference", "ml"):
               model(x)

    2. Iterable benchmark loop (requires ``repeat``)::

           for _ in gpufl.Scope("matmul", "math", repeat=10, warmup=3):
               matmul_kernel[bpg, tpb](A, B, C)

       The scope opens once, brackets all ``repeat`` measured iterations,
       and closes when the loop ends - even if the body raises.

    Args:
        name:   Scope name shown in the report / dashboard.
        tag:    Optional category tag (e.g. "math", "ml").
        repeat: If set, the scope becomes iterable and yields this many
                *measured* iterations (indices ``0 .. repeat-1``). Required
                to iterate; a ``with`` block ignores it.
        warmup: Iterations run BEFORE the scope opens - their work executes
                but is excluded from the scope's timing/profiling. They yield
                negative indices (``-warmup .. -1``) so ``i >= 0`` marks a
                measured iteration. Only meaningful together with ``repeat``.
    """

    def __init__(self, name, tag="", *, repeat=None, warmup=0):
        if repeat is not None and repeat < 0:
            raise ValueError("Scope(repeat=...) must be >= 0")
        if warmup < 0:
            raise ValueError("Scope(warmup=...) must be >= 0")
        self._name = name
        self._tag = tag
        self._repeat = repeat
        self._warmup = warmup
        self._inner = None

    # --- context-manager protocol (backward compatible) ---
    def __enter__(self):
        # Disabled mode: skip the C++ scope so we don't touch the
        # uninitialised runtime. `with gpufl.Scope(...):` still works,
        # just instrumentation-free.
        if _disabled:
            return self
        self._inner = _CScope(self._name, self._tag)
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._inner is not None:
            inner, self._inner = self._inner, None
            return inner.__exit__(exc_type, exc_value, traceback)
        return False

    # --- iterator protocol (new in 1.0.3) ---
    def __iter__(self):
        # Validate eagerly so `for _ in Scope("x")` (no repeat) raises on
        # loop entry rather than deferring until the first next().
        if self._repeat is None:
            raise TypeError(
                "Scope is only iterable when constructed with repeat=N. "
                "Use it as a context manager - `with gpufl.Scope(...):` - "
                "otherwise."
            )
        return self._iterate()

    def _iterate(self):
        # Disabled mode: iterate the same warmup-then-measured indices
        # the caller's benchmark loop expects, but skip the C++ scopes
        # entirely. Same yield contract → caller code is unchanged.
        if _disabled:
            for w in range(self._warmup):
                yield w - self._warmup
            for i in range(self._repeat):
                yield i
            return
        # Warmup: open a "<name>_warmup" sub-scope so the kernel events
        # emitted during warmup are attributed to a separately
        # identifiable bucket - same convention as the C++ BenchInvoker,
        # keeps Python and C++ logs interchangeable. The sub-scope's
        # BEGIN row carries repeat=warmup so per-iteration cold-start
        # cost can be computed by the analyzer.
        if self._warmup > 0:
            warmup_inner = _CScope(self._name + "_warmup", self._tag,
                                   repeat=self._warmup, warmup=0)
            warmup_inner.__enter__()
            try:
                for w in range(self._warmup):
                    yield w - self._warmup  # -warmup .. -1
            finally:
                warmup_inner.__exit__(None, None, None)
        # Measured: open the main scope, yield repeat times, always close.
        # Pass repeat/warmup through to the C++ scope so they land on the
        # BEGIN row of scope_event_batch - the analyzer / backend then
        # derive per-iteration metrics without the caller doing math.
        self._inner = _CScope(self._name, self._tag,
                              repeat=self._repeat, warmup=self._warmup)
        self._inner.__enter__()
        try:
            for i in range(self._repeat):
                yield i  # 0 .. repeat-1
        finally:
            inner, self._inner = self._inner, None
            inner.__exit__(None, None, None)


# ── clean_logs: wipe a session's NDJSON logs to start fresh ──────────────────
def clean_logs(log_path=None, log_prefix=None, *, dry_run=False):
    """Delete this app's NDJSON log files so the next run starts clean.

    v1.2 disk layout: logs live at ``<log_path>/<session_id>/<channel>.log``
    (and rotated ``<channel>.<N>.log[.gz]``). ``clean_logs()`` walks each
    session subdirectory and removes its channel files and the now-empty
    subdir. Unrelated files in ``<log_path>`` itself are never touched,
    and unrelated subdirectories whose contents don't match the channel
    pattern are skipped intact.

    Target resolution:
        * No args  → the ``log_path`` (or ``app_name``) of the most recent
          ``gpufl.init()`` call in this process.
        * ``log_path`` → the directory that contains session subdirs.
        * ``log_prefix`` → DEPRECATED in v1.2 (the layout has no filename
          prefix anymore). Kept as a no-op argument for one release so
          callers don't break; remove in v1.3.

    Safety:
        * **Refuses while a session is active in THIS process** - if you've
          called ``init()`` without a matching ``shutdown()`` it raises
          ``RuntimeError`` rather than delete logs you're still writing.
        * **Does NOT detect the ``gpufl-monitor`` sidecar / agent running in
          another process.** On Windows a file the agent has open cannot be
          deleted, so the OS protects you (the file is skipped with a
          warning). On Linux an unlink succeeds even while the agent holds
          the file open, which can confuse a live tail - **stop the agent
          before calling this, or use ``dry_run=True`` first to preview.**

    Args:
        log_path:   Directory whose session subdirs to remove. Defaults to
                    the last ``init()``'s location.
        log_prefix: Ignored in v1.2 (kept for backward-compat with v1.1
                    callers - emits a DeprecationWarning if passed).
        dry_run:    If True, return the list of matching files WITHOUT
                    deleting anything.

    Returns:
        list[str]: The files removed (or, with ``dry_run``, the files that
        would be removed). Empty subdirectories that contained only
        removed channel files are also deleted (and listed).
    """
    if _session_active:
        raise RuntimeError(
            "clean_logs() refused: a gpufl session is active in this process "
            "(init() was called without a matching shutdown()). Call "
            "gpufl.shutdown() first, then clean_logs()."
        )

    if log_prefix is not None:
        import warnings as _w
        _w.warn(
            "clean_logs(log_prefix=...) is deprecated in v1.2 - the new "
            "disk layout has no filename prefix (logs live at "
            "<log_path>/<session_id>/<channel>.log). The argument is "
            "ignored. Pass only log_path=... going forward.",
            DeprecationWarning,
            stacklevel=2,
        )

    if log_path is None:
        log_path = _last_log_path or _last_app_name
    if not log_path:
        raise ValueError(
            "clean_logs(): no log_path given and no prior init() to infer "
            "one from. Pass log_path=... explicitly."
        )

    # v1.2: log_path is the directory of session subdirs. v1.1 callers
    # might have passed `<dir>/<prefix>` or `<dir>/<prefix>.log` - strip
    # a trailing `.log` so either still resolves to a sensible dir.
    if log_path.endswith(".log"):
        log_path = log_path[:-4]

    import re
    import warnings

    # Each session subdir's channel files match this pattern.
    # device.log / device.log.gz / device.5.log / sass.1.log.gz / etc.
    channel_re = re.compile(r"^(?:device|scope|system|sass)(?:\.\d+)?\.log(?:\.gz)?$")

    matched = []
    if os.path.isdir(log_path):
        for entry in sorted(os.listdir(log_path)):
            sub = os.path.join(log_path, entry)
            if not os.path.isdir(sub):
                continue
            # Collect this subdir's channel files. If ALL its contents
            # match the channel pattern, the subdir itself goes too.
            subdir_files = []
            try:
                names = os.listdir(sub)
            except OSError:
                continue
            all_match = bool(names)
            for fname in names:
                if channel_re.match(fname):
                    subdir_files.append(os.path.join(sub, fname))
                else:
                    all_match = False
            matched.extend(sorted(subdir_files))
            if all_match:
                matched.append(sub)   # subdir itself is removable after files

    if dry_run:
        return matched

    removed = []
    for p in matched:
        try:
            if os.path.isdir(p):
                os.rmdir(p)            # only empty dirs reach here
            else:
                os.remove(p)
            removed.append(p)
        except OSError as e:
            # On Windows an open file (e.g. the sidecar agent tailing it)
            # raises PermissionError - skipping it IS the safety net against
            # deleting a file that's in use. Warn rather than crash.
            warnings.warn(f"[gpufl] clean_logs: could not remove {p}: {e}")
    return removed


# ── Targeting: bounded multi-pass profiling for embedded jobs ───────────
# Defined in a submodule (keeps __init__ lean); imported here, after Scope /
# session exist, so `gpufl.targeting(...)` resolves. The submodule lazy-imports
# Scope/session at call time, so this import is cycle-free.
from ._targeting import targeting  # noqa: E402

__all__ = [
    "Scope", "init", "shutdown", "session", "clean_logs", "targeting",
    "system_start", "system_stop",
    "BackendKind", "InitOptions", "ProfilingEngine",
    "upload_logs", "UploadOptions", "UploadResult",
]
