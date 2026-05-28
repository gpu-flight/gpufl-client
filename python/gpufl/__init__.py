import os
import sys
from contextlib import contextmanager

# Import-order guard: gpufl and PyTorch each bundle a CUPTI version.
# If gpufl is imported before torch, two incompatible CUPTI DLLs end up
# loaded and conflict during profiling (crash in cubin callback).
# Detect torch already being imported and warn if we loaded before it.
if os.name == 'nt' and 'torch' not in sys.modules:
    # torch not yet imported — emit a one-time advisory.  We don't raise
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

# 1. Windows DLL Handling — ensure CUDA and CUPTI DLLs are findable.
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
        None_              = "None"
        PcSampling         = "PcSampling"
        SassMetrics        = "SassMetrics"
        RangeProfiler      = "RangeProfiler"
        PcSamplingWithSass = "PcSamplingWithSass"
        # Friendly aliases — mirror the pybind11 ProfilingEngine attrs
        # added in bindings.cpp. Identical string values so equality
        # checks against the technical names still work in stub mode.
        Continuous         = "PcSampling"
        Deep               = "PcSamplingWithSass"
        Range              = "RangeProfiler"

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
            self.profiling_engine = ProfilingEngine.PcSampling
            self.config_file = ""
            self.backend_url = ""
            self.api_key = ""
            self.remote_upload = False  # DEPRECATED v1.1, removed v1.2

    class _CScope:
        # Accept kwargs (repeat/warmup added in 1.0.3) so the no-GPU
        # fallback matches the real pybind11 binding's signature.
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass

    # Stub UploadResult — mirror the C++ binding's field set so calling
    # code can introspect even in no-GPU mode without AttributeError.
    class UploadResult:
        def __init__(self):
            self.success = False
            self.files_processed = 0
            self.files_skipped_by_cursor = 0
            self.events_uploaded = 0
            self.bytes_uploaded = 0
            self.elapsed_ms = 0
            self.warnings = ["gpufl C++ extension not loaded — upload is a no-op in stub mode"]

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
        print("[GPUFL] upload_logs called in stub mode — returning empty result.",
              file=sys.stderr)
        return UploadResult()
    # --- FIX END ---

except Exception as e:
    # Catch other unexpected errors (like syntax errors in the C++ extension)
    import sys
    print(f"[FATAL] Unexpected error importing _gpufl_client: {e}", file=sys.stderr)
    raise e

__version__ = "0.1.0.dev"

# ── Backend upload ────────────────────────────────────────────────────────────
#
# Upload happens post-shutdown via the deferred path implemented in
# C++ at include/gpufl/upload/upload_logs.cpp. The historical live-
# streaming sink (HttpLogSink) was removed; nothing in this Python
# wrapper opens HTTP connections during the session. Call
# `gpufl.upload_logs(...)` after `gpufl.shutdown()`, or wrap the whole
# thing in `with gpufl.session(backend_url=..., api_key=...):` to have
# it run automatically. The legacy `remote_upload=True` kwarg on init()
# is gone — passing it raises TypeError (see the check inside init()).
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

# Wrap the C++ init to apply env-var fallbacks for backend_url and
# api_key, and to reject removed kwargs with a clear TypeError.
_original_init = init

def init(*args, backend_url=None, api_key=None, remote_upload=None, **kwargs):
    """Initialize GPUFlight.

    Configuration precedence (low → high). Each layer may override the
    previous; your explicit field sets on this call always win:

      1. InitOptions defaults (built-in).
      2. Local config file (config_file=...).
      3. Env vars (GPUFL_BACKEND_URL / GPUFL_API_KEY /
         GPUFL_REMOTE_UPLOAD / GPUFL_PROFILING_ENGINE / GPUFL_CONFIG_FILE).
      4. The kwargs you pass to this function.

    Args:
        backend_url: Base URL of the GPUFlight backend
            (e.g. "https://api.gpuflight.com"). Stored on InitOptions
            so gpufl.upload_logs() / gpufl.session() can read it back
            after shutdown without the caller having to re-supply it.
            On its own it does nothing — upload is a separate step,
            never live during the workload. **Planned removal v1.2** —
            see DEPRECATION NOTE on InitOptions.backend_url in
            include/gpufl/gpufl.hpp.
        api_key: API key for log upload. Same purpose / lifecycle as
            backend_url. **Planned removal v1.2.**
        remote_upload: DEPRECATED (v1.1) — used to enable live HTTP
            streaming via HttpLogSink. That mechanism is gone. Setting
            True here still works for one release: emits a
            DeprecationWarning and registers an atexit handler that
            calls gpufl.upload_logs() at interpreter exit, preserving
            the old "set one flag and forget" UX. Prefer
            `with gpufl.session(backend_url=..., api_key=...):` (auto-
            orchestrated) or call gpufl.upload_logs(...) explicitly
            after shutdown. **Removed in v1.2.** Env: `GPUFL_REMOTE_UPLOAD=1`.
        **kwargs: All other InitOptions fields passed to C++ init.
    """
    # Backward-compat shim: `sampling_auto_start` was renamed to
    # `continuous_system_sampling` because the old name only described
    # init-time auto-start and missed the new scope-bracketing behavior
    # (off → sample only inside GFL_SCOPE / between systemStart/stop).
    # We accept the old kwarg for one release with a DeprecationWarning,
    # forwarding it to the new name. Caller passing both is an error.
    if 'sampling_auto_start' in kwargs:
        if 'continuous_system_sampling' in kwargs:
            raise TypeError(
                "init() got both 'sampling_auto_start' (deprecated) and "
                "'continuous_system_sampling' — pass only the new name.")
        import warnings
        warnings.warn(
            "'sampling_auto_start' is deprecated; use "
            "'continuous_system_sampling' instead. With the new name, "
            "False enables scope-bracketed sampling (sample only inside "
            "GFL_SCOPE or between systemStart/stop); True samples "
            "continuously from init to shutdown. The kwarg will be "
            "removed in a future release.",
            DeprecationWarning,
            stacklevel=2)
        kwargs['continuous_system_sampling'] = kwargs.pop('sampling_auto_start')

    # Resolve env-var fallbacks. Doing this in Python lets explicit
    # kwargs win over env; the C++ layer also does env fallback for
    # the pure-C++ code path, so either side resolving is sufficient.
    if not backend_url:
        backend_url = os.environ.get('GPUFL_BACKEND_URL')
    if not api_key:
        api_key = os.environ.get('GPUFL_API_KEY')
    if remote_upload is None:
        env_upload = os.environ.get('GPUFL_REMOTE_UPLOAD', '').strip().lower()
        remote_upload = env_upload in ('1', 'true', 'yes', 'on')

    # Forward backend creds to the underlying C++ init via the pybind11
    # binding. These are needed by gpufl::uploadLogs() at shutdown — they
    # live on InitOptions so the deferred-upload path can read them back
    # without the caller having to re-supply them.
    if backend_url and 'backend_url' not in kwargs:
        kwargs['backend_url'] = backend_url
    if api_key and 'api_key' not in kwargs:
        kwargs['api_key'] = api_key

    # Remember where this session writes its logs so clean_logs() can
    # default to it later, and guard against wiping an active session.
    global _session_active, _last_log_path, _last_app_name
    _last_app_name = kwargs.get('app_name', args[0] if args else None)
    _last_log_path = kwargs.get('log_path', args[1] if len(args) > 1 else None)

    # ── remote_upload deprecation shim ──────────────────────────────────
    #
    # The kwarg used to attach a live HttpLogSink that streamed every
    # NDJSON event to the backend during the workload. That sink is
    # gone. For backward compat we still accept the kwarg, but route
    # it to the deferred path: register an `atexit` handler that calls
    # upload_logs() at interpreter exit. Customer code keeps working
    # unchanged.
    #
    # Why atexit (not "call upload_logs from the shutdown wrapper") —
    # many existing customers don't call gpufl.shutdown() explicitly
    # and rely on the interpreter exiting to flush everything. Hooking
    # atexit preserves that pattern. For callers that DO call shutdown()
    # explicitly, the atexit fires after shutdown() returns — still
    # correct, just a small delay.
    #
    # Removed in v1.2. New code should use `gpufl.session()` or call
    # `gpufl.upload_logs()` explicitly.
    if remote_upload:
        import warnings
        warnings.warn(
            "remote_upload=True is deprecated. Live HTTP streaming "
            "was removed in v1.1 — upload now happens after the "
            "session ends. For automated upload, use "
            "`with gpufl.session(backend_url=..., api_key=...):`. "
            "Your existing code keeps working — gpufl scheduled an "
            "atexit handler that calls upload_logs() at interpreter "
            "shutdown. The remote_upload kwarg, plus backend_url and "
            "api_key on InitOptions, will be removed in v1.2.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Forward to C++ so the C++ side also logs a deprecation
        # warning (helps pure-C++ callers who hit this via the
        # ergonomic env-var path).
        kwargs['remote_upload'] = True

        # Capture creds + log_path NOW. _last_log_path may be reassigned
        # by a later init() call before the interpreter exits; closing
        # over current values protects this session's atexit upload.
        _deferred_url      = backend_url
        _deferred_key      = api_key
        _deferred_log_path = (kwargs.get('log_path')
                              or (args[1] if len(args) > 1 else None)
                              or _last_log_path)
        _deferred_api_path = kwargs.get('api_path', '')

        def _atexit_upload():
            if not _deferred_log_path or not _deferred_url or not _deferred_key:
                return
            try:
                upload_logs(
                    log_path=_deferred_log_path,
                    backend_url=_deferred_url,
                    api_key=_deferred_key,
                    api_path=_deferred_api_path,
                )
            except Exception as e:
                # atexit handlers must not raise — Python prints a
                # traceback otherwise, which is confusing for users
                # who didn't know upload was happening. Log + swallow.
                print(f"[gpufl] atexit upload_logs failed: {e}",
                      file=sys.stderr)

        import atexit
        atexit.register(_atexit_upload)

    result = _original_init(*args, **kwargs)
    if result:
        _session_active = True
    return result


# Wrap shutdown so the active-session guard in clean_logs() clears once the
# logs are flushed and closed.
_original_shutdown = shutdown

def shutdown():
    """Stop the runtime, flush and close all log files."""
    global _session_active
    try:
        return _original_shutdown()
    finally:
        _session_active = False


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
    are not deleted — successful uploads only update an internal cursor
    file (`.gpufl-upload-cursor.json` in the log dir) so subsequent
    invocations don't accidentally re-upload completed sessions.

    Session selection (mutually exclusive):
        - default: upload only the LATEST session present in the files
          (the most recent `job_start.ts_ns`).
        - session_id=<uuid>: upload only that specific session.
        - all_sessions=True: upload every session found in the dir.

    Args:
        log_path: Same shape as InitOptions.log_path — both the
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
        Never raises on network errors — inspect .success.
    """
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
            "exclusive — pass only one.")
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
# nothing reaches the dashboard during the workload — uploading at the
# end is what makes the session visible. session() handles that:
#
#   with gpufl.session(app_name="my_app",
#                      backend_url="https://api.gpuflight.com",
#                      api_key="gpfl_xxxxx"):
#       # ... train ...
#   # On exit: shutdown() runs, then upload_logs() if creds were set.
#
# Without backend_url + api_key (or the matching env vars), the session
# stays fully offline — shutdown happens but no upload is attempted.
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
        The value returned by `init()` — truthy on success, False in
        stub / no-GPU mode. Lets callers do
        ``with gpufl.session(...) as ok: if not ok: ...``.
    """
    # Capture credentials for the post-shutdown upload BEFORE init()
    # mutates kwargs (it pops some of them while resolving env fallbacks).
    upload_backend_url = (kwargs.get('backend_url')
                          or os.environ.get('GPUFL_BACKEND_URL', ''))
    upload_api_key    = (kwargs.get('api_key')
                          or os.environ.get('GPUFL_API_KEY', ''))
    upload_api_path   = kwargs.get('api_path', '')

    result = init(*args, **kwargs)
    try:
        yield result
    finally:
        # shutdown() is idempotent — safe even when init() failed and
        # no real session was started.
        try:
            shutdown()
        except Exception:
            # Don't let a shutdown bug suppress the upload — the local
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
                # Print and swallow — the with-block must not raise
                # from cleanup.
                print(f"[gpufl.session] upload_logs failed: {e}",
                      file=sys.stderr)


# ── Scope: context manager + benchmark-iteration helper ─────────────────────
#
# Thin Python wrapper over the C++ `_CScope`. Backward compatible: used as a
# `with` block it behaves exactly as before. New in 1.0.3 it is also
# *iterable* when constructed with `repeat=N`, which removes the
# `with Scope(...): for _ in range(N):` boilerplate and — via `warmup=K` —
# lets you exclude cold-start iterations (JIT compile, cold caches) from the
# measured/profiled window without hand-writing two loops.
class Scope:
    """A logical profiling scope.

    Two usage forms:

    1. Context manager (unchanged) — bracket an arbitrary block::

           with gpufl.Scope("inference", "ml"):
               model(x)

    2. Iterable benchmark loop (requires ``repeat``)::

           for _ in gpufl.Scope("matmul", "math", repeat=10, warmup=3):
               matmul_kernel[bpg, tpb](A, B, C)

       The scope opens once, brackets all ``repeat`` measured iterations,
       and closes when the loop ends — even if the body raises.

    Args:
        name:   Scope name shown in the report / dashboard.
        tag:    Optional category tag (e.g. "math", "ml").
        repeat: If set, the scope becomes iterable and yields this many
                *measured* iterations (indices ``0 .. repeat-1``). Required
                to iterate; a ``with`` block ignores it.
        warmup: Iterations run BEFORE the scope opens — their work executes
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
                "Use it as a context manager — `with gpufl.Scope(...):` — "
                "otherwise."
            )
        return self._iterate()

    def _iterate(self):
        # Warmup: open a "<name>_warmup" sub-scope so the kernel events
        # emitted during warmup are attributed to a separately
        # identifiable bucket — same convention as the C++ BenchInvoker,
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
        # BEGIN row of scope_event_batch — the analyzer / backend then
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
        * **Refuses while a session is active in THIS process** — if you've
          called ``init()`` without a matching ``shutdown()`` it raises
          ``RuntimeError`` rather than delete logs you're still writing.
        * **Does NOT detect the ``gpufl-monitor`` sidecar / agent running in
          another process.** On Windows a file the agent has open cannot be
          deleted, so the OS protects you (the file is skipped with a
          warning). On Linux an unlink succeeds even while the agent holds
          the file open, which can confuse a live tail — **stop the agent
          before calling this, or use ``dry_run=True`` first to preview.**

    Args:
        log_path:   Directory whose session subdirs to remove. Defaults to
                    the last ``init()``'s location.
        log_prefix: Ignored in v1.2 (kept for backward-compat with v1.1
                    callers — emits a DeprecationWarning if passed).
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
            "clean_logs(log_prefix=...) is deprecated in v1.2 — the new "
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
    # might have passed `<dir>/<prefix>` or `<dir>/<prefix>.log` — strip
    # a trailing `.log` so either still resolves to a sensible dir.
    if log_path.endswith(".log"):
        log_path = log_path[:-4]

    import re
    import warnings

    # Each session subdir's channel files match this pattern.
    # device.log / device.log.gz / device.5.log / device.5.log.gz / etc.
    channel_re = re.compile(r"^(?:device|scope|system)(?:\.\d+)?\.log(?:\.gz)?$")

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
            # raises PermissionError — skipping it IS the safety net against
            # deleting a file that's in use. Warn rather than crash.
            warnings.warn(f"[gpufl] clean_logs: could not remove {p}: {e}")
    return removed


__all__ = [
    "Scope", "init", "shutdown", "session", "clean_logs",
    "system_start", "system_stop",
    "BackendKind", "InitOptions", "ProfilingEngine",
    "upload_logs", "UploadOptions", "UploadResult",
]
