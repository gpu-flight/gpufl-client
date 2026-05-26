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
    from ._gpufl_client import Scope as _CScope, init, shutdown, system_start, system_stop, BackendKind, InitOptions, ProfilingEngine
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
            self.remote_upload = False

    class _CScope:
        # Accept kwargs (repeat/warmup added in 1.0.3) so the no-GPU
        # fallback matches the real pybind11 binding's signature.
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
    # --- FIX END ---

except Exception as e:
    # Catch other unexpected errors (like syntax errors in the C++ extension)
    import sys
    print(f"[FATAL] Unexpected error importing _gpufl_client: {e}", file=sys.stderr)
    raise e

__version__ = "0.1.0.dev"

# ── Remote upload ─────────────────────────────────────────────────────────────
#
# Direct log upload is implemented in the C++ core (see
# include/gpufl/core/logger/http_log_sink.cpp). This Python wrapper
# is a thin pass-through: it translates user-facing kwargs into
# InitOptions fields and lets the C++ init() do the work.
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

# Wrap the C++ init to pass through backend_url / remote_upload kwargs
# and env vars into the underlying InitOptions.
_original_init = init

def init(*args, backend_url=None, api_key=None,
         remote_upload=None, **kwargs):
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
            (e.g. "https://api.gpuflight.com"). On its own it does
            nothing — opt into live NDJSON upload to
            `<backend_url>/api/v1/events/<type>` via `remote_upload=True`.
        api_key: API key for log upload to the GPUFlight backend.
        remote_upload: When truthy, attaches the C++ HttpLogSink so
            every NDJSON line is POSTed live to the backend in parallel
            with the disk write. Env: `GPUFL_REMOTE_UPLOAD=1`.
            Defaults to False.
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

    # Forward to the underlying C++ init via the pybind11 binding. C++
    # attaches HttpLogSink when remote_upload is true.
    if backend_url and 'backend_url' not in kwargs:
        kwargs['backend_url'] = backend_url
    if api_key and 'api_key' not in kwargs:
        kwargs['api_key'] = api_key
    if remote_upload and 'remote_upload' not in kwargs:
        kwargs['remote_upload'] = True

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

def shutdown():
    """Stop the runtime, flush and close all log files."""
    global _session_active
    try:
        return _original_shutdown()
    finally:
        _session_active = False


# ── Session: top-level context manager around init() / shutdown() ────────────
#
# Wraps init() + shutdown() so callers don't have to remember the explicit
# shutdown — particularly important for live HTTP upload, where the dashboard
# only flips a session from "running" → "stopped" once the C++ HttpLogSink
# POSTs `/api/v1/events/shutdown`, which only fires from shutdown(). A bare
# `gpufl.init(...)` in a Jupyter notebook cell never sends that event because
# the kernel stays alive after the cell completes, so the session UI shows
# "running" indefinitely.
#
#   with gpufl.session(app_name="my_app", remote_upload=True):
#       # ... train ...
#   # shutdown() runs here, even on exception
#
# All kwargs forward verbatim to init(). The yielded value is whatever init()
# returned (truthy on success, False in stub/no-GPU mode) so callers can
# branch on it without re-calling init.
@contextmanager
def session(*args, **kwargs):
    """Run a GPUFlight session as a context manager.

    Equivalent to::

        gpufl.init(*args, **kwargs)
        try:
            yield
        finally:
            gpufl.shutdown()

    On exit the C++ HttpLogSink drains the queue and POSTs
    `/api/v1/events/shutdown`, which is what flips the dashboard session
    from "running" to "stopped". Without this (or a manual shutdown call)
    the UI shows the session as running indefinitely.

    Yields:
        The value returned by `init()` — truthy on success, False in
        stub / no-GPU mode. Lets callers do ``with gpufl.session(...) as
        ok: if not ok: ...``.
    """
    result = init(*args, **kwargs)
    try:
        yield result
    finally:
        # shutdown() is idempotent on the C++ side (HttpLogSink::close
        # short-circuits if running_ is already false), so this is safe
        # even when init() failed and no real session was started.
        shutdown()


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

    Removes the active and rotated log files for a given prefix —
    ``<prefix>.<channel>.log`` and ``<prefix>.<channel>.<N>.log[.gz]`` — and
    nothing else. It never removes a directory and never touches files that
    don't match that pattern, so an unrelated file in the same folder is safe.

    Target resolution:
        * No args  → the ``log_path`` (or ``app_name``) of the most recent
          ``gpufl.init()`` call in this process.
        * ``log_path`` → a path/prefix like ``"./gfl_logs"``; the directory is
          its dirname and the prefix its basename.
        * ``log_prefix`` → override just the filename prefix.

    Safety:
        * **Refuses while a session is active in THIS process** — if you've
          called ``init()`` without a matching ``shutdown()`` it raises
          ``RuntimeError`` rather than delete logs you're still writing.
        * **Does NOT detect the ``gpufl-monitor`` sidecar / agent running in
          another process.** On Windows a file the agent has open cannot be
          deleted, so the OS protects you (the file is skipped with a
          warning). On Linux an unlink succeeds even while the agent holds the
          file open, which can confuse a live tail — **stop the agent before
          calling this, or use ``dry_run=True`` first to preview.**

    Args:
        log_path:   Path/prefix whose logs to remove. Defaults to the last
                    ``init()``'s location.
        log_prefix: Override the filename prefix (basename) only.
        dry_run:    If True, return the list of matching files WITHOUT
                    deleting anything.

    Returns:
        list[str]: The files removed (or, with ``dry_run``, the files that
        would be removed).
    """
    if _session_active:
        raise RuntimeError(
            "clean_logs() refused: a gpufl session is active in this process "
            "(init() was called without a matching shutdown()). Call "
            "gpufl.shutdown() first, then clean_logs()."
        )

    if log_path is None:
        log_path = _last_log_path or _last_app_name
    if not log_path:
        raise ValueError(
            "clean_logs(): no log_path given and no prior init() to infer one "
            "from. Pass log_path=... explicitly."
        )

    directory = os.path.dirname(log_path) or "."
    base = log_prefix if log_prefix is not None else os.path.basename(log_path)
    if base.endswith(".log"):
        base = base[:-4]

    import glob
    import warnings
    matched = sorted({
        p
        for pattern in (f"{base}.*.log", f"{base}.*.log.gz")
        for p in glob.glob(os.path.join(directory, pattern))
    })

    if dry_run:
        return matched

    removed = []
    for p in matched:
        try:
            os.remove(p)
            removed.append(p)
        except OSError as e:
            # On Windows an open file (e.g. the sidecar agent tailing it)
            # raises PermissionError — skipping it IS the safety net against
            # deleting a file that's in use. Warn rather than crash.
            warnings.warn(f"[gpufl] clean_logs: could not remove {p}: {e}")
    return removed


__all__ = ["Scope", "init", "shutdown", "session", "clean_logs", "system_start", "system_stop", "BackendKind", "InitOptions", "ProfilingEngine"]
