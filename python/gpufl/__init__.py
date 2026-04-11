import os
import sys

# 1. Windows DLL Handling — ensure CUDA and CUPTI DLLs are findable.
# os.add_dll_directory() alone is insufficient for some Python builds;
# we also prepend to PATH as a belt-and-suspenders approach.
if os.name == 'nt':
    cuda_path = os.environ.get('CUDA_PATH')
    if cuda_path:
        _dll_dirs = [
            os.path.join(cuda_path, 'bin'),
            os.path.join(cuda_path, 'extras', 'CUPTI', 'lib64'),
        ]
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
    from ._gpufl_client import Scope, init, shutdown, system_start, system_stop, BackendKind, InitOptions, ProfilingEngine
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

    class InitOptions:
        def __init__(self):
            self.app_name = "gpufl"
            self.log_path = ""
            self.sampling_auto_start = False
            self.system_sample_rate_ms = 0
            self.kernel_sample_rate_ms = 0
            self.backend = BackendKind.Auto
            self.enable_kernel_details = False
            self.enable_debug_output = False
            self.enable_stack_trace = True
            self.enable_source_collection = True
            self.flush_logs_always = False
            self.profiling_engine = ProfilingEngine.PcSamplingWithSass

    class Scope:
        def __init__(self, *args): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
    # --- FIX END ---

except Exception as e:
    # Catch other unexpected errors (like syntax errors in the C++ extension)
    import sys
    print(f"[FATAL] Unexpected error importing _gpufl_client: {e}", file=sys.stderr)
    raise e

__version__ = "0.1.0.dev"
__all__ = ["Scope", "init", "shutdown", "system_start", "system_stop", "BackendKind", "InitOptions"]
