import os
import sys
if os.name == 'nt':
    cuda_path = os.environ.get('CUDA_PATH')
    if cuda_path:
        bin_path = os.path.join(cuda_path, 'bin')
        if os.path.exists(bin_path):
            try:
                os.add_dll_directory(bin_path)
            except AttributeError:
                pass

# 2. Import C++ Core Bindings
try:
    from ._gpufl_client import Scope, init, shutdown, log_kernel
except Exception as e:
    if os.environ.get("GITHUB_ACTIONS") == "true":
        raise RuntimeError(
            "Failed to import native extension gpufl._gpufl_client. "
            "This usually means a missing DLL dependency on Windows "
            "(e.g., nvml.dll or CUDA runtime). Original error: "
            f"{repr(e)}"
        ) from e

    # For local dev, keep a safe fallback (optional)
    def init(*args, **kwargs): return False
    def shutdown(): return None
    def log_kernel(*args): return None
    class Scope:
        def __init__(self, *args): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass

# 3. Import Python Utilities (The Numba Wrapper)
try:
    from .utils import launch_kernel
except ImportError:
    launch_kernel = None

# 4. Define Public API
__all__ = ["Scope", "init", "shutdown", "log_kernel", "launch_kernel"]
