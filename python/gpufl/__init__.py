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
    from ._gpufl_client import Scope, init, shutdown
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
    class Scope:
        def __init__(self, *args): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass

__all__ = ["Scope", "init", "shutdown"]
