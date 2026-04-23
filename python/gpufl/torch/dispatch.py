"""
TorchDispatchMode subclass that emits NVTX ranges for every torch op.

The dispatch mode is a process-global switch: `attach()` enables it,
`detach()` disables. Nesting is supported (stacking attach() calls is a
no-op — only the first enables, only the last disables).
"""

from __future__ import annotations

import threading
from typing import Optional

from .stack import capture_python_site

# We import torch lazily so `import gpufl.torch` doesn't fail on systems
# without torch installed. The attach() call will raise a clearer error
# than ImportError if torch isn't available.
_TORCH_IMPORT_ERROR_MESSAGE = (
    "gpufl.torch requires the `torch` package (version 2.1 or higher).\n"
    "Install it via: pip install gpufl[torch]"
)


class _GpuflDispatchMode:
    """Lazily-constructed holder for the TorchDispatchMode subclass.

    Constructed on first call to attach() so `import gpufl.torch` works
    even without torch installed (just calling attach would then fail).
    """

    _lock = threading.Lock()
    _instance: Optional[object] = None  # TorchDispatchMode instance
    _attach_count = 0
    _max_stack_depth = 16

    @classmethod
    def _build_mode_class(cls):
        try:
            import torch  # noqa: F401
            from torch.utils._python_dispatch import TorchDispatchMode
            from torch.cuda import nvtx
        except ImportError as e:
            raise RuntimeError(_TORCH_IMPORT_ERROR_MESSAGE) from e

        class GpuflDispatchMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # Skip meta tensors / fake tensors — they don't launch
                # kernels so the NVTX overhead is wasted. The op still
                # runs normally.
                kwargs = kwargs or {}

                # Capture the Python site that called into torch. This
                # lets the dashboard show "aten::matmul at model.py:42"
                # rather than just "aten::matmul".
                site = capture_python_site(max_depth=_GpuflDispatchMode._max_stack_depth)

                # Compose the NVTX range name:
                #   "torch::<op-name>@<file:line>" or just "torch::<op-name>"
                # The "torch::" prefix lets the backend catalog match
                # these ranges as torch-origin vs GFL_SCOPE-origin.
                op_name = getattr(func, "__name__", str(func))
                range_name = (f"torch::{op_name}@{site}"
                              if site is not None else f"torch::{op_name}")

                nvtx.range_push(range_name)
                try:
                    return func(*args, **kwargs)
                finally:
                    nvtx.range_pop()

        return GpuflDispatchMode


def attach(max_stack_depth: int = 16) -> None:
    """
    Enable PyTorch op-level capture. Idempotent (safe to call multiple
    times — only the first activates; subsequent calls increment a
    reference count).

    Parameters
    ----------
    max_stack_depth : int
        How far up the Python stack to search for the user's call site.
        Higher values find user code in deeply nested frameworks at the
        cost of a few extra frame lookups per op. Default 16 is fine for
        typical PyTorch use. Raise to 32+ if you're using deep model
        wrappers (e.g. HuggingFace Accelerate + DeepSpeed stacks).
    """
    with _GpuflDispatchMode._lock:
        _GpuflDispatchMode._max_stack_depth = max_stack_depth
        _GpuflDispatchMode._attach_count += 1
        if _GpuflDispatchMode._attach_count == 1:
            mode_cls = _GpuflDispatchMode._build_mode_class()
            instance = mode_cls()
            instance.__enter__()
            _GpuflDispatchMode._instance = instance


def detach() -> None:
    """
    Disable PyTorch op-level capture. Balances a previous `attach()`
    call. Safe to call more times than attach() (extra calls are no-ops).
    """
    with _GpuflDispatchMode._lock:
        if _GpuflDispatchMode._attach_count == 0:
            return
        _GpuflDispatchMode._attach_count -= 1
        if _GpuflDispatchMode._attach_count == 0 and _GpuflDispatchMode._instance is not None:
            try:
                _GpuflDispatchMode._instance.__exit__(None, None, None)
            finally:
                _GpuflDispatchMode._instance = None
