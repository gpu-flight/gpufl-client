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

        # Lazy-bind the push/pop. `_gpufl_client` is the C++ extension;
        # if it's not available (no-GPU stub install) or it's an older
        # build that predates bindings, we fall back to no-op lambdas
        # so the dispatch hook still runs — but we WARN loudly because
        # this almost always means the user forgot to reinstall the
        # extension after pulling the changes, and silently degrading
        # to "no chips" is the worst possible UX.
        try:
            from gpufl._gpufl_client import (
                _push_external_corr_id as _push_ext,
                _pop_external_corr_id  as _pop_ext,
            )
        except ImportError as e:
            import warnings
            warnings.warn(
                "[gpufl.torch] CUPTI external-correlation bindings not found "
                f"in the loaded gpufl extension ({e}). The dashboard's 'op #N' "
                "chips on kernel rows will be absent until you rebuild + "
                "reinstall the gpufl Python extension "
                "(typically: `pip install . --force-reinstall --no-cache-dir`). "
                "NVTX-based attribution is unaffected and continues to work.",
                RuntimeWarning,
                stacklevel=2,
            )
            def _push_ext(kind, id_):  # type: ignore
                pass
            def _pop_ext(kind):        # type: ignore
                pass

        # CUpti_ExternalCorrelationKind values (from <cupti_activity.h>):
        #   1 = UNKNOWN, 2 = OPENACC, 3 = CUSTOM0 (used by torch.profiler),
        #   4 = CUSTOM1 (we reserve this for gpufl), 5 = CUSTOM2.
        # Picking CUSTOM1 keeps us out of conflict if torch.profiler is
        # also active in the same process (each kind has its own stack).
        _GPUFL_CORR_KIND = 4

        # Per-process op-name → 64-bit id table. The id is what the
        # dashboard's KernelTable surfaces in its "op #N" chip; making
        # it a stable hash of the op name means every `aten::matmul`
        # call across the session shares the same N, which is the
        # property users want when grouping kernels by op.
        _op_id_cache: dict[str, int] = {}
        def _op_id_for(name: str) -> int:
            cached = _op_id_cache.get(name)
            if cached is not None:
                return cached
            # FNV-1a 64-bit. Stable across runs, no cryptographic claim;
            # collision risk for our op-name space is effectively zero.
            h = 0xcbf29ce484222325
            for b in name.encode("utf-8"):
                h ^= b
                h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
            # Avoid 0 — that's the "no attribution" sentinel in the
            # backend / dashboard. If the hash hits 0 (vanishingly
            # unlikely), nudge to 1.
            if h == 0:
                h = 1
            _op_id_cache[name] = h
            return h

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

                # push a CUPTI external-correlation id derived from
                # the op name BEFORE the dispatch runs. Every CUDA kernel
                # launched inside this op (most aten ops launch one or
                # more kernels) will be tagged with this id, surfacing
                # in the dashboard as the "op #N" chip on each kernel
                # row. We push the same id we use for the NVTX label,
                # so the two attribution mechanisms agree.
                op_id = _op_id_for(op_name)
                _push_ext(_GPUFL_CORR_KIND, op_id)

                nvtx.range_push(range_name)
                try:
                    return func(*args, **kwargs)
                finally:
                    nvtx.range_pop()
                    _pop_ext(_GPUFL_CORR_KIND)

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
