"""
Higher-level convenience wrappers over attach() / detach().

Provides a context manager (`profile`) and a model-watch helper
(`watch`) for users who prefer scoped enable/disable rather than
process-global attach/detach.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional, TYPE_CHECKING

from .dispatch import attach, detach

if TYPE_CHECKING:
    import torch.nn as nn


@contextmanager
def profile(name: Optional[str] = None):
    """
    Context manager: enable PyTorch op-level capture for the duration
    of the `with` block. If `name` is provided, wrap the whole block
    in a GFL_SCOPE so all ops inside nest under that named region.

    Usage:
        with gpufl.torch.profile("forward_pass"):
            out = model(batch)
            loss = criterion(out, target)

    Equivalent manual form:
        with gpufl.Scope("forward_pass"):
            gpufl.torch.attach()
            try:
                out = model(batch)
                loss = criterion(out, target)
            finally:
                gpufl.torch.detach()
    """
    # Optional outer GFL_SCOPE for named grouping in the dashboard.
    scope_cm = None
    if name is not None:
        # Lazy import to keep `gpufl.torch` usable when the C++ extension
        # isn't loaded (e.g. importing for introspection).
        import gpufl
        scope_cm = gpufl.Scope(name)
        scope_cm.__enter__()

    attach()
    try:
        yield
    finally:
        detach()
        if scope_cm is not None:
            scope_cm.__exit__(None, None, None)


def watch(model: "nn.Module") -> "nn.Module":
    """
    Wrap a model's forward pass with op-level capture. Returns the
    same model (for chaining); no forward-hook magic is installed,
    so this is equivalent to calling `attach()` once before training
    and `detach()` when done.

    This helper exists for API symmetry with frameworks like
    `torch.profiler`; for most workflows calling `attach()` at program
    start and `detach()` at shutdown is simpler.

    Usage:
        model = gpufl.torch.watch(model)  # same model back, capture enabled
        # ... training / inference as usual ...
        # (capture stays on until program exit or explicit detach())
    """
    attach()
    return model
