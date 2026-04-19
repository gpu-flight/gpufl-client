"""
Python stack-site capture.

The goal: when a torch op fires, find the user's Python file:line — NOT
torch internals, autograd internals, or gpufl internals. Returns a
short "file.py:42" string suitable for embedding in an NVTX range name.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

# Frames we want to skip when walking up the stack. Any frame whose
# filename contains one of these path fragments is treated as "framework
# plumbing" and skipped until we find a user frame.
_FRAMEWORK_PATH_FRAGMENTS = (
    os.path.join("torch", ""),
    os.path.join("gpufl", ""),
    os.path.join("site-packages", "torch"),
    os.path.join("site-packages", "torch", ""),
    os.path.join("torch", "utils"),
    os.path.join("torch", "_subclasses"),
    os.path.join("torch", "autograd"),
    os.path.join("torch", "nn", "modules"),
)


def _is_framework_frame(filename: str) -> bool:
    """True if this filename looks like torch / gpufl internals rather
    than the user's own code."""
    lower = filename.replace("\\", "/").lower()
    # Fast path: normalize backslashes for the cross-platform fragments.
    norm_fragments = (f.replace("\\", "/").lower() for f in _FRAMEWORK_PATH_FRAGMENTS)
    return any(frag in lower for frag in norm_fragments)


def capture_python_site(max_depth: int = 16) -> Optional[str]:
    """
    Walk up the Python call stack from the caller and return a short
    `"basename.py:lineno"` string identifying the first user-code frame.

    Returns None if no user frame is found within `max_depth` frames —
    this happens when the op fires from deep inside torch internals
    with no application code on the stack (rare, mostly during torch
    compile).
    """
    try:
        # Start at frame[1] to skip this capture_python_site() itself.
        frame = sys._getframe(1)
    except ValueError:
        return None

    depth = 0
    while frame is not None and depth < max_depth:
        filename = frame.f_code.co_filename
        if not _is_framework_frame(filename):
            return f"{os.path.basename(filename)}:{frame.f_lineno}"
        frame = frame.f_back
        depth += 1
    return None
