"""Visualization and offline analysis helpers for GPU Flight (gpufl).

This subpackage is intentionally optional: it requires extra dependencies (pandas/matplotlib).
Install with:
  pip install gpufl[viz]

API (stable-ish):
  - read_events(...)
  - read_df(...)
  - summarize_kernels(...)
  - summarize_scopes(...)
  - plot_kernel_timeline(...)
"""

from .reader import read_events, read_df
