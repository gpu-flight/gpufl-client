import os
import glob
import re
import json
from typing import Optional, List, Union, Any
import pandas as pd

from .reader import read_df
from .timeline import (
    plot_combined_timeline,
    _explode_device_samples,
    _explode_host_samples
)

# --- Global State ---
_GLOBAL_DF = None

def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:
        raise ImportError("Visualization requires matplotlib. Run: pip install pandas matplotlib") from e

def _check_init():
    if _GLOBAL_DF is None:
        raise RuntimeError("Global data not loaded. Call viz.init('path/to/logs') first.")

def init(log_pattern: Union[str, List[str]]):
    """
    Load log files into memory.
    Args:
        log_pattern: File path, directory, or glob pattern (e.g., "logs/*.log").
    """
    global _GLOBAL_DF
    if isinstance(log_pattern, str) and os.path.isdir(log_pattern):
        pattern = os.path.join(log_pattern, "*.log")
    else:
        pattern = log_pattern

    print(f"Loading logs from: {pattern} ...")
    df = read_df(pattern)

    if len(df) == 0:
        print("[Warn] No events found.")
        _GLOBAL_DF = None
    else:
        # Pre-convert timestamps to numeric for speed
        for col in ["ts_ns", "start_ns", "end_ns"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Sort by time immediately
        if "ts_ns" in df.columns:
            df = df.sort_values("ts_ns")

        _GLOBAL_DF = df
        print(f"Loaded {len(df)} events.")

def get_data() -> Any:
    """Return the raw global DataFrame."""
    _check_init()
    return _GLOBAL_DF

def _parse_duration(duration_str: str) -> int:
    """Convert '1m', '1h', '30s' to nanoseconds."""
    if not duration_str: return 0
    units = {"s": 1e9, "m": 60 * 1e9, "h": 3600 * 1e9, "d": 86400 * 1e9}
    match = re.match(r"(\d+)([smhd])", duration_str)
    if match:
        val, unit = match.groups()
        return int(float(val) * units[unit])
    return 0

def show(last: Optional[str] = None, **filters):
    """
    Visualize the Stacked Timeline.

    Args:
        last: Time window from the end of the log (e.g., "1m", "30s", "1h").
              If None, shows all data.
        **filters: Key-value pairs to filter data (e.g., app="MyApp").
    """
    _check_init()
    plt = _require_matplotlib()

    df = _GLOBAL_DF.copy()

    # 1. Apply Tag/App Filters
    for k, v in filters.items():
        if k in df.columns:
            if isinstance(v, list): df = df[df[k].isin(v)]
            else: df = df[df[k] == v]

    if len(df) == 0:
        print("No data matching filters.")
        return

    # 2. Apply Time Window (Tail)
    if last:
        duration_ns = _parse_duration(last)
        if duration_ns > 0:
            max_ts = df["ts_ns"].max()
            if pd.isna(max_ts):
                max_ts = df[["start_ns", "end_ns"]].max().max()

            cutoff = max_ts - duration_ns

            # Filter rows overlapping with the window
            cond_ts = df["ts_ns"] >= cutoff
            cond_start = df["start_ns"] >= cutoff
            cond_end = df["end_ns"] >= cutoff

            df = df[cond_ts | cond_start | cond_end]

    if len(df) == 0:
        print(f"No data found in the last {last}.")
        return

    # 3. Generate Combined Plot
    fig = plot_combined_timeline(df, title=f"Timeline (last={last})" if last else "Full Timeline")

    if fig:
        plt.show()
    else:
        print("Not enough data to generate plot.")

def compare(group_by="app", metric="gpu", **filters):
    """
    Compare a specific metric across different groups (e.g. apps, tags).

    Args:
        group_by: Column to group by (default: "app").
        metric: "cpu", "gpu", "ram"
        **filters: Additional filters.
    """
    _check_init()
    plt = _require_matplotlib()

    # 1. Filter
    df = _GLOBAL_DF.copy()
    for k, v in filters.items():
        if k in df.columns:
            if isinstance(v, list): df = df[df[k].isin(v)]
            else: df = df[df[k] == v]

    if len(df) == 0:
        print("No data matches filters.")
        return

    if group_by not in df.columns:
        print(f"Cannot group by '{group_by}': column not found.")
        return

    groups = df[group_by].unique()
    if len(groups) == 0:
        print("No groups found.")
        return

    fig = plt.figure(figsize=(10, 5))
    has_plot = False

    # 2. Plot lines for each group
    for g in groups:
        sub_df = df[df[group_by] == g]
        label = str(g)

        if metric == "gpu":
            # Uses helper from timeline.py
            s = _explode_device_samples(sub_df, gpu_id=0)
            if not s.empty:
                # Normalize time to start at 0 for comparison
                start_t = s["ts_ns"].min()
                t_axis = (s["ts_ns"] - start_t) / 1e9
                plt.plot(t_axis, s["util_gpu"], label=label)
                has_plot = True

        elif metric == "cpu":
            s = _explode_host_samples(sub_df)
            if not s.empty:
                start_t = s["ts_ns"].min()
                t_axis = (s["ts_ns"] - start_t) / 1e9
                plt.plot(t_axis, s["cpu_pct"], label=label)
                has_plot = True

    if has_plot:
        plt.title(f"Comparison: {metric.upper()} by {group_by}")
        plt.xlabel("Time (s) [Relative start]")
        plt.ylabel("Utilization %")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        print(f"No data found for metric '{metric}' in the selected groups.")