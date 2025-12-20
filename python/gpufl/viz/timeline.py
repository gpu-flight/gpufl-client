from __future__ import annotations
import json
from typing import Iterable, Optional

def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError("Visualization requires matplotlib.")

def _require_pandas():
    try:
        import pandas as pd
        return pd
    except ImportError:
        raise ImportError("Visualization requires pandas.")

# ==========================================
# 1. HELPERS
# ==========================================

def _ensure_event_type_col(df):
    if df is None: return df
    # Map 'type' to 'event_type' if missing
    if "event_type" not in df.columns and "type" in df.columns:
        df = df.copy()
        df["event_type"] = df["type"]
    return df

def _coerce_devices_cell(x):
    if isinstance(x, list): return x
    if isinstance(x, str):
        try: return json.loads(x)
        except: return []
    return []

def _coerce_host_cell(x):
    if isinstance(x, dict): return x
    if isinstance(x, str):
        try: return json.loads(x)
        except: return {}
    return {}

def _explode_device_samples(df, gpu_id=0):
    pd = _require_pandas()
    df = _ensure_event_type_col(df)

    # Events that might have device info
    target_types = ["scope_sample", "system_sample", "kernel_start", "kernel_end", "init"]
    if "event_type" not in df.columns: return pd.DataFrame()

    d = df[df["event_type"].isin(target_types)].copy()
    if len(d) == 0: return pd.DataFrame()

    if "devices" in d.columns:
        d["devices"] = d["devices"].apply(_coerce_devices_cell)

    rows = []
    for _, r in d.iterrows():
        ts = r.get("ts_ns")
        devs = r.get("devices", [])
        found = None
        if isinstance(devs, list):
            for dev in devs:
                if isinstance(dev, dict) and dev.get("id") == gpu_id:
                    found = dev
                    break
        if found:
            rows.append({
                "ts_ns": ts,
                "util_gpu": found.get("util_gpu", 0),
                "util_mem": found.get("util_mem", 0),
                "used_mib": found.get("used_mib", 0),
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("ts_ns")
        min_ts = out["ts_ns"].min()
        out["t_s_abs"] = (out["ts_ns"] - min_ts) / 1e9
    return out

def _explode_host_samples(df):
    pd = _require_pandas()
    df = _ensure_event_type_col(df)

    # Events that typically carry host metrics
    target_types = ["scope_sample", "system_sample", "kernel_start", "init", "shutdown"]
    if "event_type" not in df.columns: return pd.DataFrame()

    d = df[df["event_type"].isin(target_types)].copy()
    if len(d) == 0 or "host" not in d.columns: return pd.DataFrame()

    d["host"] = d["host"].apply(_coerce_host_cell)
    rows = []
    for _, r in d.iterrows():
        h = r["host"]
        if not h: continue
        rows.append({
            "ts_ns": r.get("ts_ns") or r.get("ts_start_ns"),
            "cpu_pct": h.get("cpu_pct", 0),
            "ram_used_mib": h.get("ram_used_mib", 0)
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.dropna(subset=["ts_ns"]).sort_values("ts_ns")
        # Relative time (s)
        out["t_s_abs"] = (out["ts_ns"] - out["ts_ns"].min()) / 1e9
    return out

def _reconstruct_intervals(df, start_type, end_type, name_col="name", fallback_name="Scope"):
    """
    Pairs Start/End events into (start_sec, duration_sec, label).
    Handles multiple nesting levels naively (stack).
    """
    pd = _require_pandas() # [FIXED: Was missing]

    subset = df[df["event_type"].isin([start_type, end_type])].copy()
    if subset.empty: return []

    intervals = []
    stack = {} # name -> start_ns

    # Global min for relative time
    min_ts = df["ts_ns"].min()
    if pd.isna(min_ts): min_ts = 0

    for _, r in subset.iterrows():
        etype = r["event_type"]
        name = r.get(name_col, fallback_name)
        if pd.isna(name): name = fallback_name

        # Get timestamp
        ts = r.get("ts_ns")
        if pd.isna(ts): ts = r.get("ts_start_ns")
        if pd.isna(ts): continue

        if etype == start_type:
            # If name already in stack, it's a nested call or collision.
            # Simple overwrite for now, or use a list for stack if needed.
            stack[name] = ts
        elif etype == end_type:
            if name in stack:
                start_ns = stack.pop(name)
                start_sec = (start_ns - min_ts) / 1e9
                dur_sec = (ts - start_ns) / 1e9
                intervals.append((start_sec, dur_sec, name))

    # If using init/shutdown, usually there is only one "App" or "Kernel_Test"
    return intervals

# ==========================================
# 2. PLOTTERS
# ==========================================

def plot_combined_timeline(df, title="GPUFL Timeline"):
    pd = _require_pandas()
    plt = _require_matplotlib()

    df = _ensure_event_type_col(df)
    if "event_type" not in df.columns:
        print("[Viz] Error: No event_type column found.")
        return None

    # Calculate Global Start Time
    min_ts = df["ts_ns"].min()
    if pd.isna(min_ts): min_ts = 0

    # --- Prepare Data ---

    # 1. Scopes: Try explicit Scopes FIRST, then fallback to App (Init/Shutdown)
    scope_data = _reconstruct_intervals(df, "scope_start", "scope_end")
    if not scope_data:
        # Fallback to App lifespan if no specific scopes exist
        app_data = _reconstruct_intervals(df, "init", "shutdown", name_col="app", fallback_name="App")
        scope_data.extend(app_data)

    # 2. Kernels
    kernel_data = _reconstruct_intervals(df, "kernel_start", "kernel_end")

    # 3. GPU Metrics
    gpu_samples = _explode_device_samples(df, gpu_id=0)

    # 4. Host Metrics
    host_samples = _explode_host_samples(df)

    # --- Plotting (2 Rows) ---
    # Heights: GPU=2, Host=2
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 2]})

    # Row 1: GPU Metrics
    if not gpu_samples.empty:
        # Re-calc relative time just to be safe
        t = (gpu_samples["ts_ns"] - min_ts) / 1e9
        ax1.plot(t, gpu_samples["util_gpu"], label="GPU %", color='tab:green')
        ax1.plot(t, gpu_samples["util_mem"], label="Mem %", color='tab:purple', linestyle="--")
        ax1.set_ylabel("GPU Util %")
        ax1.set_ylim(-5, 105)
        ax1.legend(loc="upper left", fontsize='x-small')

        # Memory MiB on right axis
        ax1b = ax1.twinx()
        ax1b.fill_between(t, gpu_samples["used_mib"], color='tab:gray', alpha=0.1, label="VRAM Used")
        ax1b.set_ylabel("VRAM (MiB)", color='gray')
        ax1b.set_ylim(bottom=0)

    # Overlay Scopes as vertical lines on GPU metrics
    if scope_data:
        y_top = ax1.get_ylim()[1] if len(ax1.get_lines()) > 0 else 100
        for start_sec, dur_sec, name in scope_data:
            ax1.axvline(x=start_sec, color='tab:red', linestyle='--', alpha=0.6, linewidth=1)
            ax1.text(start_sec, y_top * 0.95, name, rotation=90, va='top', ha='center', fontsize=7,
                     color='tab:red', alpha=0.9,
                     bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.3, edgecolor='none'))

    # Overlay Kernels as vertical lines (start & end) on GPU metrics
    if kernel_data:
        y_top_k = ax1.get_ylim()[1] if len(ax1.get_lines()) > 0 else 100
        for start_sec, dur_sec, name in kernel_data:
            end_sec = start_sec + (dur_sec if dur_sec is not None else 0)
            # Start line (solid orange)
            ax1.axvline(x=start_sec, color='tab:orange', linestyle='-', linewidth=1.2)
            ax1.text(start_sec, y_top_k * 0.9, name, rotation=90, va='top', ha='center', fontsize=7,
                     color='tab:orange', alpha=0.9)
            # End line (dashed orange)
            if dur_sec and dur_sec > 0:
                ax1.axvline(x=end_sec, color='tab:orange', linestyle='--', linewidth=1.2)
                ax1.text(end_sec, y_top_k * 0.9, f"{name} end", rotation=90, va='top', ha='center', fontsize=6,
                         color='tab:orange', alpha=0.7)

    ax1.grid(True, alpha=0.3)
    ax1.set_title("GPU Metrics", fontsize=10)

    # Row 2: Host Metrics
    if not host_samples.empty:
        t_host = (host_samples["ts_ns"] - min_ts) / 1e9
        # CPU on Left
        ax2.plot(t_host, host_samples["cpu_pct"], label="CPU %", color='tab:red')
        ax2.set_ylabel("CPU Util %", color='tab:red')
        ax2.set_ylim(-5, 105)
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.legend(loc="upper left", fontsize='x-small')
        # RAM on Right
        ax2b = ax2.twinx()
        ax2b.plot(t_host, host_samples["ram_used_mib"] / 1024, label="RAM (GiB)", color='tab:blue', linestyle="--")
        ax2b.set_ylabel("Sys RAM (GiB)", color='tab:blue')
        ax2b.tick_params(axis='y', labelcolor='tab:blue')
        ax2b.set_ylim(bottom=0)
        ax2b.legend(loc="upper right", fontsize='x-small')

    # Overlay Scopes on Host metrics as well (subtle gray)
    if scope_data:
        y_top_host = ax2.get_ylim()[1] if len(ax2.get_lines()) > 0 else 100
        for start_sec, dur_sec, name in scope_data:
            ax2.axvline(x=start_sec, color='gray', linestyle=':', alpha=0.4, linewidth=1)
            ax2.text(start_sec, y_top_host * 0.95, name, rotation=90, va='top', ha='center', fontsize=6,
                     color='gray', alpha=0.7)

    # Overlay Kernels on Host metrics (subtle)
    if kernel_data:
        y_top_host_k = ax2.get_ylim()[1] if len(ax2.get_lines()) > 0 else 100
        for start_sec, dur_sec, name in kernel_data:
            end_sec = start_sec + (dur_sec if dur_sec is not None else 0)
            ax2.axvline(x=start_sec, color='tab:orange', linestyle='-', alpha=0.3, linewidth=1)
            if dur_sec and dur_sec > 0:
                ax2.axvline(x=end_sec, color='tab:orange', linestyle='--', alpha=0.25, linewidth=1)

    ax2.set_xlabel("Time (seconds)")
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Host Metrics", fontsize=10)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)
    return fig

# Legacy wrappers (required by __init__)
def plot_kernel_timeline(df, title="Kernels"): return plot_combined_timeline(df, title)
def plot_scope_timeline(df, title="Scopes"): return plot_combined_timeline(df, title)
def plot_host_timeline(df, title="Host"): return plot_combined_timeline(df, title)
def plot_memory_timeline(df, gpu_id=0, title="Mem"): return None
def plot_utilization_timeline(df, gpu_id=0, title="Util"): return None