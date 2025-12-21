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
                # [NEW] Extract Bandwidth and convert B/s -> GB/s
                "pcie_rx_gbps": found.get("pcie_rx_bw", 0) / 1e9,
                "pcie_tx_gbps": found.get("pcie_tx_bw", 0) / 1e9,
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
        out["t_s_abs"] = (out["ts_ns"] - out["ts_ns"].min()) / 1e9
    return out

def _reconstruct_intervals(df, start_type, end_type, name_col="name", fallback_name="Scope"):
    pd = _require_pandas()
    subset = df[df["event_type"].isin([start_type, end_type])].copy()
    if subset.empty: return []

    intervals = []
    stack = {}
    min_ts = df["ts_ns"].min()
    if pd.isna(min_ts): min_ts = 0

    for _, r in subset.iterrows():
        etype = r["event_type"]
        name = r.get(name_col, fallback_name)
        if pd.isna(name): name = fallback_name

        ts = r.get("ts_ns")
        if pd.isna(ts): ts = r.get("ts_start_ns")
        if pd.isna(ts): continue

        if etype == start_type:
            stack[name] = ts
        elif etype == end_type:
            if name in stack:
                start_ns = stack.pop(name)
                start_sec = (start_ns - min_ts) / 1e9
                dur_sec = (ts - start_ns) / 1e9
                intervals.append((start_sec, dur_sec, name))
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

    min_ts = df["ts_ns"].min()
    if pd.isna(min_ts): min_ts = 0

    # --- Prepare Data ---
    scope_data = _reconstruct_intervals(df, "scope_start", "scope_end")
    if not scope_data:
        app_data = _reconstruct_intervals(df, "init", "shutdown", name_col="app", fallback_name="App")
        scope_data.extend(app_data)

    kernel_data = _reconstruct_intervals(df, "kernel_start", "kernel_end")
    gpu_samples = _explode_device_samples(df, gpu_id=0)
    host_samples = _explode_host_samples(df)

    # --- Plotting (3 Rows) ---
    # Heights: GPU=2, PCIe=1.5, Host=2
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                                        gridspec_kw={'height_ratios': [2, 1.5, 2]})

    # --- Helper to Overlay Markers ---
    def overlay_markers(ax, y_lim_ref=None):
        """Draws vertical lines for Scopes and Kernels on the given axis."""
        # Get Y-limit to position text
        y_top = y_lim_ref if y_lim_ref else (ax.get_ylim()[1] if len(ax.get_lines()) > 0 else 100)

        # Scopes (Red dashed)
        if scope_data:
            for start_sec, dur_sec, name in scope_data:
                ax.axvline(x=start_sec, color='tab:red', linestyle='--', alpha=0.6, linewidth=1)
                ax.text(start_sec, y_top * 0.95, name, rotation=90, va='top', ha='center', fontsize=7,
                        color='tab:red', alpha=0.9,
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.3, edgecolor='none'))

        # Kernels (Orange solid/dashed)
        if kernel_data:
            for start_sec, dur_sec, name in kernel_data:
                end_sec = start_sec + (dur_sec if dur_sec is not None else 0)
                ax.axvline(x=start_sec, color='tab:orange', linestyle='-', linewidth=1.2)
                ax.text(start_sec, y_top * 0.85, name, rotation=90, va='top', ha='center', fontsize=7,
                        color='tab:orange', alpha=0.9)
                if dur_sec and dur_sec > 0:
                    ax.axvline(x=end_sec, color='tab:orange', linestyle='--', linewidth=1.2)
                    ax.text(end_sec, y_top * 0.85, f"{name} end", rotation=90, va='top', ha='center', fontsize=6,
                            color='tab:orange', alpha=0.7)

    # --- Row 1: GPU Metrics ---
    if not gpu_samples.empty:
        t = gpu_samples["t_s_abs"]
        ax1.plot(t, gpu_samples["util_gpu"], label="GPU %", color='tab:green')
        ax1.plot(t, gpu_samples["util_mem"], label="Mem %", color='tab:purple', linestyle="--")
        ax1.set_ylabel("GPU Util %")
        ax1.set_ylim(-5, 105)
        ax1.legend(loc="upper left", fontsize='x-small')

        ax1b = ax1.twinx()
        ax1b.fill_between(t, gpu_samples["used_mib"], color='tab:gray', alpha=0.1, label="VRAM Used")
        ax1b.set_ylabel("VRAM (MiB)", color='gray')
        ax1b.set_ylim(bottom=0)

    ax1.grid(True, alpha=0.3)
    ax1.set_title("GPU Metrics", fontsize=10)
    overlay_markers(ax1, y_lim_ref=105)

    # --- Row 2: PCIe Bandwidth (NEW) ---
    if not gpu_samples.empty:
        t = gpu_samples["t_s_abs"]
        # Plot RX (Host -> Device) and TX (Device -> Host)
        ax2.plot(t, gpu_samples["pcie_rx_gbps"], label="PCIe RX (Upload)", color='tab:blue')
        ax2.plot(t, gpu_samples["pcie_tx_gbps"], label="PCIe TX (Download)", color='tab:cyan', linestyle="--")

        ax2.set_ylabel("BW (GB/s)")
        # Dynamically scale Y-axis but keep min at 0
        ax2.set_ylim(bottom=0)
        ax2.legend(loc="upper left", fontsize='x-small')

    ax2.grid(True, alpha=0.3)
    ax2.set_title("PCIe Bandwidth", fontsize=10)
    # Overlay markers (passing None lets helper figure out Y-max from data)
    overlay_markers(ax2)

    # --- Row 3: Host Metrics ---
    if not host_samples.empty:
        t_host = host_samples["t_s_abs"]
        ax3.plot(t_host, host_samples["cpu_pct"], label="CPU %", color='tab:red')
        ax3.set_ylabel("CPU Util %", color='tab:red')
        ax3.set_ylim(-5, 105)
        ax3.tick_params(axis='y', labelcolor='tab:red')
        ax3.legend(loc="upper left", fontsize='x-small')

        ax3b = ax3.twinx()
        ax3b.plot(t_host, host_samples["ram_used_mib"] / 1024, label="RAM (GiB)", color='tab:blue', linestyle="--")
        ax3b.set_ylabel("Sys RAM (GiB)", color='tab:blue')
        ax3b.tick_params(axis='y', labelcolor='tab:blue')
        ax3b.set_ylim(bottom=0)
        ax3b.legend(loc="upper right", fontsize='x-small')

    ax3.set_xlabel("Time (seconds)")
    ax3.grid(True, alpha=0.3)
    ax3.set_title("Host Metrics", fontsize=10)
    overlay_markers(ax3, y_lim_ref=105)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.25)
    return fig

# Legacy wrappers
def plot_kernel_timeline(df, title="Kernels"): return plot_combined_timeline(df, title)
def plot_scope_timeline(df, title="Scopes"): return plot_combined_timeline(df, title)
def plot_host_timeline(df, title="Host"): return plot_combined_timeline(df, title)
def plot_memory_timeline(df, gpu_id=0, title="Mem"): return None
def plot_utilization_timeline(df, gpu_id=0, title="Util"): return None