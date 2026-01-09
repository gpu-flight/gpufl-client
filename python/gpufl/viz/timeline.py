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

    target_types = ["scope_sample", "system_sample", "system_start", "system_stop", "kernel_start", "kernel_end", "kernel_event", "init"]
    if "event_type" not in df.columns: return pd.DataFrame()

    d = df[df["event_type"].isin(target_types)].copy()
    if len(d) == 0: return pd.DataFrame()

    if "devices" in d.columns:
        d["devices"] = d["devices"].apply(_coerce_devices_cell)

    rows = []
    for _, r in d.iterrows():
        ts = r.get("ts_ns")
        # If ts_ns is missing, try start_ns (common for kernel_event)
        if pd.isna(ts):
            ts = r.get("start_ns")
        
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
                "temp_c": found.get("temp_c", 0),
                "power_mw": found.get("power_mw", 0),
                "clk_sm": found.get("clk_sm", 0),
                # [NEW] Extract Bandwidth and convert B/s -> GB/s
                "pcie_rx_gbps": found.get("pcie_rx_bw", 0) / 1e9,
                "pcie_tx_gbps": found.get("pcie_tx_bw", 0) / 1e9,
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.dropna(subset=["ts_ns"]).sort_values("ts_ns")
        min_ts = out["ts_ns"].min()
        out["t_s_abs"] = (out["ts_ns"] - min_ts) / 1e9
    return out

def _explode_host_samples(df):
    pd = _require_pandas()
    df = _ensure_event_type_col(df)

    target_types = ["scope_sample", "system_sample", "system_start", "system_stop", "kernel_start", "kernel_event", "init", "shutdown"]
    if "event_type" not in df.columns: return pd.DataFrame()

    d = df[df["event_type"].isin(target_types)].copy()
    if len(d) == 0 or "host" not in d.columns: return pd.DataFrame()

    d["host"] = d["host"].apply(_coerce_host_cell)
    rows = []
    for _, r in d.iterrows():
        h = r["host"]
        if not h: continue
        
        ts = r.get("ts_ns") or r.get("ts_start_ns") or r.get("start_ns")
        
        rows.append({
            "ts_ns": ts,
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
    # Support both "scope_start" and "scope_begin" for compatibility
    start_types = [start_type]
    if start_type == "scope_start":
        start_types.append("scope_begin")
    
    # [NEW] Handle single-event intervals like kernel_event
    is_kernel = (start_type == "kernel_start" or start_type == "kernel_event")
    if is_kernel:
        # Include kernel_event which has both start and end
        target_types = start_types + [end_type, "kernel_event"]
    else:
        target_types = start_types + [end_type]

    subset = df[df["event_type"].isin(target_types)].copy()
    if subset.empty: return []

    intervals = []
    # Use a dictionary of lists to handle multiple nested intervals with the same name
    stacks = {} 
    min_ts = df["ts_ns"].min()
    if pd.isna(min_ts):
        # try start_ns if ts_ns is all NaN
        if "start_ns" in df.columns:
            min_ts = df["start_ns"].min()
    if pd.isna(min_ts): min_ts = 0

    for _, r in subset.iterrows():
        etype = r["event_type"]
        name = r.get(name_col, fallback_name)
        if pd.isna(name): name = fallback_name

        if etype == "kernel_event" and "start_ns" in r and "end_ns" in r:
            start_ns = r["start_ns"]
            end_ns = r["end_ns"]
            if not pd.isna(start_ns) and not pd.isna(end_ns):
                start_sec = (start_ns - min_ts) / 1e9
                dur_sec = (end_ns - start_ns) / 1e9
                
                # Add extra metrics if present
                metrics = {
                    "occupancy": r.get("occupancy", 0),
                    "grid": r.get("grid", ""),
                    "block": r.get("block", ""),
                    "num_regs": r.get("num_regs", 0),
                    "dyn_shared": r.get("dyn_shared_bytes", 0),
                    "static_shared": r.get("static_shared_bytes", 0),
                }
                intervals.append((start_sec, dur_sec, name, metrics))
            continue

        ts = r.get("ts_ns")
        if pd.isna(ts): ts = r.get("ts_start_ns")
        if pd.isna(ts): ts = r.get("start_ns")
        if pd.isna(ts): continue

        if etype in start_types:
            if name not in stacks:
                stacks[name] = []
            stacks[name].append(ts)
        elif etype == end_type:
            if name in stacks and stacks[name]:
                start_ns = stacks[name].pop()
                start_sec = (start_ns - min_ts) / 1e9
                dur_sec = (ts - start_ns) / 1e9
                intervals.append((start_sec, dur_sec, name, {}))
                if not stacks[name]:
                    del stacks[name]
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
    # Try both "scope_start" and "scope_begin"
    scope_data = _reconstruct_intervals(df, "scope_start", "scope_end")
    if not scope_data:
        scope_data = _reconstruct_intervals(df, "scope_begin", "scope_end")
    
    if not scope_data:
        app_data = _reconstruct_intervals(df, "init", "shutdown", name_col="app", fallback_name="App")
        scope_data.extend(app_data)

    # [NEW] Handle kernel_event
    kernel_data = _reconstruct_intervals(df, "kernel_event", "kernel_end")
    if not kernel_data:
        kernel_data = _reconstruct_intervals(df, "kernel_start", "kernel_end")
    
    gpu_samples = _explode_device_samples(df, gpu_id=0)
    host_samples = _explode_host_samples(df)

    # --- Plotting (3 Rows) ---
    # Heights: GPU=2, PCIe=1.5, Host=2
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                                        gridspec_kw={'height_ratios': [2, 1.5, 2]})

    # --- Helper to Overlay Markers ---
    kernel_markers = [] # List of (vline, annotation)

    def overlay_markers(ax, y_lim_ref=None):
        """Draws vertical lines for Scopes and Kernels on the given axis."""
        # Get Y-limit to position text
        y_top = y_lim_ref if y_lim_ref else (ax.get_ylim()[1] if len(ax.get_lines()) > 0 else 100)

        # Scopes (Red dashed)
        if scope_data:
            for start_sec, dur_sec, name, _ in scope_data:
                ax.axvline(x=start_sec, color='tab:red', linestyle='--', alpha=0.6, linewidth=1)
                ax.text(start_sec, y_top * 0.95, name, rotation=90, va='top', ha='center', fontsize=7,
                        color='tab:red', alpha=0.9,
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.3, edgecolor='none'))

        # Kernels (Orange solid/dashed)
        if kernel_data:
            for start_sec, dur_sec, name, metrics in kernel_data:
                end_sec = start_sec + (dur_sec if dur_sec is not None else 0)
                vl = ax.axvline(x=start_sec, color='tab:orange', linestyle='-', linewidth=1.2, picker=True)
                
                # Enrich text with occupancy if available
                display_name = name
                if metrics and metrics.get("occupancy", 0) > 0:
                    display_name += f" ({metrics['occupancy']*100:.1f}%)"
                
                # Create annotation but set it invisible by default
                ann = ax.annotate(display_name, xy=(start_sec, y_top * 0.85), 
                                  xytext=(5, 0), textcoords="offset points",
                                  rotation=90, va='top', ha='left', fontsize=7,
                                  color='tab:orange', fontweight='bold',
                                  bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.8),
                                  visible=False)
                
                kernel_markers.append((vl, ann))

                if dur_sec and dur_sec > 0:
                    ax.axvline(x=end_sec, color='tab:orange', linestyle='--', linewidth=1.2)
                    # We usually don't need hover for "end" markers, but could add it.

    # --- Row 1: GPU Metrics ---
    if not gpu_samples.empty:
        t = gpu_samples["t_s_abs"]
        ax1.plot(t, gpu_samples["util_gpu"], label="GPU %", color='tab:green')
        ax1.plot(t, gpu_samples["util_mem"], label="Mem %", color='tab:purple', linestyle="--")
        
        # [NEW] Optional metrics from system log
        if "temp_c" in gpu_samples.columns and gpu_samples["temp_c"].max() > 0:
             ax1.plot(t, gpu_samples["temp_c"], label="Temp (C)", color='tab:red', alpha=0.3)
        if "clk_sm" in gpu_samples.columns and gpu_samples["clk_sm"].max() > 0:
             # Scale clock for visibility if needed, or use another axis. Let's just plot it.
             ax1.plot(t, gpu_samples["clk_sm"] / 10, label="SM Clock (x10 MHz)", color='tab:orange', alpha=0.3)

        # [NEW] Visualize Kernel Occupancy points on the timeline
        if kernel_data:
            k_t = [k[0] for k in kernel_data if k[3] and k[3].get("occupancy", 0) > 0]
            k_occ = [k[3]["occupancy"] * 100 for k in kernel_data if k[3] and k[3].get("occupancy", 0) > 0]
            if k_t:
                ax1.scatter(k_t, k_occ, color='tab:orange', marker='o', s=20, label="Kernel Occupancy", zorder=5)

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

    # --- Hover Interaction ---
    def on_hover(event):
        if event.inaxes is None: return
        
        changed = False
        for vl, ann in kernel_markers:
            # Check if mouse is near the vertical line (x-axis distance)
            if vl.axes == event.inaxes:
                # Calculate distance in pixels for better UX
                try:
                    # Convert data x to display x
                    x_display = vl.axes.transData.transform((vl.get_xdata()[0], 0))[0]
                    mouse_x = event.x
                    
                    is_near = abs(x_display - mouse_x) < 5 # 5 pixels tolerance
                    
                    if ann.get_visible() != is_near:
                        ann.set_visible(is_near)
                        changed = True
                except:
                    pass
        
        if changed:
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_hover)

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