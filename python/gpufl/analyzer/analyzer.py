import pandas as pd
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout

class GpuFlightSession:
    def __init__(self, log_dir: str, session_id: str = None, log_prefix: str = "gfl_block", max_stack_depth: int = 5):
        self.log_dir = Path(log_dir)
        self.console = Console()
        self.max_stack_depth = max_stack_depth

        # 1. Load DataFrames
        self.kernels = self._load_log(f"{log_prefix}.kernel.0.log")
        self.scopes = self._load_log(f"{log_prefix}.scope.0.log")
        self.system = self._load_log(f"{log_prefix}.system.0.log")

        # 2. Filter by Session ID if provided (or pick the latest)
        if session_id:
            self.kernels = self.kernels[self.kernels['session_id'] == session_id]

        # 3. Pre-Calculate Metrics (The "Secret Sauce")
        self._enrich_data()

    def _load_log(self, filename):
        """Efficiently loads JSONL into Pandas"""
        path = self.log_dir / filename
        if not path.exists():
            return pd.DataFrame()

        data = []
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except: pass
        return pd.DataFrame(data)

    def _enrich_data(self):
        """Calculates derived metrics (Latency, Bandwidth, Duration)"""
        if self.kernels.empty: return

        # Time conversions (ns -> ms)
        k = self.kernels
        k['duration_ms'] = (k['end_ns'] - k['start_ns']) / 1e6
        k['cpu_overhead_ms'] = (k['api_exit_ns'] - k['api_start_ns']) / 1e6

        # Queue Latency: The gap between CPU Dispatch and GPU Start
        # Note: We clamp to 0 because clock drift can make this negative
        k['queue_latency_ms'] = ((k['start_ns'] - k['api_exit_ns']) / 1e6).clip(lower=0)

        # Theoretical Bandwidth / Throughput hints
        # (Assuming you add bytes transferred in future versions)

        self.kernels = k

    def print_summary(self):
        """Prints an 'Executive Summary' of the session"""
        if self.kernels.empty:
            self.console.print("[bold red]No kernel data found![/bold red]")
            return

        total_duration = self.kernels['end_ns'].max() - self.kernels['start_ns'].min()
        total_duration_ms = total_duration / 1e6
        gpu_busy_time = self.kernels['duration_ms'].sum()

        # Calculate global GPU Utilization % from logs if available, or estimate
        def get_device_stat(devices, key, agg='mean'):
            if not isinstance(devices, list) or len(devices) == 0:
                return 0
            stats = [d.get(key, 0) for d in devices if isinstance(d, dict)]
            if not stats: return 0
            return sum(stats) / len(stats) if agg == 'mean' else max(stats)

        avg_gpu_util = self.system['devices'].apply(lambda x: get_device_stat(x, 'util_gpu')).mean()
        peak_mem = self.system['devices'].apply(lambda x: get_device_stat(x, 'used_mib', 'max')).max()

        # Create Dashboard
        grid = Table.grid(expand=True)
        grid.add_column()
        grid.add_column()

        stats = Table(show_header=False, box=None)
        stats.add_row("Total Duration:", f"[bold cyan]{total_duration_ms/1000:.2f} s[/bold cyan]")
        stats.add_row("Total Kernels:", f"[bold]{len(self.kernels)}[/bold]")
        stats.add_row("GPU Busy Time:", f"[green]{gpu_busy_time/1000:.2f} s[/green]")
        stats.add_row("Avg GPU Util:", f"[yellow]{avg_gpu_util:.1f}%[/yellow]")
        stats.add_row("Peak VRAM:", f"[red]{peak_mem} MiB[/red]")

        self.console.print(Panel(stats, title="[bold]GPUFlight Session Report[/bold]", subtitle=self.kernels.iloc[0]['app']))

    def inspect_hotspots(self, top_n=5, max_stack_depth=None):
        """Identify the most expensive kernels and show their stack traces"""
        if self.kernels.empty:
            self.console.print("[yellow]No kernel data to analyze hotspots.[/yellow]")
            return

        depth = max_stack_depth or self.max_stack_depth

        # Group by Kernel Name and Stack Trace
        # We include stack_trace in groupby to see hotspots per call site
        group_cols = ['name']
        if 'stack_trace' in self.kernels.columns:
            group_cols.append('stack_trace')

        summary = self.kernels.groupby(group_cols).agg(
            count=('name', 'count'),
            total_time_ms=('duration_ms', 'sum'),
            avg_time_ms=('duration_ms', 'mean'),
            max_time_ms=('duration_ms', 'max'),
            avg_occupancy=('occupancy', 'mean'),
            grid=('grid', 'first'),
            block=('block', 'first'),
            dyn_shared=('dyn_shared_bytes', 'first'),
            static_shared=('static_shared_bytes', 'first'),
            num_regs=('num_regs', 'first'),
            local_bytes=('local_bytes', 'first'),
            const_bytes=('const_bytes', 'first')
        ).sort_values('total_time_ms', ascending=False).head(top_n)

        table = Table(title=f"🔥 Top {top_n} Kernel Hotspots (Time Consuming)")
        table.add_column("Kernel Name / Stack Trace", style="cyan", no_wrap=False)
        table.add_column("Calls", justify="right")
        table.add_column("Total Time", justify="right", style="green")
        table.add_column("Occupancy", justify="right", style="magenta")
        table.add_column("Grid/Block", justify="center")
        table.add_column("Resources (Reg/SMem/DMem/LMem/CMem)", justify="left")

        for (name, *rest), row in summary.iterrows():
            stack_trace = rest[0] if rest else None
            
            # Clean up C++ Mangled names if possible
            clean_name = name[:60] + "..." if len(name) > 60 else name
            
            # Format display: Kernel Name followed by visualized stack
            display_content = f"[bold]{clean_name}[/bold]"
            
            if stack_trace and isinstance(stack_trace, str):
                frames = stack_trace.split('|')
                # Left most is top stack
                limited_frames = frames[:depth]
                stack_viz = ""
                for i, frame in enumerate(limited_frames):
                    indent = "  " * i
                    prefix = "└─ " if i > 0 else "Top: "
                    stack_viz += f"\n{indent}{prefix}[dim]{frame}[/dim]"
                
                if len(frames) > depth:
                    stack_viz += f"\n  {'  ' * depth}[dim]... ({len(frames) - depth} more levels)[/dim]"
                
                display_content += stack_viz

            resource_str = (
                f"Reg: {row['num_regs']} | "
                f"SMem: {row['static_shared']} | "
                f"DMem: {row['dyn_shared']} | "
                f"LMem: {row['local_bytes']} | "
                f"CMem: {row['const_bytes']}"
            )

            table.add_row(
                display_content,
                str(row['count']),
                f"{row['total_time_ms']:.2f} ms",
                f"{row['avg_occupancy']*100:.1f}%",
                f"{row['grid']}\n{row['block']}",
                resource_str
            )

        self.console.print(table)

    def inspect_scopes(self):
        """Analyze time spent in user-defined Scopes (e.g. 'Training_Epoch')"""
        if self.kernels.empty or 'user_scope' not in self.kernels.columns:
            self.console.print("[yellow]No scope data found or 'user_scope' column missing.[/yellow]")
            return

        # Aggregate metrics by user scope
        scope_stats = self.kernels.groupby('user_scope').agg(
            kernels=('name', 'count'),
            gpu_time_ms=('duration_ms', 'sum'),
            avg_queue_ms=('queue_latency_ms', 'mean'),
            cpu_overhead_ms=('cpu_overhead_ms', 'sum')
        ).sort_index()

        table = Table(title="📂 Scope Analysis (Hierarchical)")
        table.add_column("Scope / Phase", style="bold white")
        table.add_column("GPU Time", style="green", justify="right")
        table.add_column("Queue Latency", style="red", justify="right")
        table.add_column("CPU Overhead", style="yellow", justify="right")

        for scope, row in scope_stats.iterrows():
            # format the scope (e.g. replace | with >)
            formatted_scope = scope.replace("|", " [dim]>[/dim] ")
            table.add_row(
                formatted_scope,
                f"{row['gpu_time_ms']:.2f} ms",
                f"{row['avg_queue_ms']:.3f} ms",
                f"{row['cpu_overhead_ms']:.2f} ms"
            )

        self.console.print(table)