try:
    from .visualizer import init, show, compare, get_data
    from .reader import read_df, read_events
    # Import the new timeline plotter
    from .timeline import (
        plot_combined_timeline,
        plot_kernel_timeline,
        plot_scope_timeline,
        plot_host_timeline,
        plot_memory_timeline,
        plot_utilization_timeline
    )
except ImportError as e:
    # [FIX] Convert exception to string IMMEDIATELY.
    # Python 3 deletes the variable 'e' after the block, causing a crash later.
    err_msg = str(e)

    print(f"[GPUFL Warning] Visualization module disabled. Reason: {err_msg}")

    # Fallback dummies using the saved string
    def show(*args, **kwargs):
        print(f"Error: Visualization disabled. Cause: {err_msg}")

    def init(*args, **kwargs):
        print(f"Error: Visualization disabled. Cause: {err_msg}")

    compare = show