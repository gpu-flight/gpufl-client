import pytest
import json
import os
from pathlib import Path

@pytest.fixture
def mock_log_dir(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    prefix = "test_run"

    # ── device.log ────────────────────────────────────────────────────────────
    device_log = log_dir / f"{prefix}.device.0.log"
    device_events = [
        # Session start
        {
            "version": 1, "type": "job_start",
            "session_id": "test-session", "app": "test_app", "pid": 1234,
            "ts_ns": 100,
            "host": {},
            "devices": [],
            "cuda_static_devices": [{"name": "NVIDIA Test GPU", "multi_processor_count": 108}]
        },
        # String dictionary (kernel names)
        {
            "version": 1, "type": "dictionary_update",
            "session_id": "test-session",
            "kernel_dict":     {"1": "vectorAdd", "2": "matrixMul"},
            "scope_name_dict": {"1": "main_loop"},
            "function_dict":   {},
            "metric_dict":     {}
        },
        # Kernel batch:
        #   row 0 — vectorAdd:  start=1000, end=2000, corr_id=101
        #   row 1 — matrixMul:  start=3000, end=5000, corr_id=102
        {
            "version": 1, "type": "kernel_event_batch",
            "session_id": "test-session", "batch_id": 1, "base_time_ns": 1000,
            "columns": ["dt_ns","kernel_id","stream_id","duration_ns","corr_id","dyn_shared","num_regs","has_details"],
            "rows": [
                [0,    1, 0, 1000, 101,    0, 32, 1],
                [2000, 2, 0, 2000, 102, 1024, 64, 1]
            ]
        },
        # Kernel details (occupancy, grid/block, scope info)
        {
            "version": 1, "type": "kernel_detail",
            "session_id": "test-session", "pid": 1234, "app": "test_app",
            "corr_id": 101,
            "grid": "(1,1,1)", "block": "(256,1,1)",
            "static_shared": 100, "local_bytes": 0, "const_bytes": 0,
            "occupancy": 0.8, "reg_occupancy": 0.9, "smem_occupancy": 1.0,
            "warp_occupancy": 0.8, "block_occupancy": 0.8,
            "limiting_resource": "REGISTERS",
            "local_mem_total_bytes": 0, "local_mem_per_thread_bytes": 0,
            "shared_mem_executed_bytes": 100, "max_active_blocks": 4,
            "cache_config_requested": 0, "cache_config_executed": 0,
            "user_scope": "global|main_loop|vectorAdd",
            "stack_trace": "main|vectorAdd"
        },
        {
            "version": 1, "type": "kernel_detail",
            "session_id": "test-session", "pid": 1234, "app": "test_app",
            "corr_id": 102,
            "grid": "(1,1,1)", "block": "(16,16,1)",
            "static_shared": 200, "local_bytes": 0, "const_bytes": 128,
            "occupancy": 0.6, "reg_occupancy": 0.7, "smem_occupancy": 0.8,
            "warp_occupancy": 0.6, "block_occupancy": 0.6,
            "limiting_resource": "SHARED_MEMORY",
            "local_mem_total_bytes": 32768, "local_mem_per_thread_bytes": 32,
            "shared_mem_executed_bytes": 200, "max_active_blocks": 2,
            "cache_config_requested": 0, "cache_config_executed": 0,
            "user_scope": "global|main_loop|matrixMul",
            "stack_trace": "main|matrixMul"
        },
        # Memcpy batch: start=500, duration=400, end=900, 4096 bytes HtoD (copy_kind=1)
        {
            "version": 1, "type": "memcpy_event_batch",
            "session_id": "test-session", "batch_id": 1, "base_time_ns": 500,
            "columns": ["dt_ns","stream_id","duration_ns","bytes","copy_kind","corr_id"],
            "rows": [
                [0, 0, 400, 4096, 1, 100]
            ]
        },
        # Session end
        {
            "type": "shutdown",
            "session_id": "test-session", "app": "test_app", "pid": 1234,
            "ts_ns": 10000
        },
    ]
    with open(device_log, "w") as f:
        for ev in device_events:
            f.write(json.dumps(ev) + "\n")

    # ── scope.log ─────────────────────────────────────────────────────────────
    scope_log = log_dir / f"{prefix}.scope.0.log"
    scope_events = [
        # Dictionary (scope names duplicated here for completeness — agent sends to all channels)
        {
            "version": 1, "type": "dictionary_update",
            "session_id": "test-session",
            "kernel_dict":     {},
            "scope_name_dict": {"1": "main_loop"},
            "function_dict":   {},
            "metric_dict":     {}
        },
        # Scope event batch: begin(ts=500) + end(ts=6000) for main_loop (name_id=1)
        {
            "version": 1, "type": "scope_event_batch",
            "session_id": "test-session", "batch_id": 1, "base_time_ns": 500,
            "columns": ["dt_ns","scope_instance_id","name_id","event_type","depth"],
            "rows": [
                [0,    1, 1, 0, 0],
                [5500, 1, 1, 1, 0]
            ]
        },
    ]
    with open(scope_log, "w") as f:
        for ev in scope_events:
            f.write(json.dumps(ev) + "\n")

    # ── system.log ────────────────────────────────────────────────────────────
    system_log = log_dir / f"{prefix}.system.0.log"
    system_events = [
        # Device metric batch: two samples for device 0
        {
            "version": 1, "type": "device_metric_batch",
            "session_id": "test-session", "batch_id": 1, "base_time_ns": 1000,
            "columns": ["dt_ns","device_id","gpu_util","mem_util","temp_c","power_mw","used_mib"],
            "rows": [
                [0,    0, 50, 30, 70, 150000, 1024],
                [3000, 0, 80, 40, 75, 200000, 2048]
            ]
        },
        # Host metric batch
        {
            "version": 1, "type": "host_metric_batch",
            "session_id": "test-session", "batch_id": 1, "base_time_ns": 1000,
            "columns": ["dt_ns","cpu_pct_x100","ram_used_mib","ram_total_mib"],
            "rows": [
                [0,    2500, 4096, 16384],
                [3000, 3000, 4096, 16384]
            ]
        },
    ]
    with open(system_log, "w") as f:
        for ev in system_events:
            f.write(json.dumps(ev) + "\n")

    return str(log_dir), prefix
