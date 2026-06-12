import sys
import os
import json
from pathlib import Path
import pytest

# Add the python directory to sys.path so we can import gpufl
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

from gpufl.analyzer import GpuFlightSession

def test_session_loading(mock_log_dir):
    log_dir, prefix = mock_log_dir
    session = GpuFlightSession(log_dir, log_prefix=prefix)
    
    assert not session.kernels.empty
    assert len(session.kernels) == 2
    assert "vectorAdd" in session.kernels["name"].values
    assert "matrixMul" in session.kernels["name"].values
    assert session.static_devices[0]["name"] == "NVIDIA Test GPU"

def test_session_metrics(mock_log_dir):
    log_dir, prefix = mock_log_dir
    session = GpuFlightSession(log_dir, log_prefix=prefix)

    # Check enriched metrics
    assert "duration_ms" in session.kernels.columns
    # queue_latency_ms only present in old per-event format (api_start/exit not in batch)

    # vectorAdd: start=1000, duration=1000 ns -> 0.001 ms
    vector_add = session.kernels[session.kernels["name"] == "vectorAdd"].iloc[0]
    assert vector_add["duration_ms"] == pytest.approx(0.001)

    assert "clock_sm" in session.device_metrics.columns
    assert session.device_metrics["clock_sm"].dropna().mean() == pytest.approx(2000.0)

def test_session_summary(mock_log_dir, capsys):
    log_dir, prefix = mock_log_dir
    session = GpuFlightSession(log_dir, log_prefix=prefix)
    
    # This should print to console without error
    session.print_summary()
    captured = capsys.readouterr()
    # GpuFlightSession uses rich, which might bypass capsys or use its own console.
    # But print_summary uses self.console.print.

def test_session_hotspots(mock_log_dir):
    log_dir, prefix = mock_log_dir
    session = GpuFlightSession(log_dir, log_prefix=prefix)
    
    # Should not crash
    session.inspect_hotspots()

def test_session_scopes(mock_log_dir):
    log_dir, prefix = mock_log_dir
    session = GpuFlightSession(log_dir, log_prefix=prefix)
    
    # Should not crash
    session.inspect_scopes()

def test_session_loads_canonical_static_devices(mock_log_dir):
    log_dir, prefix = mock_log_dir
    session = GpuFlightSession(log_dir, log_prefix=prefix)

    assert session.static_devices == [
        {"name": "NVIDIA Test GPU", "vendor": "NVIDIA", "multi_processor_count": 108}
    ]

def test_session_loads_rocm_static_devices(mock_log_dir_rocm_only):
    log_dir, prefix = mock_log_dir_rocm_only
    session = GpuFlightSession(log_dir, log_prefix=prefix)

    assert session.static_devices == [
        {"name": "AMD Test GPU", "vendor": "AMD", "multi_processor_count": 120}
    ]

def test_session_loads_legacy_device_batch_without_clock(mock_log_dir_legacy_device_batch):
    log_dir, prefix = mock_log_dir_legacy_device_batch
    session = GpuFlightSession(log_dir, log_prefix=prefix)

    assert not session.device_metrics.empty
    assert "clock_sm" in session.device_metrics.columns
    assert session.device_metrics["clock_sm"].isna().all()


def test_session_can_load_all_rotated_windows(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    prefix = "rotated"

    first = [
        {
            "version": 1, "type": "job_start",
            "session_id": "rotated-session", "app": "test", "pid": 1,
            "ts_ns": 100, "host": {}, "devices": []
        },
        {
            "version": 1, "type": "dictionary_update",
            "session_id": "rotated-session",
            "kernel_dict": {"1": "firstKernel", "2": "secondKernel"},
            "scope_name_dict": {}, "function_dict": {}, "metric_dict": {}
        },
        {
            "version": 1, "type": "kernel_event_batch",
            "session_id": "rotated-session", "batch_id": 1,
            "base_time_ns": 1000,
            "columns": ["dt_ns", "kernel_id", "stream_id", "duration_ns",
                        "corr_id", "dyn_shared", "num_regs", "has_details"],
            "rows": [[0, 1, 0, 100, 11, 0, 16, 0]]
        },
    ]
    second = [
        {
            "version": 1, "type": "kernel_event_batch",
            "session_id": "rotated-session", "batch_id": 2,
            "base_time_ns": 2000,
            "columns": ["dt_ns", "kernel_id", "stream_id", "duration_ns",
                        "corr_id", "dyn_shared", "num_regs", "has_details"],
            "rows": [[0, 2, 0, 200, 22, 0, 32, 0]]
        },
        {
            "type": "shutdown",
            "session_id": "rotated-session", "app": "test", "pid": 1,
            "ts_ns": 3000
        },
    ]

    for name, events in [
        (f"{prefix}.device.1.log", first),
        (f"{prefix}.device.2.log", second),
    ]:
        with open(log_dir / name, "w") as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")

    light = GpuFlightSession(log_dir, log_prefix=prefix)
    assert len(light.kernels) == 1

    full = GpuFlightSession(log_dir, log_prefix=prefix,
                            load_all_rotated=True)
    assert len(full.kernels) == 2
    assert list(full.kernels["name"]) == ["firstKernel", "secondKernel"]
