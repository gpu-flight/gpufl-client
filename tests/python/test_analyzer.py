import sys
import os
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
