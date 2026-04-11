"""
Tests for the gpufl Python bindings.

These tests verify that the Python API (init kwargs, InitOptions fields,
enum values, Scope context manager) match the C++ API, even when running
in stub mode (no GPU).
"""
import sys
from pathlib import Path

import pytest

# Add the python directory to sys.path so we can import gpufl
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import gpufl


# ---------------------------------------------------------------------------
# InitOptions field completeness
# ---------------------------------------------------------------------------

# Every field in the C++ InitOptions struct must exist on the Python object.
EXPECTED_INIT_OPTIONS_FIELDS = {
    "app_name":               ("gpufl", str),
    "log_path":               ("",      str),
    "sampling_auto_start":    (False,   bool),
    "system_sample_rate_ms":  (0,       int),
    "kernel_sample_rate_ms":  (0,       int),
    "enable_kernel_details":  (False,   bool),
    "enable_debug_output":    (False,   bool),
    "enable_stack_trace":     (True,    bool),
    "enable_source_collection": (True,  bool),
    "flush_logs_always":      (False,   bool),
}


class TestInitOptions:
    def test_all_fields_exist(self):
        """InitOptions stub/binding exposes every C++ field."""
        opts = gpufl.InitOptions()
        for field in EXPECTED_INIT_OPTIONS_FIELDS:
            assert hasattr(opts, field), f"Missing field: InitOptions.{field}"

    def test_default_values(self):
        """Default values match the C++ defaults."""
        opts = gpufl.InitOptions()
        for field, (expected_default, _) in EXPECTED_INIT_OPTIONS_FIELDS.items():
            actual = getattr(opts, field)
            assert actual == expected_default, (
                f"InitOptions.{field}: expected {expected_default!r}, got {actual!r}"
            )

    def test_fields_are_writable(self):
        """All fields can be set (not read-only)."""
        opts = gpufl.InitOptions()
        opts.app_name = "test_app"
        assert opts.app_name == "test_app"
        opts.enable_source_collection = False
        assert opts.enable_source_collection is False
        opts.system_sample_rate_ms = 100
        assert opts.system_sample_rate_ms == 100


# ---------------------------------------------------------------------------
# Enum completeness
# ---------------------------------------------------------------------------

EXPECTED_PROFILING_ENGINES = [
    "PcSampling",
    "SassMetrics",
    "RangeProfiler",
    "PcSamplingWithSass",
]

EXPECTED_BACKEND_KINDS = [
    "Auto",
    "Nvidia",
    "Amd",
]


class TestEnums:
    def test_profiling_engine_values(self):
        """All C++ ProfilingEngine enum values are accessible."""
        for name in EXPECTED_PROFILING_ENGINES:
            assert hasattr(gpufl.ProfilingEngine, name), (
                f"Missing ProfilingEngine.{name}"
            )

    def test_profiling_engine_none(self):
        """ProfilingEngine.None_ exists (Python keyword workaround)."""
        assert hasattr(gpufl.ProfilingEngine, "None_")

    def test_backend_kind_values(self):
        """All C++ BackendKind enum values are accessible."""
        for name in EXPECTED_BACKEND_KINDS:
            assert hasattr(gpufl.BackendKind, name), (
                f"Missing BackendKind.{name}"
            )


# ---------------------------------------------------------------------------
# Scope context manager
# ---------------------------------------------------------------------------

class TestScope:
    def test_scope_context_manager(self):
        """Scope works as a context manager without errors."""
        with gpufl.Scope("test_scope"):
            pass

    def test_scope_with_tag(self):
        """Scope accepts an optional tag argument."""
        with gpufl.Scope("test_scope", "test_tag"):
            pass

    def test_nested_scopes(self):
        """Scopes can nest."""
        with gpufl.Scope("outer"):
            with gpufl.Scope("inner"):
                pass


# ---------------------------------------------------------------------------
# init() function signature
# ---------------------------------------------------------------------------

class TestInitFunction:
    def test_init_exists(self):
        """gpufl.init is callable."""
        assert callable(gpufl.init)

    def test_init_returns_bool(self):
        """init() returns a bool (True on success, False on failure/stub)."""
        # In stub mode this returns False; with GPU it returns True.
        result = gpufl.init("test_app")
        assert isinstance(result, bool)

    def test_shutdown_exists(self):
        """gpufl.shutdown is callable."""
        assert callable(gpufl.shutdown)
        gpufl.shutdown()

    def test_init_accepts_all_kwargs(self):
        """init() accepts all documented keyword arguments without error."""
        # This tests that the function signature matches.
        # In stub mode it returns False but should not raise.
        result = gpufl.init(
            "test_app",
            log_path="./test_logs",
            sampling_auto_start=False,
            system_sample_rate_ms=50,
            kernel_sample_rate_ms=50,
            enable_kernel_details=True,
            enable_debug_output=False,
            enable_profiling=True,
            enable_stack_trace=True,
            enable_source_collection=True,
        )
        assert isinstance(result, bool)
        gpufl.shutdown()

    def test_init_with_profiling_engine(self):
        """init() accepts profiling_engine kwarg."""
        result = gpufl.init(
            "test_engine",
            profiling_engine=gpufl.ProfilingEngine.SassMetrics,
        )
        assert isinstance(result, bool)
        gpufl.shutdown()


# ---------------------------------------------------------------------------
# System start/stop
# ---------------------------------------------------------------------------

class TestSystemMonitoring:
    def test_system_start_exists(self):
        assert callable(gpufl.system_start)

    def test_system_stop_exists(self):
        assert callable(gpufl.system_stop)

    def test_system_start_stop(self):
        """system_start/stop don't crash in stub mode."""
        gpufl.system_start("test")
        gpufl.system_stop("test")
