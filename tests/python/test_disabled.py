"""
Tests for the gpufl disable flag.

Two equivalent kill-switches:
  * gpufl.init(..., enabled=False)
  * GPUFL_DISABLED env var (1/true/yes/on)

Env wins over kwarg — set via env, you can disable gpufl in someone
else's code without editing it. When disabled, every public entry point
must be a true no-op: no daemon spawn, no C++ runtime, no network calls,
no log files, no atexit handler.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import gpufl


@pytest.fixture(autouse=True)
def _reset_state():
    """Each test gets a clean disable state + clean env."""
    saved = os.environ.pop("GPUFL_DISABLED", None)
    gpufl._disabled = False
    try:
        yield
    finally:
        gpufl._disabled = False
        try:
            gpufl.shutdown()
        except Exception:
            pass
        if saved is not None:
            os.environ["GPUFL_DISABLED"] = saved


# ── enabled kwarg ───────────────────────────────────────────────────────────

class TestEnabledKwarg:
    def test_enabled_false_returns_false(self):
        """init(enabled=False) returns False without raising."""
        assert gpufl.init("disabled_app", enabled=False) is False

    def test_enabled_false_sets_disabled_flag(self):
        """After init(enabled=False) the module-level flag is True."""
        gpufl.init("disabled_app", enabled=False)
        assert gpufl._disabled is True

    def test_enabled_true_is_default(self):
        """enabled=True is the default — same shape as the original init."""
        # In stub mode the return is False (no GPU), but the disabled
        # flag must NOT be set — that path is reserved for the toggle.
        gpufl.init("normal_app")
        assert gpufl._disabled is False

    def test_enabled_true_clears_prior_disabled_state(self):
        """init(enabled=True) after init(enabled=False) re-enables gpufl."""
        gpufl.init("a", enabled=False)
        assert gpufl._disabled is True
        gpufl.init("b", enabled=True)
        assert gpufl._disabled is False


# ── GPUFL_DISABLED env var ──────────────────────────────────────────────────

class TestEnvVar:
    @pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on", "  yes  "])
    def test_truthy_values_disable(self, val):
        os.environ["GPUFL_DISABLED"] = val
        assert gpufl.init("env_app") is False
        assert gpufl._disabled is True

    @pytest.mark.parametrize("val", ["0", "false", "no", "off", ""])
    def test_falsy_values_do_not_disable(self, val):
        os.environ["GPUFL_DISABLED"] = val
        gpufl.init("env_app")
        assert gpufl._disabled is False

    def test_env_overrides_kwarg(self):
        """Env=1 wins even when caller passed enabled=True explicitly."""
        os.environ["GPUFL_DISABLED"] = "1"
        result = gpufl.init("env_wins", enabled=True)
        assert result is False
        assert gpufl._disabled is True

    def test_kwarg_disables_when_env_unset(self):
        """No env var → kwarg controls."""
        assert "GPUFL_DISABLED" not in os.environ
        gpufl.init("kw_app", enabled=False)
        assert gpufl._disabled is True


# ── no-op semantics: nothing should raise, nothing should hit network ───────

class TestNoOpSurfaces:
    def setup_method(self):
        gpufl.init("noop_app", enabled=False)
        assert gpufl._disabled is True

    def test_shutdown_is_noop(self):
        assert gpufl.shutdown() is None

    def test_system_start_stop_are_noop(self):
        assert gpufl.system_start("s") is None
        assert gpufl.system_stop("s") is None

    def test_scope_context_manager_is_noop(self):
        with gpufl.Scope("any_scope", "any_tag"):
            pass

    def test_scope_iterable_yields_correct_indices(self):
        """Disabled scope still yields warmup + measured indices so the
        caller's benchmark loop stays unchanged."""
        seen = list(gpufl.Scope("bench", repeat=3, warmup=2))
        assert seen == [-2, -1, 0, 1, 2]

    def test_upload_logs_is_noop_and_returns_success(self):
        """No network call. Returns an UploadResult-shaped object with a
        clear warning."""
        result = gpufl.upload_logs(
            log_path="/nonexistent/path",
            backend_url="https://this-host-must-not-be-contacted.invalid",
            api_key="garbage",
        )
        assert result.success is True
        assert result.events_uploaded == 0
        assert any("disabled" in w.lower() for w in result.warnings)

    def test_upload_logs_noop_returns_pure_python_class(self):
        """Regression test for the production crash on GPUFL_DISABLED=1.

        The C++ binding's :class:`UploadResult` has no exposed
        ``py::init<>()`` and all its fields are ``def_readonly`` — so
        constructing it from Python and assigning to its fields throws.
        The disabled path MUST therefore avoid touching that class and
        return a pure-Python stand-in instead.

        Without this assertion the bug could regress silently in CI
        because the test environment runs in stub mode (no C++ extension
        loaded), where ``UploadResult`` is a regular mutable Python
        class. Real installs hit the binding and crash.
        """
        result = gpufl.upload_logs(
            log_path="/nonexistent/path",
            backend_url="https://anywhere.invalid",
            api_key="x",
        )
        # The disabled path uses _NoopUploadResult — a private class
        # whose only purpose is to not BE the C++ binding.
        assert isinstance(result, gpufl._NoopUploadResult), (
            f"disabled upload_logs returned a {type(result).__name__}; "
            "this likely means the disabled path is constructing the "
            "C++ UploadResult binding, which crashes in non-stub mode."
        )
        # Round-trip every field the binding exposes — if a future C++
        # change adds a field, the stub class must keep parity.
        for field in ("success", "files_processed", "files_skipped_by_cursor",
                      "events_uploaded", "bytes_uploaded", "elapsed_ms",
                      "warnings", "spool_ids"):
            assert hasattr(result, field), (
                f"_NoopUploadResult missing '{field}' — the C++ binding "
                "exposes it, so callers depending on duck-typing parity "
                "will break.")

    def test_session_context_manager_runs_clean(self):
        """`with gpufl.session(enabled=False)` enters and exits without
        contacting the network, even with creds set."""
        with gpufl.session(
            "noop_session",
            backend_url="https://this-host-must-not-be-contacted.invalid",
            api_key="garbage",
            enabled=False,
        ) as ok:
            assert ok is False
