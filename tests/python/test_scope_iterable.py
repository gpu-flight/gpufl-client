"""
Tests for the pure-Python wrapper layer in python/gpufl/__init__.py:

  * gpufl.Scope context-manager form (legacy, unchanged)
  * gpufl.Scope iterable form (repeat + warmup, new in 1.0.3)
  * The "<name>_warmup" sub-scope opened automatically when warmup > 0
  * gpufl.clean_logs() - prefix-scoped deletion + active-session guard

These all run without a GPU by monkey-patching the C++ _CScope binding
with a recording fake. The fake captures every constructor call and every
__enter__ / __exit__ so we can assert on (a) the kwargs forwarded to C++,
and (b) the open/close ordering between the warmup sub-scope and the
main scope.
"""
import sys
from pathlib import Path

import pytest

# Mirror test_bindings.py - add the source python/ dir to sys.path so a
# git-checkout dev environment doesn't need an installed wheel.
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import gpufl


# ---------------------------------------------------------------------------
# Fixture: replace the C++ _CScope with a recording fake
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_cscope(monkeypatch):
    """Patch gpufl._CScope so we can observe ctor args + open/close order."""
    events = []      # list of ("enter"|"exit", scope_name)
    ctor_calls = []  # list of (positional_args, kwargs)

    class FakeCScope:
        def __init__(self, *args, **kwargs):
            ctor_calls.append((args, kwargs))
            self._name = args[0] if args else "?"

        def __enter__(self):
            events.append(("enter", self._name))
            return self

        def __exit__(self, *_):
            events.append(("exit", self._name))
            return False

    # The Scope wrapper looks up `_CScope` as a module global, resolved at
    # call time - monkeypatching gpufl._CScope is enough; no reload needed.
    monkeypatch.setattr(gpufl, "_CScope", FakeCScope)
    return events, ctor_calls


# ---------------------------------------------------------------------------
# Context-manager form - must be byte-identical to pre-1.0.3 behavior
# ---------------------------------------------------------------------------

class TestScopeContextManager:
    def test_plain_with_no_kwargs(self, fake_cscope):
        events, ctor_calls = fake_cscope
        with gpufl.Scope("inference"):
            pass
        # Legacy fast path: no repeat/warmup kwargs forwarded to C++.
        assert ctor_calls == [(("inference", ""), {})]
        assert events == [("enter", "inference"), ("exit", "inference")]

    def test_with_tag_passes_positionally(self, fake_cscope):
        events, ctor_calls = fake_cscope
        with gpufl.Scope("forward", "ml"):
            pass
        assert ctor_calls == [(("forward", "ml"), {})]


# ---------------------------------------------------------------------------
# Iterable form - repeat without warmup
# ---------------------------------------------------------------------------

class TestScopeIterableRepeatOnly:
    def test_repeat_yields_zero_to_n_minus_one(self, fake_cscope):
        seen = list(gpufl.Scope("hot", repeat=5))
        assert seen == [0, 1, 2, 3, 4]

    def test_repeat_forwards_kwargs_to_cscope(self, fake_cscope):
        events, ctor_calls = fake_cscope
        list(gpufl.Scope("hot", repeat=5))
        # Exactly one scope is constructed (no warmup sub-scope when warmup=0)
        assert len(ctor_calls) == 1
        args, kwargs = ctor_calls[0]
        assert args == ("hot", "")
        assert kwargs == {"repeat": 5, "warmup": 0}

    def test_repeat_zero_runs_no_body_but_opens_scope(self, fake_cscope):
        events, ctor_calls = fake_cscope
        seen = list(gpufl.Scope("empty", repeat=0))
        assert seen == []
        # Main scope still opens and closes (empty BEGIN/END pair in the log).
        assert events == [("enter", "empty"), ("exit", "empty")]


# ---------------------------------------------------------------------------
# Iterable form - warmup>0 opens the "<name>_warmup" sub-scope
# ---------------------------------------------------------------------------

class TestScopeIterableWarmup:
    def test_warmup_yields_negative_indices_then_measured(self, fake_cscope):
        seen = list(gpufl.Scope("matmul", repeat=10, warmup=3))
        # Warmup: -3, -2, -1 (so `i >= 0` marks a measured iteration)
        # Measured: 0..9
        assert seen == [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_warmup_opens_sub_scope_before_main_scope(self, fake_cscope):
        """The key 1.0.3+ invariant - warmup events get their own bucket."""
        events, ctor_calls = fake_cscope
        list(gpufl.Scope("matmul", repeat=10, warmup=3))

        # Two scopes constructed in order: warmup sub-scope, then main.
        assert len(ctor_calls) == 2

        warmup_args, warmup_kwargs = ctor_calls[0]
        assert warmup_args == ("matmul_warmup", "")
        # Sub-scope carries repeat=warmup_count so the analyzer's
        # per-iter math works for the cold-start cost. warmup=0 because
        # the sub-scope IS the warmup phase (not a nested benchmark).
        assert warmup_kwargs == {"repeat": 3, "warmup": 0}

        main_args, main_kwargs = ctor_calls[1]
        assert main_args == ("matmul", "")
        # Main scope keeps the original metadata - warmup count records
        # for audit ("3 warmup launches were excluded from this scope").
        assert main_kwargs == {"repeat": 10, "warmup": 3}

        # Ordering: warmup scope is fully opened AND closed before the
        # main scope opens. This is the contract that lets the analyzer
        # cleanly partition kernel events into cold-start vs steady-state.
        assert events == [
            ("enter", "matmul_warmup"),
            ("exit",  "matmul_warmup"),
            ("enter", "matmul"),
            ("exit",  "matmul"),
        ]

    def test_warmup_only_no_repeat_still_opens_both(self, fake_cscope):
        """warmup=K with repeat=0 still produces sub-scope + empty main."""
        events, ctor_calls = fake_cscope
        seen = list(gpufl.Scope("name", repeat=0, warmup=2))
        assert seen == [-2, -1]
        assert len(ctor_calls) == 2  # sub-scope + main (empty)
        assert events == [
            ("enter", "name_warmup"),
            ("exit",  "name_warmup"),
            ("enter", "name"),
            ("exit",  "name"),
        ]

    def test_tag_inherits_into_warmup_subscope(self, fake_cscope):
        """User-provided tag flows to both warmup and main scopes."""
        events, ctor_calls = fake_cscope
        list(gpufl.Scope("op", "ml", repeat=2, warmup=1))
        assert ctor_calls[0][0] == ("op_warmup", "ml")
        assert ctor_calls[1][0] == ("op", "ml")


# ---------------------------------------------------------------------------
# Iterable form - error paths
# ---------------------------------------------------------------------------

class TestScopeIterableErrors:
    def test_iter_without_repeat_raises_eagerly(self, fake_cscope):
        events, _ = fake_cscope
        # Validation happens in __iter__, BEFORE the generator starts,
        # so the for-loop raises on entry rather than swallowing into
        # a later StopIteration.
        with pytest.raises(TypeError, match=r"repeat"):
            for _ in gpufl.Scope("no_repeat"):
                pass  # pragma: no cover - should not execute
        # No scopes were constructed.
        assert events == []

    def test_exception_in_body_closes_main_scope(self, fake_cscope):
        events, _ = fake_cscope
        with pytest.raises(ValueError, match="kaboom"):
            for i in gpufl.Scope("crash", repeat=5):
                if i == 1:
                    raise ValueError("kaboom")
        # The main scope's finally-block ran the exit even on exception.
        assert events[-1] == ("exit", "crash")

    def test_negative_repeat_rejected(self):
        with pytest.raises(ValueError, match=r">= 0"):
            gpufl.Scope("x", repeat=-1)

    def test_negative_warmup_rejected(self):
        with pytest.raises(ValueError, match=r">= 0"):
            gpufl.Scope("x", warmup=-1)


# ---------------------------------------------------------------------------
# clean_logs
# ---------------------------------------------------------------------------

class TestCleanLogs:
    def test_refuses_during_active_session(self):
        """Calling clean_logs between init() and shutdown() raises."""
        # Manually set the guard the init() wrapper would set on success.
        gpufl._session_active = True
        try:
            with pytest.raises(RuntimeError, match=r"session is active"):
                gpufl.clean_logs("./irrelevant")
        finally:
            gpufl._session_active = False

    def test_dry_run_lists_matches_without_deleting(self, tmp_path):
        # v1.2 layout: log_path is a directory containing session subdirs.
        # Each session subdir's channel files (device/scope/system, with
        # optional .N rotation index and .gz suffix) are matched. Files
        # at the top level of log_path are NEVER touched, and subdirs
        # whose contents don't match the channel pattern are skipped.
        log_root = tmp_path / "gfl_logs"
        sid_a = log_root / "session-aaaa"
        sid_b = log_root / "session-bbbb"
        sid_a.mkdir(parents=True)
        sid_b.mkdir(parents=True)

        # Session A: one active + one rotated.gz + one unrelated file.
        (sid_a / "device.log").touch()
        (sid_a / "scope.1.log.gz").touch()
        (sid_a / "notes.txt").touch()    # unrelated within subdir - preserves subdir

        # Session B: only channel files (subdir will be fully removable).
        (sid_b / "device.log").touch()
        (sid_b / "system.log.gz").touch()

        # Unrelated top-level entries - must NOT be touched.
        (log_root / "README.md").touch()
        (log_root / "other_app").mkdir()
        (log_root / "other_app" / "data.bin").touch()

        preview = gpufl.clean_logs(str(log_root), dry_run=True)

        names = sorted(Path(p).name for p in preview)
        # Expected: device.log + scope.1.log.gz from session-aaaa (subdir
        # NOT in list because notes.txt prevents removal), and
        # device.log + system.log.gz + the session-bbbb subdir itself
        # (all its contents match → subdir is removable).
        assert names == [
            "device.log",
            "device.log",
            "scope.1.log.gz",
            "session-bbbb",
            "system.log.gz",
        ]
        # Dry-run must NOT delete anything.
        assert (sid_a / "device.log").exists()
        assert (sid_b / "device.log").exists()
        assert (log_root / "README.md").exists()

    def test_actual_delete_keeps_unrelated_files(self, tmp_path):
        # Same layout as the dry-run test; verify real deletion semantics.
        log_root = tmp_path / "gfl_logs"
        sid_a = log_root / "session-aaaa"
        sid_b = log_root / "session-bbbb"
        sid_a.mkdir(parents=True)
        sid_b.mkdir(parents=True)

        (sid_a / "device.log").touch()
        (sid_a / "scope.1.log.gz").touch()
        (sid_b / "device.log").touch()
        (sid_b / "system.log.gz").touch()

        (log_root / "README.md").touch()
        (log_root / "other_app").mkdir()
        (log_root / "other_app" / "data.bin").touch()

        removed = gpufl.clean_logs(str(log_root))

        # 4 channel files + 2 subdirs (both fully removable) = 6 entries.
        assert len(removed) == 6
        # Subdirs themselves are gone.
        assert not sid_a.exists()
        assert not sid_b.exists()
        # Unrelated top-level entries untouched.
        assert (log_root / "README.md").exists()
        assert (log_root / "other_app" / "data.bin").exists()

    def test_defaults_to_last_init_path(self, tmp_path, monkeypatch):
        """No args → uses the path stashed by the most recent init()."""
        # v1.2 layout: my_app is the directory; sessions are subdirs inside.
        log_root = tmp_path / "my_app"
        session = log_root / "session-1234"
        session.mkdir(parents=True)
        (session / "device.log").touch()

        monkeypatch.setattr(gpufl, "_last_log_path", str(log_root))
        monkeypatch.setattr(gpufl, "_last_app_name", "my_app")

        removed = gpufl.clean_logs()
        # One file + the now-empty session subdir = 2 entries.
        assert len(removed) == 2
        names = sorted(Path(p).name for p in removed)
        assert names == ["device.log", "session-1234"]

    def test_no_path_and_no_prior_init_raises(self, monkeypatch):
        monkeypatch.setattr(gpufl, "_last_log_path", None)
        monkeypatch.setattr(gpufl, "_last_app_name", None)
        with pytest.raises(ValueError, match=r"log_path"):
            gpufl.clean_logs()
