"""Smoke test for direct-to-backend upload (HttpLogSink).

Validates the Python kwargs / env-var plumbing that reaches the C++
HttpLogSink. We spin up a tiny stub HTTP server on localhost that
accepts any POST to /api/v1/events/<type>, records the request
details, and returns 200. Then we run gpufl.init(..., remote_upload=True)
pointed at it, trigger a couple of events, and assert the server saw
them with the right auth header and path.

The test is defensively skipped when either:
  - the C++ extension is not available (no-GPU CI); or
  - the pybind11 init() binding doesn't accept the remote_* kwargs
    (older builds) — we check this with inspect.signature.
"""
from __future__ import annotations

import inspect
import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


# ──────────────────────────────────────────────────────────────────────────────


def _free_port() -> int:
    """Grab an ephemeral localhost port so parallel tests don't collide."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _CaptureHandler(BaseHTTPRequestHandler):
    """Minimal ingestion-endpoint stub. Records each POST body + headers."""

    # Populated by the fixture so tests can inspect what arrived.
    captured: list[dict] = []

    def do_POST(self):  # noqa: N802 — name required by BaseHTTPRequestHandler
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8", errors="replace")
        self.captured.append({
            "path": self.path,
            "body": body,
            "auth": self.headers.get("Authorization", ""),
        })
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def log_message(self, *_args, **_kwargs):  # silence stderr spam
        return


@pytest.fixture
def stub_backend():
    """Start a thread-backed HTTP stub on a free port; tear down after test."""
    _CaptureHandler.captured = []
    port = _free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), _CaptureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield {
            "url": f"http://127.0.0.1:{port}",
            "captured": _CaptureHandler.captured,
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# ──────────────────────────────────────────────────────────────────────────────


def _init_accepts_remote_kwargs() -> bool:
    """True iff the currently-loaded gpufl.init() exposes the new kwargs."""
    import gpufl
    try:
        sig = inspect.signature(gpufl.init)
    except (TypeError, ValueError):
        return False
    return "remote_upload" in sig.parameters


def test_remote_upload_routes_events_to_stub_backend(stub_backend, tmp_path):
    """Happy path: kwargs flow through, events POSTed with Bearer auth."""
    try:
        import gpufl
    except ImportError:
        pytest.skip("gpufl C++ extension not available in this environment")
    if not _init_accepts_remote_kwargs():
        pytest.skip("gpufl.init does not yet expose remote_upload kwargs")
    # Skip unless we can actually initialize (no-GPU builds return False from
    # the stub init function, leaving the NDJSON pipeline dormant).
    if getattr(gpufl, "_gpufl_client", None) is None:
        pytest.skip("no C++ extension loaded — stub init cannot exercise HttpLogSink")

    log_prefix = str(tmp_path / "remote_upload_smoke")
    ok = gpufl.init(
        app_name="remote_upload_smoke",
        log_path=log_prefix,
        backend_url=stub_backend["url"],
        api_key="gpfl_test_abc123",
        remote_upload=True,
        sampling_auto_start=False,
        enable_debug_output=False,
    )
    if not ok:
        pytest.skip("gpufl.init returned False — no GPU backend available")

    try:
        with gpufl.Scope("smoke_scope"):
            pass
    finally:
        gpufl.shutdown()

    # Give the worker thread a moment to drain the queue.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not stub_backend["captured"]:
        time.sleep(0.05)

    captured = list(stub_backend["captured"])
    assert captured, "expected at least one event POSTed to stub backend"

    # Every POST must carry the Bearer auth header.
    for c in captured:
        assert c["auth"] == "Bearer gpfl_test_abc123", c

    # The stub backend saw a job_start first (issued by gpufl::init).
    job_start = [c for c in captured if c["path"].endswith("/job_start")]
    assert job_start, f"expected at least one job_start POST, got paths={[c['path'] for c in captured]}"

    # The body is the EventWrapper envelope: {"data":"<ndjson>",
    # "agentSendingTime":<ms>,"hostname":"","ipAddr":""}. The inner
    # ndjson string must carry the matching `type`.
    for c in captured:
        envelope = json.loads(c["body"])
        assert "data" in envelope, envelope
        assert "agentSendingTime" in envelope, envelope
        inner = json.loads(envelope["data"])
        expected_type = c["path"].rsplit("/", 1)[-1]
        assert inner.get("type") == expected_type, (
            c["path"], inner.get("type"))


def test_env_var_enables_remote_upload(stub_backend, tmp_path, monkeypatch):
    """GPUFL_REMOTE_UPLOAD=1 should flip the flag without a kwarg."""
    try:
        import gpufl
    except ImportError:
        pytest.skip("gpufl C++ extension not available")
    if not _init_accepts_remote_kwargs():
        pytest.skip("gpufl.init does not yet expose remote_upload kwargs")
    if getattr(gpufl, "_gpufl_client", None) is None:
        pytest.skip("no C++ extension loaded")

    monkeypatch.setenv("GPUFL_REMOTE_UPLOAD", "1")
    monkeypatch.setenv("GPUFL_BACKEND_URL", stub_backend["url"])
    monkeypatch.setenv("GPUFL_API_KEY", "gpfl_env_key")

    log_prefix = str(tmp_path / "remote_upload_env")
    ok = gpufl.init(app_name="remote_upload_env", log_path=log_prefix,
                    sampling_auto_start=False)
    if not ok:
        pytest.skip("gpufl.init returned False — no GPU")
    try:
        with gpufl.Scope("env_scope"):
            pass
    finally:
        gpufl.shutdown()

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not stub_backend["captured"]:
        time.sleep(0.05)
    assert stub_backend["captured"], "expected env-var path to upload too"
    for c in stub_backend["captured"]:
        assert c["auth"] == "Bearer gpfl_env_key"
