"""
Import a torch.profiler Chrome Trace Event JSON file into GPUFlight.

This is the zero-friction onboarding path: users who already run
torch.profiler can upload their existing trace files directly — no
need to install or run the GPUFlight C++ client in their training
loop.

The heavy lifting (parsing Chrome Trace events, extracting kernels and
ops, correlating them) happens on the backend. This module is a thin
HTTP client that POSTs the trace file and returns the created session
URL.

The backend endpoint (`POST /api/v1/events/import/torch-profiler`)
accepts multipart/form-data with a single "trace" field containing the
Chrome Trace JSON. See the plan's "Backend — Chrome Trace import"
section for the parser design.

Backend endpoint is v1-deferred — this client-side helper ships but
may return a clear "endpoint not yet deployed" message until the
server lands. Safe no-op prior to backend rollout.
"""

from __future__ import annotations

import os
from typing import Optional


def _default_endpoint() -> str:
    """Resolve the upload endpoint from env or fall back to the public
    app URL. Mirrors how `gpufl.init()` resolves its ingestion URL."""
    env = os.environ.get("GPUFL_TRACE_IMPORT_URL")
    if env:
        return env
    base = os.environ.get("GPUFL_API_URL", "https://api.gpuflight.com")
    return f"{base}/api/v1/events/import/torch-profiler"


def import_trace(
    path: str,
    *,
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout_sec: int = 60,
) -> str:
    """
    Upload a Chrome Trace Event JSON file to GPUFlight and return the
    URL of the resulting session page.

    Parameters
    ----------
    path : str
        Local path to the Chrome Trace JSON file produced by
        `torch.profiler.profile(...).export_chrome_trace("trace.json")`.
    endpoint : str, optional
        Override the upload endpoint. Defaults to the value of the
        `GPUFL_TRACE_IMPORT_URL` env var, or
        `https://api.gpuflight.com/api/v1/events/import/torch-profiler`.
    api_key : str, optional
        Bearer token for authentication. Defaults to the value of the
        `GPUFL_API_KEY` env var. Required for the hosted backend.
    timeout_sec : int
        HTTP timeout. Default 60s; raise for large traces (>100MB).

    Returns
    -------
    str
        The URL of the imported session's dashboard page.

    Raises
    ------
    FileNotFoundError
        If `path` doesn't exist.
    RuntimeError
        If the backend rejects the upload or the endpoint isn't yet
        deployed (returns 404 / 501).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Trace file not found: {path}")

    # Lazy import — keeps `import gpufl.torch` working for users who
    # only need attach/detach and don't have requests installed.
    try:
        import requests
    except ImportError as e:
        raise RuntimeError(
            "import_trace() requires the `requests` package.\n"
            "Install it via: pip install requests"
        ) from e

    resolved_endpoint = endpoint or _default_endpoint()
    resolved_key = api_key or os.environ.get("GPUFL_API_KEY")

    headers = {}
    if resolved_key:
        headers["Authorization"] = f"Bearer {resolved_key}"

    with open(path, "rb") as f:
        files = {"trace": (os.path.basename(path), f, "application/json")}
        try:
            resp = requests.post(
                resolved_endpoint,
                headers=headers,
                files=files,
                timeout=timeout_sec,
            )
        except requests.RequestException as e:
            raise RuntimeError(
                f"Failed to upload trace to {resolved_endpoint}: {e}"
            ) from e

    if resp.status_code == 404 or resp.status_code == 501:
        raise RuntimeError(
            "Trace import endpoint not yet deployed on the backend.\n"
            "This feature is shipping in a future release. In the meantime, "
            "use `gpufl.torch.attach()` in your training script for live capture."
        )
    if not resp.ok:
        raise RuntimeError(
            f"Trace upload failed: HTTP {resp.status_code} — {resp.text[:200]}"
        )

    body = resp.json()
    return body.get("session_url", "")
