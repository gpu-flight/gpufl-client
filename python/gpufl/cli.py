"""gpufl CLI — entry point for `gpufl upload <logdir>` and related tools.

Installed via [project.scripts] in pyproject.toml as `gpufl`. Today the
only subcommand is `upload` (ad-hoc / post-mortem deferred upload of an
existing log directory). Production fleet sync should still use
`gpufl-agent` — the JVM service is the recommended path for clusters.

Run `gpufl --help` for usage.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import gpufl


# Exit codes — chosen for shell pipeline use:
#   0 = success (every event uploaded)
#   1 = partial success (some warnings, but events were uploaded)
#   2 = full failure (auth error, missing dir, timeout, etc.)
_EXIT_OK = 0
_EXIT_PARTIAL = 1
_EXIT_FAIL = 2


def _cmd_upload(args: argparse.Namespace) -> int:
    backend_url = args.backend_url or os.environ.get('GPUFL_BACKEND_URL', '')
    api_key     = args.api_key     or os.environ.get('GPUFL_API_KEY', '')

    if not backend_url:
        print("error: --backend-url required (or set GPUFL_BACKEND_URL)",
              file=sys.stderr)
        return _EXIT_FAIL
    if not api_key:
        print("error: --api-key required (or set GPUFL_API_KEY)",
              file=sys.stderr)
        return _EXIT_FAIL

    if args.session_id and args.all_sessions:
        print("error: --session-id and --all-sessions are mutually exclusive",
              file=sys.stderr)
        return _EXIT_FAIL

    try:
        result = gpufl.upload_logs(
            log_path=args.log_path,
            backend_url=backend_url,
            api_key=api_key,
            api_path=args.api_path,
            total_timeout_ms=args.timeout * 1000,
            max_retries=args.retries,
            report_progress=not args.quiet,
            session_id=args.session_id,
            all_sessions=args.all_sessions,
            force=args.force,
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return _EXIT_FAIL
    except Exception as e:
        # upload_logs() shouldn't raise on network errors, but defensive
        # against unexpected internal errors.
        print(f"error: upload_logs raised unexpectedly: {e}",
              file=sys.stderr)
        return _EXIT_FAIL

    # Summary to stdout (machine-friendlier than the per-progress stderr lines)
    print(f"Uploaded {result.events_uploaded} events "
          f"({result.bytes_uploaded / (1024 * 1024):.1f} MB) "
          f"across {result.files_processed} file(s) "
          f"in {result.elapsed_ms / 1000:.1f}s.")
    if result.files_skipped_by_cursor:
        print(f"Skipped {result.files_skipped_by_cursor} file(s) "
              f"already uploaded (per cursor).")
    if result.warnings:
        print(f"\n{len(result.warnings)} warning(s):", file=sys.stderr)
        for w in result.warnings:
            print(f"  - {w}", file=sys.stderr)

    if result.success and not result.warnings:
        return _EXIT_OK
    if result.success and result.warnings:
        return _EXIT_PARTIAL
    return _EXIT_FAIL


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gpufl",
        description="GPUFlight client utilities.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    up = sub.add_parser(
        "upload",
        help="Upload a session's NDJSON logs to the GPUFlight backend.",
        description=(
            "Bulk-uploads the NDJSON log files produced by a previous "
            "gpufl session to the configured backend. Recommended for "
            "dev / one-off use; for production fleets use gpufl-agent."
        ),
    )
    up.add_argument(
        "log_path",
        help=("Path prefix for the session's logs — same as the "
              "InitOptions.log_path used when the session ran. Both the "
              "directory and filename-prefix; e.g. "
              "'/tmp/runs/pytorch_demo' matches "
              "'/tmp/runs/pytorch_demo.device.log' plus rotated files."),
    )
    up.add_argument(
        "--backend-url", default=None,
        help="Backend base URL. Env: GPUFL_BACKEND_URL.",
    )
    up.add_argument(
        "--api-key", default=None,
        help="Bearer token. Env: GPUFL_API_KEY.",
    )
    up.add_argument(
        "--api-path", default="",
        help="Reverse-proxy path mount. Defaults to /api/v1.",
    )
    up.add_argument(
        "--timeout", type=int, default=300,
        help="Total wall budget for the upload, in seconds. Default 300 (5min).",
    )
    up.add_argument(
        "--retries", type=int, default=1,
        help="Retries per failing POST. Default 1.",
    )
    up.add_argument(
        "--quiet", action="store_true",
        help="Suppress periodic progress lines on stderr.",
    )

    # ── Session selection ───────────────────────────────────────────────
    #
    # By default `gpufl upload <dir>` uploads only the LATEST session
    # present in the directory (the most recent job_start.ts_ns). The
    # next two flags override that:
    #   --session-id <id>   only upload that specific session
    #   --all-sessions      upload every session in the dir
    # They're mutually exclusive — passing both is an error.
    up.add_argument(
        "--session-id", default=None,
        help=("Upload only the session with this session_id. Mutually "
              "exclusive with --all-sessions. By default 'gpufl upload' "
              "ships only the latest session found in the directory."),
    )
    up.add_argument(
        "--all-sessions", action="store_true",
        help=("Upload every session present in the directory (default: "
              "only the latest). Already-uploaded sessions (per the "
              "cursor file) are skipped silently unless --force is set."),
    )
    up.add_argument(
        "--force", action="store_true",
        help=("Re-upload sessions even if the cursor file says they've "
              "already shipped. Default: refuse to re-upload (single-"
              "session modes) or skip silently (--all-sessions)."),
    )

    up.set_defaults(func=_cmd_upload)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
