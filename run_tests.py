#!/usr/bin/env python3
"""Single-entry test runner for gpufl-client (C++ + Python).

Builds the GoogleTest binary, runs it, then runs the pytest suite that
exercises the pybind11 bindings. Designed to be the one command a
contributor needs to run before pushing -- and the one CI runs in lieu
of separate test stages.

Usage:
    python run_tests.py                  # build + run everything
    python run_tests.py --cpp-only       # skip Python entirely
    python run_tests.py --python-only    # skip C++ build + run
    python run_tests.py --no-build       # rerun existing binaries (fast loop)
    python run_tests.py --cpp-filter 'UploadLogs*'   # GoogleTest filter
    python run_tests.py --py-filter 'analyzer'       # pytest -k expression
    python run_tests.py --reinstall      # force a fresh pip install of gpufl

Exit codes:
    0 -- every requested step passed
    1 -- at least one step failed (build, gtest, or pytest)
    2 -- bad CLI args / environment misconfigured

Cross-platform: works on Windows MSVC and Linux/Mac single-config
generators (Ninja, Make) without modification -- the script asks CMake
where the binary landed rather than guessing.
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
DEFAULT_BUILD_DIR = ROOT / "build_tests"
PY_TESTS_DIR = ROOT / "tests" / "python"
GTEST_TARGET = "gpufl_tests"


# -- tiny TTY helpers (avoid a dependency just for colors) -------------

def _supports_color() -> bool:
    if not sys.stdout.isatty():
        return False
    if platform.system() == "Windows":
        # Modern Windows Terminal / VS Code / Git Bash all honour ANSI.
        return os.environ.get("TERM") != "dumb"
    return True


_COLOR = _supports_color()


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _COLOR else s


def header(msg: str) -> None:
    # ASCII-only -- Windows cp1252 chokes on box-drawing / arrows, and
    # adding `chcp 65001` / PYTHONIOENCODING gymnastics for a status
    # bar isn't worth it.
    bar = "-" * max(8, 72 - len(msg))
    print(f"\n{_c('1;36', '>>> ' + msg)} {_c('2', bar)}", flush=True)


def ok(msg: str) -> None:
    print(f"  {_c('32', '[ok]')} {msg}", flush=True)


def fail(msg: str) -> None:
    print(f"  {_c('31', '[FAIL]')} {msg}", flush=True)


def info(msg: str) -> None:
    print(f"  {_c('2', '...')} {msg}", flush=True)


# -- shell helpers -----------------------------------------------------

def run(cmd: List[str], cwd: Optional[Path] = None, *, env=None,
        check: bool = False) -> int:
    """Run a subprocess, stream its output. Return the exit code."""
    info(f"$ {' '.join(_quote(x) for x in cmd)}")
    proc = subprocess.run(cmd, cwd=cwd, env=env)
    if check and proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return proc.returncode


def _quote(s: str) -> str:
    return f'"{s}"' if " " in s and not (s.startswith('"') and s.endswith('"')) else s


# -- steps -------------------------------------------------------------

def configure_cpp(build_dir: Path) -> bool:
    """Run CMake configure if the build dir hasn't been configured yet.

    Idempotent -- a re-configure on an already-configured dir is cheap
    (CMake's own cache check), but we skip when the CMakeCache exists
    just to keep the output quiet on the common path."""
    if (build_dir / "CMakeCache.txt").exists():
        info(f"reusing existing CMake build dir at {build_dir}")
        return True
    build_dir.mkdir(parents=True, exist_ok=True)
    rc = run([
        "cmake", "-S", str(ROOT), "-B", str(build_dir),
        "-DBUILD_TESTS=ON",
    ])
    return rc == 0


def build_cpp(build_dir: Path, config: str) -> bool:
    """Build the gpufl_tests target. Works for both single-config (Ninja /
    Make on Linux) and multi-config (Visual Studio / Xcode) generators --
    `--config` is silently ignored when not applicable."""
    rc = run([
        "cmake", "--build", str(build_dir),
        "--target", GTEST_TARGET,
        "--config", config,
    ])
    return rc == 0


def find_gtest_binary(build_dir: Path, config: str) -> Optional[Path]:
    """Locate the test binary CMake produced. Multi-config generators
    nest under `<config>/`; single-config drops it at the target's
    output dir. We probe both."""
    name = GTEST_TARGET + (".exe" if platform.system() == "Windows" else "")
    candidates = [
        build_dir / "tests" / config / name,   # Visual Studio
        build_dir / "tests" / name,            # Ninja / Make
        build_dir / config / name,             # some toolchains
        build_dir / name,                       # rare
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def run_gtest(binary: Path, filter_expr: Optional[str]) -> bool:
    cmd = [str(binary)]
    if filter_expr:
        cmd.append(f"--gtest_filter={filter_expr}")
    rc = run(cmd)
    return rc == 0


def ensure_python_module(reinstall: bool) -> bool:
    """Install the gpufl Python module if it's not importable, or
    reinstall on demand. Uses the same flags the project's
    `build.sh` does so the module's feature surface matches what
    the CLI/notebook user gets."""
    if not reinstall:
        try:
            subprocess.run(
                [sys.executable, "-c", "import gpufl"],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            info("gpufl module already importable -- skipping pip install "
                 "(pass --reinstall to force)")
            return True
        except subprocess.CalledProcessError:
            info("gpufl module not importable -- installing")

    cmd = [
        sys.executable, "-m", "pip", "install", str(ROOT), "-v",
        "-C", "cmake.define.BUILD_PYTHON=ON",
        "-C", "cmake.define.GPUFL_ENABLE_CUDA=ON",
        "-C", "cmake.define.GPUFL_ENABLE_NVML=ON",
    ]
    rc = run(cmd)
    return rc == 0


def run_pytest(filter_expr: Optional[str], verbose: bool) -> bool:
    if not PY_TESTS_DIR.exists():
        fail(f"Python test dir missing: {PY_TESTS_DIR}")
        return False
    cmd = [sys.executable, "-m", "pytest", str(PY_TESTS_DIR)]
    if verbose:
        cmd.append("-v")
    if filter_expr:
        cmd.extend(["-k", filter_expr])
    rc = run(cmd)
    return rc == 0


# -- main --------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build + run the C++ (GoogleTest) and Python (pytest) "
                    "test suites for gpufl-client.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:", 1)[1] if "Usage:" in __doc__ else "",
    )
    side = p.add_mutually_exclusive_group()
    side.add_argument("--cpp-only", action="store_true",
                      help="Run only the C++ side; skip Python.")
    side.add_argument("--python-only", action="store_true",
                      help="Run only the Python side; skip C++.")

    p.add_argument("--no-build", action="store_true",
                   help="Skip configure/build steps for both sides. Useful "
                        "for re-running the same binaries after a quick edit.")
    p.add_argument("--reinstall", action="store_true",
                   help="Force `pip install .` even if gpufl is already "
                        "importable. Required after editing bindings.cpp.")

    p.add_argument("--build-dir", default=str(DEFAULT_BUILD_DIR), type=Path,
                   help=f"CMake build directory. Default: {DEFAULT_BUILD_DIR.name}/")
    p.add_argument("--config", default="Release",
                   choices=("Debug", "Release", "RelWithDebInfo", "MinSizeRel"),
                   help="CMake build config (multi-config generators only). "
                        "Default: Release.")

    p.add_argument("--cpp-filter", default=None,
                   help="GoogleTest --gtest_filter expression (e.g. 'UploadLogs*').")
    p.add_argument("--py-filter", default=None,
                   help="pytest -k expression (e.g. 'analyzer').")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Pass -v to pytest for per-test output.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    do_cpp    = not args.python_only
    do_python = not args.cpp_only

    start = time.monotonic()
    results: List[Tuple[str, bool]] = []

    # -- C++ side ------------------------------------------------------
    if do_cpp:
        if not args.no_build:
            header("C++: CMake configure")
            if not configure_cpp(args.build_dir):
                fail("CMake configure failed")
                return summarize(results + [("C++ configure", False)], start)
            ok("configured")

            header(f"C++: build {GTEST_TARGET}")
            if not build_cpp(args.build_dir, args.config):
                fail("C++ build failed")
                return summarize(results + [("C++ build", False)], start)
            ok("built")

        header("C++: run GoogleTest")
        binary = find_gtest_binary(args.build_dir, args.config)
        if not binary:
            fail(f"could not locate {GTEST_TARGET} under {args.build_dir} -- "
                 "did the build succeed?")
            results.append(("C++ run", False))
        else:
            info(f"binary: {binary}")
            passed = run_gtest(binary, args.cpp_filter)
            results.append(("C++ tests", passed))
            (ok if passed else fail)(
                "all GoogleTest cases passed" if passed
                else "one or more GoogleTest cases failed")

    # -- Python side ---------------------------------------------------
    if do_python:
        if not args.no_build:
            header("Python: install gpufl module")
            installed = ensure_python_module(args.reinstall)
            results.append(("Python install", installed))
            if not installed:
                fail("pip install failed -- skipping pytest")
                return summarize(results, start)
            ok("installed")

        header("Python: run pytest")
        passed = run_pytest(args.py_filter, args.verbose)
        results.append(("Python tests", passed))
        (ok if passed else fail)(
            "all pytest cases passed" if passed
            else "one or more pytest cases failed")

    return summarize(results, start)


def summarize(results: List[Tuple[str, bool]], start: float) -> int:
    elapsed = time.monotonic() - start
    header(f"Summary ({elapsed:.1f}s)")
    if not results:
        info("nothing ran")
        return 0
    all_ok = True
    for name, passed in results:
        (ok if passed else fail)(name)
        all_ok = all_ok and passed
    return 0 if all_ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
