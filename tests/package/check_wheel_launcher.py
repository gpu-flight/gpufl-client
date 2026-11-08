"""Check the gpufl command an installed Linux wheel provides.

Run it with the Python of the environment the wheel was installed into. CI runs
it as cibuildwheel's Linux test command (release.yml and build.yml):
cibuildwheel installs each repaired wheel into a fresh virtual environment and
runs this script with that environment's Python. By hand:

    python3 -m venv /tmp/gpufl-venv
    /tmp/gpufl-venv/bin/pip install dist/gpufl-*.whl
    /tmp/gpufl-venv/bin/python tests/package/check_wheel_launcher.py

It checks that:

1. `gpufl` resolves on PATH to the file pip put in this environment's scripts
   directory, and that file is native (ELF), not a Python script.
2. `gpufl --version` prints `gpufl <version> (wire v1)` for the installed
   distribution's version.
3. `gpufl upload --help` reaches the upload subcommand.
4. `gpufl trace` finds libgpufl_inject.so inside the installed package. It
   traces `true`, which needs no GPU, with --verbose so the launcher prints the
   library it resolved. The trace must exit 0 and the library must preload.

Before that, `ldd` must resolve every library the entry, the launcher and the
injection library load. The release build links libcuda.so.1 and
libnvidia-ml.so.1, which come with the NVIDIA driver; where the driver is
missing, as on CI runners, the CUDA toolkit's link stubs stand in for them.
None of these checks calls either library. Any other missing library fails
the check, because auditwheel should have bundled it.
"""

from __future__ import annotations

import importlib.metadata
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

DIST = "gpufl"
# Driver library SONAME -> the CUDA toolkit stub that provides it.
DRIVER_LIBS = {
    "libcuda.so.1": "libcuda.so",
    "libnvidia-ml.so.1": "libnvidia-ml.so",
}
INJECT_LIB_LINE = "[gpufl] inject lib: "


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr, flush=True)
    sys.exit(1)


def run(cmd: list[str], env: dict[str, str], cwd: Path) -> subprocess.CompletedProcess:
    print("$ " + " ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd, env=env, cwd=cwd, capture_output=True, text=True, timeout=300
    )
    for name, text in (("stdout", proc.stdout), ("stderr", proc.stderr)):
        for line in text.splitlines():
            print(f"  {name}| {line}")
    print(f"  exit {proc.returncode}", flush=True)
    return proc


def stub_dirs() -> list[Path]:
    roots = [os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"), "/usr/local/cuda"]
    return [
        Path(root) / sub
        for root in dict.fromkeys(r for r in roots if r)
        for sub in ("lib64/stubs", "targets/x86_64-linux/lib/stubs")
    ]


def linked_libs(binaries: list[Path], env: dict[str, str]) -> dict[str, dict[str, str | None]]:
    """binary name -> {SONAME it loads: resolved path, or None if not found}."""
    linked: dict[str, dict[str, str | None]] = {}
    for binary in binaries:
        proc = subprocess.run(["ldd", str(binary)], env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            fail(f"ldd {binary} failed: {proc.stderr.strip()}")
        libs: dict[str, str | None] = {}
        for line in proc.stdout.splitlines():
            name, arrow, target = line.strip().partition(" => ")
            if not arrow:
                continue
            libs[name] = None if target.startswith("not found") else target.split(" (")[0]
        linked[binary.name] = libs
    return linked


def unresolved(linked: dict[str, dict[str, str | None]]) -> dict[str, list[str]]:
    """SONAME -> the binaries that would fail to load for lack of it."""
    missing: dict[str, list[str]] = {}
    for binary, libs in linked.items():
        for name, path in libs.items():
            if path is None:
                missing.setdefault(name, []).append(binary)
    return missing


def with_driver_libs(
    env: dict[str, str], scratch: Path, binaries: list[Path]
) -> tuple[dict[str, str], str]:
    """env, plus CUDA toolkit stubs for driver libraries this host lacks.

    Any other library the binaries need and cannot find is a packaging bug:
    auditwheel should have bundled it in gpufl.libs/.
    """
    linked = linked_libs(binaries, env)
    missing = unresolved(linked)
    others = {lib: users for lib, users in missing.items() if lib not in DRIVER_LIBS}
    if others:
        fail(
            "libraries neither bundled nor on this system: "
            + "; ".join(f"{lib} (needed by {', '.join(u)})" for lib, u in sorted(others.items()))
        )
    if not missing:
        driver = sorted({p for libs in linked.values() for n, p in libs.items() if n in DRIVER_LIBS and p})
        return env, (
            f"the NVIDIA driver's ({', '.join(driver)})"
            if driver
            else "none (these binaries link no NVIDIA driver library)"
        )

    link_dir = scratch / "driver-stubs"
    link_dir.mkdir()
    for soname in sorted(missing):
        stub = next(
            (d / DRIVER_LIBS[soname] for d in stub_dirs() if (d / DRIVER_LIBS[soname]).is_file()),
            None,
        )
        if stub is None:
            fail(
                f"{soname} is not installed (no NVIDIA driver), and no CUDA toolkit "
                f"stub for it was found in: {', '.join(map(str, stub_dirs()))}"
            )
        (link_dir / soname).symlink_to(stub)
    env = dict(env)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        p for p in (str(link_dir), env.get("LD_LIBRARY_PATH", "")) if p
    )
    still = unresolved(linked_libs(binaries, env))
    if still:
        fail(f"still unresolved with the stubs in place: {', '.join(sorted(still))}")
    return env, f"CUDA toolkit stubs for {', '.join(sorted(missing))} (no NVIDIA driver here)"


def main() -> None:
    if not sys.platform.startswith("linux"):
        fail("only Linux wheels install the gpufl command")

    version = importlib.metadata.version(DIST)
    dist = importlib.metadata.distribution(DIST)
    # Located through the distribution, so gpufl itself is never imported.
    launcher = Path(dist.locate_file("gpufl/bin/gpufl")).resolve()
    inject_lib = Path(dist.locate_file("gpufl/lib/libgpufl_inject.so")).resolve()
    scripts_dir = Path(sysconfig.get_path("scripts"))
    entry = scripts_dir / "gpufl"
    print(f"gpufl {version} from {dist.locate_file('')}")
    print(f"python {sys.version.split()[0]} at {sys.executable}")

    for path, what in ((entry, "PATH entry"), (launcher, "launcher"), (inject_lib, "injection library")):
        if not path.is_file():
            fail(f"the wheel installed no {what} at {path}")

    results = []
    with tempfile.TemporaryDirectory(prefix="gpufl-wheel-check-") as tmp:
        scratch = Path(tmp)
        env = dict(os.environ)
        env["PATH"] = os.pathsep.join(p for p in (str(scripts_dir), env.get("PATH", "")) if p)
        env, driver = with_driver_libs(env, scratch, [entry, launcher, inject_lib])

        # 1. The command users run is native and comes from this environment.
        found = shutil.which("gpufl", path=env["PATH"])
        if found is None or Path(found).resolve() != entry.resolve():
            fail(f"`gpufl` on PATH is {found}, not {entry}")
        with entry.open("rb") as f:
            magic = f.read(4)
        if magic != b"\x7fELF":
            fail(f"{entry} is not a native executable (starts with {magic!r})")
        if not os.access(entry, os.X_OK):
            fail(f"{entry} is not executable")
        results.append(f"`gpufl` on PATH: {entry} (ELF)")

        # 2. --version, for the version pip installed.
        expected = f"gpufl {version} (wire v1)"
        proc = run(["gpufl", "--version"], env, scratch)
        if proc.returncode != 0 or proc.stdout.strip() != expected:
            fail(f"`gpufl --version` should print `{expected}` and exit 0")
        results.append(f"`gpufl --version`: {proc.stdout.strip()}")

        # 3. The upload subcommand is there.
        proc = run(["gpufl", "upload", "--help"], env, scratch)
        if proc.returncode != 0 or "gpufl upload" not in proc.stdout:
            fail("`gpufl upload --help` should print the upload help and exit 0")
        results.append("`gpufl upload --help`: exit 0")

        # 4. trace resolves the injection library inside the package.
        proc = run(
            ["gpufl", "trace", "--verbose", "--name", "wheel-check",
             "--output", str(scratch / "trace"), "--", "true"],
            env,
            scratch,
        )
        if "cannot find" in proc.stderr:
            fail("`gpufl trace` could not find libgpufl_inject.so")
        reported = [
            line[len(INJECT_LIB_LINE):]
            for line in proc.stderr.splitlines()
            if line.startswith(INJECT_LIB_LINE)
        ]
        if not reported:
            fail("`gpufl trace --verbose` did not report the injection library it uses")
        if Path(reported[-1]).resolve() != inject_lib:
            fail(f"`gpufl trace` used {reported[-1]}, not {inject_lib}")
        if "cannot be preloaded" in proc.stderr:
            fail(f"{inject_lib} did not preload into the traced command")
        if proc.returncode != 0:
            fail("`gpufl trace -- true` should exit 0")
        results.append(f"`gpufl trace --verbose -- true`: exit 0, injection library {reported[-1]}")

    print()
    print(f"driver libraries: {driver}")
    for line in results:
        print(f"PASS {line}")


if __name__ == "__main__":
    main()
