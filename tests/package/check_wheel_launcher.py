"""Check the gpufl command an installed Linux wheel provides, and its OpenSSL.

Run it with the Python of the environment the wheel was installed into. CI runs
it as cibuildwheel's Linux test command (release.yml and build.yml):
cibuildwheel installs each repaired wheel into a fresh virtual environment and
runs this script with that environment's Python. build.yml also runs it on an
Ubuntu runner (linux-wheels-ubuntu-host). By hand:

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
5. The wheel keeps OpenSSL to itself. No libssl or libcrypto is bundled; no
   ELF in the wheel NEEDs one, imports or exports an OpenSSL symbol, or carries
   an OPENSSL_* symbol version; and the launcher, gpufl-monitor, the injection
   library and the Python extension each contain their private OpenSSL.
6. A program traced by `gpufl trace` keeps its own OpenSSL. A Python started by
   a traced command dlopens libssl.so.3 under LD_DEBUG=bindings: none of its
   OpenSSL symbols may bind into the wheel, and nothing of the wheel's named
   libssl or libcrypto may be mapped. Then Python's own ssl module must see the
   same CA paths and certificates traced and untraced.
7. gpufl's own HTTPS works, against a TLS server this script runs on
   127.0.0.1: gpufl.upload_logs untraced and traced, and the version probe that
   gpufl.init starts inside a traced program, each with the CA given in
   SSL_CERT_FILE, with a wrong CA (which must fail verification) and with the
   CA added to the system trust store and no SSL_CERT_* variables (which
   exercises gpufl's CA bundle discovery). A traced Python's own urllib must
   reach the server too.

Before that, `ldd` must resolve every library the wheel's ELFs load. The
release build links libcuda.so.1 and libnvidia-ml.so.1, which come with the
NVIDIA driver; where the driver is missing, as on CI runners, the CUDA
toolkit's link stubs stand in for them, or, with GPUFL_WHEEL_CHECK_STANDIN_DRIVER=1,
small libraries this script compiles whose functions all return an error. No
check here needs a GPU. Any other missing library fails the check, because
auditwheel should have bundled it.

Environment:
  GPUFL_WHEEL_CHECK_STRICT=1   a check that cannot run fails instead of being
                               skipped. Implied by CIBUILDWHEEL=1.
  GPUFL_WHEEL_CHECK_SUDO=1     change the system trust store with `sudo -n`
                               when not root.
  GPUFL_WHEEL_CHECK_STANDIN_DRIVER=1
                               compile stand-in driver libraries when the CUDA
                               toolkit's stubs are missing (needs `cc`).
  GPUFL_WHEEL_CHECK_REQUIRE_OPENSSL3_PYTHON=1
                               check 6 must include a Python that loads
                               libssl.so.3 from a module, as the system Python
                               on Ubuntu does. That is the kind of program the
                               shared OpenSSL of earlier wheels took over.
"""

from __future__ import annotations

import contextlib
import http.server
import importlib.metadata
import json
import os
import re
import shutil
import ssl
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
import uuid
from pathlib import Path

DIST = "gpufl"
# Driver library SONAME -> the CUDA toolkit stub that provides it.
DRIVER_LIBS = {
    "libcuda.so.1": "libcuda.so",
    "libnvidia-ml.so.1": "libnvidia-ml.so",
}
INJECT_LIB_LINE = "[gpufl] inject lib: "

# Symbol names OpenSSL exports. Matching a wheel ELF's dynamic symbols or a
# traced program's bindings against it finds OpenSSL wherever it leaks.
OSSL_SYMBOL = re.compile(
    r"^(SSL|SSL_CTX|TLS|DTLS|OPENSSL|OSSL|ossl|X509|EVP|BIO|CRYPTO|ERR|PEM|ASN1"
    r"|BN|RSA|EC|HMAC|RAND|OBJ|d2i|i2d)_"
)
OSSL_FILE = re.compile(r"^lib(ssl|crypto)[-.]")
OSSL_VERSION_TEXT = re.compile(rb"OpenSSL 3\.\d+\.\d+")
LD_BINDING = re.compile(
    r"binding file (\S+) \[\d+\] to (\S+) \[\d+\]: normal symbol `([^']+)'(?: \[([^\]]*)\])?"
)
DYNSYM = re.compile(r"^\s*\d+: [0-9a-f]+\s+\S+\s+\S+\s+\S+\s+\S+\s+(\S+)\s+(\S+)")

STRICT = os.environ.get("GPUFL_WHEEL_CHECK_STRICT") == "1" or os.environ.get("CIBUILDWHEEL") == "1"
USE_SUDO = os.environ.get("GPUFL_WHEEL_CHECK_SUDO") == "1"
STANDIN_DRIVER = os.environ.get("GPUFL_WHEEL_CHECK_STANDIN_DRIVER") == "1"
REQUIRE_OPENSSL3_PYTHON = os.environ.get("GPUFL_WHEEL_CHECK_REQUIRE_OPENSSL3_PYTHON") == "1"

RESULTS: list[str] = []


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr, flush=True)
    sys.exit(1)


def passed(message: str) -> None:
    print(f"PASS {message}", flush=True)
    RESULTS.append(f"PASS {message}")


def skip(message: str) -> None:
    if STRICT:
        fail(f"{message} (GPUFL_WHEEL_CHECK_STRICT: a check that cannot run fails)")
    print(f"SKIP {message}", flush=True)
    RESULTS.append(f"SKIP {message}")


def run(
    cmd: list[str], env: dict[str, str], cwd: Path, quiet: bool = False
) -> subprocess.CompletedProcess:
    print("$ " + " ".join(cmd if len(" ".join(cmd)) < 400 else cmd[:3] + ["..."]), flush=True)
    proc = subprocess.run(
        cmd, env=env, cwd=cwd, capture_output=True, text=True, timeout=300
    )
    for name, text in (("stdout", proc.stdout), ("stderr", proc.stderr)):
        lines = text.splitlines()
        if quiet and len(lines) > 20:
            lines = lines[-20:]
        for line in lines:
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


def readelf(path: Path) -> str:
    tool = shutil.which("readelf")
    if tool is None:
        fail("readelf is not installed (binutils)")
    proc = subprocess.run(
        [tool, "-W", "-d", "--dyn-syms", "-V", str(path)], capture_output=True, text=True
    )
    if proc.returncode != 0:
        fail(f"readelf {path} failed: {proc.stderr.strip()}")
    return proc.stdout


def needed(elf_text: str) -> list[str]:
    return re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", elf_text)


def dynsyms(elf_text: str) -> list[tuple[str, str]]:
    """(section index, name without @version) of each dynamic symbol."""
    out = []
    for line in elf_text.splitlines():
        m = DYNSYM.match(line)
        if m:
            out.append((m.group(1), m.group(2).split("@")[0]))
    return out


def linked_libs(binaries: list[Path], env: dict[str, str]) -> dict[str, dict[str, str | None]]:
    """binary name -> {SONAME it loads: resolved path, or None if not found}."""
    linked: dict[str, dict[str, str | None]] = {}
    for binary in binaries:
        proc = subprocess.run(["ldd", str(binary)], env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            # ldd runs the dynamic loader on the binary; a binary whose
            # segments the loader cannot set up crashes it here too.
            fail(
                f"ldd {binary} failed (exit {proc.returncode}): {proc.stderr.strip()} "
                f"{proc.stdout.strip()}".rstrip()
            )
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


def standin_driver(soname: str, binaries: list[Path], out_dir: Path) -> Path:
    """Compile a library named soname that defines every function the
    binaries import from it, each returning an error code."""
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        fail(f"no C compiler to build a stand-in {soname}")
    prefix = "cu" if soname.startswith("libcuda") else "nvml"
    names = sorted(
        {
            name
            for binary in binaries
            for ndx, name in dynsyms(readelf(binary))
            if ndx == "UND" and re.match(rf"^{prefix}[A-Z]", name)
        }
    )
    lines = ['static const char kStandIn[] = "gpufl wheel check stand-in driver";']
    for name in names:
        if name.endswith("ErrorString") and prefix == "nvml":
            lines.append(f"const char *{name}(int r) {{ (void)r; return kStandIn; }}")
        else:
            # CUDA_ERROR_UNKNOWN and NVML_ERROR_UNKNOWN are both 999.
            lines.append(f"int {name}(void) {{ return 999; }}")
    src = out_dir / f"{soname}.c"
    src.write_text("\n".join(lines) + "\n")
    lib = out_dir / soname
    proc = subprocess.run(
        [cc, "-shared", "-fPIC", "-o", str(lib), f"-Wl,-soname,{soname}", str(src)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        fail(f"building a stand-in {soname} failed: {proc.stderr.strip()}")
    return lib


def with_driver_libs(
    env: dict[str, str], scratch: Path, binaries: list[Path]
) -> tuple[dict[str, str], str]:
    """env, plus stand-ins for driver libraries this host lacks.

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
    used = []
    for soname in sorted(missing):
        stub = next(
            (d / DRIVER_LIBS[soname] for d in stub_dirs() if (d / DRIVER_LIBS[soname]).is_file()),
            None,
        )
        if stub is not None:
            (link_dir / soname).symlink_to(stub)
            used.append(f"the CUDA toolkit stub for {soname}")
        elif STANDIN_DRIVER:
            standin_driver(soname, binaries, link_dir)
            used.append(f"a compiled stand-in {soname}")
        else:
            fail(
                f"{soname} is not installed (no NVIDIA driver), and no CUDA toolkit "
                f"stub for it was found in: {', '.join(map(str, stub_dirs()))} "
                "(GPUFL_WHEEL_CHECK_STANDIN_DRIVER=1 compiles one)"
            )
    env = dict(env)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        p for p in (str(link_dir), env.get("LD_LIBRARY_PATH", "")) if p
    )
    still = unresolved(linked_libs(binaries, env))
    if still:
        fail(f"still unresolved with the stand-ins in place: {', '.join(sorted(still))}")
    return env, f"{', '.join(used)} (no NVIDIA driver here)"


def wheel_elfs(dist: importlib.metadata.Distribution) -> list[Path]:
    """Every ELF the distribution installed, from its RECORD, auditwheel's
    gpufl.libs included."""
    elfs = set()
    for entry in dist.files or []:
        path = Path(dist.locate_file(entry))
        try:
            with path.open("rb") as f:
                if f.read(4) == b"\x7fELF":
                    elfs.add(path.resolve())
        except OSError:
            continue
    return sorted(elfs)


def inside(path: str | Path, roots: list[Path]) -> bool:
    try:
        resolved = Path(path).resolve()
    except OSError:
        return False
    return any(resolved == root or root in resolved.parents for root in roots)


def without_ssl_env(env: dict[str, str]) -> dict[str, str]:
    # manylinux images set SSL_CERT_FILE for every process.
    env = dict(env)
    env.pop("SSL_CERT_FILE", None)
    env.pop("SSL_CERT_DIR", None)
    return env


# --------------------------------------------------------------------------
# Check 5: the ELF invariants.
# --------------------------------------------------------------------------


def check_elf_invariants(site: Path, elfs: list[Path]) -> None:
    libs_dir = site / "gpufl.libs"
    bundled = sorted(p.name for p in libs_dir.iterdir()) if libs_dir.is_dir() else []
    leaked = [n for n in bundled if OSSL_FILE.match(n)]
    if leaked:
        fail(f"gpufl.libs bundles OpenSSL: {', '.join(leaked)}")

    problems = []
    for elf in elfs:
        text = readelf(elf)
        for lib in needed(text):
            if OSSL_FILE.match(lib):
                problems.append(f"{elf.name} NEEDs {lib}")
        names = sorted({name for _, name in dynsyms(text) if OSSL_SYMBOL.match(name)})
        if names:
            problems.append(
                f"{elf.name} has {len(names)} OpenSSL dynamic symbols, e.g. {', '.join(names[:5])}"
            )
        versions = sorted(set(re.findall(r"OPENSSL_\d[\w.]*", text)))
        if versions:
            problems.append(f"{elf.name} references symbol versions {', '.join(versions)}")
    if problems:
        fail("the wheel shares OpenSSL with the programs it loads into:\n  " + "\n  ".join(problems))

    private = [
        site / "gpufl/bin/gpufl",
        site / "gpufl/bin/gpufl-monitor",
        site / "gpufl/lib/libgpufl_inject.so",
        *sorted((site / "gpufl").glob("_gpufl_client*.so")),
    ]
    for path in private:
        if not path.is_file():
            fail(f"the wheel installed no {path}")
        found = OSSL_VERSION_TEXT.search(path.read_bytes())
        if found is None:
            fail(f"{path} contains no private OpenSSL: its HTTPS support is missing")
    passed(
        f"OpenSSL kept private: gpufl.libs holds {', '.join(bundled) or 'nothing'}; "
        f"{len(elfs)} ELFs need, import and export no OpenSSL; "
        f"{len(private)} binaries contain {found.group().decode()}"
    )


# --------------------------------------------------------------------------
# Check 6: what a traced program's OpenSSL resolves to.
# --------------------------------------------------------------------------

# Run under a traced command. It must keep dlopen'ing libssl.so.3: a program
# that NEEDs libssl itself puts its own copy ahead of the injection library's
# dependencies, so it passes even against wheels that share their OpenSSL.
SSL_SCOPE_CONSUMER = r"""
import ctypes, os, sys
roots = [os.path.realpath(r) for r in sys.argv[1:]]
ctypes.CDLL("libssl.so.3")
paths = set()
with open("/proc/self/maps") as f:
    for line in f:
        parts = line.split(None, 5)
        if len(parts) == 6 and parts[5].startswith("/"):
            paths.add(parts[5].strip())
inject = any(os.path.basename(p).startswith("libgpufl_inject") for p in paths)
ours = sorted(
    p for p in paths
    if os.path.basename(p).startswith(("libssl", "libcrypto"))
    and any(os.path.realpath(p).startswith(r + os.sep) for r in roots)
)
print("SSL-SCOPE inject=%d gpufl_openssl=%s" % (inject, ",".join(ours) or "-"))
"""

PYTHON_SSL_PROBE = r"""
import json, os, ssl
paths = ssl.get_default_verify_paths()
count = ssl.create_default_context().cert_store_stats()["x509_ca"]
libssl = set()
with open("/proc/self/maps") as f:
    for line in f:
        parts = line.split(None, 5)
        if len(parts) == 6 and os.path.basename(parts[5].strip()).startswith("libssl"):
            libssl.add(parts[5].strip())
print("PY-SSL " + json.dumps({
    "openssl": ssl.OPENSSL_VERSION,
    "cafile": paths.openssl_cafile,
    "capath": paths.openssl_capath,
    "x509_ca": count,
    "libssl": sorted(libssl),
}, sort_keys=True))
"""


def check_traced_openssl(
    env: dict[str, str], scratch: Path, roots: list[Path]
) -> None:
    probe = subprocess.run(
        [sys.executable, "-I", "-c", "import ctypes; ctypes.CDLL('libssl.so.3')"],
        env=env, capture_output=True, text=True,
    )
    if probe.returncode != 0:
        skip("check 6: this host has no libssl.so.3 for a traced program to load")
        return

    ld_dir = scratch / "ld-bindings"
    ld_dir.mkdir()
    proc = run(
        ["gpufl", "trace", "--name", "ssl-scope", "--output", str(scratch / "trace-ssl-scope"),
         "--", "/usr/bin/env", "LD_BIND_NOW=1", "LD_DEBUG=bindings",
         f"LD_DEBUG_OUTPUT={ld_dir / 'b'}", sys.executable, "-I", "-c", SSL_SCOPE_CONSUMER,
         *map(str, roots)],
        env, scratch, quiet=True,
    )
    if proc.returncode != 0:
        fail("the traced OpenSSL consumer should exit 0")
    scope = [line for line in proc.stdout.splitlines() if line.startswith("SSL-SCOPE ")]
    if scope != ["SSL-SCOPE inject=1 gpufl_openssl=-"]:
        fail(
            "a traced program should have the injection library mapped and no OpenSSL "
            f"of the wheel's; it reported {scope}"
        )

    outside, into = 0, []
    for log in ld_dir.glob("b.*"):
        for m in LD_BINDING.finditer(log.read_text(errors="replace")):
            src, dst, name, version = m.groups()
            if not OSSL_SYMBOL.match(name) or inside(src, roots):
                continue
            if inside(dst, roots):
                into.append(f"{Path(src).name} -> {Path(dst).name}: {name}")
            elif (version or "").startswith("OPENSSL_3"):
                outside += 1
    if into:
        fail(
            f"{len(into)} OpenSSL symbols of a traced program bound into the wheel, e.g. "
            + "; ".join(into[:5])
        )
    if outside == 0:
        fail(
            "the traced program's libssl.so.3 bound no OPENSSL_3 symbol at all "
            "(LD_DEBUG=bindings produced nothing to check)"
        )
    passed(
        f"traced program keeps its OpenSSL: {outside} OPENSSL_3 bindings to the system's, "
        "0 into the wheel, none of the wheel's libssl or libcrypto mapped"
    )

    pythons = [Path(sys.executable)]
    system_python = Path("/usr/bin/python3")
    if system_python.is_file() and system_python.resolve() != Path(sys.executable).resolve():
        pythons.append(system_python)
    level2 = []
    for python in pythons:
        seen = {}
        for label, cmd in (
            ("untraced", [str(python), "-I", "-c", PYTHON_SSL_PROBE]),
            ("traced", ["gpufl", "trace", "--name", "py-ssl", "--output",
                        str(scratch / f"trace-py-ssl-{len(level2)}-{python.name}-{uuid.uuid4().hex[:6]}"),
                        "--", str(python), "-I", "-c", PYTHON_SSL_PROBE]),
        ):
            proc = run(cmd, without_ssl_env(env), scratch)
            lines = [l for l in proc.stdout.splitlines() if l.startswith("PY-SSL ")]
            if proc.returncode != 0 or len(lines) != 1:
                fail(f"{python} ({label}) should print its ssl settings and exit 0")
            seen[label] = json.loads(lines[0][len("PY-SSL "):])
        if seen["untraced"]["x509_ca"] <= 0:
            fail(f"{python} loads no CA certificates even untraced: {seen['untraced']}")
        if seen["traced"] != seen["untraced"]:
            fail(
                f"`gpufl trace` changed {python}'s OpenSSL: untraced {seen['untraced']}, "
                f"traced {seen['traced']}"
            )
        exe_needs = needed(readelf(python.resolve()))
        if any(Path(p).name.startswith("libssl.so.3") for p in seen["untraced"]["libssl"]) and not any(
            n.startswith("libssl") for n in exe_needs
        ):
            level2.append(str(python))
        passed(
            f"{python} under trace: {seen['traced']['openssl']}, cafile "
            f"{seen['traced']['cafile']}, {seen['traced']['x509_ca']} CA certificates, as untraced"
        )
    if REQUIRE_OPENSSL3_PYTHON and not level2:
        fail(
            "GPUFL_WHEEL_CHECK_REQUIRE_OPENSSL3_PYTHON: none of "
            f"{', '.join(map(str, pythons))} loads libssl.so.3 from a module"
        )


# --------------------------------------------------------------------------
# Check 7: gpufl's own HTTPS against a local TLS server.
# --------------------------------------------------------------------------

CA_CONFIG = """[req]
distinguished_name = dn
prompt = no
x509_extensions = v3_ca
[dn]
CN = {cn}
[v3_ca]
basicConstraints = critical,CA:TRUE
keyUsage = critical,keyCertSign,cRLSign
subjectKeyIdentifier = hash
"""

SERVER_CONFIG = """[req]
distinguished_name = dn
prompt = no
[dn]
CN = 127.0.0.1
[v3_server]
basicConstraints = critical,CA:FALSE
keyUsage = critical,digitalSignature,keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = IP:127.0.0.1
subjectKeyIdentifier = hash
authorityKeyIdentifier = keyid
"""


def make_certificates(out: Path) -> bool:
    """ca.pem, a server certificate it signs for 127.0.0.1, and an unrelated
    other.pem. Explicit configs keep the openssl CLI's defaults (1.1.1 in the
    manylinux image, 3.x elsewhere) out of it."""
    tool = shutil.which("openssl")
    if tool is None:
        return False
    newkey = ["-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1", "-nodes"]

    def sh(*args: str) -> None:
        proc = subprocess.run([tool, *args], cwd=out, capture_output=True, text=True)
        if proc.returncode != 0:
            fail(f"openssl {' '.join(args[:2])} failed: {proc.stderr.strip()}")

    for name, cn in (("ca", "gpufl wheel check CA"), ("other", "gpufl wheel check other CA")):
        (out / f"{name}.cnf").write_text(CA_CONFIG.format(cn=cn))
        sh("req", "-x509", *newkey, "-days", "2", "-config", f"{name}.cnf",
           "-keyout", f"{name}.key", "-out", f"{name}.pem")
    (out / "srv.cnf").write_text(SERVER_CONFIG)
    sh("req", "-new", *newkey, "-config", "srv.cnf", "-keyout", "srv.key", "-out", "srv.csr")
    sh("x509", "-req", "-in", "srv.csr", "-CA", "ca.pem", "-CAkey", "ca.key", "-set_serial", "2",
       "-days", "2", "-extfile", "srv.cnf", "-extensions", "v3_server", "-out", "srv.pem")
    return True


class TlsServer:
    """An HTTPS server on 127.0.0.1 that answers gpufl's two requests and
    counts what reached it. Marker files let a traced program wait for it.

    A client that rejects the certificate leaves a dropped connection: one
    that sent no request. Python's ssl fails the handshake itself;
    cpp-httplib completes it, checks the certificate afterwards and hangs up.
    """

    def __init__(self, certs: Path, markers: Path) -> None:
        self.context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self.context.load_cert_chain(certs / "srv.pem", certs / "srv.key")
        self.markers = markers
        self.lock = threading.Lock()
        self.requests: list[tuple[str, str, str, str]] = []
        self.dropped = 0
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            answered = False

            def handle(self) -> None:
                try:
                    super().handle()
                except OSError:
                    pass
                if not self.answered:
                    outer.drop()

            def _reply(self, code: int, body: bytes) -> None:
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _record(self) -> None:
                self.answered = True
                with outer.lock:
                    outer.requests.append((
                        self.command, self.path,
                        self.headers.get("X-GpuFlight-Session-Id", ""),
                        self.headers.get("User-Agent", ""),
                    ))

            def do_POST(self) -> None:  # noqa: N802
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                self._record()
                if self.path == "/api/v1/events/stream":
                    self._reply(200, b'{"accepted":1,"rejected":0,"errors":[]}')
                else:
                    self._reply(404, b"{}")

            def do_GET(self) -> None:  # noqa: N802
                self._record()
                if self.path == "/api/v1/info/version":
                    (outer.markers / "served").touch()
                    self._reply(200, b'{"supported":["1"]}')
                else:
                    self._reply(404, b"{}")

            def log_message(self, *args) -> None:
                pass

        class Server(http.server.ThreadingHTTPServer):
            daemon_threads = True

            def get_request(self):
                sock, addr = self.socket.accept()
                sock.settimeout(15)
                try:
                    return outer.context.wrap_socket(sock, server_side=True), addr
                except OSError:
                    outer.drop()
                    sock.close()
                    raise

            def handle_error(self, request, client_address) -> None:
                pass

        self.httpd = Server(("127.0.0.1", 0), Handler)
        self.url = f"https://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def drop(self) -> None:
        with self.lock:
            self.dropped += 1
        (self.markers / "dropped").touch()

    def reset(self) -> None:
        with self.lock:
            self.requests.clear()
            self.dropped = 0
        for marker in ("served", "dropped"):
            (self.markers / marker).unlink(missing_ok=True)

    def wait_dropped(self, seconds: float = 10) -> None:
        """The client is gone as soon as it rejects the certificate; the
        server notices a moment later."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and not (self.markers / "dropped").exists():
            time.sleep(0.05)

    def matching(self, method: str, path: str) -> list[tuple[str, str, str, str]]:
        with self.lock:
            return [r for r in self.requests if r[:2] == (method, path)]

    def snapshot(self) -> str:
        with self.lock:
            return f"requests {self.requests}, connections dropped without a request {self.dropped}"


UPLOAD = r"""
import sys, gpufl
r = gpufl.upload_logs(log_path=sys.argv[1], backend_url=sys.argv[2], api_key="ci",
                      all_sessions=True, max_retries=0, total_timeout_ms=20000,
                      connect_timeout_ms=5000, read_timeout_ms=5000, report_progress=False)
print("UPLOAD", r.events_uploaded)
for warning in r.warnings:
    print("WARNING", warning)
"""

# gpufl.init starts the backend version probe on a thread when
# GPUFL_BACKEND_URL is set. os._exit, because a traced Python that calls
# gpufl.init aborts in its exit-time destructors (a separate, known issue).
PROBE = r"""
import os, sys, tempfile, time, gpufl
gpufl.init("wheel-check-probe", tempfile.mkdtemp(), backend=getattr(gpufl.BackendKind, "None"))
deadline = time.time() + 15
while time.time() < deadline and not any(os.path.exists(m) for m in sys.argv[1:]):
    time.sleep(0.1)
time.sleep(0.5)
sys.stdout.flush()
os._exit(0)
"""

URLLIB = r"""
import ssl, sys, urllib.request
ctx = ssl.create_default_context(cafile=sys.argv[2])
print("URLLIB", urllib.request.urlopen(sys.argv[1], context=ctx, timeout=10).status)
"""


@contextlib.contextmanager
def ca_in_system_store(ca: Path):
    """Adds ca to the system trust store for the duration; yields False if
    this host's store cannot be changed."""
    if Path("/etc/pki/ca-trust/source/anchors").is_dir() and shutil.which("update-ca-trust"):
        dest = Path("/etc/pki/ca-trust/source/anchors/gpufl-wheel-check.pem")
        refresh = ["update-ca-trust", "extract"]
    elif Path("/usr/local/share/ca-certificates").is_dir() and shutil.which("update-ca-certificates"):
        dest = Path("/usr/local/share/ca-certificates/gpufl-wheel-check.crt")
        refresh = ["update-ca-certificates"]
    else:
        yield False
        return
    if os.geteuid() == 0:
        prefix: list[str] = []
    elif USE_SUDO and shutil.which("sudo"):
        prefix = ["sudo", "-n"]
    else:
        yield False
        return

    def sh(cmd: list[str]) -> None:
        proc = subprocess.run(prefix + cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            fail(f"{' '.join(prefix + cmd)} failed: {proc.stderr.strip()}")

    sh(["install", "-m", "0644", str(ca), str(dest)])
    try:
        sh(refresh)
        yield True
    finally:
        sh(["rm", "-f", str(dest)])
        sh(refresh)


def check_own_https(env: dict[str, str], scratch: Path, version: str) -> None:
    certs = scratch / "certs"
    certs.mkdir()
    if not make_certificates(certs):
        skip("check 7: no openssl command to make test certificates")
        return
    markers = scratch / "markers"
    markers.mkdir()
    server = TlsServer(certs, markers)
    no_dir = scratch / "no-certs"
    no_dir.mkdir()
    py = sys.executable
    runs = 0

    def fixture() -> tuple[Path, str]:
        nonlocal runs
        runs += 1
        session = str(uuid.uuid4())
        logs = scratch / f"logs-{runs}"
        (logs / session).mkdir(parents=True)
        (logs / session / "device.log").write_text(json.dumps({
            "type": "job_start", "session_id": session, "ts_ns": 1, "app": "wheel-check", "pid": 1,
        }) + "\n")
        return logs, session

    def traced(name: str, cmd: list[str]) -> list[str]:
        return ["gpufl", "trace", "--name", name,
                "--output", str(scratch / f"trace-{name}-{runs}"), "--", *cmd]

    def upload(mode: str, mode_env: dict[str, str], trace: bool, trusted: bool) -> None:
        logs, session = fixture()
        server.reset()
        cmd = [py, "-I", "-c", UPLOAD, str(logs), server.url]
        proc = run(traced(f"upload-{mode}", cmd) if trace else cmd, mode_env, scratch)
        label = f"gpufl.upload_logs, {mode} CA, {'traced' if trace else 'untraced'}"
        if proc.returncode != 0:
            fail(f"{label}: should exit 0; {server.snapshot()}")
        lines = proc.stdout.splitlines()
        uploaded = [l.split()[1] for l in lines if l.startswith("UPLOAD ")]
        # httplib::Error::SSLServerVerification
        refused = any(l.startswith("WARNING ") and "httplib::Error=10" in l for l in lines)
        posts = server.matching("POST", "/api/v1/events/stream")
        if trusted:
            if not uploaded or int(uploaded[-1]) < 1 or not posts:
                fail(f"{label}: should upload over HTTPS; printed {uploaded}; {server.snapshot()}")
            if any(r[2] != session for r in posts):
                fail(f"{label}: posted a session other than {session}; {server.snapshot()}")
            passed(f"{label}: {uploaded[-1]} event(s) uploaded over HTTPS")
        else:
            server.wait_dropped()
            if uploaded != ["0"] or posts or not refused or server.dropped < 1:
                fail(
                    f"{label}: certificate verification should refuse the server; "
                    f"printed {lines}; {server.snapshot()}"
                )
            passed(f"{label}: refused the untrusted server (TLS certificate verification failed)")

    def version_probe(mode: str, mode_env: dict[str, str], trusted: bool) -> None:
        server.reset()
        probe_env = dict(mode_env, GPUFL_BACKEND_URL=server.url)
        proc = run(
            traced(f"probe-{mode}", [py, "-I", "-c", PROBE,
                                     str(markers / "served"), str(markers / "dropped")]),
            probe_env, scratch,
        )
        label = f"version probe inside a traced program, {mode} CA"
        if proc.returncode != 0:
            fail(f"{label}: the trace should exit 0; {server.snapshot()}")
        gets = server.matching("GET", "/api/v1/info/version")
        if trusted:
            if not gets or any(r[3] != f"gpufl/{version}" for r in gets):
                fail(f"{label}: should GET /api/v1/info/version as gpufl/{version}; {server.snapshot()}")
            passed(f"{label}: {len(gets)} GET /api/v1/info/version over HTTPS")
        else:
            server.wait_dropped()
            if gets or server.dropped < 1:
                fail(f"{label}: certificate verification should refuse the server; {server.snapshot()}")
            passed(f"{label}: connected and refused the untrusted server")

    given = dict(env, SSL_CERT_FILE=str(certs / "ca.pem"), SSL_CERT_DIR=str(no_dir))
    wrong = dict(env, SSL_CERT_FILE=str(certs / "other.pem"), SSL_CERT_DIR=str(no_dir))

    upload("SSL_CERT_FILE", given, trace=False, trusted=True)
    upload("SSL_CERT_FILE", given, trace=True, trusted=True)
    version_probe("SSL_CERT_FILE", given, trusted=True)
    upload("wrong", wrong, trace=False, trusted=False)
    upload("wrong", wrong, trace=True, trusted=False)
    version_probe("wrong", wrong, trusted=False)

    server.reset()
    proc = run(
        traced("urllib", [py, "-I", "-c", URLLIB, f"{server.url}/api/v1/info/version",
                          str(certs / "ca.pem")]),
        given, scratch,
    )
    if proc.returncode != 0 or "URLLIB 200" not in proc.stdout.splitlines():
        fail(f"a traced Python's own urllib should reach the server; {server.snapshot()}")
    passed("a traced Python's own HTTPS (urllib, its own ssl module): HTTP 200")

    with ca_in_system_store(certs / "ca.pem") as added:
        if not added:
            skip(
                "check 7, system trust store: cannot add a CA here "
                "(needs root, or GPUFL_WHEEL_CHECK_SUDO=1 with passwordless sudo)"
            )
            return
        system = without_ssl_env(env)
        upload("system store", system, trace=False, trusted=True)
        upload("system store", system, trace=True, trusted=True)
        version_probe("system store", system, trusted=True)


def main() -> None:
    if not sys.platform.startswith("linux"):
        fail("only Linux wheels install the gpufl command")

    version = importlib.metadata.version(DIST)
    dist = importlib.metadata.distribution(DIST)
    site = Path(dist.locate_file("")).resolve()
    # Located through the distribution, so gpufl itself is never imported.
    launcher = Path(dist.locate_file("gpufl/bin/gpufl")).resolve()
    inject_lib = Path(dist.locate_file("gpufl/lib/libgpufl_inject.so")).resolve()
    scripts_dir = Path(sysconfig.get_path("scripts"))
    entry = scripts_dir / "gpufl"
    roots = [site / "gpufl", site / "gpufl.libs"]
    print(f"gpufl {version} from {site}")
    print(f"python {sys.version.split()[0]} at {sys.executable}")
    print(f"strict: {STRICT}")

    for path, what in ((entry, "PATH entry"), (launcher, "launcher"), (inject_lib, "injection library")):
        if not path.is_file():
            fail(f"the wheel installed no {what} at {path}")
    elfs = wheel_elfs(dist)
    # The libraries auditwheel grafted into gpufl.libs find each other only
    # once loaded; ldd sees them through the ELFs that load them.
    loaders = [e for e in elfs if not inside(e, [site / "gpufl.libs"])]

    with tempfile.TemporaryDirectory(prefix="gpufl-wheel-check-") as tmp:
        scratch = Path(tmp)
        env = dict(os.environ)
        env["PATH"] = os.pathsep.join(p for p in (str(scripts_dir), env.get("PATH", "")) if p)
        env, driver = with_driver_libs(env, scratch, loaders)
        print(f"driver libraries: {driver}")

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
        passed(f"`gpufl` on PATH: {entry} (ELF)")

        # 2. --version, for the version pip installed.
        expected = f"gpufl {version} (wire v1)"
        proc = run(["gpufl", "--version"], env, scratch)
        if proc.returncode != 0 or proc.stdout.strip() != expected:
            fail(f"`gpufl --version` should print `{expected}` and exit 0")
        passed(f"`gpufl --version`: {proc.stdout.strip()}")

        # 3. The upload subcommand is there.
        proc = run(["gpufl", "upload", "--help"], env, scratch)
        if proc.returncode != 0 or "gpufl upload" not in proc.stdout:
            fail("`gpufl upload --help` should print the upload help and exit 0")
        passed("`gpufl upload --help`: exit 0")

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
        passed(f"`gpufl trace --verbose -- true`: exit 0, injection library {reported[-1]}")

        # 5-7. OpenSSL.
        check_elf_invariants(site, elfs)
        check_traced_openssl(env, scratch, roots)
        check_own_https(env, scratch, version)

    print()
    print(f"driver libraries: {driver}")
    for line in RESULTS:
        print(line)


if __name__ == "__main__":
    main()
