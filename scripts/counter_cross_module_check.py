# Proves the counter registry is shared across module boundaries.
#
# gpufl is a static library, so the Python extension and the injection library
# each hold their own copy of it. Without the shared runtime they would also
# hold their own counter registries, and a target ticking a counter would be
# invisible to the injected evaluator - the case counters exist for.
#
# Run standalone, or under the launcher:
#
#   set GPUFL_REPO=<repo root>
#   python scripts/counter_cross_module_check.py
#   gpufl trace -o out --passes Trace -- python scripts/counter_cross_module_check.py
#
# Exits non-zero on any failure, so it is usable as a CTest case rather than
# something a human has to read the output of. A check nobody runs, or one that
# prints a wrong answer and still succeeds, is not evidence of anything.
import ctypes
import os
import pathlib
import sys

EXPECTED = 42

repo = os.environ.get("GPUFL_REPO")
if not repo:
    print("XMOD: set GPUFL_REPO to the repository root", file=sys.stderr)
    sys.exit(2)
sys.path.insert(0, str(pathlib.Path(repo) / "python"))
import gpufl  # noqa: E402

tokens = gpufl.counter("xmod_token")
tokens.add(41)
tokens.add(1)


def runtime_names():
    """Platform library names, most specific first."""
    if sys.platform == "win32":
        return ["gpufl_counter_runtime.dll"]
    if sys.platform == "darwin":
        return ["libgpufl_counter_runtime.dylib"]
    return ["libgpufl_counter_runtime.so"]


lib = None
for name in runtime_names():
    try:
        lib = ctypes.CDLL(name)
        break
    except OSError:
        continue
if lib is None:
    # Not a pass. The whole point is that this library is reachable; if it is
    # not, the two modules are already on separate registries.
    print("XMOD: shared runtime not loadable (%s)" % ", ".join(runtime_names()),
          file=sys.stderr)
    sys.exit(1)

lib.gpufl_get_counter_provider_v1.restype = ctypes.c_void_p
p = lib.gpufl_get_counter_provider_v1()
if not p:
    print("XMOD: no provider", file=sys.stderr)
    sys.exit(1)


class Provider(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("register_counter", ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t)),
        ("add", ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint64)),
        ("load", ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p)),
        ("load_since_baseline", ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p)),
        ("begin_session", ctypes.CFUNCTYPE(None)),
        ("end_session", ctypes.CFUNCTYPE(None)),
        ("session_active", ctypes.CFUNCTYPE(ctypes.c_int)),
        ("lookup", ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t)),
    ]


prov = ctypes.cast(p, ctypes.POINTER(Provider)).contents
name = b"xmod_token"

# lookup, not register: if the extension had its own registry, the counter
# would be absent here rather than present with a value of 0. Those are
# different failures and the message should say which one happened.
h = prov.lookup(name, len(name))
if not h:
    print("XMOD: counter absent from the shared runtime - the extension is on "
          "its own registry", file=sys.stderr)
    sys.exit(1)

value = prov.load(h)
print("XMOD: abi=%d value_via_abi=%d" % (prov.abi_version, value))
if value != EXPECTED:
    print("XMOD: expected %d, got %d" % (EXPECTED, value), file=sys.stderr)
    sys.exit(1)
