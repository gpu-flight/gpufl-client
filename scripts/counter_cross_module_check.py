# Proves the counter registry is shared across module boundaries.
#
# gpufl is a static library, so the Python extension and gpufl_inject.dll each
# hold their own copy of it. Without the shared runtime they would also hold
# their own counter registries, and a target ticking a counter would be
# invisible to the injected evaluator - the case counters exist for.
#
# Run standalone, or under the launcher:
#
#   set GPUFL_REPO=<repo root>
#   python scripts/counter_cross_module_check.py
#   gpufl trace -o out --passes Trace -- python scripts/counter_cross_module_check.py
#
# Expected: "value_via_abi=42". A private registry in the extension reads 0.
import ctypes, os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(os.environ["GPUFL_REPO"]) / "python"))
import gpufl

tokens = gpufl.counter("xmod_token")
tokens.add(41)
tokens.add(1)

# Read the slot back through the runtime DLL directly - a third binder, using
# only the C ABI. If the extension had its own registry this would read 0.
lib = ctypes.CDLL("gpufl_counter_runtime.dll")
lib.gpufl_get_counter_provider_v1.restype = ctypes.c_void_p
p = lib.gpufl_get_counter_provider_v1()
if not p:
    print("XMOD: no provider"); sys.exit(1)

class Provider(ctypes.Structure):
    _fields_ = [("abi_version", ctypes.c_uint32), ("struct_size", ctypes.c_uint32),
                ("register_counter", ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t)),
                ("add", ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint64)),
                ("load", ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p)),
                ("load_since_baseline", ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p)),
                ("begin_session", ctypes.CFUNCTYPE(None)),
                ("end_session", ctypes.CFUNCTYPE(None)),
                ("session_active", ctypes.CFUNCTYPE(ctypes.c_int))]
prov = ctypes.cast(p, ctypes.POINTER(Provider)).contents
name = b"xmod_token"
h = prov.register_counter(name, len(name))
print("XMOD: abi=%d value_via_abi=%d" % (prov.abi_version, prov.load(h)))
