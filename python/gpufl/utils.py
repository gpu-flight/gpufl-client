import time
import gpufl as gfl
import sys

try:
    from numba import cuda
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

def _to_dim3_str(val):
    if isinstance(val, int):
        return f"({val},1,1)"
    if isinstance(val, (tuple, list)):
        x = val[0] if len(val) > 0 else 1
        y = val[1] if len(val) > 1 else 1
        z = val[2] if len(val) > 2 else 1
        return f"({x},{y},{z})"
    return "(1,1,1)"
