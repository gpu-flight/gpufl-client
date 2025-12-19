import gpufl as gfl
import time

# 1. Initialize the library
# Arguments: (AppName, LogFilePath, SampleIntervalMs)
# Interval=0 means "Only log start/end", no background sampling.
res = gfl.init("PythonDemo", "gpufl_basic", 5)

print("Starting Trace...")

# 2. Define a Scope
# This will write a "scope_begin" event immediately.
with gfl.Scope("Initialization"):
    print("  inside scope 'Initialization'")
    time.sleep(0.5)

# 3. Define another Scope with a Tag
# Tags are useful for filtering (e.g., "loading", "compute")
with gfl.Scope("DataLoading", "io-bound"):
    print("  inside scope 'DataLoading'")
    time.sleep(0.2)

# 4. Cleanup (Optional, but good practice)
gfl.shutdown()
print("Trace finished. Check 'gpufl_basic'")