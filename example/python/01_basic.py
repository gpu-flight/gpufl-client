import gpumon
import time

# 1. Initialize the library
# Arguments: (AppName, LogFilePath, SampleIntervalMs)
# Interval=0 means "Only log start/end", no background sampling.
gpumon.init("PythonDemo", "gpumon_basic.log", 5)

print("Starting Trace...")

# 2. Define a Scope
# This will write a "scope_begin" event immediately.
with gpumon.Scope("Initialization"):
    print("  inside scope 'Initialization'")
    time.sleep(0.5)

# 3. Define another Scope with a Tag
# Tags are useful for filtering (e.g., "loading", "compute")
with gpumon.Scope("DataLoading", "io-bound"):
    print("  inside scope 'DataLoading'")
    time.sleep(0.2)

# 4. Cleanup (Optional, but good practice)
gpumon.shutdown()
print("Trace finished. Check 'gpumon_basic.log'")