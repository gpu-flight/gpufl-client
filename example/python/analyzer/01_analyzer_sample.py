from gpufl.analyzer import GpuFlightSession
import os

# Create dummy log files with expected names from existing stress logs if they don't exist
# or just use the stress logs by renaming/linking them if the analyzer supported it.
# However, the analyzer expects "gfl_block.log.kernel.0.log" etc.
# Let's try to copy them for the sample to work.

# The GpuFlightSession now supports log_prefix parameter.
# We can use "stress" as prefix since we have stress.kernel.0.log etc.
analyzer = GpuFlightSession("./", log_prefix="stress", max_stack_depth=5)

analyzer.print_summary()

analyzer.inspect_scopes()

analyzer.inspect_hotspots()