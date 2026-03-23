import os
from gpufl.analyzer import GpuFlightSession

analyzer = GpuFlightSession("./", log_prefix="sass_divergence", max_stack_depth=5)

analyzer.print_summary()

analyzer.inspect_scopes()

analyzer.inspect_hotspots()

analyzer.inspect_profile_samples()

analyzer.inspect_perf_metrics()
