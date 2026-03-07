from gpufl.analyzer import GpuFlightSession
import os

analyzer = GpuFlightSession("./", log_prefix="gfl_block.log", max_stack_depth=5)

analyzer.print_summary()

analyzer.inspect_scopes()

analyzer.inspect_hotspots()

analyzer.inspect_profile_samples()

analyzer.inspect_perf_metrics()
