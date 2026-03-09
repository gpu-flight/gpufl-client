import torch
import gpufl
from gpufl import ProfilingEngine

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")

# Initialize GPU Flight
gpufl.init("smoke-test",
           log_path="./smoke_test",
           sampling_auto_start=True,
           enable_kernel_details=True,
           enable_stack_trace=True,
           profiling_engine = ProfilingEngine.RangeProfiler)

# Run a simple operation
with gpufl.Scope("RandomGeneration"):
    a = torch.randn(1024, 1024, device="cuda")
    b = torch.randn(1024, 1024, device="cuda")
with gpufl.Scope("a @ b"):
    c = a @ b
    torch.cuda.synchronize()

gpufl.shutdown()
print("GPU Flight logs written!")