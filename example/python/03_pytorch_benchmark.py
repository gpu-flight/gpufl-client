import torch
import gpufl
import time
import os

def run_stress_test():
    print("--- GpuFlight: Heavy Stress Test (RTX 5060 Optimized) ---")

    if not torch.cuda.is_available():
        print("[ERROR] PyTorch (CUDA) not found. Did you install the cu124 version?")
        return

    device = torch.device("cuda")
    print(f"Target: {torch.cuda.get_device_name(0)}")

    # 1. Init GpuFlight
    # 15ms is the fastest reliable timer on Windows
    gpufl.init("Heavy_Stress_App",
               log_path="./stress",
               sampling_auto_start=True,
               system_sample_rate_ms=15,
               kernel_sample_rate_ms=15,
               enable_kernel_details=True,
               enable_debug_output=True,
               enable_profiling=True,
               enable_stack_trace=True)

    try:
        # 2. Allocate (Uses approx 3GB VRAM)
        # N = 16384 * 16384 * 4 bytes = 1 GB per matrix
        # A + B + Result = 3 GB Total
        N = 16384
        print(f"Allocating 3GB of Tensors ({N}x{N})...")

        with gpufl.Scope("Allocation_Phase", "setup"):
            a = torch.randn(N, N, device=device)
            b = torch.randn(N, N, device=device)
            torch.cuda.synchronize()

        print("Warmup (1 iteration)...")
        _ = torch.matmul(a, b)
        torch.cuda.synchronize()

        # 3. Heavy Compute Loop
        iterations = 50
        print(f"Starting {iterations} iterations of Matrix Multiplication...")
        print("This should take about 5-10 seconds. Check Task Manager!")

        # One big scope for the whole benchmark
        with gpufl.Scope("Heavy_Compute_Loop", "stress"):
            start_t = time.time()

            for i in range(iterations):
                # Optional: Add sub-scope for granular detail
                # with gpufl.Scope(f"Iter_{i}", "step"):
                c = torch.matmul(a, b)
                torch.cuda.synchronize()

                # Print progress every 10 steps so you know it's alive
                if i % 10 == 0:
                    print(f"  -> Finished iteration {i}/{iterations}")

            end_t = time.time()
            print(f"Loop finished in {end_t - start_t:.2f} seconds.")

    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        print("If this is an Out of Memory error, try reducing N to 12288.")

    finally:
        gpufl.shutdown()
        print(f"\n[DONE] Logs generated at: {os.path.abspath('./stress.scope.log')}")

if __name__ == "__main__":
    run_stress_test()