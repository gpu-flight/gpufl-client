# IMPORTANT: import torch (and other GPU frameworks) BEFORE importing gpufl.
# Both gpufl and PyTorch bundle a CUPTI version; importing gpufl first loads
# CUDA 13.x CUPTI (2025.4) before PyTorch loads its own CUPTI (e.g. 2025.1),
# causing a CUPTI version conflict that leads to a crash during profiling.
# Correct order: framework first, then profiler.
import os
import time
import torch       # load framework (and its CUPTI) before gpufl
import gpufl       # now gpufl's CUPTI loads into an already-initialized CUDA ctx
import gpufl.torch


def run_stress_test():
    print("--- GpuFlight: Heavy Stress Test (RTX 5060 Optimized) ---")

    if not torch.cuda.is_available():
        print("[ERROR] PyTorch (CUDA) not found. Did you install the cu124 version?")
        return

    gpufl.torch.attach()
    device = torch.device("cuda")
    print(f"Target: {torch.cuda.get_device_name(0)}")

    api_key = os.environ.get("GPUFL_API_KEY", "")
    backend_url = os.environ.get("GPUFL_BACKEND_URL", "http://localhost:8080")
    remote_upload = bool(api_key)
    print(f"Remote upload: {remote_upload}")

    gpufl.init("Heavy_Stress_App",
               log_path="./stress",
               sampling_auto_start=True,
               system_sample_rate_ms=50,
               kernel_sample_rate_ms=50,
               enable_kernel_details=True,
               enable_debug_output=True,
               enable_profiling=True,
               enable_stack_trace=True,
               # opt-in to memory tracking. Default-off in v1
               # because TF eager and similar workloads can produce
               # high record volume; PyTorch with the caching
               # allocator stays comfortably under 1k events per
               # session, so it's safe to flip on for this benchmark.
               enable_memory_tracking=True,
               # opt-in to CUDA graphs tracking. Default-off in v1
               # because CUDA Graphs interact with PC sampling on some
               # Blackwell driver builds; we've tested it here so
               # it's safe to enable for this benchmark.
               enable_cuda_graphs_tracking=True,
               remote_upload=remote_upload,
               api_key=api_key,
               backend_url=backend_url,
               profiling_engine=gpufl.ProfilingEngine.PcSamplingWithSass)

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

        # NOTE: gpufl.torch.attach() above already pushes CUPTI external
        # correlation IDs around every aten dispatch (see
        # python/gpufl/torch/dispatch.py). That means each kernel
        # captured here will surface in the dashboard's Kernels tab
        # with an accent-blue `op #N` chip showing the framework op id —
        # no separate `torch.profiler.profile()` context needed.

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

        # ── F4 demo: capture matmul into a CUDA graph and replay it ───
        # CUDA Graphs amortize host-side launch overhead by capturing a
        # sequence of CUDA calls once and replaying them as a single
        # launch. CUPTI emits one CUPTI_ACTIVITY_KIND_GRAPH_TRACE record
        # per replay — repeated replays of the same captured graph all
        # share the same `graph_id`, so the dashboard can aggregate.
        #
        # We run this AFTER the regular loop so users get to see both:
        # eager-mode kernels (op #1, op #2 chips) AND graph launches
        # (in the InsightsPanel "CUDA graphs" KPI tile).
        try:
            with gpufl.Scope("Graph_Replay_Loop", "graph"):
                print("\nCapturing matmul into a CUDA graph + replaying 20x...")
                # Warm up the stream before capture (required by torch
                # CUDA-graph contract; reduces capture-time errors).
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        _ = torch.matmul(a, b)
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()

                # Capture
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    c_g = torch.matmul(a, b)

                # Replay
                for _ in range(20):
                    g.replay()
                torch.cuda.synchronize()
                print("Graph replays finished.")
        except Exception as e:
            # CUDA Graph capture is sensitive to PC sampling on some
            # Blackwell driver builds — log + continue rather than
            # tearing down the whole benchmark.
            print(f"[WARN] CUDA graph demo skipped: {e}")

    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        print("If this is an Out of Memory error, try reducing N to 12288.")

    finally:
        gpufl.shutdown()
        gpufl.torch.detach()
        print(f"\n[DONE] Logs generated at: {os.path.abspath('./stress.scope.log')}")


if __name__ == "__main__":
    run_stress_test()
