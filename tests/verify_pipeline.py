import gpufl as gfl
import os
import json
import time
import tempfile
import shutil

def test_pipeline():
    print("--- Starting GPUFL Pipeline Verification ---")

    # 1. Setup a temporary directory for logs
    # We use a temp dir to ensure we don't pollute the CI runner
    temp_dir = tempfile.mkdtemp()
    log_base = os.path.join(temp_dir, "ci_test")

    print(f"1. Log path set to: {log_base}")

    try:
        # 2. Initialize GPUFL
        # We pass the base path. Expectation: ci_test.scope.log, ci_test.kernel.log, etc.
        print("2. Initializing GPUFL...")
        gfl.init("CI_Test_App", log_base, 0) # 0 interval = no background sampler (simpler for CI)

        # 3. Trigger a Scope (This writes to .scope.log)
        print("3. Running Scope...")
        with gfl.Scope("ci_scope_01", "test_tag"):
            # Simulate 'work' (just time passing)
            time.sleep(0.1)
            x = 0
            for i in range(1000): x += i

        # 4. Shutdown (Flushes logs)
        print("4. Shutting down...")
        gfl.shutdown()

        # 5. Verify Files Exist
        print("5. Verifying Log Files...")

        # The library may name logs as:
        #   ci_test.scope.log (preferred)
        # or other patterns depending on platform/build.
        candidates = [
            f"{log_base}.scope.log",
            f"{log_base}.0.log",  # legacy/alternate naming
        ]

        scope_path = next((p for p in candidates if os.path.exists(p)), None)

        if not scope_path:
            print("FAILED: Missing scope log file. Tried:")
            for p in candidates:
                print("  -", p)

            # Help debugging: list what was actually produced
            produced = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir)]
            print("Files in temp_dir:")
            for f in produced:
                print("  -", f)

            exit(1)

        print(f"Found scope log: {scope_path}")

    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    test_pipeline()