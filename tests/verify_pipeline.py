import gpumon
import os
import json
import time
import tempfile
import shutil

def test_pipeline():
    print("--- Starting GPUMON Pipeline Verification ---")

    # 1. Setup a temporary directory for logs
    # We use a temp dir to ensure we don't pollute the CI runner
    temp_dir = tempfile.mkdtemp()
    log_base = os.path.join(temp_dir, "ci_test")

    print(f"1. Log path set to: {log_base}")

    try:
        # 2. Initialize GPUMON
        # We pass the base path. Expectation: ci_test.scope.log, ci_test.kernel.log, etc.
        print("2. Initializing GPUMON...")
        gpumon.init("CI_Test_App", log_base, 0) # 0 interval = no background sampler (simpler for CI)

        # 3. Trigger a Scope (This writes to .scope.log)
        print("3. Running Scope...")
        with gpumon.Scope("ci_scope_01", "test_tag"):
            # Simulate 'work' (just time passing)
            time.sleep(0.1)
            x = 0
            for i in range(1000): x += i

        # 4. Shutdown (Flushes logs)
        print("4. Shutting down...")
        gpumon.shutdown()

        # 5. Verify Files Exist
        print("5. Verifying Log Files...")
        expected_files = {
            "scope": f"{log_base}.scope.log",
            "kernel": f"{log_base}.kernel.log",
            "system": f"{log_base}.system.log"
        }

        for cat, path in expected_files.items():
            if not os.path.exists(path):
                print(f"FAILED: Missing {cat} log file at {path}")
                exit(1)
            else:
                print(f"Found {cat} log: {path}")

        # 6. Verify Content (JSON Parsing)
        print("6. Verifying JSON Content...")

        # Check Scope Log
        with open(expected_files["scope"], 'r') as f:
            lines = f.readlines()
            # We expect at least: Init, Scope Begin, Scope End, Shutdown
            if len(lines) < 4:
                print(f"FAILED: Scope log has insufficient lines. Found {len(lines)}")
                exit(1)

            # Parse line by line
            for line in lines:
                try:
                    data = json.loads(line)
                    print(f"   - Validated JSON event: {data.get('type', 'unknown')}")

                    # specific check for the scope we ran
                    if data.get('name') == "ci_scope_01":
                        if data.get('type') == 'scope_end':
                            duration = data.get('duration_ns', 0)
                            print(f"     -> Captured Duration: {duration} ns")
                            if duration <= 0:
                                print("FAILED: Duration is invalid")
                                exit(1)
                except json.JSONDecodeError:
                    print(f"FAILED: Invalid JSON line: {line}")
                    exit(1)

        print("\nSUCCESS: GPUMON Python Bindings are working correctly.")

    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    test_pipeline()