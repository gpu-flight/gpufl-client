import gpufl as gfl
import os
import time
import tempfile
import shutil
import glob
import json


def test_pipeline():
    print("--- Starting GPUFL Multi-Log Pipeline Verification ---")
    temp_dir = tempfile.mkdtemp()
    log_base_name = "ci_test"
    log_base_path = os.path.join(temp_dir, log_base_name)
    print(f"1. Log path set to: {log_base_path}")

    keep = False
    try:
        print("2. Initializing GPUFL...")
        # Passing 0 for interval
        res = gfl.init("CI_Test_App", log_base_path, 5)
        print(f"   Result: {res}")
        print("3. Running Scope...")
        with gfl.Scope("ci_scope_01", "test_tag"):
            time.sleep(0.1)
            x = 0
            for i in range(1000): x += i

        print("4. Shutting down...")
        gfl.shutdown()

        print("5. Verifying Log Files...")
        files = sorted(os.listdir(temp_dir))
        print(f" files = {files}")
        print(f"   Files found in {temp_dir}:")
        for f in files:
            full = os.path.join(temp_dir, f)
            ftype = "DIR" if os.path.isdir(full) else "FILE"
            fsize = os.path.getsize(full) if os.path.isfile(full) else 0
            print(f"    - {f} [{ftype}, size={fsize}]")

        if not files:
            print("FAILED: No log files were created at all.")
            keep = True
            raise SystemExit(1)

        categories = ["scope", "system"]
        for cat in categories:
            expected_name = f"{log_base_name}.{cat}.0.log"
            full_path = os.path.join(temp_dir, expected_name)

            if not os.path.exists(full_path):
                print(f"FAILED: Expected log file missing: {expected_name}")
                keep = True
                raise SystemExit(1)

            print(f"   [OK] Found {cat} log: {expected_name}")

            with open(full_path, 'r') as f:
                lines = f.readlines()

            has_init = any('"type":"init"' in line for line in lines)
            if not has_init:
                print(f"FAILED: {cat} log missing init event.")
                keep = True
                raise SystemExit(1)

        print("\nSUCCESS: All log files created and content verified.")

    except Exception as e:
        print(f"\nCRITICAL FAILURE: {e}")
        keep = True
        raise

    finally:
        if keep:
            print(f"Keeping temp dir for inspection: {temp_dir}")
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    test_pipeline()