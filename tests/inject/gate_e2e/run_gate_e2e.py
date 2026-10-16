"""Capture-gate end-to-end tests on CPU.

Runs the target under the stand-in injection library (real gate, Linux glue
and typed CUDA wrappers; fake gpufl::init) with fake libcuda/libcudart, and
optionally under the real `gpufl trace` launcher. Fake functions prove the
ordering and state behaviour, not the absence of real driver locks.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ARGS = None

# Every first-touch runtime wrapper the gate supports, plus waiting copy and
# sync wrappers (launches are covered by the unit tests: their by-value dim3
# ABI is not faked here).
RUNTIME_WRAPPERS = [
    "cudaMalloc", "cudaMallocManaged", "cudaMallocHost", "cudaHostAlloc", "cudaFree",
    "cudaSetDevice", "cudaGetDeviceCount", "cudaGetDeviceProperties_v2", "cudaMemGetInfo",
    "cudaEventCreate", "cudaEventCreateWithFlags", "cudaStreamCreate",
    "cudaStreamCreateWithFlags", "cudaDeviceSynchronize", "cudaStreamSynchronize",
    "cudaMemcpy", "cudaMemcpyAsync",
]
DRIVER_WRAPPERS = ["cuMemAlloc_v2", "cuModuleLoadData", "cuModuleLoadDataEx", "cuCtxSynchronize"]


def read_status(status_dir):
    files = sorted(Path(status_dir).glob("status-*.json"))
    if not files:
        return None
    return json.loads(files[0].read_text())


class Run:
    def __init__(self, rc, log, status, elapsed, stderr=""):
        self.rc = rc
        self.log = log
        self.status = status
        self.elapsed = elapsed
        self.stderr = stderr

    def index(self, event):
        return self.log.index(event)

    def before(self, first, second):
        return self.index(first) < self.index(second)


class GateEndToEnd(unittest.TestCase):
    def setUp(self):
        self.work = Path(tempfile.mkdtemp(prefix="gpufl-gate-e2e-"))

    def tearDown(self):
        shutil.rmtree(self.work, ignore_errors=True)

    def work_reset(self):
        shutil.rmtree(self.work, ignore_errors=True)
        self.work.mkdir()

    def base_env(self, **extra):
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("GPUFL_") and not k.startswith("FAKE_")
               and k not in ("LD_PRELOAD", "CUDA_INJECTION64_PATH")}
        env.update({
            "FAKE_LOG": str(self.work / "events.log"),
            "GPUFL_INJECT_LAUNCH_WAIT_MS": "5000",
        })
        env.update({k: str(v) for k, v in extra.items()})
        return env

    def direct(self, scenario, release_after=None, **extra):
        status_dir = self.work / "status"
        status_dir.mkdir(exist_ok=True)
        env = self.base_env(**extra)
        env.update({
            "LD_PRELOAD": ARGS.inject,
            "CUDA_INJECTION64_PATH": ARGS.inject,
            "GPUFL_INJECT": "1",
            "GPUFL_INJECT_PROTOCOL": "2",
            "GPUFL_INJECT_STATUS_DIR": str(status_dir),
            "GPUFL_INJECT_STATUS_NONCE": "e2e",
        })
        release = self.work / "release"
        if release_after is not None:
            env["FAKE_INIT_RELEASE_FILE"] = str(release)
            threading.Timer(release_after, release.touch).start()
        start = time.monotonic()
        proc = subprocess.run([ARGS.target, scenario], env=env, timeout=60,
                              capture_output=True, text=True)
        elapsed = time.monotonic() - start
        log_path = self.work / "events.log"
        log = log_path.read_text().split() if log_path.exists() else []
        return Run(proc.returncode, log, read_status(status_dir), elapsed, proc.stderr)

    def assert_status(self, run, readiness, owner, reason="", notes=()):
        self.assertIsNotNone(run.status, run.log)
        self.assertEqual(run.status["readiness"], readiness, run.status)
        self.assertEqual(run.status["owner"], owner, run.status)
        self.assertEqual(run.status["reason"], reason, run.status)
        self.assertEqual(sorted(run.status["coverage_notes"]), sorted(notes), run.status)

    def test_first_runtime_call_waits_for_capture(self):
        run = self.direct("runtime")
        self.assertEqual(run.rc, 0, run.stderr)
        self.assert_status(run, "ready", "gate")
        self.assertEqual(run.status["lifecycle"], "finished")
        self.assertTrue(run.before("cudaGetDeviceCount", "cuInit:callback"))
        self.assertTrue(run.before("init:ok", "cudaMalloc"))
        self.assertTrue(run.before("cudaMalloc", "app:after_first_call"))
        self.assertTrue(run.before("shutdown", "destroyed"))

    def test_set_device_first_is_a_supported_route(self):
        run = self.direct("setdevice")
        self.assert_status(run, "ready", "gate")
        self.assertTrue(run.before("init:ok", "cudaSetDevice"))

    def test_unwrapped_first_touch_is_recorded(self):
        run = self.direct("unwrapped")
        self.assert_status(run, "ready", "callback", notes=["unsupported_first_touch"])
        self.assertTrue(run.before("init:ok", "cudaMalloc"))

    def test_driver_api_host_without_handshake_is_unsupported(self):
        run = self.direct("driver")
        self.assert_status(run, "ready", "callback", notes=["unsupported_first_touch"])
        self.assertTrue(run.before("init:ok", "cuMemAlloc_v2"))

    def test_rust_bootstrap_order_uses_one_synchronous_initializer(self):
        run = self.direct("manual")
        self.assertIn("app:handshake_1", run.log)
        self.assert_status(run, "ready", "manual")
        self.assertTrue(run.before("cuInit:callback", "cuCtxSetCurrent"))
        self.assertTrue(run.before("cuCtxSetCurrent", "init:begin"))
        self.assertEqual(run.log.count("init:begin"), 1)

    def test_trigger_failure_after_the_worker_ran_stays_failed(self):
        run = self.direct("runtime", FAKE_CUDART="fail")
        self.assert_status(run, "failed", "gate", reason="runtime_init_failed:35")
        self.assertEqual(run.status["lifecycle"], "finished")

    def test_driver_init_without_callback_fails(self):
        run = self.direct("runtime", FAKE_CUINIT="nocallback")
        self.assert_status(run, "failed", "gate", reason="no_injection_callback")
        self.assertNotIn("init:begin", run.log)

    def test_timeout_lets_the_application_go_on_and_exit_does_not_tear_down(self):
        run = self.direct("runtime", FAKE_INIT="block", GPUFL_INJECT_LAUNCH_WAIT_MS=300)
        self.assertIn("app:after_first_call", run.log)
        self.assertNotIn("init:released", run.log)
        self.assertEqual(run.rc, 86)
        self.assertNotIn("destroyed", run.log)
        self.assertEqual(run.status["reason"], "capture_wait_timeout")
        self.assertIn("exit_while_initializing", run.status["coverage_notes"])

    def test_exit_with_a_wedged_initializer_skips_teardown(self):
        run = self.direct("exit_early", FAKE_INIT="block", GPUFL_INJECT_LAUNCH_WAIT_MS=300)
        self.assertEqual(run.rc, 86)
        self.assertLess(run.elapsed, 10)
        self.assertNotIn("destroyed", run.log)
        # Documented overlap: a handler registered after the guard runs first.
        self.assertIn("app_late_atexit", run.log)
        self.assertIn("exit_while_initializing", run.status["coverage_notes"])

    def test_exit_waits_for_a_late_initializer_before_teardown(self):
        run = self.direct("exit_early", release_after=0.3, FAKE_INIT="block")
        self.assertEqual(run.rc, 0, run.stderr)
        self.assertTrue(run.before("touch:alive", "destroyed"))
        self.assertNotIn("touch:after_destroy", run.log)
        self.assertEqual(run.log.count("destroyed"), 1)

    def test_every_first_touch_runtime_wrapper_waits_for_capture(self):
        for name in RUNTIME_WRAPPERS:
            with self.subTest(name):
                self.work_reset()
                run = self.direct("first:" + name)
                self.assertEqual(run.rc, 0, run.stderr)
                self.assert_status(run, "ready", "gate")
                # The application's own call is the last one with that name
                # (cudaGetDeviceCount is also the trigger).
                last = len(run.log) - 1 - run.log[::-1].index(name)
                self.assertLess(run.index("init:ok"), last, run.log)

    def test_every_driver_wrapper_waits_after_an_unwrapped_cuinit(self):
        for name in DRIVER_WRAPPERS:
            with self.subTest(name):
                self.work_reset()
                run = self.direct("driver:" + name)
                self.assertEqual(run.rc, 0, run.stderr)
                self.assert_status(run, "ready", "callback",
                                   notes=["unsupported_first_touch"])
                self.assertTrue(run.before("init:ok", name), run.log)

    def test_exit_during_the_warmup_stands_the_worker_down(self):
        run = self.direct("exit_early", GPUFL_INJECT_WARMUP_MS=5000)
        self.assertEqual(run.rc, 0, run.stderr)
        self.assertLess(run.elapsed, 3)
        self.assertNotIn("init:begin", run.log)
        self.assertIn("destroyed", run.log)
        self.assertEqual(run.status["reason"], "exit_before_capture")
        self.assertIn("warmup_unqualified", run.status["coverage_notes"])

    def test_a_process_without_cuda_writes_nothing(self):
        run = self.direct("none")
        self.assertEqual(run.rc, 0)
        self.assertIsNone(run.status)

    def test_accidental_preload_stays_inert(self):
        status_dir = self.work / "status"
        env = self.base_env(LD_PRELOAD=ARGS.inject, CUDA_INJECTION64_PATH=ARGS.inject,
                            GPUFL_INJECT_STATUS_DIR=status_dir)
        status_dir.mkdir()
        proc = subprocess.run([ARGS.target, "runtime"], env=env, timeout=60)
        self.assertEqual(proc.returncode, 0)
        self.assertIsNone(read_status(status_dir))

    # -- The real launcher -------------------------------------------------

    def launch(self, scenario, *trace_args, **extra):
        if not ARGS.launcher:
            self.skipTest("launcher not built")
        bin_dir = self.work / "bin"
        bin_dir.mkdir()
        launcher = bin_dir / "gpufl"
        shutil.copy2(ARGS.launcher, launcher)
        shutil.copy2(ARGS.inject, bin_dir / "libgpufl_inject.so")
        env = self.base_env(GPUFL_INJECT_WATCHDOG_GRACE_MS=200, **extra)
        start = time.monotonic()
        proc = subprocess.run(
            [str(launcher), "trace", "-o", str(self.work / "out"), "--passes", "Trace",
             *trace_args, "--", ARGS.target, scenario],
            env=env, timeout=120, capture_output=True, text=True)
        elapsed = time.monotonic() - start
        log_path = self.work / "events.log"
        log = log_path.read_text().split() if log_path.exists() else []
        return Run(proc.returncode, log, None, elapsed, proc.stderr)

    def test_launcher_reports_readiness(self):
        run = self.launch("runtime")
        self.assertIn("capture readiness: ready owner=gate", run.stderr)

    def test_launcher_watchdog_kills_a_wedged_driver_init_on_natural_exit(self):
        run = self.launch("runtime", FAKE_CUINIT="block_first",
                          GPUFL_INJECT_LAUNCH_WAIT_MS=300)
        self.assertLess(run.elapsed, 60)
        self.assertIn("watchdog killed the target", run.stderr)

    def test_launcher_watchdog_also_bounds_a_windowed_run(self):
        run = self.launch("runtime", "--window-timeout", "60s",
                          FAKE_CUINIT="block_first", GPUFL_INJECT_LAUNCH_WAIT_MS=300)
        self.assertLess(run.elapsed, 30)
        self.assertIn("watchdog killed the target", run.stderr)

    def test_launcher_keeps_a_healthy_warmup_longer_than_the_init_allowance(self):
        run = self.launch("sleep:2500", "--warmup", "1s", GPUFL_INJECT_LAUNCH_WAIT_MS=300)
        self.assertEqual(run.rc, 0, run.stderr)
        self.assertNotIn("watchdog killed", run.stderr)
        self.assertTrue(run.before("init:ok", "app:after_sleep"), run.log)
        self.assertIn("coverage=warmup_unqualified", run.stderr)

    def test_launcher_kills_an_initializer_that_wedges_after_the_warmup(self):
        run = self.launch("sleep:20000", "--warmup", "1s", FAKE_INIT="block",
                          GPUFL_INJECT_LAUNCH_WAIT_MS=300)
        self.assertIn("init:begin", run.log)  # the warmup itself was allowed
        self.assertLess(run.elapsed, 10)
        self.assertIn("watchdog killed the target", run.stderr)

    def test_launcher_watchdog_covers_a_blocked_trigger_after_worker_success(self):
        run = self.launch("runtime", FAKE_CUDART="block", GPUFL_INJECT_LAUNCH_WAIT_MS=300)
        self.assertIn("init:ok", run.log)
        self.assertLess(run.elapsed, 60)
        self.assertIn("watchdog killed the target", run.stderr)


def main():
    global ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--inject", required=True)
    parser.add_argument("--fake-dir", required=True)
    parser.add_argument("--launcher")
    ARGS, rest = parser.parse_known_args()
    unittest.main(argv=[sys.argv[0], "-v", *rest])


if __name__ == "__main__":
    main()
