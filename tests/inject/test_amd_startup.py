"""CPU-only ABI/order regression for the production AMD/glibc preload shim."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class AmdStartupTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("GPUFL_TEST_AMD_INJECT"), "built AMD injection library required")
    def test_real_preload_starts_and_shuts_down_without_devices(self):
        library = Path(os.environ["GPUFL_TEST_AMD_INJECT"]).resolve(strict=True)
        with tempfile.TemporaryDirectory() as directory:
            for constructor in ("0", "1"):
                env = {"PATH": os.defpath, "HOME": directory,
                       "LD_PRELOAD": str(library), "LD_LIBRARY_PATH": str(library.parent) + ":/opt/rocm/lib",
                       "GPUFL_INJECT": "1", "GPUFL_INJECT_USE_CONSTRUCTOR": constructor,
                       "GPUFL_PROFILING_ENGINE": "Trace", "GPUFL_LOG_DIR": directory,
                       "ROCR_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""}
                for command, expected in ((["/usr/bin/true"], 0), (["/bin/sh", "-c", "exit 23"], 23)):
                    with self.subTest(constructor=constructor, command=command):
                        result = subprocess.run(command, env=env, capture_output=True, timeout=20)
                        self.assertEqual(result.returncode, expected, result.stderr.decode(errors="replace"))

    def test_dso_state_ready_before_executable_constructors(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            (tmp / "state.cpp").write_text(r'''#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
static std::string state = "constructed";
static int initialized;
namespace gpufl::inject {
void initializeAmdInjection() noexcept {
    if (state != "constructed") _exit(81);
    const char* enabled = std::getenv("GPUFL_INJECT");
    if (enabled && std::strcmp(enabled, "1") == 0) ++initialized;
}
}
extern "C" int injectionReady() { return initialized; }
''')
            (tmp / "target.cpp").write_text(r'''#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <unistd.h>
static void check() {
    auto ready = reinterpret_cast<int (*)()>(dlsym(RTLD_DEFAULT, "injectionReady"));
    const char* enabled = std::getenv("GPUFL_INJECT");
    int expected = enabled && std::strcmp(enabled, "1") == 0 ? 1 : 0;
    if (!ready || ready() != expected) _exit(82);
}
struct BeforeMain { BeforeMain() { check(); } } before_main;
int main(int argc, char** argv) {
    check();
    return argc == 2 && std::strcmp(argv[1], "argument preserved") == 0 ? 23 : 83;
}
''')
            subprocess.run(["c++", "-std=c++17", "-shared", "-fPIC",
                            str(ROOT / "include/gpufl/inject/amd_startup_linux.cpp"),
                            str(tmp / "state.cpp"), "-ldl", "-o", str(tmp / "inject.so")], check=True)
            subprocess.run(["c++", str(tmp / "target.cpp"), "-ldl", "-o", str(tmp / "target")], check=True)
            for sentinel in (None, "0", "1"):
                for constructor in (None, "1"):
                    with self.subTest(sentinel=sentinel, constructor=constructor):
                        env = {"PATH": os.defpath, "LD_PRELOAD": str(tmp / "inject.so")}
                        if sentinel is not None:
                            env["GPUFL_INJECT"] = sentinel
                        if constructor is not None:
                            env["GPUFL_INJECT_USE_CONSTRUCTOR"] = constructor
                        result = subprocess.run([str(tmp / "target"), "argument preserved"], env=env, timeout=10)
                        self.assertEqual(result.returncode, 23)


if __name__ == "__main__":
    unittest.main()
