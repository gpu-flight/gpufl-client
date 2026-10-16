#pragma once

// The Linux injection glue around CaptureGate: driver-init triggers, the
// status file, the exit guard and the process-wide gate. Separate from
// inject_entry.cpp so the CPU end-to-end tests can load it, with the real
// typed wrappers, into a process with fake CUDA libraries.

#include "gpufl/inject/capture_gate.hpp"

namespace gpufl::inject {

// The documented exit status when the process ends at exit with an
// initializer still inside CUDA/CUPTI after the deadline.
constexpr int kInitWedgedExitCode = 86;

struct LinuxGateBindings {
    // gpufl::init plus the Trace facility check, on the initializer thread.
    GateStep (*initialize)() = nullptr;
    // At exit, after every initialization participant returned.
    void (*after_initialization_exit)() = nullptr;
    // Called once when the gate is created (e.g. to register a capability
    // contributor).
    void (*on_gate_created)() = nullptr;
};

/**
 * Binds the initializer. Called from the injection library's static
 * initialization, before any CUDA call can reach the gate.
 */
void BindLinuxCaptureGate(const LinuxGateBindings& bindings);

/** GPUFL_INJECT=1: anything else is an accidental preload and stays inert. */
bool LinuxInjectionRequested();

/** The process-lifetime gate (created on first use). */
CaptureGate& LinuxCaptureGate();

/** InitializeInjection's Linux body. */
void LinuxInjectionCallback();

/** GpuFlightInitializeInjectionAfterCuda's Linux body. */
int LinuxManualHandshake();

}  // namespace gpufl::inject
