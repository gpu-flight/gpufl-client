#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "gpufl/inject/capture_gate.hpp"

using gpufl::inject::CaptureGate;
using gpufl::inject::GateConfig;
using gpufl::inject::GateHooks;
using gpufl::inject::GateLifecycle;
using gpufl::inject::GateMode;
using gpufl::inject::GateOwner;
using gpufl::inject::GateReadiness;
using gpufl::inject::GateSnapshot;
using gpufl::inject::GateStep;

namespace {

using Clock = std::chrono::steady_clock;

int64_t RealNowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               Clock::now().time_since_epoch())
        .count();
}

// A one-shot latch the fakes block on.
class Latch {
   public:
    void Open() {
        {
            std::lock_guard lock(mu_);
            open_ = true;
        }
        cv_.notify_all();
    }
    void Wait() {
        std::unique_lock lock(mu_);
        cv_.wait(lock, [this] { return open_; });
    }

   private:
    std::mutex mu_;
    std::condition_variable cv_;
    bool open_ = false;
};

// Real threads and the real clock (plus an offset a test can push forward),
// fake driver and CUPTI steps.
struct Fakes {
    std::atomic<int64_t> clock_offset{0};
    std::atomic<int> spawned{0};
    std::atomic<int> initialized{0};
    std::function<GateStep()> barrier = [] { return GateStep{true, ""}; };
    std::function<GateStep()> init = [] { return GateStep{true, ""}; };
    std::mutex mu;
    std::vector<std::string> events;
    std::vector<GateSnapshot> published;
    std::vector<std::thread> threads;

    ~Fakes() { JoinAll(); }

    void JoinAll() {
        std::vector<std::thread> pending;
        {
            std::lock_guard lock(mu);
            pending.swap(threads);
        }
        for (auto& t : pending) t.join();
    }

    void Event(const std::string& e) {
        std::lock_guard lock(mu);
        events.push_back(e);
    }

    GateHooks Hooks() {
        GateHooks hooks;
        hooks.now_ns = [this] { return RealNowNs() + clock_offset.load(); };
        hooks.spawn = [this](std::function<void()> body) {
            ++spawned;
            std::lock_guard lock(mu);
            threads.emplace_back(std::move(body));
            return true;
        };
        hooks.driver_barrier = [this] {
            Event("barrier");
            return barrier();
        };
        hooks.sleep_ms = [](int) {};
        hooks.initialize = [this] {
            ++initialized;
            Event("init");
            return init();
        };
        hooks.publish = [this](const GateSnapshot& s) {
            std::lock_guard lock(mu);
            published.push_back(s);
        };
        return hooks;
    }

    bool EverPublishedReady() {
        std::lock_guard lock(mu);
        for (const auto& s : published) {
            if (s.readiness == GateReadiness::Ready) return true;
        }
        return false;
    }
};

GateConfig Config(GateMode mode = GateMode::Automatic,
                  int64_t wait_ms = 10'000) {
    GateConfig config;
    config.mode = mode;
    config.wait_ns = wait_ms * 1'000'000LL;
    config.protocol = "2";
    return config;
}

template <typename Predicate>
void WaitFor(Predicate predicate) {
    const auto until = Clock::now() + std::chrono::seconds(10);
    while (!predicate()) {
        ASSERT_LT(Clock::now(), until) << "condition never became true";
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

}  // namespace

TEST(CaptureGate, FirstCallInitializesCaptureBeforeTheApplicationProceeds) {
    Fakes fakes;
    CaptureGate gate(Config(), fakes.Hooks());

    gate.Enter([&] {
        fakes.Event("driver_init");
        gate.OnInjectionCallback();  // the driver calls InitializeInjection
        return GateStep{true, ""};
    });
    fakes.Event("application_call");

    fakes.JoinAll();
    const auto s = gate.Snapshot();
    EXPECT_EQ(s.readiness, GateReadiness::Ready);
    EXPECT_EQ(s.lifecycle, GateLifecycle::Finished);
    EXPECT_EQ(s.owner, GateOwner::Gate);
    EXPECT_TRUE(s.coverage_notes.empty());
    EXPECT_EQ(fakes.events, (std::vector<std::string>{
                                "driver_init", "barrier", "init", "application_call"}));
}

TEST(CaptureGate, TriggerFailureBeforeAnyCallbackSettlesAtOnce) {
    Fakes fakes;
    CaptureGate gate(Config(), fakes.Hooks());
    const auto start = Clock::now();

    gate.Enter([] { return GateStep{false, "runtime_init_failed:35"}; });

    EXPECT_LT(Clock::now() - start, std::chrono::seconds(1));
    const auto s = gate.Snapshot();
    EXPECT_EQ(s.readiness, GateReadiness::Failed);
    EXPECT_EQ(s.reason, "runtime_init_failed:35");
    EXPECT_EQ(s.lifecycle, GateLifecycle::Finished);  // no worker ever started
    EXPECT_FALSE(s.worker.expected);
    EXPECT_TRUE(gate.AwaitAtExit());
}

TEST(CaptureGate, DriverInitWithoutTheInjectionCallbackFails) {
    Fakes fakes;
    CaptureGate gate(Config(), fakes.Hooks());

    gate.Enter([] { return GateStep{true, ""}; });

    const auto s = gate.Snapshot();
    EXPECT_EQ(s.readiness, GateReadiness::Failed);
    EXPECT_EQ(s.reason, "no_injection_callback");
    EXPECT_EQ(s.lifecycle, GateLifecycle::Finished);
    EXPECT_EQ(fakes.spawned.load(), 0);
}

TEST(CaptureGate, WorkerSuccessDoesNotHideALaterTriggerFailure) {
    Fakes fakes;
    CaptureGate gate(Config(), fakes.Hooks());

    gate.Enter([&] {
        gate.OnInjectionCallback();
        WaitFor([&] { return gate.Snapshot().worker.returned; });
        EXPECT_EQ(gate.Snapshot().lifecycle, GateLifecycle::CaptureInitializing);
        return GateStep{false, "runtime_init_failed:100"};
    });

    fakes.JoinAll();
    const auto s = gate.Snapshot();
    EXPECT_EQ(s.readiness, GateReadiness::Failed);
    EXPECT_EQ(s.reason, "runtime_init_failed:100");
    EXPECT_EQ(s.lifecycle, GateLifecycle::Finished);
    EXPECT_FALSE(fakes.EverPublishedReady());
}

TEST(CaptureGate, ABlockedTriggerKeepsInitializationUnfinished) {
    Fakes fakes;
    CaptureGate gate(Config(GateMode::Automatic, 200), fakes.Hooks());
    Latch release;

    std::thread trigger_thread([&] {
        gate.Enter([&] {
            gate.OnInjectionCallback();
            release.Wait();  // never returns until the test lets it
            return GateStep{true, ""};
        });
    });
    WaitFor([&] { return gate.Snapshot().worker.returned; });

    // An independent waiter is what observes the deadline in-process.
    gate.Enter([] { return GateStep{true, ""}; });
    auto s = gate.Snapshot();
    EXPECT_EQ(s.readiness, GateReadiness::Failed);
    EXPECT_EQ(s.reason, "capture_wait_timeout");
    EXPECT_NE(s.lifecycle, GateLifecycle::Finished);
    EXPECT_FALSE(gate.AwaitAtExit());  // the process must not tear down
    EXPECT_EQ(gate.Snapshot().coverage_notes.count("exit_while_initializing"), 1u);

    release.Open();
    trigger_thread.join();
    s = gate.Snapshot();
    EXPECT_EQ(s.lifecycle, GateLifecycle::Finished);
    EXPECT_EQ(s.reason, "capture_wait_timeout");
    EXPECT_FALSE(fakes.EverPublishedReady());
}

TEST(CaptureGate, TimeoutStaysFailedAfterALateSuccessAndNothingWaitsAgain) {
    Fakes fakes;
    Latch release;
    fakes.init = [&] {
        release.Wait();
        return GateStep{true, ""};
    };
    CaptureGate gate(Config(GateMode::Automatic, 200), fakes.Hooks());

    gate.Enter([&] {
        gate.OnInjectionCallback();
        return GateStep{true, ""};
    });
    EXPECT_EQ(gate.Snapshot().reason, "capture_wait_timeout");

    release.Open();
    fakes.JoinAll();
    EXPECT_EQ(gate.Snapshot().readiness, GateReadiness::Failed);
    EXPECT_EQ(gate.Snapshot().lifecycle, GateLifecycle::Finished);

    const auto start = Clock::now();
    gate.Enter([] { return GateStep{true, ""}; });
    EXPECT_LT(Clock::now() - start, std::chrono::milliseconds(50));
}

TEST(CaptureGate, SuccessAfterTheDeadlineFailsWithoutAnyWaiter) {
    Fakes fakes;
    CaptureGate gate(Config(), fakes.Hooks());
    fakes.init = [&] {
        fakes.clock_offset = 60'000'000'000LL;  // a minute later
        return GateStep{true, ""};
    };

    gate.OnInjectionCallback();  // driver init from an unwrapped route
    fakes.JoinAll();

    const auto s = gate.Snapshot();
    EXPECT_EQ(s.owner, GateOwner::Callback);
    EXPECT_EQ(s.readiness, GateReadiness::Failed);
    EXPECT_EQ(s.reason, "capture_init_deadline_exceeded");
    EXPECT_EQ(s.coverage_notes.count("unsupported_first_touch"), 1u);
}

TEST(CaptureGate, ConcurrentFirstCallersShareOneInitializer) {
    Fakes fakes;
    CaptureGate gate(Config(), fakes.Hooks());
    std::atomic<int> triggers{0};
    std::vector<std::thread> callers;
    for (int i = 0; i < 8; ++i) {
        callers.emplace_back([&] {
            gate.Enter([&] {
                ++triggers;
                gate.OnInjectionCallback();
                gate.OnInjectionCallback();  // duplicate callback
                return GateStep{true, ""};
            });
            EXPECT_EQ(gate.Snapshot().readiness, GateReadiness::Ready);
        });
    }
    for (auto& t : callers) t.join();
    fakes.JoinAll();
    EXPECT_EQ(triggers.load(), 1);
    EXPECT_EQ(fakes.spawned.load(), 1);
    EXPECT_EQ(fakes.initialized.load(), 1);
}

// The Rust bootstrap order: MANUAL=1, cuInit (its callback fires), primary
// context preparation, then the handshake.
TEST(CaptureGate, ManualModeRunsOneSynchronousInitializer) {
    Fakes fakes;
    CaptureGate gate(Config(GateMode::Manual), fakes.Hooks());

    gate.OnInjectionCallback();
    gate.OnInjectionCallback();
    EXPECT_EQ(gate.Snapshot().lifecycle, GateLifecycle::NotRequested);
    EXPECT_EQ(gate.ManualHandshake(nullptr), 1);

    int concurrent = -1;
    std::thread other([&] { concurrent = gate.ManualHandshake(nullptr); });
    other.join();
    EXPECT_EQ(concurrent, 1);

    const auto s = gate.Snapshot();
    EXPECT_EQ(fakes.spawned.load(), 0);
    EXPECT_EQ(fakes.initialized.load(), 1);
    EXPECT_EQ(s.owner, GateOwner::Manual);
    EXPECT_EQ(s.callbacks_seen, 2u);
    EXPECT_TRUE(s.coverage_notes.empty());
}

TEST(CaptureGate, ManualHandshakeReentryAndOwnCallsDoNotWait) {
    Fakes fakes;
    CaptureGate* gate_ptr = nullptr;
    int reentry = -1;
    fakes.init = [&] {
        reentry = gate_ptr->ManualHandshake(nullptr);
        gate_ptr->Enter(nullptr);  // gpufl's own CUDA call during init
        return GateStep{true, ""};
    };
    CaptureGate gate(Config(GateMode::Manual), fakes.Hooks());
    gate_ptr = &gate;

    EXPECT_EQ(gate.ManualHandshake(nullptr), 1);
    EXPECT_EQ(reentry, 0);
    EXPECT_EQ(gate.Snapshot().handshake_reentries, 1u);
    EXPECT_TRUE(gate.Snapshot().coverage_notes.empty());
}

TEST(CaptureGate, ManualCallFromAnotherThreadBeforeReadyIsAViolation) {
    Fakes fakes;
    Latch in_init;
    Latch release;
    fakes.init = [&] {
        in_init.Open();
        release.Wait();
        return GateStep{true, ""};
    };
    CaptureGate gate(Config(GateMode::Manual), fakes.Hooks());

    int result = -1;
    std::thread host([&] { result = gate.ManualHandshake(nullptr); });
    in_init.Wait();
    gate.Enter(nullptr);  // application work while the handshake runs
    release.Open();
    host.join();

    EXPECT_EQ(result, 0);
    EXPECT_EQ(gate.Snapshot().readiness, GateReadiness::Ready);
    EXPECT_EQ(gate.Snapshot().coverage_notes.count("manual_protocol_violation"), 1u);
}

TEST(CaptureGate, WarmupNeverWaitsAndIsUnqualified) {
    Fakes fakes;
    CaptureGate gate(Config(GateMode::Warmup), fakes.Hooks());
    bool triggered = false;

    gate.Enter([&] {
        triggered = true;
        return GateStep{true, ""};
    });

    EXPECT_FALSE(triggered);
    EXPECT_EQ(gate.Snapshot().coverage_notes.count("warmup_unqualified"), 1u);
}

// An unwrapped entry holds a lock the initializer needs and then reaches a
// wrapper. The TLS flags cannot see that; the deadline bounds the stall.
TEST(CaptureGate, AHeldLockStallsOnlyUntilTheDeadline) {
    Fakes fakes;
    std::mutex driver_lock;
    fakes.init = [&] {
        std::lock_guard lock(driver_lock);
        return GateStep{true, ""};
    };
    CaptureGate gate(Config(GateMode::Automatic, 200), fakes.Hooks());

    const auto start = Clock::now();
    {
        std::lock_guard held(driver_lock);
        gate.OnInjectionCallback();  // the unwrapped entry ran cuInit
        gate.Enter([] { return GateStep{true, ""}; });
    }
    EXPECT_LT(Clock::now() - start, std::chrono::seconds(5));
    fakes.JoinAll();

    const auto s = gate.Snapshot();
    EXPECT_EQ(s.reason, "capture_wait_timeout");
    EXPECT_EQ(s.lifecycle, GateLifecycle::Finished);
    EXPECT_EQ(s.coverage_notes.count("unsupported_first_touch"), 1u);
}

TEST(CaptureGate, ExitWithNothingRequestedOrFinishedTearsDownNormally) {
    Fakes fakes;
    CaptureGate idle(Config(), fakes.Hooks());
    EXPECT_TRUE(idle.AwaitAtExit());

    CaptureGate done(Config(), fakes.Hooks());
    done.Enter([&] {
        done.OnInjectionCallback();
        return GateStep{true, ""};
    });
    EXPECT_TRUE(done.AwaitAtExit());
}

TEST(CaptureGateConfig, ReadsModesDelaysAndRejectsBadWaits) {
    const auto config = [](std::vector<std::pair<const char*, const char*>> vars) {
        return gpufl::inject::GateConfigFromEnv([vars](const char* name) -> const char* {
            for (const auto& [k, v] : vars) {
                if (std::string(k) == name) return v;
            }
            return nullptr;
        });
    };

    auto c = config({{"GPUFL_INJECT_PROTOCOL", "2"}, {"GPUFL_INJECT_INIT_DELAY_MS", "100"}});
    EXPECT_EQ(c.mode, GateMode::Automatic);
    EXPECT_EQ(c.delay_ms, 100);
    EXPECT_FALSE(c.legacy_launcher_delay);
    EXPECT_EQ(c.wait_ns, 15'000'000'000LL);

    c = config({{"GPUFL_INJECT_INIT_DELAY_MS", "5000"}});
    EXPECT_TRUE(c.legacy_launcher_delay);

    c = config({{"GPUFL_INJECT_PROTOCOL", "2"}, {"GPUFL_INJECT_WARMUP_MS", "3000"},
                {"GPUFL_INJECT_INIT_DELAY_MS", "100"}});
    EXPECT_EQ(c.mode, GateMode::Warmup);
    EXPECT_EQ(c.delay_ms, 3000);

    c = config({{"GPUFL_INJECT_MANUAL_INIT", "1"}, {"GPUFL_INJECT_WARMUP_MS", "3000"}});
    EXPECT_EQ(c.mode, GateMode::Manual);

    for (const char* bad : {"0", "-5", "abc", "600001", "99999999999999999999"}) {
        c = config({{"GPUFL_INJECT_LAUNCH_WAIT_MS", bad}});
        EXPECT_TRUE(c.wait_config_invalid) << bad;
        EXPECT_EQ(c.wait_ns, 15'000'000'000LL) << bad;
    }
    c = config({{"GPUFL_INJECT_LAUNCH_WAIT_MS", "250"}});
    EXPECT_FALSE(c.wait_config_invalid);
    EXPECT_EQ(c.wait_ns, 250'000'000LL);
}

TEST(CaptureGateStatus, WritesTheSnapshotAtomically) {
    GateSnapshot s;
    s.seq = 7;
    s.lifecycle = GateLifecycle::Finished;
    s.owner = GateOwner::Gate;
    s.readiness = GateReadiness::Failed;
    s.reason = "cuinit_symbol_missing:\"quoted\"\n";
    s.coverage_notes = {"unsupported_first_touch"};
    s.deadline_ns = 123;
    const std::string json = gpufl::inject::GateStatusJson(s, "abc", 42);
    EXPECT_NE(json.find("\"schema\":1,\"nonce\":\"abc\",\"pid\":42,\"seq\":7"),
              std::string::npos);
    EXPECT_NE(json.find("\"readiness\":\"failed\""), std::string::npos);
    EXPECT_NE(json.find("\\\"quoted\\\"\\n"), std::string::npos);
    EXPECT_NE(json.find("\"coverage_notes\":[\"unsupported_first_touch\"]"),
              std::string::npos);
    EXPECT_NE(json.find("\"deadline_monotonic_ns\":123"), std::string::npos);

    const auto dir = std::filesystem::temp_directory_path() /
                     ("gpufl-gate-status-" + std::to_string(RealNowNs()));
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "status-42.json").string();
    ASSERT_TRUE(gpufl::inject::WriteGateStatusFile(path, json));
    std::ifstream in(path);
    const std::string read((std::istreambuf_iterator<char>(in)),
                           std::istreambuf_iterator<char>());
    EXPECT_EQ(read, json);
    EXPECT_FALSE(std::filesystem::exists(path + ".tmp"));
    EXPECT_FALSE(gpufl::inject::WriteGateStatusFile(path, std::string(5000, 'x')));
    std::filesystem::remove_all(dir);
}
