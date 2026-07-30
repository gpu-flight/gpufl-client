# Long-Running Session Segmentation — Client Contract

Status: **PROPOSED — WIRE AND LIFECYCLE CONTRACT**  
Repository baseline reviewed: 2026-07-29  
Companion frontend plan:
`gpufl-product-front/docs/session-segmentation-front-plan.md`

This document defines the client-side identity, lifecycle events, boundary
semantics, and runtime ownership required to split one long-running profiling
run into independently uploadable and queryable session segments.

It does not authorize implementation yet. Backend ingestion and agent
transport must accept this contract before segmentation can be enabled outside
an opt-in test path.

---

## 1. Outcome

A ten-minute profiling target configured with a five-minute cadence produces:

```text
run R
├── session S0 · segment 0 · 0–5 min
└── session S1 · segment 1 · 5–10 min
```

Each segment:

- has its own `session_id` and log directory;
- begins with an ordinary `job_start`;
- ends with an ordinary `shutdown`;
- can upload, finalize, and open in the existing session detail page without
  waiting for the target process to exit;
- carries enough metadata and dictionaries to be interpreted independently;
- remains linked to the logical run by `run_id` and `segment_index`.

The GPU runtime, CUDA context, CUPTI subscription, profiling engines, metric
baselines, and deep-window evaluator are not restarted at a boundary.

---

## 2. Fixed Product and Runtime Decisions

1. The client chooses segment boundaries. The backend never slices a completed
   session by timestamps.
2. Segmentation rotates session identity and output ownership only. It never
   calls `gpufl::shutdown()` followed by `gpufl::init()` at a boundary.
3. `analysis_id` and `run_id` are orthogonal:
   - analysis passes overlay the same workload interval;
   - run segments concatenate adjacent workload intervals.
4. V1 rejects multi-pass plus segmentation before the target launches.
5. V1 supports the launcher/injection plus agent path. Embedded self-upload is
   deferred.
6. Segmentation is opt-in and off by default.
7. Time and row-budget triggers may coexist; the first due trigger requests a
   boundary.
8. A deep window is never split. A requested boundary waits until the window
   closes.
9. No boundary inserts `cudaDeviceSynchronize` or otherwise changes target
   CUDA behavior.
10. `run_end` is the only authority for the final segment index.
11. A crashed or killed process may have no `segment_end` and no `run_end`.
    Backend liveness/finality timeout is mandatory.
12. Event and batch IDs remain unique across the run. They are not reset per
    segment.

---

## 3. Configuration Contract

Draft launcher options:

```text
--segment-every <duration>
--segment-max-rows <count>
```

Internal environment contract:

```text
GPUFL_RUN_ID
GPUFL_SEGMENT_EVERY_MS
GPUFL_SEGMENT_MAX_ROWS
```

Rules:

- the launcher generates one UUIDv4 `GPUFL_RUN_ID` before starting the target;
- the injected runtime requires that ID when segmentation is enabled;
- an absent `GPUFL_RUN_ID` while a segment option is present is a startup
  error, not a request to generate unrelated IDs in multiple modules;
- a normal non-segmented invocation does not set these variables and preserves
  today's wire format;
- `GPUFL_ANALYSIS_ID` plus either segment option is rejected before target
  execution;
- until `SegmentCoordinator` lands, one shared compile-time readiness gate is
  enforced by both the launcher and `gpufl::init()`. Directly injecting the
  internal environment variables must not bypass the launcher and emit a
  misleading one-session segmented run;
- a zero or absent time/row value disables that trigger;
- both disabled means segmentation is off;
- production CLI requires a non-zero `--segment-every` cadence of at least 60
  seconds to prevent an accidental session storm; unit tests use a fake
  coordinator clock instead of weakening that minimum.

The launcher owns configuration and validation. The target runtime owns
individual `session_id` generation after the initial session.

---

## 4. Identity and Clock Domains

### 4.1 Identity

```text
run_id        UUIDv4 · constant for the target process
session_id    UUIDv4 · unique for each segment
segment_index uint32 · zero-based and contiguous within the run
```

The invariant is:

```text
UNIQUE(run_id, segment_index)
UNIQUE(session_id)
```

`segment_index` is logical order. Arrival and finalization order may differ.

### 4.2 Time

Two clocks have different responsibilities:

- `std::chrono::steady_clock` decides when a duration boundary is due and
  measures deferral;
- the existing `detail::GetTimestampNs()` event clock stamps wire events and
  segment diagnostic bounds.

Never compare a raw steady-clock value with a telemetry timestamp.

Wire fields:

```text
actual_start_ns       event-clock timestamp at context cutover
actual_end_ns         event-clock timestamp at context cutover
requested_boundary_ns event-clock projection of the steady deadline
boundary_delay_ns     monotonic elapsed delay after the requested deadline
```

`boundary_delay_ns` is authoritative for “how late did this boundary occur?”
because wall time may adjust during a long run. Diagnostics use only
`actual_start_ns` and `actual_end_ns`.

---

## 5. Existing Lifecycle Compatibility

Every segment is deliberately an ordinary session to the current transport and
backend pipeline.

### 5.1 Start

`job_start` is the first NDJSON event in every new segment channel. It gains
two optional fields:

```json
{
  "version": 1,
  "type": "job_start",
  "session_id": "S1",
  "run_id": "R",
  "segment_index": 1
}
```

The fields are emitted together only when segmentation is enabled. An ordinary
session remains byte-compatible with the current wire.

`segment_start` follows `job_start`. It never precedes it, because current
upload ordering and backend placeholder creation depend on `job_start`.

### 5.2 End

Every segment ends with the existing `shutdown` event so current finalization,
retention, status, and session-complete behavior continue to work.

Segmentation does not add provenance fields to `shutdown`. Its wire shape and
meaning remain unchanged: it is a session terminal record, not an authority
for whether the profiled process continues.

Segment-boundary provenance belongs only to `segment_end.end_reason`. Process
finality belongs only to `run_end`. This remains unambiguous when launcher
crash repair appends an existing synthetic `shutdown`, and it avoids storing a
stale process-continuation claim in an earlier segment.

The final event order is:

```text
... data
segment-local terminal metadata
segment_end
run_end          # final segment only
shutdown         # last lifecycle record for this session
```

The agent sends session-complete only after the segment sink is closed and its
`.tmp` directory is gone.

---

## 6. Wire Events

All new NDJSON events use `"version":1`, include `session_id`, and go to
`Channel::All` unless specified otherwise.

### 6.1 `segment_start`

```json
{
  "version": 1,
  "type": "segment_start",
  "session_id": "S1",
  "run_id": "R",
  "segment_index": 1,
  "ts_ns": 300000000001,
  "actual_start_ns": 300000000000,
  "previous_session_id": "S0",
  "requested_boundary_ns": 300000000000,
  "boundary_delay_ns": 12500000,
  "deferred_by": "deep_window"
}
```

Rules:

- Segment 0 has `previous_session_id: null`.
- Segment 0 has `requested_boundary_ns: null`, `boundary_delay_ns: 0`, and
  `deferred_by: null`.
- `deferred_by` is `deep_window` or null.
- There is no `start_reason`. The previous segment's `segment_end.end_reason`
  is the single authority for why the cut occurred.

### 6.2 `segment_end`

```json
{
  "version": 1,
  "type": "segment_end",
  "session_id": "S0",
  "run_id": "R",
  "segment_index": 0,
  "ts_ns": 300000000020,
  "actual_end_ns": 300000000000,
  "requested_boundary_ns": 300000000000,
  "boundary_delay_ns": 12500000,
  "end_reason": "time",
  "deferred_by": "deep_window",
  "records_outside_segment_window": 3
}
```

`end_reason` is one of:

```text
time
row_budget
process_shutdown
```

When both time and row budget become due, the row-budget crossing is
timestamped with the steady-clock time at which the batch that caused the
crossing commits. That timestamp, not a later coordinator observation time,
is compared with the time deadline. The earlier timestamp wins; exact equality
resolves to `time`.

`records_outside_segment_window` counts stored records whose complete event
interval falls outside `[actual_start_ns, actual_end_ns]`. It is a data-quality
count, not a reason to discard the records.

### 6.3 `run_end`

```json
{
  "version": 1,
  "type": "run_end",
  "session_id": "S1",
  "run_id": "R",
  "final_segment_index": 1,
  "ts_ns": 600000000000,
  "ended_ns": 600000000000
}
```

Rules:

- emitted exactly once, in the final segment;
- emitted only during the runtime's terminal capture shutdown;
- emitted before that segment's ordinary `shutdown`;
- never synthesized by log salvage or the launcher crash-repair path;
- absent after `SIGKILL`, fatal process loss, or a crash before terminal
  capture shutdown;
- means “the profiling capture ended cleanly,” not “the target returned exit
  code zero.” Target exit provenance remains separate.

The backend marks a run complete only when `run_end.final_segment_index` is
known and every segment `0..final_segment_index` is finalized. Delivery is
order-independent: segment directories are uploaded by independent Agent
drains, so `run_end` may become queryable before an earlier segment even
though the client closes prior segments before writing the final `run_end`.
Seeing `run_end` is therefore never sufficient by itself to mark the run
complete.

---

## 7. Segment Bootstrap and Terminal Snapshots

### 7.1 Bootstrap order

The client creates the new session directory and acquires its
`SessionOwnershipLock` before writing any visible bootstrap artifact. Before a
new `SegmentContext` becomes visible to producers, its sink receives:

1. `job_start`;
2. `segment_start`;
3. cached host/device/static capture configuration;
4. a full dictionary snapshot;
5. continuation-open scope rows;
6. `rule_state_checkpoint`;
7. counter-quality carry-in metadata where applicable.

Only after these records are durable in the new active files may the
coordinator publish the new context.

The host and static GPU inventory is cached from initial runtime setup.
Boundaries must not rerun slow NVML/NVAPI/CUDA inventory calls.

### 7.2 Terminal order

After old-context writers drain, the coordinator emits into the retiring
segment:

1. every remaining buffered batch;
2. continuation-close scope rows;
3. per-segment capture capability outcome;
4. per-segment deep-window rule and counter-quality deltas;
5. `segment_end`;
6. `run_end` when terminal;
7. `shutdown`;
8. sink close/retirement and transport reconciliation;
9. release the retiring segment's `SessionOwnershipLock`.

Compression, publish retry, and `.tmp` cleanup execute on the existing
retirement/export worker, never on a CUPTI callback or application hot path.

The retiring lock is released only after channel writers are closed, pending
windows have been reconciled or deliberately left visible for salvage, and
the segment `.tmp` directory has been removed or intentionally retained as an
incomplete transport state. Releasing it earlier lets the Agent salvage files
that the client may still mutate. Holding it until process exit defeats early
segment availability.

---

## 8. `SegmentContext` and Cutover Linearization

Current session identity is split across `Runtime::session_id`,
`Runtime::logger`, `MonitorBatchManager::FlushSink`, sampler configuration, and
event builders. Updating these fields independently can put an old
`session_id` into a new directory or vice versa.

One immutable context becomes the ownership unit:

```cpp
struct SegmentContext {
    std::string run_id;
    std::string session_id;
    uint32_t segment_index;
    int64_t actual_start_ns;
    std::shared_ptr<Logger> logger;
    std::shared_ptr<SegmentDictionaryEmitter> dictionary;
};
```

Publication uses C++17 `std::atomic_load`/`std::atomic_store` overloads for
`shared_ptr`. Writer drainage does **not** infer liveness from
`shared_ptr::use_count()`: reference count observation is not a synchronization
primitive and unrelated owner copies would make its target value ambiguous.
Each context instead owns an explicit sealed writer-lease counter and drain
condition.

### 8.1 Writer contract

A writer:

1. acquires one move-only `SegmentWriteLease`;
2. uses that same context to build the event/batch JSON;
3. writes through that context's logger;
4. releases the lease only after the complete record or batch is committed.

Publication seals the old context before storing the new one. An acquire that
races with sealing either increments the old context before the seal and is
included in its drain, or observes the seal and retries against the new
context. The retirement worker waits on the explicit counter with a bounded
timeout. A timeout emits an ERROR and leaves that segment deliberately
incomplete; it must not close a logger underneath a live writer or block
process teardown forever.

Batch-scoped acquisition is preferred. Per-kernel shared-pointer reference
traffic is prohibited until benchmarked.

### 8.2 Linearization point

The atomic publication of the new context is the boundary's storage
linearization point.

- acquisitions after publication use the new segment;
- a writer that acquired the old context before publication may finish writing
  to the old segment afterward;
- the retiring sink remains open until all such references drain;
- sink close never happens on the thread that releases the last producer
  reference.

Therefore the precise attribution rule is:

> A record belongs to the immutable segment context acquired for its complete
> serialization/write operation.

This replaces the less implementable phrase “the context active when the write
finished.”

### 8.3 Boundary sequence

```text
1. boundary request becomes due
2. if deep window active, defer
3. coordinator serializes against periodic CUPTI flush
4. flush/drain available activity without device synchronization
5. capture open-scope snapshot
6. choose actual boundary timestamp
7. create the next segment directory and acquire its ownership lock
8. prepare new context and write its bootstrap records
9. atomically publish new context and seal the old context from new boundaries
10. new producers continue immediately
11. wait for old producer references on retirement coordinator
12. finish old batches/snapshots/lifecycle records
13. close old sink and reconcile its transport/.tmp state
14. release old segment ownership lock; Agent may complete/upload it
```

The coordinator does not wait for gzip or backend network activity before
publishing the new context.

For a bounded handoff interval, one process therefore owns two
`SessionOwnershipLock`s for two different session directories. This is
intentional. The in-process ownership registry must reject duplicate
acquisition of the same directory but permit distinct old/new segment paths.
The new segment lock must be acquired before its bootstrap is visible;
otherwise the Agent can classify the directory as legacy and bypass the
window-identity contract.

---

## 9. Dictionary Contract

The current `DictionaryManager` has one global dirty map. That is insufficient
for segmentation:

- a new segment needs a full mapping even when no IDs are globally dirty;
- an old-context flush can consume a dirty entry after the new segment's
  snapshot;
- a producer may intern an ID around cutover and write through either context.

The implementation must separate:

```text
GlobalDictionaryRegistry
  stable name ↔ id mappings for the run

SegmentDictionaryEmitter
  which mappings have been emitted into one segment
```

Requirements:

1. Numeric dictionary IDs remain stable for the full run.
2. Every new segment receives a full registry snapshot before context
   publication.
3. Each segment independently tracks emitted IDs after bootstrap.
4. Before writing a batch, the batch's segment emitter writes every referenced
   mapping not yet present in that segment.
5. Emission precedes the referencing batch in that segment's channel.
6. A flush in Segment N cannot clear emission state required by Segment N+1.
7. Source collection privacy settings apply identically to every segment.

Adding only `flushFullDictionary()` beside the existing global dirty maps is
not sufficient; the cutover race would remain.

---

## 10. Scope Continuation

An open scope must be locally balanced in every segment it spans.

Extend the scope batch contract:

```text
event_type 0 = begin
event_type 1 = end
event_type 2 = continuation_open
event_type 3 = continuation_close
```

Add `original_start_ns` to the row schema. Continue using the same
`scope_instance_id` for the logical scope across all segments; it is already
run-global and does not reset.

Example:

```text
Segment 0
  begin(id=17, ts=1s, original_start_ns=1s)
  continuation_close(id=17, ts=5m)

Segment 1
  continuation_open(id=17, ts=5m, original_start_ns=1s)
  continuation_close(id=17, ts=10m)

Segment 2
  continuation_open(id=17, ts=10m, original_start_ns=1s)
  end(id=17, ts=11m)
```

Rules:

- continuation-close and continuation-open use the exact same boundary
  timestamp;
- name ID, logical instance ID, depth, and benchmark metadata remain stable;
- the new segment's continuation-open is written before context publication;
- the old segment's continuation-close is emitted after old writers drain but
  carries the boundary timestamp;
- final process shutdown closes remaining scopes with ordinary `end` rows;
- backend session views pair `begin|continuation_open` with
  `end|continuation_close`;
- a future run-wide view stitches by `(run_id, scope_instance_id)`.

The open-scope snapshot must be captured under the scope-state mutex. It must
not rely only on a thread-local name stack because scopes may exist on multiple
application threads.

---

## 11. Event Attribution and Diagnostics

No device synchronization is added. A kernel, memcpy, or scope may straddle a
boundary.

Storage attribution follows the acquired `SegmentContext`. In particular:

- a CUPTI record drained before cutover normally lands in the old segment;
- a running kernel whose completion record arrives after cutover normally
  lands in the new segment;
- a pre-cutover writer holding the old context may commit after the boundary.

Diagnostics use:

```text
window = [segment_start.actual_start_ns, segment_end.actual_end_ns]
contribution = intersection(event interval, window)
```

Records entirely outside that interval:

- remain available for timeline forensics;
- contribute zero to segment diagnostics;
- increment `records_outside_segment_window`.

Timeline bounds may union full timestamps for visualization, but data bounds
must never replace the diagnostic denominator.

---

## 12. Deep Window and Rule State

### 12.1 Boundary deferral

- A segment boundary does not close an active deep window.
- The coordinator records the original requested deadline and waits for the
  window's normal close.
- `boundary_delay_ns` measures the monotonic deferral.
- The following segment records `deferred_by: "deep_window"`.
- Segmentation rejects any configuration capable of opening an unbounded deep
  window; every supported window must have a duration or launch-count bound.

### 12.2 Rule state

`DeepWindowRules::Finish()` remains terminal and is never called at an
intermediate boundary.

Required APIs:

```cpp
SnapshotSegment(closing_context)
BeginNextSegment(new_context)
Checkpoint(new_context)
```

Behavior:

- evaluator state, warm-up, rate baseline, cooldown, and `max_windows` budget
  continue across boundaries;
- `SnapshotSegment` emits segment-local windows opened, samples, and quality
  reset deltas plus explicitly named cumulative values;
- `BeginNextSegment` resets delta accumulators only;
- `Checkpoint` writes carry-in state to the new segment before publication and
  contains no segment delta;
- terminal `Finish()` writes the final summary into the final segment.

Counter data quality follows the same snapshot/delta pattern. Process-lifetime
tracked-counter context is not misrepresented as a segment-local count.

---

## 13. Capture Capabilities and Run Artifacts

Current capability emission is effectively one-shot. Segmentation requires:

- cached requested/configured capability information at segment start;
- segment-local collected/partial/no-data outcomes at segment end;
- counters reset only for per-segment observations, not engine/runtime state.

Large artifacts such as source content, PTX, and cubin disassembly must not be
blindly duplicated into every segment.

The first implementation slice is limited to capture modes whose segment is
independently useful without an unresolved cross-segment artifact reference.
Before enabling a source/SASS-producing mode, implement a content-addressed
run artifact plus a per-segment artifact-reference contract. Do not silently
show an empty Source/SASS tab in later segments.

The V1 launcher/injection whitelist is intentionally narrow:

- `Monitor`;
- one native `Trace` pass;
- one native `PmSampling` pass;
- the adaptive plan with native `Trace` as the base and window-only
  `PmSampling` as the prepared deep engine.

V1 rejects `PcSampling`, `SassMetrics`, `RangeProfiler`,
`RangeProfilerKernelReplay`, `Deep`, composites containing any of them, and
all user-specified multi-pass lists. An opened `sass` channel alone does not
reject a run; producing a source/SASS artifact outside the whitelisted plan
does. Rejections occur before the target starts and name the unsupported
engine or pass.

---

## 14. Row Budget

The row trigger counts logical telemetry data rows, not NDJSON lines:

- kernel rows;
- memcpy/memset rows;
- scope rows;
- synchronization rows;
- allocation rows;
- profile/PM sample rows;
- system metric rows.

It excludes lifecycle events, dictionary mappings, capability/configuration
events, and artifact payloads.

The count belongs to the `SegmentContext`. A batch crossing the budget is
written atomically to one segment; the boundary is requested after that batch.
Rows are never split across two segment contexts solely to hit an exact count.
Every new context starts its counter at zero. Once a context retires,
`boundary_requests_enabled` becomes false: writers already holding it may
still commit late rows and increment its final statistics, but those rows
must never request another cutover.

---

## 15. Crash and Recovery Contract

### 15.1 Process dies before a boundary

- active transport windows and raw retired files follow the existing salvage
  contract;
- `segment_end`, `run_end`, and `shutdown` may all be absent;
- launcher repair may append the existing synthetic `shutdown`;
- launcher repair must not fabricate `segment_end` or `run_end`;
- backend timeout makes the segment/run stale, crashed, or incomplete.

### 15.2 Process dies during cutover

Possible durable states must converge:

- old segment complete, new segment absent;
- old segment retiring, new segment has bootstrap only;
- both segment directories exist, with one awaiting salvage.

Requirements:

- a published `segment_start` always has a preceding `job_start`;
- `(run_id, segment_index)` prevents duplicate logical segments;
- sidecar/tombstone sequence prevents transport-window identity reuse;
- salvage never invents a new `session_id`, `segment_index`, or `run_end`;
- an orphan bootstrap-only segment becomes incomplete after timeout rather than
  completed.

### 15.3 Transport health

Backend timeout is mandatory even if no health record can leave the machine.

A later Agent control-plane contract may upload precise durable loss reasons.
It must bypass the normal session-complete gate; otherwise the loss marker that
blocks completion would also block its own report.

---

## 16. Agent Contract

Each segment directory is an ordinary session directory. Existing window
identity, ACK retention, retry, and session-complete behavior remain
session-scoped.

Required verification:

- multiple live segment directories from one target process are discovered;
- the new segment lock is acquired before bootstrap becomes visible;
- a cutover may temporarily hold locks for distinct old and new directories;
- the retiring lock remains held through sink close and `.tmp` reconciliation;
- an older segment can complete and upload while a newer segment remains live;
- out-of-order segment finalization is allowed;
- final process exit prompts the remaining segment promptly;
- Agent never assumes only one live session per process or log root;
- loss in one segment does not allow run completion to be reported as healthy.

No Agent grouping logic is required. Run lifecycle is a backend concern.

---

## 17. Implementation Boundaries

New client components:

```text
SegmentCoordinator
SegmentContext
SegmentDictionaryEmitter
SegmentLifecycleModels
```

Existing areas requiring refactor:

- `Runtime`
  - replace mutable `session_id`/`logger` reads with current context access;
  - cache bootstrap inventory.
- `MonitorBatchManager`
  - remove the single cached `FlushSink`;
  - flush a complete batch through one acquired context;
  - preserve run-global batch and scope IDs.
- `Sampler`
  - acquire the current context for each sample batch instead of caching one
    session ID and logger forever.
- `Logger`
  - one logger/sink per segment context;
  - retirement remains asynchronous.
- `DictionaryManager`
  - split run-global registry from segment-local emission state.
- lifecycle/capability/deep-rule models
  - add versioned segment events and snapshot APIs.
- launcher parser
  - options, environment, incompatibility validation, and run ID generation.

---

## 18. Test and Mutation Plan

### 18.1 Pure contract tests

- exact JSON shape for all three events;
- segment fields omitted from non-segmented `job_start`;
- segmented and non-segmented `shutdown` have the same existing wire shape;
- `job_start` is first and `shutdown` last in every segment;
- `run_end` occurs once and only in the final segment;
- no `start_reason` duplicate authority;
- row-budget crossing uses batch-commit steady time, not coordinator poll time;
- exact time/row trigger ties resolve to `time`;
- the V1 engine/pass whitelist rejects unsupported modes before target start.

### 18.2 Context concurrency

- a writer holding old context cannot write into the new directory;
- a new writer cannot emit before new `job_start`;
- new bootstrap is not visible before its ownership lock is acquired;
- distinct old/new ownership locks overlap during handoff;
- old sink closes only after old references drain;
- a leaked writer reaches the bounded timeout, emits a diagnostic, publishes
  no false `segment_end`/`run_end`, and does not hang shutdown;
- old ownership lock releases only after sink/transport retirement completes;
- a retiring context cannot request a second boundary from late rows;
- last producer release does not execute filesystem close/compression;
- batch-scoped context acquisition has measured overhead within budget.

Mutation checks:

- build JSON with one context and write with another;
- close sink on last-reference thread;
- reset batch IDs at segment start.

### 18.3 Dictionary

- Segment 1 resolves IDs created only in Segment 0;
- IDs interned immediately before, during, and after cutover resolve in every
  segment that references them;
- an old flush cannot consume a new segment's required mapping;
- full snapshot precedes the first referencing batch.

Mutation check: replace per-segment emission state with the current global dirty
map and require the race fixture to fail.

### 18.4 Scopes

- scope spanning one, two, and three boundaries is balanced in every segment;
- same `scope_instance_id`, name, depth, metadata, and original start persist;
- multiple threads with overlapping open scopes snapshot correctly;
- final shutdown produces ordinary ends rather than continuation closes.

### 18.5 Deep window and counters

- due boundary defers during a bounded deep window;
- evaluator baseline, cooldown, and max-window budget survive the boundary;
- checkpoint contains no prior segment deltas;
- segment summaries contain only their own deltas;
- terminal `Finish()` still runs once.

### 18.6 Crash states

Kill at:

- before new bootstrap;
- after new `job_start` but before context publish;
- immediately after context publish;
- while old context retires;
- while gzip `.part` is being written.

After salvage, assert no fabricated `run_end`, no duplicate segment identity,
and no session promoted without its first `job_start`.

### 18.7 Hardware/full-stack release gate

On L4 and RTX 3090:

1. run a ten-minute target with five-minute segmentation;
2. verify Segment 0 becomes queryable before target exit;
3. measure boundary-to-queryable latency, initial target <= 60 seconds;
4. verify kernels, memcpy, scopes, dictionaries, capabilities, and diagnostics
   in both segments;
5. verify a scope and a kernel spanning the boundary;
6. verify a conditional deep window that crosses the requested boundary;
7. verify Agent restart and backend outage between segments;
8. verify final run status after out-of-order ingest;
9. use real upload/ingest only—no synthetic database rows.

---

## 19. Delivery Order

1. Commit this contract after review.
2. Add wire structs/models and exact serialization tests without enabling
   runtime segmentation.
3. Add launcher parsing, run ID generation, and invalid-combination tests.
   This slice remains behind an explicit execution-boundary gate until the
   coordinator lands. The launcher and injected runtime consult the same gate;
   neither the CLI nor direct environment injection may silently accept
   segmentation while still producing only one session.
4. Implement `SegmentContext` and refactor producers to acquire it while still
   running one segment.
5. Implement global dictionary registry plus segment-local emission.
6. Add coordinator boundary state machine and asynchronous retirement.
7. Add scope continuation and segment-local rule/capability snapshots.
8. Verify Agent multi-directory behavior.
9. Implement backend lifecycle, timeout, uniqueness, quota, retention, and
   paginated read APIs.
10. Implement frontend plan.
11. Run L4/3090 and full-stack early-queryability gates.

Segmentation remains opt-in until the transport-health timeout contract and
time-to-first-queryable-segment release gate both pass.
