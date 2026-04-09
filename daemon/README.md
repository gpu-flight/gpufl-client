# gpufl-monitor — Standalone GPU Monitoring Daemon

`gpufl-monitor` is a lightweight always-on daemon that continuously samples GPU and host metrics (utilization, memory, temperature, power, CPU, RAM) and writes them as JSONL event logs. A bundled Java agent (`gpufl-agent`) tails those logs and ships the data to a GPUFlight backend.

Both processes run inside a single Docker container managed by `supervisord`.

GPU vendor support:

| Vendor | Dockerfile | Compose file |
|---|---|---|
| NVIDIA | `Dockerfile.monitor` | `docker-compose.monitor.yml` |
| AMD | `Dockerfile.monitor.amd` | `docker-compose.monitor.amd.yml` |

---

## Prerequisites

### NVIDIA

- Docker 24+ with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed
- A running GPUFlight backend reachable from the container
- An API token for the backend (create one under **Settings → API Keys**)

Verify the NVIDIA runtime is available:

```bash
docker run --rm --gpus all nvidia/cuda:13.1.0-base-ubuntu24.04 nvidia-smi
```

### AMD

- Docker 24+ with a ROCm-enabled kernel (Linux 5.15+ recommended)
- `/dev/kfd` and `/dev/dri` present on the host (standard with ROCm driver)
- A running GPUFlight backend reachable from the container
- An API token for the backend (create one under **Settings → API Keys**)

Verify AMD GPU access:

```bash
docker run --rm --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  rocm/dev-ubuntu-24.04:6.4-complete rocm-smi
```

---

## Building the image

The build uses multiple stages: a C++ stage (builds `gpufl-monitor`), a JRE stage, and a slim runtime image. The Java agent JAR is pulled automatically from the pre-built `ghcr.io/gpu-flight/gpufl-agent` image — no local checkout of `gpufl-agent` is required.

### NVIDIA

From the **repository root**:

```bash
docker build \
  -f Dockerfile.monitor \
  -t gpufl/monitor:latest \
  .
```

### AMD

From the **repository root**:

```bash
docker build \
  -f Dockerfile.monitor.amd \
  -t gpufl/monitor-amd:latest \
  .
```

---

## Running with docker compose (recommended)

### NVIDIA

Copy `.env.example` to `.env` and set the required variables, then:

```bash
GPUFL_HTTP_URL=https://your-backend/api/v1/events/ \
GPUFL_HTTP_TOKEN=gfl_your_token_here \
docker compose -f docker-compose.monitor.yml up -d
```

View logs from both processes:

```bash
docker compose -f docker-compose.monitor.yml logs -f
```

Stop and remove the container:

```bash
docker compose -f docker-compose.monitor.yml down
```

### AMD

```bash
GPUFL_HTTP_URL=https://your-backend/api/v1/events/ \
GPUFL_HTTP_TOKEN=gfl_your_token_here \
docker compose -f docker-compose.monitor.amd.yml up -d
```

View logs:

```bash
docker compose -f docker-compose.monitor.amd.yml logs -f
```

Stop:

```bash
docker compose -f docker-compose.monitor.amd.yml down
```

---

## Running with docker run

### NVIDIA

```bash
docker run -d \
  --name gpufl-monitor \
  --gpus all \
  --restart unless-stopped \
  -e GPUFL_HTTP_URL=https://your-backend/api/v1/events/ \
  -e GPUFL_HTTP_TOKEN=gfl_your_token_here \
  -v gpufl-cursor:/var/gpufl/monitor \
  gpufl/monitor:latest
```

### AMD

```bash
docker run -d \
  --name gpufl-monitor-amd \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  --restart unless-stopped \
  -e GPUFL_HTTP_URL=https://your-backend/api/v1/events/ \
  -e GPUFL_HTTP_TOKEN=gfl_your_token_here \
  -v gpufl-cursor-amd:/var/gpufl/monitor \
  gpufl/monitor-amd:latest
```

The named volume persists the agent's read cursor so it resumes from where it left off after a restart.

---

## Environment variables

### Daemon (`gpufl-monitor` C++ binary)

| Variable | Default | Description |
|---|---|---|
| `GPUFL_MONITOR_APP` | `gpufl-monitor` | App name tag written into every event |
| `GPUFL_MONITOR_LOG_DIR` | `/var/gpufl/monitor/session` | Directory where JSONL log files are written |
| `GPUFL_MONITOR_INTERVAL_MS` | `5000` | Sampling interval in milliseconds |

### Agent — log source

| Variable | Default | Description |
|---|---|---|
| `GPUFL_SOURCE_FOLDER` | same as `GPUFL_MONITOR_LOG_DIR` | Directory the agent watches for log files |
| `GPUFL_SOURCE_PREFIX` | `session` | Filename prefix of the log files (e.g. `session.device.log`) |
| `GPUFL_LOG_TYPES` | `device,scope,system` | Comma-separated log channels to tail |
| `GPUFL_CURSOR_FILE` | `/var/gpufl/monitor/cursor.json` | Path to the cursor file for resume-on-restart |

### Agent — HTTP publisher

| Variable | Default | Description |
|---|---|---|
| `GPUFL_PUBLISHER_TYPE` | `http` | Publisher backend: `http` or `kafka` |
| `GPUFL_HTTP_URL` | *(required)* | Backend ingest URL, e.g. `https://app.gpuflight.io/api/v1/events/` |
| `GPUFL_HTTP_TOKEN` | *(empty)* | Bearer token for the backend API |
| `GPUFL_HTTP_TIMEOUT_SEC` | `10` | HTTP request timeout in seconds |

### Agent — Kafka publisher (optional)

Set `GPUFL_PUBLISHER_TYPE=kafka` and configure:

| Variable | Default | Description |
|---|---|---|
| `GPUFL_KAFKA_BROKERS` | *(empty)* | Comma-separated broker list, e.g. `broker1:9092,broker2:9092` |
| `GPUFL_KAFKA_TOPIC_PREFIX` | `gpu-trace` | Topic prefix; channel is appended (e.g. `gpu-trace-device`) |
| `GPUFL_KAFKA_COMPRESSION` | `snappy` | Compression codec: `none`, `gzip`, `snappy`, `lz4`, `zstd` |
| `GPUFL_KAFKA_LINGER_MS` | `100` | Producer linger time in milliseconds |

---

## Verifying it works

Replace `gpufl-monitor` with `gpufl-monitor-amd` for the AMD container.

1. Check both processes are running:

```bash
docker exec gpufl-monitor supervisorctl status
# gpufl-monitor    RUNNING   pid 7,  uptime 0:01:23
# gpufl-agent      RUNNING   pid 12, uptime 0:01:22
```

2. Confirm log files are being written:

```bash
docker exec gpufl-monitor ls /var/gpufl/monitor/session/
# session.device.log  session.system.log
```

3. (AMD only) Verify ROCm sees the GPU inside the container:

```bash
docker exec gpufl-monitor-amd rocm-smi
```

4. Tail the container output to see events being shipped:

```bash
docker logs -f gpufl-monitor
```

5. Open the GPUFlight dashboard — the **Monitoring → Overview** page should show live GPU utilization within one sampling interval.

---

## Log file layout

The daemon writes one file per channel under `GPUFL_MONITOR_LOG_DIR`:

```
/var/gpufl/monitor/session/
  session.device.log   ← per-GPU metrics (util, mem, temp, power)
  session.system.log   ← host metrics (CPU %, RAM)
  session.scope.log    ← scope events (empty when profiling engine is None)
```

Each line is a JSON object. The agent reads these files, tracks its position in `cursor.json`, and POSTs each line to the backend as it arrives.

---

## Tuning the sampling interval

The default interval is 5 seconds. For higher-resolution monitoring reduce it — but be aware of increased backend ingest load:

```bash
-e GPUFL_MONITOR_INTERVAL_MS=1000   # 1 s — fine-grained
-e GPUFL_MONITOR_INTERVAL_MS=30000  # 30 s — low-overhead fleet monitoring
```
