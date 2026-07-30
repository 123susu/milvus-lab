# VectorDBBench Milvus baseline

This benchmark uses an isolated Python environment.

## Environment

- VectorDBBench: `1.0.22`
- PyMilvus: `2.6.17`
- Milvus URI: `http://localhost:19530`
- Authentication: disabled
- Milvus deployment: local distributed Cluster (`v2.6.21`)
- Cluster service limits: `8 CPU`, `10.5 GiB memory` in total
- Benchmark case: `Performance1536D50K`
- Index: HNSW (`M=16`, `efConstruction=128`, `ef=128`)
- Top K: `100`
- Concurrent clients: `1` (`30s`)

The selected case contains 50,000 real 1,536-dimensional text embeddings,
query vectors, and exact-neighbor ground truth.

## Run

Each database/index subcommand has its own YAML configuration. The current
HNSW profile lives in:

```text
benchmark/vectordbbench/config/milvushnsw.yml
```

Validate a profile without executing the benchmark:

```powershell
.\benchmark\vectordbbench\run_benchmark.ps1 -Command milvushnsw -DryRun
```

Run the configured HNSW profile:

```powershell
.\benchmark\vectordbbench\run_benchmark.ps1 -Command milvushnsw
```

## Monitoring

Start the monitoring stack before a benchmark run:

```powershell
docker compose -f .\deployments\monitoring\docker-compose.yml up -d
```

Open the provisioned Grafana dashboard:

```text
http://localhost:3001/d/milvus-cluster-local/milvus-cluster-local
```

### Single-query Trace test

The local dashboard includes a **single-query trace test** when the FastAPI
backend is enabled. It:

1. reads one existing vector from the selected Collection;
2. executes one read-only Milvus Search;
3. resolves the matching Proxy Search trace from local Jaeger;
4. displays every returned span on a shared waterfall timeline.

The default Collection is `TraceDemo`. On its first use, the API creates this
small, isolated HNSW Collection from deterministic text-derived vectors. It
does not modify or delete the `VDBBench` experiment Collection. A different
Collection selected in the UI must already exist, contain data, and be loaded.
Jaeger is available at `http://localhost:16686`.

It samples all Milvus Cluster components every second and shows component
health plus per-container CPU and memory. Prometheus target status is available
at:

```text
http://localhost:9090/targets
```

The YAML `task_label` is used as a prefix. At runtime the runner appends
`HHmmssfff`, for example `hnsw-performance-cluster-local-214530127`. Because
VectorDBBench already includes the date in the filename, this time suffix
prevents repeated runs on the same day from overwriting each other.

By default, `-Command <name>` loads `config/<name>.yml`. For example, a future
`-Command milvusivfflat` profile belongs in `config/milvusivfflat.yml`. A
different path can be supplied explicitly with `-ConfigFile`.

Datasets are cached under `data/vectordbbench/`; results, reports, and logs are
written under `results/vectordbbench/Milvus/`. Both generated-data roots are
excluded from Git.
The runner forces Python and PowerShell native-process output to UTF-8, so the
log files should be opened as UTF-8 rather than GBK.
After a successful run, Milvus result JSON files are automatically formatted
as UTF-8 with two-space indentation.

Each invocation has an independent UTF-8 log using the same filename stem as
its result:

```text
result_20260726_hnsw-performance-cluster-local-214530127_milvus.json
result_20260726_hnsw-performance-cluster-local-214530127_milvus.log
```

## SQLite metric collection

Each benchmark configuration can contain its own `_metrics` section:

```yaml
_metrics:
  prometheus_url: http://localhost:9090
  timezone: "+08:00"
  querynode_cpu_limit_cores: 2
  query_range_step_seconds: 1
  rate_window: 10s
  timeout_seconds: 15
```

After a successful benchmark, the runner saves structured metrics to:

```text
results/vectordbbench/benchmark_metrics.sqlite3
```

The `benchmark_runs` table contains configuration, load/index duration,
Recall, nDCG, serial latency, maximum QPS, the complete source JSON, and an
instant QueryNode vector-index memory sample. The
`concurrency_stage_metrics` table contains each concurrency stage's QPS,
latency, exact log window, and QueryNode CPU average/peak values.

Concurrent stage windows begin at `Syncing all process and start concurrency
search` and end at `End search in concurrency`. CPU is queried over that range.
Vector-index memory is queried once when the SQLite record is created.

Set `VDBBENCH_PROMETHEUS_URL` to override the YAML URL. If Prometheus is
temporarily unavailable, the VectorDBBench metrics are still saved; unavailable
monitoring values remain SQL `NULL` with the error stored alongside them.
Saving the same `run_id` and case again updates the existing row and its stages.
Every VectorDBBench execution is retained as an independent raw run. A stable
configuration key groups runs with the same database label, case, index/search
parameters, load settings, enabled stages, concurrency list, duration, and
timeout. Generated task labels, timestamps, result values, and monitoring
samples do not affect grouping. The aggregate API reports count, mean, sample
standard deviation, minimum, and maximum without deleting the underlying runs.

## Local metrics API

Install the API dependencies into the benchmark environment:

```powershell
.\.venv-bench\Scripts\python.exe -m pip install -r `
  .\benchmark\vectordbbench\api-requirements.txt
```

Start the local FastAPI service:

```powershell
.\.venv-bench\Scripts\python.exe `
  .\benchmark\vectordbbench\benchmark_metrics_api.py
```

The service binds to `127.0.0.1:8765` by default and exposes:

```text
GET /api/health
GET /api/benchmarks?limit=20&offset=0
GET /api/benchmark-aggregates?limit=20&offset=0
GET /api/benchmarks/{run_id}?case_index=0
GET /api/benchmark-profiles
POST /api/benchmark-jobs
GET /api/benchmark-jobs
GET /api/benchmark-jobs/{job_id}
POST /api/benchmark-jobs/{job_id}/cancel
POST /api/tuning-agent/recommend
GET /docs
```

The Recall tuning endpoint is a small Deep Agents agent running on LangGraph.
Its only benchmark-data tool is a constrained, read-only SQLite aggregate
query; it cannot execute arbitrary SQL, start a benchmark, or update records.
The agent reuses the OpenAI-compatible Qwen configuration that was previously
used by `run_benchmark.ps1`: `qwen-plus`, the Aliyun Bailian compatible-mode
endpoint, and the `DASHSCOPE_API_KEY` environment variable. Set the key before
starting the API:

```powershell
$env:DASHSCOPE_API_KEY = "<your-api-key>"
```

The defaults can be overridden with `MILVUS_TUNING_AGENT_MODEL`,
`MILVUS_TUNING_AGENT_BASE_URL`, and
`MILVUS_TUNING_AGENT_API_KEY_ENV`. The older `VDBBENCH_LLM_MODEL` and
`VDBBENCH_LLM_BASE_URL` variables are also accepted. Submit a target as a
decimal:

```json
{
  "recall_target": 0.95
}
```

Create a CPU-index matrix job with:

```json
{
  "command": "milvusivfflat",
  "parameters": {
    "uri": "http://localhost:19530",
    "num_shards": 1,
    "replica_number": 1,
    "case_type": "Performance1536D50K",
    "drop_old": true,
    "load": true,
    "load_concurrency": 4,
    "search_serial": true,
    "search_concurrent": true,
    "k": 100,
    "concurrency_duration": 30,
    "num_concurrency": [1],
    "concurrency_timeout": 3600,
    "db_label": "local-cluster-8c10_5g-2_6_21"
  },
  "index_matrix": {
    "nlist": [128, 256],
    "nprobe": [8, 16, 32]
  },
  "repetitions": 3
}
```

The API supports `milvushnsw`, `milvushnswsq`, `milvushnswpq`,
`milvushnswprq`, `milvusivfflat`, `milvusivfsq8`, `milvusautoindex`, and
`milvusflat`. It runs the Cartesian product of the selected index parameter
lists sequentially, repeating each configuration from 1 to 5 times. Numeric,
boolean, and enum parameter lists are supported. One request
is limited to 30 benchmark executions. Only one benchmark job can run at a
time. Every execution gets an independent generated configuration and log under
`results/vectordbbench/jobs/<job_id>/run-NNN/`. The source
profiles under `config/` are never overwritten. Job state is kept in API memory,
while every completed raw measurement is appended to SQLite by the existing
collector.

Use `--database <path>` to select another SQLite file. Benchmark queries open
SQLite in read-only/query-only mode. Browser requests are accepted only from
localhost or 127.0.0.1 origins.

VectorDBBench uses the collection name `VDBBench`. Its required `--drop-old`
option only replaces that benchmark collection; it does not affect collections
with other names. Starting a job from the web page therefore rebuilds
`VDBBench`.

The current Cluster topology and per-service resource limits are versioned in:

```text
deployments/milvus-cluster/docker-compose.yml
```
