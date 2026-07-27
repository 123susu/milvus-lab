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
- Concurrent clients: `1, 5, 10, 20` (`10s` each)

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
its result. A completed run can therefore have:

```text
result_20260726_hnsw-performance-cluster-local-214530127_milvus.json
result_20260726_hnsw-performance-cluster-local-214530127_milvus.md
result_20260726_hnsw-performance-cluster-local-214530127_milvus.log
```

## LLM report

Each benchmark configuration can contain its own `_report` section:

```yaml
_report:
  enabled: false
  base_url: ""
  model: ""
  api_key_env: DASHSCOPE_API_KEY
  language: 简体中文
  temperature: 0.2
  timeout_seconds: 120
```

Fill `base_url` and `model`, then set the API key in the environment variable
named by `api_key_env`. Do not put the API key in YAML. Environment variables
`VDBBENCH_LLM_BASE_URL` and `VDBBENCH_LLM_MODEL` can override the YAML values.

LLM report generation is currently disabled. Set `enabled: true` only when an
automatic Markdown report is required.

After a successful benchmark, the runner sends the newly produced JSON to an
OpenAI-compatible `chat/completions` endpoint and writes a UTF-8 Markdown file
next to it using the same filename. A missing or failed LLM configuration does
not invalidate the completed benchmark result.

VectorDBBench uses the collection name `VDBBench`. Its required `--drop-old`
option only replaces that benchmark collection; it does not affect collections
with other names.

The current Cluster topology and per-service resource limits are versioned in:

```text
deployments/milvus-cluster/docker-compose.yml
```
