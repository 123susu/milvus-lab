# Backend source

`src/` contains the MilvusTune backend and benchmark implementation.

- `vectordbbench/benchmark_metrics_api.py`: FastAPI entry point
- `vectordbbench/agent/`: Recall tuning workflow, configuration, prompts, and models
- `vectordbbench/agent/tools/`: model-facing benchmark tool implementation
- `vectordbbench/metrics/`: metrics collection, persistence, jobs, and tracing
- `vectordbbench/config/`: VectorDBBench and agent configuration
- `vectordbbench/run_*.ps1`: local benchmark runners
- `vectordbbench/tests/`: backend tests

Start the backend from the repository root:

```powershell
.\.venv-bench\Scripts\python.exe .\src\vectordbbench\benchmark_metrics_api.py
```
