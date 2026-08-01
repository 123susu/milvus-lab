"""Data models for one VectorDBBench run and its concurrent stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class ConcurrencyStageMetrics:
    """Metrics collected for one concurrent search stage."""

    stage_index: int
    concurrency: int
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_seconds: float | None = None
    qps: float | None = None
    latency_avg_ms: float | None = None
    latency_p95_ms: float | None = None
    latency_p99_ms: float | None = None
    querynode_cpu_avg_cores: float | None = None
    querynode_cpu_peak_cores: float | None = None
    querynode_cpu_avg_percent: float | None = None
    querynode_cpu_peak_percent: float | None = None
    querynode_cpu_sample_count: int | None = None
    monitoring_error: str | None = None


@dataclass(slots=True)
class BenchmarkRunMetrics:
    """Metrics and configuration for one VectorDBBench case result."""

    run_id: str
    case_index: int
    task_label: str
    status: str
    result_file: str
    log_file: str
    raw_result: dict[str, Any]
    database: str
    db_label: str | None
    case_type: str
    command: str
    index_type: str
    metric_type: str
    index_parameters: dict[str, Any]
    search_parameters: dict[str, Any]
    hnsw_m: int | None
    hnsw_ef_construction: int | None
    hnsw_ef_search: int | None
    top_k: int
    num_shards: int | None
    replica_number: int | None
    load_concurrency: int | None
    concurrency_duration_seconds: int | None
    concurrency_timeout_seconds: int | None
    executed_stages: list[str] = field(default_factory=list)
    insert_duration_seconds: float | None = None
    optimize_duration_seconds: float | None = None
    load_duration_seconds: float | None = None
    max_load_count: int | None = None
    recall: float | None = None
    ndcg: float | None = None
    serial_latency_p95_ms: float | None = None
    serial_latency_p99_ms: float | None = None
    max_qps: float | None = None
    vector_index_memory_bytes: int | None = None
    vector_index_memory_collected_at: datetime | None = None
    concurrency_stages: list[ConcurrencyStageMetrics] = field(default_factory=list)
    monitoring_error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now().astimezone())
