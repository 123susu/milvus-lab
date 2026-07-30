"""Local FastAPI service for VectorDBBench metrics and CPU-index jobs."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
from collections import defaultdict
from contextlib import closing
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import quote, urlsplit

import uvicorn
from fastapi import FastAPI, HTTPException, Path as ApiPath, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from metrics.jobs import (
    BenchmarkJob,
    BenchmarkJobConflictError,
    BenchmarkJobManager,
    BenchmarkJobParameters,
)
from metrics.index_profiles import (
    CPU_INDEX_PROFILES,
    expand_index_matrix,
    public_profiles,
)
from metrics.tracing import (
    MilvusTraceSearchService,
    TraceSearchError,
    TraceSearchRequest as TraceServiceRequest,
)
from metrics.tuning_agent import (
    BenchmarkTuningAgent,
    TuningAgentConfigurationError,
    TuningAgentDataError,
    TuningAgentError,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE = (
    PROJECT_ROOT / "results" / "vectordbbench" / "benchmark_metrics.sqlite3"
)
BASE_CONFIG = (
    PROJECT_ROOT
    / "benchmark"
    / "vectordbbench"
    / "config"
    / "milvushnsw.yml"
)
BENCHMARK_RUNNER = (
    PROJECT_ROOT / "benchmark" / "vectordbbench" / "run_benchmark.ps1"
)
JOBS_ROOT = PROJECT_ROOT / "results" / "vectordbbench" / "jobs"
JAEGER_QUERY_URL = os.environ.get(
    "MILVUS_JAEGER_QUERY_URL", "http://127.0.0.1:16686"
)
TRACE_SEARCH_SERVICE: MilvusTraceSearchService | None = None


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthResponse(ApiModel):
    status: str
    database: str
    benchmark_run_count: int
    active_job_id: str | None


class ConcurrencyStageResponse(ApiModel):
    stage_index: int
    concurrency: int
    started_at: datetime | None
    finished_at: datetime | None
    duration_seconds: float | None
    qps: float | None
    latency_avg_ms: float | None
    latency_p95_ms: float | None
    latency_p99_ms: float | None
    querynode_cpu_avg_cores: float | None
    querynode_cpu_peak_cores: float | None
    querynode_cpu_avg_percent: float | None
    querynode_cpu_peak_percent: float | None
    querynode_cpu_sample_count: int | None
    monitoring_error: str | None


class BenchmarkResponse(ApiModel):
    configuration_key: str | None
    run_id: str
    case_index: int
    task_label: str
    status: str
    created_at: datetime
    case_type: str
    database: str
    db_label: str | None
    command: str
    index_type: str
    metric_type: str
    index_parameters: dict[str, object]
    search_parameters: dict[str, object]
    hnsw_m: int | None
    hnsw_ef_construction: int | None
    hnsw_ef_search: int | None
    top_k: int
    num_shards: int | None
    replica_number: int | None
    load_concurrency: int | None
    concurrency_duration_seconds: int | None
    concurrency_timeout_seconds: int | None
    insert_duration_seconds: float | None
    optimize_duration_seconds: float | None
    load_duration_seconds: float | None
    recall: float | None
    ndcg: float | None
    serial_latency_p95_ms: float | None
    serial_latency_p99_ms: float | None
    max_qps: float | None
    vector_index_memory_bytes: int | None
    vector_index_memory_mib: float | None
    vector_index_memory_collected_at: datetime | None
    monitoring_error: str | None
    stages: list[ConcurrencyStageResponse]


class BenchmarkListResponse(ApiModel):
    items: list[BenchmarkResponse]
    total: int
    limit: int
    offset: int


class MetricSummaryResponse(ApiModel):
    mean: float | None
    stddev: float | None
    minimum: float | None
    maximum: float | None
    sample_count: int


class BenchmarkAggregateResponse(ApiModel):
    configuration_key: str
    sample_count: int
    latest_created_at: datetime
    latest_run_id: str
    case_type: str
    db_label: str | None
    command: str
    index_type: str
    metric_type: str
    index_parameters: dict[str, object]
    search_parameters: dict[str, object]
    hnsw_m: int | None
    hnsw_ef_construction: int | None
    hnsw_ef_search: int | None
    top_k: int
    num_shards: int | None
    replica_number: int | None
    load_concurrency: int | None
    concurrency_duration_seconds: int | None
    concurrency_timeout_seconds: int | None
    executed_stages: list[str]
    stage_index: int
    concurrency: int
    qps: MetricSummaryResponse
    latency_avg_ms: MetricSummaryResponse
    latency_p95_ms: MetricSummaryResponse
    latency_p99_ms: MetricSummaryResponse
    recall: MetricSummaryResponse
    ndcg: MetricSummaryResponse
    serial_latency_p99_ms: MetricSummaryResponse
    insert_duration_seconds: MetricSummaryResponse
    optimize_duration_seconds: MetricSummaryResponse
    load_duration_seconds: MetricSummaryResponse
    querynode_cpu_avg_percent: MetricSummaryResponse
    querynode_cpu_peak_percent: MetricSummaryResponse
    vector_index_memory_mib: MetricSummaryResponse


class BenchmarkAggregateListResponse(ApiModel):
    items: list[BenchmarkAggregateResponse]
    total: int
    limit: int
    offset: int


class TraceSearchRequest(ApiModel):
    uri: str = Field(min_length=8, max_length=512)
    database: str = Field(
        default="default",
        min_length=1,
        max_length=255,
        pattern=r"^[A-Za-z0-9_]+$",
    )
    collection_name: str = Field(
        default="TraceDemo",
        min_length=1,
        max_length=255,
        pattern=r"^[A-Za-z0-9_]+$",
    )
    top_k: int = Field(default=10, ge=1, le=2048)

    @field_validator("uri")
    @classmethod
    def validate_http_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("must be an HTTP or HTTPS URL")
        return value


class TraceSpanResponse(ApiModel):
    span_id: str
    parent_span_id: str | None
    service: str
    operation: str
    start_offset_ms: float
    duration_ms: float
    depth: int
    error: bool


class TraceSearchResponse(ApiModel):
    trace_id: str
    jaeger_url: str
    collection_name: str
    vector_field: str
    top_k: int
    hit_count: int
    client_latency_ms: float
    total_duration_ms: float
    spans: list[TraceSpanResponse]


class TuningAgentRequest(ApiModel):
    recall_target: float = Field(gt=0, le=1)


class TuningAgentResponse(ApiModel):
    recall_target: float
    model: str
    answer: str
    tools_used: list[str]


class BenchmarkCommonParametersRequest(ApiModel):
    uri: str = Field(min_length=8, max_length=512)
    num_shards: int = Field(ge=1, le=64)
    replica_number: int = Field(ge=1, le=16)
    case_type: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z][A-Za-z0-9_]*$",
    )
    drop_old: bool
    load: bool
    load_concurrency: int = Field(ge=0, le=256)
    search_serial: bool
    search_concurrent: bool
    k: int = Field(ge=1, le=2048)
    concurrency_duration: int = Field(ge=1, le=86400)
    num_concurrency: list[Annotated[int, Field(ge=1, le=1024)]] = Field(
        min_length=1,
        max_length=32,
    )
    concurrency_timeout: int = Field(ge=-1, le=86400)
    db_label: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
    )

    @field_validator("uri")
    @classmethod
    def validate_http_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("must be an HTTP or HTTPS URL")
        return value

    @model_validator(mode="after")
    def validate_stages(self) -> "BenchmarkCommonParametersRequest":
        if not (self.load or self.search_serial or self.search_concurrent):
            raise ValueError("at least one load or search stage must be enabled")
        return self


class CpuIndexExperimentRequest(ApiModel):
    command: str
    parameters: BenchmarkCommonParametersRequest
    index_matrix: dict[str, list[Any]]
    repetitions: int = Field(ge=1, le=5)

    @field_validator("command")
    @classmethod
    def validate_command(cls, value: str) -> str:
        if value not in CPU_INDEX_PROFILES:
            raise ValueError("unsupported CPU index command")
        return value

    @model_validator(mode="after")
    def validate_matrix(self) -> "CpuIndexExperimentRequest":
        configurations = expand_index_matrix(
            self.command,
            self.index_matrix,
            self.parameters.k,
        )
        if len(configurations) * self.repetitions > 30:
            raise ValueError("experiment is limited to 30 total benchmark runs")
        return self


class BenchmarkJobParametersResponse(BenchmarkCommonParametersRequest):
    command: str
    index_parameters: dict[str, object]


class IndexParameterDefinitionResponse(ApiModel):
    name: str
    label: str
    kind: Literal["integer", "number", "boolean", "choice"]
    default: object
    minimum: float | None = None
    maximum: float | None = None
    options: list[object] | None = None
    description: str


class IndexProfileResponse(ApiModel):
    command: str
    label: str
    index_type: str
    parameters: list[IndexParameterDefinitionResponse]


class BenchmarkJobResponse(ApiModel):
    job_id: str
    status: str
    phase: str
    parameters: BenchmarkJobParametersResponse
    task_label: str
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    elapsed_seconds: float | None
    exit_code: int | None
    error: str | None
    result_run_id: str | None
    result_case_index: int | None
    repetitions: int
    configuration_count: int
    total_runs: int
    completed_runs: int
    current_run_number: int
    result_run_ids: list[str]
    log_tail: list[str]


def database_path() -> Path:
    configured = os.getenv("VDBBENCH_METRICS_DB")
    return Path(configured).resolve() if configured else DEFAULT_DATABASE


def connect_read_only() -> sqlite3.Connection:
    path = database_path()
    if not path.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Metrics database does not exist: {path}",
        )
    uri = f"file:{quote(path.as_posix(), safe='/:')}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        return connection
    except sqlite3.Error as error:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot open metrics database: {error}",
        ) from error


JOB_MANAGER = BenchmarkJobManager(
    project_root=PROJECT_ROOT,
    base_config_path=BASE_CONFIG,
    runner_path=BENCHMARK_RUNNER,
    database_path=DEFAULT_DATABASE,
    jobs_root=JOBS_ROOT,
)


def build_job_response(job: BenchmarkJob) -> BenchmarkJobResponse:
    return BenchmarkJobResponse(
        job_id=job.job_id,
        status=job.status,
        phase=job.phase,
        parameters=BenchmarkJobParametersResponse.model_validate(
            asdict(job.parameters)
        ),
        task_label=job.task_label_prefix,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        elapsed_seconds=JOB_MANAGER.elapsed_seconds(job),
        exit_code=job.exit_code,
        error=job.error,
        result_run_id=job.result_run_id,
        result_case_index=job.result_case_index,
        repetitions=job.repetitions,
        configuration_count=len(job.parameter_sets),
        total_runs=job.total_runs,
        completed_runs=job.completed_runs,
        current_run_number=job.current_run_number,
        result_run_ids=list(job.result_run_ids),
        log_tail=JOB_MANAGER.log_tail(job.job_id),
    )


RUN_COLUMNS = """
    configuration_key,
    run_id,
    case_index,
    task_label,
    status,
    created_at,
    case_type,
    database_name,
    db_label,
    command,
    index_type,
    metric_type,
    index_parameters_json,
    search_parameters_json,
    hnsw_m,
    hnsw_ef_construction,
    hnsw_ef_search,
    top_k,
    num_shards,
    replica_number,
    load_concurrency,
    concurrency_duration_seconds,
    concurrency_timeout_seconds,
    insert_duration_seconds,
    optimize_duration_seconds,
    load_duration_seconds,
    recall,
    ndcg,
    serial_latency_p95_ms,
    serial_latency_p99_ms,
    max_qps,
    vector_index_memory_bytes,
    vector_index_memory_collected_at,
    monitoring_error
"""

AGGREGATE_COLUMNS = """
    br.configuration_key,
    br.run_id,
    br.created_at,
    br.case_type,
    br.db_label,
    br.command,
    br.index_type,
    br.metric_type,
    br.index_parameters_json,
    br.search_parameters_json,
    br.hnsw_m,
    br.hnsw_ef_construction,
    br.hnsw_ef_search,
    br.top_k,
    br.num_shards,
    br.replica_number,
    br.load_concurrency,
    br.concurrency_duration_seconds,
    br.concurrency_timeout_seconds,
    br.executed_stages_json,
    br.recall,
    br.ndcg,
    br.serial_latency_p99_ms,
    br.insert_duration_seconds,
    br.optimize_duration_seconds,
    br.load_duration_seconds,
    br.vector_index_memory_bytes,
    cs.stage_index,
    cs.concurrency,
    cs.qps,
    cs.latency_avg_ms,
    cs.latency_p95_ms,
    cs.latency_p99_ms,
    cs.querynode_cpu_avg_percent,
    cs.querynode_cpu_peak_percent
"""


def summarize_metric(
    rows: list[dict[str, object]],
    column: str,
    scale: float = 1.0,
) -> MetricSummaryResponse:
    values = [
        float(row[column]) / scale
        for row in rows
        if row.get(column) is not None
    ]
    if not values:
        return MetricSummaryResponse(
            mean=None,
            stddev=None,
            minimum=None,
            maximum=None,
            sample_count=0,
        )
    return MetricSummaryResponse(
        mean=statistics.fmean(values),
        stddev=statistics.stdev(values) if len(values) > 1 else 0.0,
        minimum=min(values),
        maximum=max(values),
        sample_count=len(values),
    )


def build_aggregate(rows: list[dict[str, object]]) -> BenchmarkAggregateResponse:
    latest = rows[0]
    return BenchmarkAggregateResponse(
        configuration_key=str(latest["configuration_key"]),
        sample_count=len(rows),
        latest_created_at=str(latest["created_at"]),
        latest_run_id=str(latest["run_id"]),
        case_type=str(latest["case_type"]),
        db_label=latest["db_label"],
        command=str(latest["command"]),
        index_type=str(latest["index_type"]),
        metric_type=str(latest["metric_type"]),
        index_parameters=json.loads(str(latest["index_parameters_json"])),
        search_parameters=json.loads(str(latest["search_parameters_json"])),
        hnsw_m=latest["hnsw_m"],
        hnsw_ef_construction=latest["hnsw_ef_construction"],
        hnsw_ef_search=latest["hnsw_ef_search"],
        top_k=int(latest["top_k"]),
        num_shards=latest["num_shards"],
        replica_number=latest["replica_number"],
        load_concurrency=latest["load_concurrency"],
        concurrency_duration_seconds=latest["concurrency_duration_seconds"],
        concurrency_timeout_seconds=latest["concurrency_timeout_seconds"],
        executed_stages=json.loads(str(latest["executed_stages_json"])),
        stage_index=int(latest["stage_index"]),
        concurrency=int(latest["concurrency"]),
        qps=summarize_metric(rows, "qps"),
        latency_avg_ms=summarize_metric(rows, "latency_avg_ms"),
        latency_p95_ms=summarize_metric(rows, "latency_p95_ms"),
        latency_p99_ms=summarize_metric(rows, "latency_p99_ms"),
        recall=summarize_metric(rows, "recall"),
        ndcg=summarize_metric(rows, "ndcg"),
        serial_latency_p99_ms=summarize_metric(rows, "serial_latency_p99_ms"),
        insert_duration_seconds=summarize_metric(
            rows,
            "insert_duration_seconds",
        ),
        optimize_duration_seconds=summarize_metric(
            rows,
            "optimize_duration_seconds",
        ),
        load_duration_seconds=summarize_metric(rows, "load_duration_seconds"),
        querynode_cpu_avg_percent=summarize_metric(
            rows,
            "querynode_cpu_avg_percent",
        ),
        querynode_cpu_peak_percent=summarize_metric(
            rows,
            "querynode_cpu_peak_percent",
        ),
        vector_index_memory_mib=summarize_metric(
            rows,
            "vector_index_memory_bytes",
            1024 * 1024,
        ),
    )


def fetch_stages(
    connection: sqlite3.Connection,
    run_id: str,
    case_index: int,
) -> list[ConcurrencyStageResponse]:
    rows = connection.execute(
        """
        SELECT
            stage_index,
            concurrency,
            started_at,
            finished_at,
            duration_seconds,
            qps,
            latency_avg_ms,
            latency_p95_ms,
            latency_p99_ms,
            querynode_cpu_avg_cores,
            querynode_cpu_peak_cores,
            querynode_cpu_avg_percent,
            querynode_cpu_peak_percent,
            querynode_cpu_sample_count,
            monitoring_error
        FROM concurrency_stage_metrics
        WHERE run_id = ? AND case_index = ?
        ORDER BY stage_index
        """,
        (run_id, case_index),
    ).fetchall()
    return [ConcurrencyStageResponse.model_validate(dict(row)) for row in rows]


def build_benchmark(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
) -> BenchmarkResponse:
    values = dict(row)
    vector_bytes = values.pop("vector_index_memory_bytes")
    values["database"] = values.pop("database_name")
    values["index_parameters"] = json.loads(
        values.pop("index_parameters_json")
    )
    values["search_parameters"] = json.loads(
        values.pop("search_parameters_json")
    )
    values["vector_index_memory_bytes"] = vector_bytes
    values["vector_index_memory_mib"] = (
        vector_bytes / (1024 * 1024) if vector_bytes is not None else None
    )
    values["stages"] = fetch_stages(
        connection,
        values["run_id"],
        values["case_index"],
    )
    return BenchmarkResponse.model_validate(values)


app = FastAPI(
    title="Milvus Lab Benchmark Metrics API",
    version="1.3.0",
    description="Local API backed by VectorDBBench, Prometheus, and SQLite.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    with closing(connect_read_only()) as connection:
        try:
            count = connection.execute(
                "SELECT COUNT(*) FROM benchmark_runs"
            ).fetchone()[0]
        except sqlite3.Error as error:
            raise HTTPException(
                status_code=503,
                detail=f"Metrics schema is unavailable: {error}",
            ) from error
    return HealthResponse(
        status="ok",
        database=str(database_path()),
        benchmark_run_count=count,
        active_job_id=JOB_MANAGER.active_job_id,
    )


@app.get("/api/benchmarks", response_model=BenchmarkListResponse)
def list_benchmarks(
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> BenchmarkListResponse:
    with closing(connect_read_only()) as connection:
        try:
            total = connection.execute(
                "SELECT COUNT(*) FROM benchmark_runs"
            ).fetchone()[0]
            rows = connection.execute(
                f"""
                SELECT {RUN_COLUMNS}
                FROM benchmark_runs
                ORDER BY created_at DESC, run_id DESC, case_index
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
            items = [build_benchmark(connection, row) for row in rows]
        except sqlite3.Error as error:
            raise HTTPException(
                status_code=503,
                detail=f"Cannot query benchmark metrics: {error}",
            ) from error
    return BenchmarkListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
    )


@app.get(
    "/api/benchmark-aggregates",
    response_model=BenchmarkAggregateListResponse,
)
def list_benchmark_aggregates(
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> BenchmarkAggregateListResponse:
    with closing(connect_read_only()) as connection:
        try:
            rows = connection.execute(
                f"""
                SELECT {AGGREGATE_COLUMNS}
                FROM benchmark_runs AS br
                INNER JOIN concurrency_stage_metrics AS cs
                    ON cs.run_id = br.run_id
                    AND cs.case_index = br.case_index
                WHERE br.configuration_key IS NOT NULL
                ORDER BY br.created_at DESC, br.run_id DESC, cs.stage_index
                """
            ).fetchall()
        except sqlite3.Error as error:
            raise HTTPException(
                status_code=503,
                detail=f"Cannot aggregate benchmark metrics: {error}",
            ) from error

    grouped: dict[tuple[str, int, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        values = dict(row)
        key = (
            str(values["configuration_key"]),
            int(values["stage_index"]),
            int(values["concurrency"]),
        )
        grouped[key].append(values)
    aggregates = [build_aggregate(group) for group in grouped.values()]
    total = len(aggregates)
    return BenchmarkAggregateListResponse(
        items=aggregates[offset : offset + limit],
        total=total,
        limit=limit,
        offset=offset,
    )


@app.get(
    "/api/benchmarks/{run_id}",
    response_model=BenchmarkResponse,
)
def get_benchmark(
    run_id: Annotated[
        str,
        ApiPath(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
    ],
    case_index: Annotated[int, Query(ge=0)] = 0,
) -> BenchmarkResponse:
    with closing(connect_read_only()) as connection:
        try:
            row = connection.execute(
                f"""
                SELECT {RUN_COLUMNS}
                FROM benchmark_runs
                WHERE run_id = ? AND case_index = ?
                """,
                (run_id, case_index),
            ).fetchone()
            if row is None:
                raise HTTPException(
                    status_code=404,
                    detail="Benchmark run was not found",
                )
            return build_benchmark(connection, row)
        except sqlite3.Error as error:
            raise HTTPException(
                status_code=503,
                detail=f"Cannot query benchmark metrics: {error}",
            ) from error


@app.post(
    "/api/benchmark-jobs",
    response_model=BenchmarkJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_benchmark_job(
    experiment: CpuIndexExperimentRequest,
) -> BenchmarkJobResponse:
    try:
        parameter_sets: list[BenchmarkJobParameters] = []
        common_values = experiment.parameters.model_dump()
        common_values["num_concurrency"] = tuple(
            experiment.parameters.num_concurrency
        )
        for index_parameters in expand_index_matrix(
            experiment.command,
            experiment.index_matrix,
            experiment.parameters.k,
        ):
            parameter_sets.append(
                BenchmarkJobParameters(
                    command=experiment.command,
                    index_parameters=index_parameters,
                    **common_values,
                )
            )
        job = JOB_MANAGER.submit(parameter_sets, experiment.repetitions)
        return build_job_response(job)
    except BenchmarkJobConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except (OSError, TypeError, ValueError) as error:
        raise HTTPException(
            status_code=500,
            detail=f"Cannot prepare benchmark job: {error}",
        ) from error


@app.get(
    "/api/benchmark-profiles",
    response_model=list[IndexProfileResponse],
)
def list_benchmark_profiles() -> list[IndexProfileResponse]:
    return [
        IndexProfileResponse.model_validate(profile)
        for profile in public_profiles()
    ]


@app.get(
    "/api/benchmark-jobs",
    response_model=list[BenchmarkJobResponse],
)
def list_benchmark_jobs() -> list[BenchmarkJobResponse]:
    return [build_job_response(job) for job in JOB_MANAGER.list()]


@app.get(
    "/api/benchmark-jobs/{job_id}",
    response_model=BenchmarkJobResponse,
)
def get_benchmark_job(
    job_id: Annotated[
        str,
        ApiPath(min_length=32, max_length=32, pattern=r"^[a-f0-9]{32}$"),
    ],
) -> BenchmarkJobResponse:
    try:
        return build_job_response(JOB_MANAGER.get(job_id))
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Benchmark job was not found") from error


@app.post(
    "/api/benchmark-jobs/{job_id}/cancel",
    response_model=BenchmarkJobResponse,
)
def cancel_benchmark_job(
    job_id: Annotated[
        str,
        ApiPath(min_length=32, max_length=32, pattern=r"^[a-f0-9]{32}$"),
    ],
) -> BenchmarkJobResponse:
    try:
        return build_job_response(JOB_MANAGER.cancel(job_id))
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Benchmark job was not found") from error


@app.post(
    "/api/tuning-agent/recommend",
    response_model=TuningAgentResponse,
)
def recommend_index_configuration(
    request: TuningAgentRequest,
) -> TuningAgentResponse:
    try:
        result = BenchmarkTuningAgent(database_path()).recommend(
            request.recall_target
        )
        return TuningAgentResponse(
            recall_target=request.recall_target,
            **result,
        )
    except TuningAgentConfigurationError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except TuningAgentDataError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except TuningAgentError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@app.post(
    "/api/trace-search",
    response_model=TraceSearchResponse,
)
def run_trace_search(request: TraceSearchRequest) -> TraceSearchResponse:
    global TRACE_SEARCH_SERVICE
    try:
        if TRACE_SEARCH_SERVICE is None:
            TRACE_SEARCH_SERVICE = MilvusTraceSearchService(
                jaeger_query_url=JAEGER_QUERY_URL,
            )
        result = TRACE_SEARCH_SERVICE.search(
            TraceServiceRequest(
                uri=request.uri,
                database=request.database,
                collection_name=request.collection_name,
                top_k=request.top_k,
            )
        )
        return TraceSearchResponse.model_validate(result)
    except TraceSearchError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=f"Trace 服务不可用：{error}",
        ) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--database", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.database:
        resolved_database = args.database.resolve()
        os.environ["VDBBENCH_METRICS_DB"] = str(resolved_database)
        JOB_MANAGER.database_path = resolved_database
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
