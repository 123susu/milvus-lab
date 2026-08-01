"""Validated benchmark parameter construction and bounded execution."""

from __future__ import annotations

import json
import os
from contextlib import closing
from pathlib import Path
from time import monotonic, sleep
from typing import Any

import sqlite3
import yaml

from metrics.index_profiles import validate_index_parameters
from metrics.jobs import (
    ACTIVE_STATUSES,
    BenchmarkJobConflictError,
    BenchmarkJobManager,
    BenchmarkJobParameters,
)
from ..history import _optional_float, _parse_json_object, _readonly_connection
from ..models import (
    IndexParameterValue,
    TuningAgentBenchmarkConflictError,
    TuningAgentConfigurationError,
    TuningAgentDataError,
    TuningAgentError,
)

def _benchmark_profile(
    base_config_path: Path,
    command: str | None = None,
) -> dict[str, Any]:
    config_path = (
        base_config_path.parent / f"{command}.yml"
        if command
        else base_config_path
    )
    try:
        with config_path.open("r", encoding="utf-8") as file:
            document = yaml.safe_load(file) or {}
    except (OSError, yaml.YAMLError) as error:
        raise TuningAgentConfigurationError(
            f"无法读取 Agent Benchmark 基础配置：{error}"
        ) from error
    profile = document.get(command) if command else next(
        (
            value
            for key, value in document.items()
            if not str(key).startswith("_") and isinstance(value, dict)
        ),
        None,
    )
    if profile is None:
        raise TuningAgentConfigurationError(
            "Agent Benchmark 基础配置中没有 command 配置"
        )
    return profile


def build_benchmark_parameters(
    job_manager: BenchmarkJobManager,
    current_config: dict[str, Any],
    search_parameters: dict[str, IndexParameterValue],
) -> BenchmarkJobParameters:
    """Build one full benchmark while keeping index parameters fixed."""

    command = str(current_config["command"])
    profile = _benchmark_profile(job_manager.base_config_path, command)
    top_k = int(current_config["top_k"])
    allowed_names = set(current_config["allowed_search_parameters"])
    if set(search_parameters) != allowed_names:
        missing = sorted(allowed_names - set(search_parameters))
        extra = sorted(set(search_parameters) - allowed_names)
        details = []
        if missing:
            details.append("缺少 " + ", ".join(missing))
        if extra:
            details.append("不允许 " + ", ".join(extra))
        raise ValueError("搜索参数不完整：" + "；".join(details))
    complete_parameters = {
        **current_config["index_parameters"],
        **search_parameters,
    }
    normalized = validate_index_parameters(
        command,
        complete_parameters,
        top_k,
    )
    return BenchmarkJobParameters(
        command=command,
        uri=str(profile.get("uri", "http://localhost:19530")),
        num_shards=int(current_config["num_shards"]),
        replica_number=int(current_config["replica_number"]),
        case_type=str(current_config["case_type"]),
        drop_old=True,
        load=True,
        load_concurrency=int(current_config["load_concurrency"]),
        search_serial=True,
        search_concurrent=True,
        k=top_k,
        concurrency_duration=int(current_config["concurrency_duration"]),
        num_concurrency=(1,),
        concurrency_timeout=int(current_config["concurrency_timeout"]),
        index_parameters=normalized,
        db_label=str(current_config["db_label"]),
    )


def query_run_result(
    database_path: Path,
    run_id: str,
    case_index: int,
) -> dict[str, Any]:
    """Read the newly completed run that the existing collector saved."""

    sql = """
        SELECT
            br.run_id,
            br.case_index,
            br.command,
            br.index_type,
            br.case_type,
            br.metric_type,
            br.index_parameters_json,
            br.search_parameters_json,
            br.executed_stages_json,
            br.top_k,
            br.recall,
            br.ndcg,
            br.insert_duration_seconds,
            br.optimize_duration_seconds,
            br.load_duration_seconds,
            br.vector_index_memory_bytes,
            br.serial_latency_p99_ms,
            br.serial_latency_p95_ms,
            cs.concurrency,
            cs.qps,
            cs.latency_avg_ms,
            cs.latency_p95_ms,
            cs.latency_p99_ms
        FROM benchmark_runs AS br
        LEFT JOIN concurrency_stage_metrics AS cs
            ON cs.run_id = br.run_id
            AND cs.case_index = br.case_index
        WHERE br.run_id = ? AND br.case_index = ?
        ORDER BY cs.concurrency DESC, cs.stage_index DESC
        LIMIT 1
    """
    try:
        with closing(_readonly_connection(database_path)) as connection:
            row = connection.execute(sql, (run_id, case_index)).fetchone()
    except sqlite3.Error as error:
        raise TuningAgentDataError(f"读取新 Benchmark 结果失败：{error}") from error
    if row is None:
        raise TuningAgentDataError(f"新 Benchmark 结果不存在：{run_id}/{case_index}")
    memory_bytes = row["vector_index_memory_bytes"]
    return {
        "run_id": row["run_id"],
        "case_index": row["case_index"],
        "command": row["command"],
        "index_type": row["index_type"],
        "case_type": row["case_type"],
        "metric_type": row["metric_type"],
        "index_parameters": _parse_json_object(row["index_parameters_json"]),
        "search_parameters": _parse_json_object(row["search_parameters_json"]),
        "top_k": row["top_k"],
        "concurrency": row["concurrency"],
        "executed_stages": json.loads(row["executed_stages_json"] or "[]"),
        "recall": (
            _optional_float(row["recall"])
            if "search_serial"
            in json.loads(row["executed_stages_json"] or "[]")
            else None
        ),
        "ndcg": _optional_float(row["ndcg"]),
        "p99_ms": (
            _optional_float(row["latency_p99_ms"])
            if row["latency_p99_ms"] is not None
            else _optional_float(row["serial_latency_p99_ms"])
        ),
        "latency_avg_ms": _optional_float(row["latency_avg_ms"]),
        "qps": _optional_float(row["qps"]),
        "vector_index_memory_mib": (
            float(memory_bytes) / (1024 * 1024)
            if memory_bytes is not None and memory_bytes > 0
            else None
        ),
        "insert_seconds": _optional_float(row["insert_duration_seconds"]),
        "optimize_seconds": _optional_float(row["optimize_duration_seconds"]),
        "load_seconds": _optional_float(row["load_duration_seconds"]),
    }


class AgentBenchmarkExecutor:
    """Run serial and concurrent stages in one job against the retained Collection."""

    def __init__(
        self,
        job_manager: BenchmarkJobManager,
        database_path: Path,
        *,
        timeout_seconds: int | None = None,
        poll_seconds: float = 1.0,
    ) -> None:
        self.job_manager = job_manager
        self.database_path = database_path
        self.timeout_seconds = timeout_seconds or int(
            os.getenv("MILVUS_TUNING_AGENT_BENCHMARK_TIMEOUT", "7200")
        )
        self.poll_seconds = poll_seconds

    def _run_one(
        self,
        current_config: dict[str, Any],
        search_parameters: dict[str, IndexParameterValue],
        parameters: BenchmarkJobParameters,
        reason: str,
    ) -> dict[str, Any]:
        try:
            job = self.job_manager.submit(parameters, repetitions=1)
        except BenchmarkJobConflictError as error:
            raise TuningAgentBenchmarkConflictError(str(error)) from error

        deadline = monotonic() + self.timeout_seconds
        while job.status in ACTIVE_STATUSES:
            if monotonic() >= deadline:
                self.job_manager.cancel(job.job_id)
                raise TuningAgentError(
                    f"Agent Benchmark 超过 {self.timeout_seconds} 秒，已请求取消"
                )
            sleep(self.poll_seconds)
            job = self.job_manager.get(job.job_id)

        base_result: dict[str, Any] = {
            "job_id": job.job_id,
            "status": job.status,
            "reason": reason,
            "command": current_config["command"],
            "requested_search_parameters": search_parameters,
            "elapsed_seconds": self.job_manager.elapsed_seconds(job),
        }
        if job.status != "succeeded":
            return {
                **base_result,
                "error": job.error or f"Benchmark {job.status}",
            }
        if job.result_run_id is None or job.result_case_index is None:
            return {
                **base_result,
                "status": "failed",
                "error": "Benchmark 完成但没有关联到 SQLite 结果",
            }
        return {
            **base_result,
            **query_run_result(
                self.database_path,
                job.result_run_id,
                job.result_case_index,
            ),
        }

    def run(
        self,
        current_config: dict[str, Any],
        search_parameters: dict[str, IndexParameterValue],
        reason: str,
    ) -> dict[str, Any]:
        return self._run_one(
            current_config,
            search_parameters,
            build_benchmark_parameters(
                self.job_manager,
                current_config,
                search_parameters,
            ),
            reason,
        )

