"""Parse VectorDBBench artifacts and collect QueryNode Prometheus metrics."""

from __future__ import annotations

import json
import math
import os
import re
import urllib.parse
import urllib.request
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from .models import BenchmarkRunMetrics, ConcurrencyStageMetrics


STAGE_START_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) .*"
    r"Syncing all process and start concurrency search, concurrency=(?P<concurrency>\d+)"
)
STAGE_END_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) .*"
    r"End search in concurrency (?P<concurrency>\d+)"
)

QUERYNODE_CPU_PROMQL = (
    "sum(rate(container_cpu_usage_seconds_total{"
    'job="cadvisor",'
    'container_label_com_docker_compose_project="milvus-cluster-local",'
    'name=~".*querynode.*"'
    "}[{rate_window}]))"
)
VECTOR_INDEX_MEMORY_PROMQL = (
    "sum(internal_cache_loaded_bytes{"
    'job="milvus-cluster",'
    'component="querynode",'
    'data_type="vector_index",'
    'location="memory"'
    "})"
)


def parse_timezone_offset(value: Any) -> timezone:
    text = str(value or "+08:00").strip()
    match = re.fullmatch(r"([+-])(\d{2}):(\d{2})", text)
    if not match:
        raise ValueError("_metrics.timezone must use an offset such as +08:00")
    sign = 1 if match.group(1) == "+" else -1
    offset = timedelta(hours=int(match.group(2)), minutes=int(match.group(3))) * sign
    return timezone(offset)


def _optional_float(value: Any, multiplier: float = 1.0) -> float | None:
    return None if value is None else float(value) * multiplier


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _metric_at(values: list[Any], index: int, multiplier: float = 1.0) -> float | None:
    return _optional_float(values[index], multiplier) if index < len(values) else None


def _normalized_index_parameters(
    index_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    index_type = str(index_config.get("index", "")).upper()
    if index_type in {"HNSW", "HNSW_SQ", "HNSW_PQ", "HNSW_PRQ"}:
        index_parameters = {
            "m": _optional_int(index_config.get("M")),
            "ef_construction": _optional_int(
                index_config.get("efConstruction")
            ),
        }
        for name in ("sq_type", "nbits", "nrq", "refine", "refine_type"):
            if index_config.get(name) is not None:
                index_parameters[name] = index_config[name]
        search_parameters = {
            "ef_search": _optional_int(index_config.get("ef")),
        }
        if index_config.get("refine_k") is not None:
            search_parameters["refine_k"] = _optional_float(
                index_config["refine_k"]
            )
        return (
            index_parameters,
            search_parameters,
        )
    if index_type in {"IVF_FLAT", "IVF_SQ8"}:
        return (
            {"nlist": _optional_int(index_config.get("nlist"))},
            {"nprobe": _optional_int(index_config.get("nprobe"))},
        )
    return {}, {}


class PrometheusClient:
    """Minimal Prometheus HTTP API client using only the Python standard library."""

    def __init__(self, base_url: str, timeout_seconds: int = 15) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _request(self, path: str, parameters: dict[str, Any]) -> dict[str, Any]:
        query = urllib.parse.urlencode(parameters)
        endpoint = f"{self.base_url}{path}?{query}"
        with urllib.request.urlopen(endpoint, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if payload.get("status") != "success":
            raise RuntimeError(f"Prometheus query failed: {payload.get('error', 'unknown error')}")
        return payload

    def query_range(
        self,
        promql: str,
        started_at: datetime,
        finished_at: datetime,
        step_seconds: int,
    ) -> list[float]:
        payload = self._request(
            "/api/v1/query_range",
            {
                "query": promql,
                "start": started_at.timestamp(),
                "end": finished_at.timestamp(),
                "step": step_seconds,
            },
        )
        values: list[float] = []
        for series in payload.get("data", {}).get("result", []):
            for _, raw_value in series.get("values", []):
                value = float(raw_value)
                if math.isfinite(value):
                    values.append(value)
        return values

    def query(self, promql: str, tz: timezone) -> tuple[float, datetime]:
        payload = self._request("/api/v1/query", {"query": promql})
        values: list[float] = []
        timestamps: list[float] = []
        for sample in payload.get("data", {}).get("result", []):
            timestamp, raw_value = sample.get("value", [None, None])
            if timestamp is None or raw_value is None:
                continue
            value = float(raw_value)
            if math.isfinite(value):
                values.append(value)
                timestamps.append(float(timestamp))
        if not values:
            raise RuntimeError("Prometheus returned no samples")
        return sum(values), datetime.fromtimestamp(max(timestamps), tz=tz)


class BenchmarkMetricsCollector:
    """Collect a complete benchmark record from JSON, log, and Prometheus."""

    def __init__(
        self,
        prometheus_client: PrometheusClient,
        tz: timezone,
        querynode_cpu_limit_cores: float,
        query_range_step_seconds: int = 1,
        rate_window: str = "10s",
    ) -> None:
        if querynode_cpu_limit_cores <= 0:
            raise ValueError("querynode_cpu_limit_cores must be positive")
        if query_range_step_seconds <= 0:
            raise ValueError("query_range_step_seconds must be positive")
        if not re.fullmatch(r"[1-9]\d*[smhd]", rate_window):
            raise ValueError("rate_window must look like 10s, 1m, or 1h")
        self.prometheus = prometheus_client
        self.tz = tz
        self.querynode_cpu_limit_cores = querynode_cpu_limit_cores
        self.query_range_step_seconds = query_range_step_seconds
        self.cpu_promql = QUERYNODE_CPU_PROMQL.replace("{rate_window}", rate_window)

    @staticmethod
    def load_config(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as file:
            document = yaml.safe_load(file) or {}
        if not isinstance(document, dict):
            raise ValueError("benchmark configuration must be a YAML object")
        return document

    def _parse_log_timestamp(self, value: str) -> datetime:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S,%f").replace(tzinfo=self.tz)

    def parse_stage_windows(self, log_path: Path) -> list[dict[str, Any]]:
        active: dict[int, deque[datetime]] = defaultdict(deque)
        windows: list[dict[str, Any]] = []
        for line in log_path.read_text(encoding="utf-8").splitlines():
            start_match = STAGE_START_RE.search(line)
            if start_match:
                concurrency = int(start_match.group("concurrency"))
                active[concurrency].append(
                    self._parse_log_timestamp(start_match.group("timestamp"))
                )
                continue

            end_match = STAGE_END_RE.search(line)
            if not end_match:
                continue
            concurrency = int(end_match.group("concurrency"))
            if not active[concurrency]:
                continue
            started_at = active[concurrency].popleft()
            finished_at = self._parse_log_timestamp(end_match.group("timestamp"))
            windows.append(
                {
                    "concurrency": concurrency,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "duration_seconds": (finished_at - started_at).total_seconds(),
                }
            )
        return sorted(windows, key=lambda item: item["started_at"])

    def _collect_cpu(self, stage: ConcurrencyStageMetrics) -> None:
        if stage.started_at is None or stage.finished_at is None:
            stage.monitoring_error = "完整的并发 Stage 开始/结束时间窗不存在"
            return
        try:
            samples = self.prometheus.query_range(
                self.cpu_promql,
                stage.started_at,
                stage.finished_at,
                self.query_range_step_seconds,
            )
            if not samples:
                raise RuntimeError("Prometheus returned no QueryNode CPU samples")
            avg_cores = sum(samples) / len(samples)
            peak_cores = max(samples)
            stage.querynode_cpu_avg_cores = avg_cores
            stage.querynode_cpu_peak_cores = peak_cores
            stage.querynode_cpu_avg_percent = (
                avg_cores / self.querynode_cpu_limit_cores * 100
            )
            stage.querynode_cpu_peak_percent = (
                peak_cores / self.querynode_cpu_limit_cores * 100
            )
            stage.querynode_cpu_sample_count = len(samples)
        except Exception as error:
            stage.monitoring_error = str(error)

    def _collect_vector_index_memory(self) -> tuple[int | None, datetime | None, str | None]:
        try:
            value, collected_at = self.prometheus.query(
                VECTOR_INDEX_MEMORY_PROMQL,
                self.tz,
            )
            return int(round(value)), collected_at, None
        except Exception as error:
            return None, None, str(error)

    def collect(
        self,
        config_path: Path,
        result_path: Path,
        log_path: Path,
    ) -> list[BenchmarkRunMetrics]:
        config = self.load_config(config_path)
        with result_path.open("r", encoding="utf-8") as file:
            result_document = json.load(file)

        configured_case = next(
            (
                value
                for key, value in config.items()
                if not key.startswith("_") and isinstance(value, dict)
            ),
            {},
        )
        configured_command = next(
            (
                key
                for key, value in config.items()
                if not key.startswith("_") and isinstance(value, dict)
            ),
            "",
        )
        windows_by_concurrency: dict[int, deque[dict[str, Any]]] = defaultdict(deque)
        for window in self.parse_stage_windows(log_path):
            windows_by_concurrency[window["concurrency"]].append(window)

        vector_bytes, vector_collected_at, vector_error = (
            self._collect_vector_index_memory()
        )
        collected_runs: list[BenchmarkRunMetrics] = []

        for case_index, case_result in enumerate(result_document.get("results", [])):
            metrics = case_result.get("metrics", {})
            task_config = case_result.get("task_config", {})
            db_config = task_config.get("db_config", {})
            index_config = task_config.get("db_case_config", {})
            case_config = task_config.get("case_config", {})
            concurrency_config = case_config.get("concurrency_search_config", {})
            index_parameters, search_parameters = (
                _normalized_index_parameters(index_config)
            )
            concurrencies = metrics.get("conc_num_list", [])
            stages: list[ConcurrencyStageMetrics] = []

            for stage_index, concurrency in enumerate(concurrencies):
                window = (
                    windows_by_concurrency[int(concurrency)].popleft()
                    if windows_by_concurrency[int(concurrency)]
                    else {}
                )
                stage = ConcurrencyStageMetrics(
                    stage_index=stage_index,
                    concurrency=int(concurrency),
                    started_at=window.get("started_at"),
                    finished_at=window.get("finished_at"),
                    duration_seconds=window.get("duration_seconds"),
                    qps=_metric_at(metrics.get("conc_qps_list", []), stage_index),
                    latency_avg_ms=_metric_at(
                        metrics.get("conc_latency_avg_list", []), stage_index, 1000
                    ),
                    latency_p95_ms=_metric_at(
                        metrics.get("conc_latency_p95_list", []), stage_index, 1000
                    ),
                    latency_p99_ms=_metric_at(
                        metrics.get("conc_latency_p99_list", []), stage_index, 1000
                    ),
                )
                self._collect_cpu(stage)
                stages.append(stage)

            run = BenchmarkRunMetrics(
                run_id=str(result_document["run_id"]),
                case_index=case_index,
                task_label=str(result_document.get("task_label", "")),
                status=str(case_result.get("label", "")),
                result_file=str(result_path.resolve()),
                log_file=str(log_path.resolve()),
                raw_result=result_document,
                database=str(task_config.get("db", "")),
                db_label=db_config.get("db_label"),
                case_type=str(
                    configured_case.get("case_type", case_config.get("case_id", ""))
                ),
                command=configured_command,
                index_type=str(index_config.get("index", "")),
                metric_type=str(index_config.get("metric_type", "")),
                index_parameters=index_parameters,
                search_parameters=search_parameters,
                hnsw_m=_optional_int(index_config.get("M")),
                hnsw_ef_construction=_optional_int(
                    index_config.get("efConstruction")
                ),
                hnsw_ef_search=_optional_int(index_config.get("ef")),
                top_k=int(case_config.get("k", 0)),
                num_shards=_optional_int(configured_case.get("num_shards")),
                replica_number=_optional_int(
                    configured_case.get("replica_number")
                ),
                load_concurrency=_optional_int(task_config.get("load_concurrency")),
                concurrency_duration_seconds=_optional_int(
                    concurrency_config.get("concurrency_duration")
                ),
                concurrency_timeout_seconds=_optional_int(
                    concurrency_config.get("concurrency_timeout")
                ),
                executed_stages=[str(stage) for stage in task_config.get("stages", [])],
                insert_duration_seconds=_optional_float(
                    metrics.get("insert_duration")
                ),
                optimize_duration_seconds=_optional_float(
                    metrics.get("optimize_duration")
                ),
                load_duration_seconds=_optional_float(metrics.get("load_duration")),
                max_load_count=_optional_int(metrics.get("max_load_count")),
                recall=_optional_float(metrics.get("recall")),
                ndcg=_optional_float(metrics.get("ndcg")),
                serial_latency_p95_ms=_optional_float(
                    metrics.get("serial_latency_p95"), 1000
                ),
                serial_latency_p99_ms=_optional_float(
                    metrics.get("serial_latency_p99"), 1000
                ),
                max_qps=_optional_float(metrics.get("qps")),
                vector_index_memory_bytes=vector_bytes,
                vector_index_memory_collected_at=vector_collected_at,
                concurrency_stages=stages,
                monitoring_error=vector_error,
            )
            collected_runs.append(run)

        if not collected_runs:
            raise ValueError("VectorDBBench result contains no case results")
        return collected_runs


def prometheus_client_from_config(config: dict[str, Any]) -> PrometheusClient:
    metrics_config = config.get("_metrics", {})
    if not isinstance(metrics_config, dict):
        raise ValueError("_metrics must be a YAML object")
    base_url = str(
        os.getenv("VDBBENCH_PROMETHEUS_URL")
        or metrics_config.get("prometheus_url")
        or "http://localhost:9090"
    ).strip()
    return PrometheusClient(
        base_url=base_url,
        timeout_seconds=int(metrics_config.get("timeout_seconds", 15)),
    )
