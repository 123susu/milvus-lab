"""Read-only access to benchmark history and retained collection state."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .models import TuningAgentDataError

def _readonly_connection(database_path: Path) -> sqlite3.Connection:
    if not database_path.is_file():
        raise TuningAgentDataError(f"SQLite 数据库不存在：{database_path}")
    uri = f"file:{quote(database_path.resolve().as_posix(), safe='/:')}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        return connection
    except sqlite3.Error as error:
        raise TuningAgentDataError(f"无法只读打开 SQLite：{error}") from error


def _parse_json_object(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _candidate_from_row(row: sqlite3.Row) -> dict[str, Any]:
    memory_mib = row["memory_mib"]
    return {
        "configuration_key": row["configuration_key"],
        "command": row["command"],
        "index_type": row["index_type"],
        "case_type": row["case_type"],
        "metric_type": row["metric_type"],
        "top_k": row["top_k"],
        "concurrency": row["concurrency"],
        "index_parameters": _parse_json_object(row["index_parameters_json"]),
        "search_parameters": _parse_json_object(row["search_parameters_json"]),
        "sample_count": row["sample_count"],
        "recall_mean": row["recall_mean"],
        "p99_ms_mean": row["p99_ms_mean"],
        "vector_index_memory_mib_mean": (
            None if memory_mib is None or memory_mib <= 0 else memory_mib
        ),
        "insert_seconds_mean": row["insert_seconds_mean"],
        "optimize_seconds_mean": row["optimize_seconds_mean"],
    }


def query_candidate_data(
    database_path: Path,
    recall_target: float,
    *,
    qualified_limit: int = 100,
    near_miss_limit: int = 100,
) -> dict[str, Any]:
    """Read aggregate history for the LangGraph preprocessing node."""

    if not 0 < recall_target <= 1:
        raise ValueError("recall_target 必须大于 0 且小于等于 1")

    sql = """
        WITH stage_aggregates AS (
            SELECT
                br.configuration_key,
                br.command,
                br.index_type,
                br.case_type,
                br.metric_type,
                br.index_parameters_json,
                br.search_parameters_json,
                br.top_k,
                cs.stage_index,
                cs.concurrency,
                COUNT(*) AS sample_count,
                AVG(br.recall) AS recall_mean,
                AVG(cs.latency_p99_ms) AS p99_ms_mean,
                AVG(br.vector_index_memory_bytes) / 1048576.0 AS memory_mib,
                AVG(br.insert_duration_seconds) AS insert_seconds_mean,
                AVG(br.optimize_duration_seconds) AS optimize_seconds_mean
            FROM benchmark_runs AS br
            INNER JOIN concurrency_stage_metrics AS cs
                ON cs.run_id = br.run_id
                AND cs.case_index = br.case_index
            WHERE br.configuration_key IS NOT NULL
                AND br.recall IS NOT NULL
                AND br.executed_stages_json LIKE '%search_serial%'
            GROUP BY
                br.configuration_key,
                cs.stage_index,
                cs.concurrency
        ),
        highest_concurrency AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY configuration_key
                    ORDER BY concurrency DESC, stage_index DESC
                ) AS stage_rank
            FROM stage_aggregates
        )
        SELECT *
        FROM highest_concurrency
        WHERE stage_rank = 1
    """
    try:
        with closing(_readonly_connection(database_path)) as connection:
            rows = connection.execute(sql).fetchall()
    except sqlite3.Error as error:
        raise TuningAgentDataError(f"查询 benchmark 聚合数据失败：{error}") from error

    candidates = [_candidate_from_row(row) for row in rows]
    qualified = [
        item for item in candidates if item["recall_mean"] >= recall_target
    ]
    qualified.sort(
        key=lambda item: (
            item["p99_ms_mean"] is None,
            item["p99_ms_mean"] or float("inf"),
            item["vector_index_memory_mib_mean"] is None,
            item["vector_index_memory_mib_mean"] or float("inf"),
        )
    )
    near_misses = [
        item for item in candidates if item["recall_mean"] < recall_target
    ]
    near_misses.sort(
        key=lambda item: (
            -item["recall_mean"],
            item["p99_ms_mean"] is None,
            item["p99_ms_mean"] or float("inf"),
        )
    )
    return {
        "recall_target": recall_target,
        "comparison_scope": (
            "每个 configuration_key 的最高并发 Stage；"
            "同配置的多次 Run 使用均值"
        ),
        "configuration_count": len(candidates),
        "qualified_count": len(qualified),
        "qualified_candidates": qualified[:qualified_limit],
        "near_misses": near_misses[:near_miss_limit],
    }


def query_current_collection_config(database_path: Path) -> dict[str, Any]:
    """Infer the currently retained VDBBench index from the latest raw run."""

    sql = """
        SELECT
            command,
            index_type,
            case_type,
            metric_type,
            index_parameters_json,
            search_parameters_json,
            top_k,
            num_shards,
            replica_number,
            load_concurrency,
            concurrency_duration_seconds,
            concurrency_timeout_seconds,
            db_label,
            created_at,
            run_id
        FROM benchmark_runs
        WHERE command IS NOT NULL
        ORDER BY created_at DESC, run_id DESC, case_index DESC
        LIMIT 1
    """
    try:
        with closing(_readonly_connection(database_path)) as connection:
            row = connection.execute(sql).fetchone()
    except sqlite3.Error as error:
        raise TuningAgentDataError(
            f"读取当前 VDBBench 索引配置失败：{error}"
        ) from error
    if row is None:
        raise TuningAgentDataError(
            "SQLite 中没有可用于判断当前 VDBBench 索引的记录"
        )
    return {
        "source": "SQLite 中最近一次 Run；要求 VDBBench Collection 此后未被外部重建",
        "run_id": row["run_id"],
        "created_at": row["created_at"],
        "command": row["command"],
        "index_type": row["index_type"],
        "case_type": row["case_type"],
        "metric_type": row["metric_type"],
        "index_parameters": _parse_json_object(row["index_parameters_json"]),
        "search_parameters": _parse_json_object(row["search_parameters_json"]),
        "top_k": int(row["top_k"]),
        "num_shards": int(row["num_shards"] or 1),
        "replica_number": int(row["replica_number"] or 1),
        "load_concurrency": int(row["load_concurrency"] or 0),
        "concurrency_duration": int(
            row["concurrency_duration_seconds"] or 30
        ),
        "concurrency_timeout": int(
            row["concurrency_timeout_seconds"] or 3600
        ),
        "db_label": row["db_label"] or "agent-local-cluster",
        "allowed_search_parameters": sorted(
            _parse_json_object(row["search_parameters_json"])
        ),
    }


