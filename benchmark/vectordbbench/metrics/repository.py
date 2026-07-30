"""SQLite repository for benchmark run metrics."""

from __future__ import annotations

import json
import hashlib
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import BenchmarkRunMetrics, ConcurrencyStageMetrics


def _iso(value: datetime | None) -> str | None:
    return value.isoformat(timespec="milliseconds") if value is not None else None


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


class BenchmarkMetricsRepository:
    """Create and update normalized benchmark records in SQLite."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def connect(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def initialize(self, connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS benchmark_runs (
                run_id TEXT NOT NULL,
                case_index INTEGER NOT NULL,
                task_label TEXT NOT NULL,
                status TEXT NOT NULL,
                result_file TEXT NOT NULL,
                log_file TEXT NOT NULL,
                raw_result_json TEXT NOT NULL,
                database_name TEXT NOT NULL,
                db_label TEXT,
                case_type TEXT NOT NULL,
                command TEXT,
                index_type TEXT NOT NULL,
                metric_type TEXT NOT NULL,
                index_parameters_json TEXT,
                search_parameters_json TEXT,
                hnsw_m INTEGER,
                hnsw_ef_construction INTEGER,
                hnsw_ef_search INTEGER,
                top_k INTEGER NOT NULL,
                num_shards INTEGER,
                replica_number INTEGER,
                load_concurrency INTEGER,
                concurrency_duration_seconds INTEGER,
                concurrency_timeout_seconds INTEGER,
                executed_stages_json TEXT NOT NULL,
                configuration_key TEXT,
                insert_duration_seconds REAL,
                optimize_duration_seconds REAL,
                load_duration_seconds REAL,
                max_load_count INTEGER,
                recall REAL,
                ndcg REAL,
                serial_latency_p95_ms REAL,
                serial_latency_p99_ms REAL,
                max_qps REAL,
                vector_index_memory_bytes INTEGER,
                vector_index_memory_collected_at TEXT,
                monitoring_error TEXT,
                created_at TEXT NOT NULL,
                PRIMARY KEY (run_id, case_index)
            );

            CREATE TABLE IF NOT EXISTS concurrency_stage_metrics (
                run_id TEXT NOT NULL,
                case_index INTEGER NOT NULL,
                stage_index INTEGER NOT NULL,
                concurrency INTEGER NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                duration_seconds REAL,
                qps REAL,
                latency_avg_ms REAL,
                latency_p95_ms REAL,
                latency_p99_ms REAL,
                querynode_cpu_avg_cores REAL,
                querynode_cpu_peak_cores REAL,
                querynode_cpu_avg_percent REAL,
                querynode_cpu_peak_percent REAL,
                querynode_cpu_sample_count INTEGER,
                monitoring_error TEXT,
                PRIMARY KEY (run_id, case_index, stage_index),
                FOREIGN KEY (run_id, case_index)
                    REFERENCES benchmark_runs (run_id, case_index)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_benchmark_runs_task_label
                ON benchmark_runs (task_label);
            CREATE INDEX IF NOT EXISTS idx_concurrency_stage_metrics_concurrency
                ON concurrency_stage_metrics (concurrency);
            """
        )
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(benchmark_runs)")
        }
        if "concurrency_timeout_seconds" not in columns:
            connection.execute(
                "ALTER TABLE benchmark_runs "
                "ADD COLUMN concurrency_timeout_seconds INTEGER"
            )
        if "configuration_key" not in columns:
            connection.execute(
                "ALTER TABLE benchmark_runs ADD COLUMN configuration_key TEXT"
            )
        if "command" not in columns:
            connection.execute(
                "ALTER TABLE benchmark_runs ADD COLUMN command TEXT"
            )
        if "index_parameters_json" not in columns:
            connection.execute(
                "ALTER TABLE benchmark_runs ADD COLUMN index_parameters_json TEXT"
            )
        if "search_parameters_json" not in columns:
            connection.execute(
                "ALTER TABLE benchmark_runs ADD COLUMN search_parameters_json TEXT"
            )
        configuration_index = next(
            (
                row
                for row in connection.execute(
                    "PRAGMA index_list(benchmark_runs)"
                )
                if row[1] == "idx_benchmark_runs_configuration_key"
            ),
            None,
        )
        if configuration_index is not None and configuration_index[2]:
            connection.execute(
                "DROP INDEX idx_benchmark_runs_configuration_key"
            )
        self._backfill_configuration_keys(connection)
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_benchmark_runs_configuration_key "
            "ON benchmark_runs (configuration_key)"
        )

    @staticmethod
    def _configuration_key_from_values(
        values: dict[str, Any],
        concurrencies: list[int],
    ) -> str:
        payload = {
            "database_name": values.get("database_name"),
            "db_label": values.get("db_label"),
            "case_type": values.get("case_type"),
            "command": values.get("command"),
            "index_type": values.get("index_type"),
            "metric_type": values.get("metric_type"),
            "index_parameters": values.get("index_parameters", {}),
            "search_parameters": values.get("search_parameters", {}),
            "hnsw_m": values.get("hnsw_m"),
            "hnsw_ef_construction": values.get("hnsw_ef_construction"),
            "hnsw_ef_search": values.get("hnsw_ef_search"),
            "top_k": values.get("top_k"),
            "num_shards": values.get("num_shards"),
            "replica_number": values.get("replica_number"),
            "load_concurrency": values.get("load_concurrency"),
            "concurrency_duration_seconds": values.get(
                "concurrency_duration_seconds"
            ),
            "concurrency_timeout_seconds": values.get(
                "concurrency_timeout_seconds"
            ),
            "executed_stages": values.get("executed_stages", []),
            "num_concurrency": concurrencies,
        }
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _backfill_configuration_keys(
        self,
        connection: sqlite3.Connection,
    ) -> None:
        rows = connection.execute(
            """
            SELECT
                run_id, case_index, raw_result_json, database_name, db_label,
                case_type, command, index_type, metric_type,
                index_parameters_json, search_parameters_json, hnsw_m,
                hnsw_ef_construction, hnsw_ef_search, top_k, num_shards,
                replica_number, load_concurrency,
                concurrency_duration_seconds, concurrency_timeout_seconds,
                executed_stages_json, created_at
            FROM benchmark_runs
            WHERE configuration_key IS NULL
               OR command IS NULL
               OR index_parameters_json IS NULL
               OR search_parameters_json IS NULL
            ORDER BY created_at DESC, rowid DESC
            """
        ).fetchall()
        for row in rows:
            values = dict(zip(
                (
                    "run_id", "case_index", "raw_result_json", "database_name",
                    "db_label", "case_type", "command", "index_type",
                    "metric_type", "index_parameters_json",
                    "search_parameters_json", "hnsw_m",
                    "hnsw_ef_construction", "hnsw_ef_search",
                    "top_k", "num_shards", "replica_number",
                    "load_concurrency", "concurrency_duration_seconds",
                    "concurrency_timeout_seconds", "executed_stages_json",
                    "created_at",
                ),
                row,
            ))
            try:
                raw_result = json.loads(values["raw_result_json"])
                raw_case = raw_result.get("results", [])[int(values["case_index"])]
                index_config = (
                    raw_case.get("task_config", {})
                    .get("db_case_config", {})
                )
                concurrency_config = (
                    raw_case.get("task_config", {})
                    .get("case_config", {})
                    .get("concurrency_search_config", {})
                )
            except (IndexError, TypeError, ValueError, json.JSONDecodeError):
                index_config = {}
                concurrency_config = {}
            index_type = str(values["index_type"]).upper()
            command = values["command"] or {
                "HNSW": "milvushnsw",
                "HNSW_SQ": "milvushnswsq",
                "HNSW_PQ": "milvushnswpq",
                "HNSW_PRQ": "milvushnswprq",
                "IVF_FLAT": "milvusivfflat",
                "IVF_SQ8": "milvusivfsq8",
                "AUTOINDEX": "milvusautoindex",
                "FLAT": "milvusflat",
            }.get(index_type, "")
            if values["index_parameters_json"] is not None:
                index_parameters = json.loads(
                    values["index_parameters_json"]
                )
            elif index_type in {"HNSW", "HNSW_SQ", "HNSW_PQ", "HNSW_PRQ"}:
                index_parameters = {
                    "m": values["hnsw_m"],
                    "ef_construction": values["hnsw_ef_construction"],
                }
            elif index_type in {"IVF_FLAT", "IVF_SQ8"}:
                index_parameters = {
                    "nlist": _optional_int(index_config.get("nlist"))
                }
            else:
                index_parameters = {}
            if values["search_parameters_json"] is not None:
                search_parameters = json.loads(
                    values["search_parameters_json"]
                )
            elif index_type in {"HNSW", "HNSW_SQ", "HNSW_PQ", "HNSW_PRQ"}:
                search_parameters = {
                    "ef_search": values["hnsw_ef_search"]
                }
            elif index_type in {"IVF_FLAT", "IVF_SQ8"}:
                search_parameters = {
                    "nprobe": _optional_int(index_config.get("nprobe"))
                }
            else:
                search_parameters = {}
            values["command"] = command
            values["index_parameters"] = index_parameters
            values["search_parameters"] = search_parameters
            timeout = values["concurrency_timeout_seconds"]
            if timeout is None:
                timeout = concurrency_config.get("concurrency_timeout")
                connection.execute(
                    """
                    UPDATE benchmark_runs
                    SET concurrency_timeout_seconds = ?
                    WHERE run_id = ? AND case_index = ?
                    """,
                    (timeout, values["run_id"], values["case_index"]),
                )
            values["concurrency_timeout_seconds"] = timeout
            values["executed_stages"] = json.loads(
                values["executed_stages_json"]
            )
            concurrencies = [
                int(stage[0])
                for stage in connection.execute(
                    """
                    SELECT concurrency
                    FROM concurrency_stage_metrics
                    WHERE run_id = ? AND case_index = ?
                    ORDER BY stage_index
                    """,
                    (values["run_id"], values["case_index"]),
                )
            ]
            if not concurrencies:
                concurrencies = [
                    int(value)
                    for value in concurrency_config.get("num_concurrency", [])
                ]
            configuration_key = self._configuration_key_from_values(
                values,
                concurrencies,
            )
            connection.execute(
                """
                UPDATE benchmark_runs
                SET configuration_key = ?,
                    command = ?,
                    index_parameters_json = ?,
                    search_parameters_json = ?
                WHERE run_id = ? AND case_index = ?
                """,
                (
                    configuration_key,
                    command,
                    json.dumps(
                        index_parameters,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    json.dumps(
                        search_parameters,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    values["run_id"],
                    values["case_index"],
                ),
            )

    @classmethod
    def _run_values(cls, metrics: BenchmarkRunMetrics) -> dict[str, Any]:
        values = {
            "run_id": metrics.run_id,
            "case_index": metrics.case_index,
            "task_label": metrics.task_label,
            "status": metrics.status,
            "result_file": metrics.result_file,
            "log_file": metrics.log_file,
            "raw_result_json": json.dumps(
                metrics.raw_result, ensure_ascii=False, separators=(",", ":")
            ),
            "database_name": metrics.database,
            "db_label": metrics.db_label,
            "case_type": metrics.case_type,
            "command": metrics.command,
            "index_type": metrics.index_type,
            "metric_type": metrics.metric_type,
            "index_parameters_json": json.dumps(
                metrics.index_parameters,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "search_parameters_json": json.dumps(
                metrics.search_parameters,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "hnsw_m": metrics.hnsw_m,
            "hnsw_ef_construction": metrics.hnsw_ef_construction,
            "hnsw_ef_search": metrics.hnsw_ef_search,
            "top_k": metrics.top_k,
            "num_shards": metrics.num_shards,
            "replica_number": metrics.replica_number,
            "load_concurrency": metrics.load_concurrency,
            "concurrency_duration_seconds": metrics.concurrency_duration_seconds,
            "concurrency_timeout_seconds": metrics.concurrency_timeout_seconds,
            "executed_stages_json": json.dumps(
                metrics.executed_stages, ensure_ascii=False
            ),
            "insert_duration_seconds": metrics.insert_duration_seconds,
            "optimize_duration_seconds": metrics.optimize_duration_seconds,
            "load_duration_seconds": metrics.load_duration_seconds,
            "max_load_count": metrics.max_load_count,
            "recall": metrics.recall,
            "ndcg": metrics.ndcg,
            "serial_latency_p95_ms": metrics.serial_latency_p95_ms,
            "serial_latency_p99_ms": metrics.serial_latency_p99_ms,
            "max_qps": metrics.max_qps,
            "vector_index_memory_bytes": metrics.vector_index_memory_bytes,
            "vector_index_memory_collected_at": _iso(
                metrics.vector_index_memory_collected_at
            ),
            "monitoring_error": metrics.monitoring_error,
            "created_at": _iso(metrics.created_at),
        }
        values["executed_stages"] = metrics.executed_stages
        values["index_parameters"] = metrics.index_parameters
        values["search_parameters"] = metrics.search_parameters
        values["configuration_key"] = cls._configuration_key_from_values(
            values,
            [stage.concurrency for stage in metrics.concurrency_stages],
        )
        return values

    @staticmethod
    def _stage_values(
        metrics: BenchmarkRunMetrics,
        stage: ConcurrencyStageMetrics,
    ) -> dict[str, Any]:
        return {
            "run_id": metrics.run_id,
            "case_index": metrics.case_index,
            "stage_index": stage.stage_index,
            "concurrency": stage.concurrency,
            "started_at": _iso(stage.started_at),
            "finished_at": _iso(stage.finished_at),
            "duration_seconds": stage.duration_seconds,
            "qps": stage.qps,
            "latency_avg_ms": stage.latency_avg_ms,
            "latency_p95_ms": stage.latency_p95_ms,
            "latency_p99_ms": stage.latency_p99_ms,
            "querynode_cpu_avg_cores": stage.querynode_cpu_avg_cores,
            "querynode_cpu_peak_cores": stage.querynode_cpu_peak_cores,
            "querynode_cpu_avg_percent": stage.querynode_cpu_avg_percent,
            "querynode_cpu_peak_percent": stage.querynode_cpu_peak_percent,
            "querynode_cpu_sample_count": stage.querynode_cpu_sample_count,
            "monitoring_error": stage.monitoring_error,
        }

    def save_all(self, runs: list[BenchmarkRunMetrics]) -> None:
        with closing(self.connect()) as connection, connection:
            self.initialize(connection)
            for metrics in runs:
                run_values = self._run_values(metrics)
                connection.execute(
                    """
                    INSERT INTO benchmark_runs (
                        run_id, case_index, task_label, status, result_file,
                        log_file, raw_result_json, database_name, db_label,
                        case_type, command, index_type, metric_type,
                        index_parameters_json, search_parameters_json, hnsw_m,
                        hnsw_ef_construction, hnsw_ef_search, top_k, num_shards,
                        replica_number, load_concurrency,
                        concurrency_duration_seconds,
                        concurrency_timeout_seconds, executed_stages_json,
                        configuration_key,
                        insert_duration_seconds, optimize_duration_seconds,
                        load_duration_seconds, max_load_count, recall, ndcg,
                        serial_latency_p95_ms, serial_latency_p99_ms, max_qps,
                        vector_index_memory_bytes,
                        vector_index_memory_collected_at, monitoring_error,
                        created_at
                    ) VALUES (
                        :run_id, :case_index, :task_label, :status, :result_file,
                        :log_file, :raw_result_json, :database_name, :db_label,
                        :case_type, :command, :index_type, :metric_type,
                        :index_parameters_json, :search_parameters_json, :hnsw_m,
                        :hnsw_ef_construction, :hnsw_ef_search, :top_k,
                        :num_shards, :replica_number, :load_concurrency,
                        :concurrency_duration_seconds,
                        :concurrency_timeout_seconds, :executed_stages_json,
                        :configuration_key,
                        :insert_duration_seconds, :optimize_duration_seconds,
                        :load_duration_seconds, :max_load_count, :recall, :ndcg,
                        :serial_latency_p95_ms, :serial_latency_p99_ms, :max_qps,
                        :vector_index_memory_bytes,
                        :vector_index_memory_collected_at, :monitoring_error,
                        :created_at
                    )
                    ON CONFLICT (run_id, case_index) DO UPDATE SET
                        task_label = excluded.task_label,
                        status = excluded.status,
                        result_file = excluded.result_file,
                        log_file = excluded.log_file,
                        raw_result_json = excluded.raw_result_json,
                        database_name = excluded.database_name,
                        db_label = excluded.db_label,
                        case_type = excluded.case_type,
                        command = excluded.command,
                        index_type = excluded.index_type,
                        metric_type = excluded.metric_type,
                        index_parameters_json = excluded.index_parameters_json,
                        search_parameters_json = excluded.search_parameters_json,
                        hnsw_m = excluded.hnsw_m,
                        hnsw_ef_construction = excluded.hnsw_ef_construction,
                        hnsw_ef_search = excluded.hnsw_ef_search,
                        top_k = excluded.top_k,
                        num_shards = excluded.num_shards,
                        replica_number = excluded.replica_number,
                        load_concurrency = excluded.load_concurrency,
                        concurrency_duration_seconds =
                            excluded.concurrency_duration_seconds,
                        concurrency_timeout_seconds =
                            excluded.concurrency_timeout_seconds,
                        executed_stages_json = excluded.executed_stages_json,
                        configuration_key = excluded.configuration_key,
                        insert_duration_seconds = excluded.insert_duration_seconds,
                        optimize_duration_seconds = excluded.optimize_duration_seconds,
                        load_duration_seconds = excluded.load_duration_seconds,
                        max_load_count = excluded.max_load_count,
                        recall = excluded.recall,
                        ndcg = excluded.ndcg,
                        serial_latency_p95_ms = excluded.serial_latency_p95_ms,
                        serial_latency_p99_ms = excluded.serial_latency_p99_ms,
                        max_qps = excluded.max_qps,
                        vector_index_memory_bytes =
                            excluded.vector_index_memory_bytes,
                        vector_index_memory_collected_at =
                            excluded.vector_index_memory_collected_at,
                        monitoring_error = excluded.monitoring_error,
                        created_at = excluded.created_at
                    """,
                    run_values,
                )
                connection.execute(
                    """
                    DELETE FROM concurrency_stage_metrics
                    WHERE run_id = ? AND case_index = ?
                    """,
                    (metrics.run_id, metrics.case_index),
                )
                connection.executemany(
                    """
                    INSERT INTO concurrency_stage_metrics (
                        run_id, case_index, stage_index, concurrency, started_at,
                        finished_at, duration_seconds, qps, latency_avg_ms,
                        latency_p95_ms, latency_p99_ms,
                        querynode_cpu_avg_cores, querynode_cpu_peak_cores,
                        querynode_cpu_avg_percent, querynode_cpu_peak_percent,
                        querynode_cpu_sample_count, monitoring_error
                    ) VALUES (
                        :run_id, :case_index, :stage_index, :concurrency,
                        :started_at, :finished_at, :duration_seconds, :qps,
                        :latency_avg_ms, :latency_p95_ms, :latency_p99_ms,
                        :querynode_cpu_avg_cores, :querynode_cpu_peak_cores,
                        :querynode_cpu_avg_percent,
                        :querynode_cpu_peak_percent,
                        :querynode_cpu_sample_count, :monitoring_error
                    )
                    """,
                    [
                        self._stage_values(metrics, stage)
                        for stage in metrics.concurrency_stages
                    ],
                )
