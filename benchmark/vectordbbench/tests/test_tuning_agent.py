from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from metrics.models import BenchmarkRunMetrics, ConcurrencyStageMetrics
from metrics.repository import BenchmarkMetricsRepository
from metrics.tuning_agent import (
    build_benchmark_parameters,
    query_candidate_data,
    query_current_collection_config,
)


def benchmark_run(
    run_id: str,
    *,
    recall: float,
    p99_ms: float,
    memory_mib: float,
    m: int,
    ef_search: int,
    concurrency: int = 1,
) -> BenchmarkRunMetrics:
    return BenchmarkRunMetrics(
        run_id=run_id,
        case_index=0,
        task_label=run_id,
        status="ok",
        result_file="result.json",
        log_file="result.log",
        raw_result={},
        database="Milvus",
        db_label="test-cluster",
        case_type="Performance1536D50K",
        command="milvushnsw",
        index_type="HNSW",
        metric_type="COSINE",
        index_parameters={"m": m, "ef_construction": 128},
        search_parameters={"ef_search": ef_search},
        hnsw_m=m,
        hnsw_ef_construction=128,
        hnsw_ef_search=ef_search,
        top_k=100,
        num_shards=1,
        replica_number=1,
        load_concurrency=4,
        concurrency_duration_seconds=30,
        concurrency_timeout_seconds=3600,
        executed_stages=["search_serial", "search_concurrent"],
        recall=recall,
        insert_duration_seconds=10,
        optimize_duration_seconds=20,
        vector_index_memory_bytes=int(memory_mib * 1024 * 1024),
        concurrency_stages=[
            ConcurrencyStageMetrics(
                stage_index=0,
                concurrency=concurrency,
                latency_p99_ms=p99_ms,
            )
        ],
    )


class TuningAgentQueryTest(unittest.TestCase):
    def test_ranks_qualified_by_p99_and_keeps_near_misses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "metrics.sqlite3"
            repository = BenchmarkMetricsRepository(database)
            repository.save_all(
                [
                    benchmark_run(
                        "qualified-slow",
                        recall=0.98,
                        p99_ms=12,
                        memory_mib=300,
                        m=32,
                        ef_search=256,
                    ),
                    benchmark_run(
                        "qualified-fast",
                        recall=0.96,
                        p99_ms=8,
                        memory_mib=280,
                        m=24,
                        ef_search=192,
                    ),
                    benchmark_run(
                        "near-miss",
                        recall=0.94,
                        p99_ms=6,
                        memory_mib=260,
                        m=16,
                        ef_search=128,
                    ),
                ]
            )

            result = query_candidate_data(database, 0.95)

        self.assertEqual(result["configuration_count"], 3)
        self.assertEqual(result["qualified_count"], 2)
        self.assertEqual(
            result["qualified_candidates"][0]["index_parameters"]["m"],
            24,
        )
        self.assertAlmostEqual(
            result["near_misses"][0]["recall_mean"],
            0.94,
        )

    def test_reads_current_collection_config_from_latest_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "metrics.sqlite3"
            repository = BenchmarkMetricsRepository(database)
            repository.save_all(
                [
                    benchmark_run(
                        "current-index",
                        recall=0.96,
                        p99_ms=8,
                        memory_mib=280,
                        m=24,
                        ef_search=192,
                    )
                ]
            )

            result = query_current_collection_config(database)

        self.assertEqual(result["command"], "milvushnsw")
        self.assertEqual(result["index_parameters"]["m"], 24)
        self.assertEqual(result["search_parameters"], {"ef_search": 192})
        self.assertEqual(result["allowed_search_parameters"], ["ef_search"])

    def test_builds_search_only_single_concurrency_job(self) -> None:
        project_root = Path(__file__).resolve().parents[3]
        manager = SimpleNamespace(
            base_config_path=(
                project_root
                / "benchmark"
                / "vectordbbench"
                / "config"
                / "milvushnsw.yml"
            )
        )
        current_config = {
            "command": "milvushnsw",
            "case_type": "Performance1536D50K",
            "top_k": 100,
            "num_shards": 1,
            "replica_number": 1,
            "load_concurrency": 4,
            "concurrency_duration": 30,
            "concurrency_timeout": 3600,
            "db_label": "test-cluster",
            "index_parameters": {"m": 24, "ef_construction": 128},
            "allowed_search_parameters": ["ef_search"],
        }

        parameters = build_benchmark_parameters(
            manager,
            current_config,
            {"ef_search": 256},
        )

        self.assertFalse(parameters.drop_old)
        self.assertFalse(parameters.load)
        self.assertTrue(parameters.search_serial)
        self.assertTrue(parameters.search_concurrent)
        self.assertEqual(parameters.num_concurrency, (1,))
        self.assertEqual(parameters.command, "milvushnsw")
        self.assertEqual(parameters.index_parameters["m"], 24)
        self.assertEqual(parameters.index_parameters["ef_search"], 256)

    def test_rejects_build_parameter_from_agent_tool(self) -> None:
        project_root = Path(__file__).resolve().parents[3]
        manager = SimpleNamespace(
            base_config_path=(
                project_root
                / "benchmark"
                / "vectordbbench"
                / "config"
                / "milvushnsw.yml"
            )
        )
        current_config = {
            "command": "milvushnsw",
            "case_type": "Performance1536D50K",
            "top_k": 100,
            "num_shards": 1,
            "replica_number": 1,
            "load_concurrency": 4,
            "concurrency_duration": 30,
            "concurrency_timeout": 3600,
            "db_label": "test-cluster",
            "index_parameters": {"m": 24, "ef_construction": 128},
            "allowed_search_parameters": ["ef_search"],
        }

        with self.assertRaisesRegex(ValueError, "不允许 m"):
            build_benchmark_parameters(
                manager,
                current_config,
                {"ef_search": 256, "m": 16},
            )


if __name__ == "__main__":
    unittest.main()
