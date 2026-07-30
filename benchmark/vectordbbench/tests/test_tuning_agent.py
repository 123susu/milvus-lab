from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from metrics.models import BenchmarkRunMetrics, ConcurrencyStageMetrics
from metrics.repository import BenchmarkMetricsRepository
from metrics.tuning_agent import query_candidate_data


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
        executed_stages=["search_concurrent"],
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


if __name__ == "__main__":
    unittest.main()
