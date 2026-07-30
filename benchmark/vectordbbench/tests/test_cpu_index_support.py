from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from typing import Any

import yaml

from metrics.collector import _normalized_index_parameters
from metrics.index_profiles import expand_index_matrix
from metrics.jobs import BenchmarkJobManager, BenchmarkJobParameters
from metrics.models import BenchmarkRunMetrics, ConcurrencyStageMetrics
from metrics.repository import BenchmarkMetricsRepository


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_ROOT = PROJECT_ROOT / "benchmark" / "vectordbbench"


def common_parameters(
    command: str,
    index_parameters: dict[str, Any],
) -> BenchmarkJobParameters:
    return BenchmarkJobParameters(
        command=command,
        uri="http://localhost:19530",
        num_shards=1,
        replica_number=1,
        case_type="Performance1536D50K",
        drop_old=True,
        load=True,
        load_concurrency=4,
        search_serial=True,
        search_concurrent=True,
        k=100,
        concurrency_duration=30,
        num_concurrency=(1,),
        concurrency_timeout=3600,
        index_parameters=index_parameters,
        db_label="test-cluster",
    )


class CpuIndexSupportTest(unittest.TestCase):
    def test_normalizes_result_index_parameters(self) -> None:
        self.assertEqual(
            _normalized_index_parameters({
                "index": "IVF_FLAT",
                "nlist": 128,
                "nprobe": 16,
            }),
            ({"nlist": 128}, {"nprobe": 16}),
        )
        self.assertEqual(
            _normalized_index_parameters({"index": "FLAT"}),
            ({}, {}),
        )
        self.assertEqual(
            _normalized_index_parameters({
                "index": "HNSW_SQ",
                "M": 16,
                "efConstruction": 128,
                "ef": 128,
                "sq_type": "SQ8",
                "refine": True,
                "refine_type": "FP32",
                "refine_k": 2.0,
            }),
            (
                {
                    "m": 16,
                    "ef_construction": 128,
                    "sq_type": "SQ8",
                    "refine": True,
                    "refine_type": "FP32",
                },
                {"ef_search": 128, "refine_k": 2.0},
            ),
        )

    def test_expands_and_validates_each_matrix(self) -> None:
        self.assertEqual(
            len(expand_index_matrix(
                "milvushnsw",
                {
                    "m": [16, 32],
                    "ef_construction": [128],
                    "ef_search": [100, 200],
                },
                100,
            )),
            4,
        )
        self.assertEqual(
            expand_index_matrix(
                "milvusivfflat",
                {"nlist": [128], "nprobe": [8, 16]},
                100,
            ),
            [
                {"nlist": 128, "nprobe": 8},
                {"nlist": 128, "nprobe": 16},
            ],
        )
        self.assertEqual(expand_index_matrix("milvusflat", {}, 100), [{}])
        self.assertEqual(
            len(expand_index_matrix(
                "milvushnswsq",
                {
                    "m": [16],
                    "ef_construction": [128],
                    "ef_search": [128],
                    "sq_type": ["SQ6", "SQ8"],
                    "refine": [True],
                    "refine_type": ["FP32"],
                    "refine_k": [1.0, 2.0],
                },
                100,
            )),
            4,
        )
        with self.assertRaisesRegex(ValueError, "nprobe"):
            expand_index_matrix(
                "milvusivfsq8",
                {"nlist": [16], "nprobe": [32]},
                100,
            )

    def test_generates_command_specific_yaml(self) -> None:
        cases = {
            "milvushnsw": {
                "m": 16,
                "ef_construction": 128,
                "ef_search": 128,
            },
            "milvusivfflat": {"nlist": 128, "nprobe": 16},
            "milvusivfsq8": {"nlist": 128, "nprobe": 16},
            "milvushnswsq": {
                "m": 16,
                "ef_construction": 128,
                "ef_search": 128,
                "sq_type": "SQ8",
                "refine": True,
                "refine_type": "FP32",
                "refine_k": 1.0,
            },
            "milvushnswpq": {
                "m": 16,
                "ef_construction": 128,
                "ef_search": 128,
                "nbits": 8,
                "refine": True,
                "refine_type": "FP32",
                "refine_k": 1.0,
            },
            "milvushnswprq": {
                "m": 16,
                "ef_construction": 128,
                "ef_search": 128,
                "nbits": 8,
                "nrq": 2,
                "refine": True,
                "refine_type": "FP32",
                "refine_k": 1.0,
            },
            "milvusautoindex": {},
            "milvusflat": {},
        }
        with tempfile.TemporaryDirectory() as temporary:
            manager = BenchmarkJobManager(
                project_root=PROJECT_ROOT,
                base_config_path=(
                    BENCHMARK_ROOT / "config" / "milvushnsw.yml"
                ),
                runner_path=BENCHMARK_ROOT / "run_benchmark.ps1",
                database_path=Path(temporary) / "metrics.sqlite3",
                jobs_root=Path(temporary) / "jobs",
            )
            for offset, (command, index_parameters) in enumerate(cases.items()):
                parameters = common_parameters(command, index_parameters)
                job = manager.prepare_experiment(
                    (parameters,),
                    repetitions=1,
                    job_id=f"{offset + 1:032x}",
                )
                manager._prepare_run_artifacts(job, parameters, 1)
                document = yaml.safe_load(
                    job.config_path.read_text(encoding="utf-8")
                )
                self.assertIn(command, document)
                for name, value in index_parameters.items():
                    self.assertEqual(document[command][name], value)
                self.assertEqual(manager._command(job)[-3], command)

    def test_persists_generic_index_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "metrics.sqlite3"
            repository = BenchmarkMetricsRepository(database)
            run = BenchmarkRunMetrics(
                run_id="run-ivf",
                case_index=0,
                task_label="ivfflat-test",
                status="ok",
                result_file="result.json",
                log_file="result.log",
                raw_result={"results": []},
                database="Milvus",
                db_label="test-cluster",
                case_type="Performance1536D50K",
                command="milvusivfflat",
                index_type="IVF_FLAT",
                metric_type="COSINE",
                index_parameters={"nlist": 128},
                search_parameters={"nprobe": 16},
                hnsw_m=None,
                hnsw_ef_construction=None,
                hnsw_ef_search=None,
                top_k=100,
                num_shards=1,
                replica_number=1,
                load_concurrency=4,
                concurrency_duration_seconds=30,
                concurrency_timeout_seconds=3600,
                concurrency_stages=[
                    ConcurrencyStageMetrics(
                        stage_index=0,
                        concurrency=1,
                        qps=100,
                    )
                ],
            )
            repository.save_all([run])
            with closing(repository.connect()) as connection:
                row = connection.execute(
                    """
                    SELECT command, index_parameters_json,
                           search_parameters_json, configuration_key
                    FROM benchmark_runs
                    """
                ).fetchone()
            self.assertEqual(row[0], "milvusivfflat")
            self.assertEqual(json.loads(row[1]), {"nlist": 128})
            self.assertEqual(json.loads(row[2]), {"nprobe": 16})
            self.assertTrue(row[3])


if __name__ == "__main__":
    unittest.main()
