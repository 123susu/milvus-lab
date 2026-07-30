"""Collect one VectorDBBench result and persist it to SQLite."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from metrics import (
    BenchmarkMetricsCollector,
    BenchmarkMetricsRepository,
    parse_timezone_offset,
)
from metrics.collector import prometheus_client_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = BenchmarkMetricsCollector.load_config(args.config)
    metrics_config = config.get("_metrics", {})
    if not isinstance(metrics_config, dict):
        raise ValueError("_metrics must be a YAML object")

    collector = BenchmarkMetricsCollector(
        prometheus_client=prometheus_client_from_config(config),
        tz=parse_timezone_offset(metrics_config.get("timezone")),
        querynode_cpu_limit_cores=float(
            metrics_config.get("querynode_cpu_limit_cores", 2)
        ),
        query_range_step_seconds=int(
            metrics_config.get("query_range_step_seconds", 1)
        ),
        rate_window=str(metrics_config.get("rate_window", "10s")),
    )
    runs = collector.collect(args.config, args.result, args.log)
    BenchmarkMetricsRepository(args.database).save_all(runs)

    for run in runs:
        print(
            "Saved benchmark metrics: "
            f"run_id={run.run_id}, case_index={run.case_index}, "
            f"stages={len(run.concurrency_stages)}"
        )
    print(f"SQLite database: {args.database.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
        json.JSONDecodeError,
        sqlite3.Error,
    ) as error:
        print(f"Benchmark metric collection failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
