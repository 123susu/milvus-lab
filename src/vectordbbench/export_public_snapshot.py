"""Export a sanitized read-only benchmark snapshot for the public frontend."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "frontend" / "public" / "benchmark-snapshot.json"


def fetch_json(url: str) -> Any:
    with urlopen(url, timeout=30) as response:
        return json.load(response)


def sanitize_run(run: dict[str, Any], index: int) -> dict[str, Any]:
    sanitized = dict(run)
    sanitized["task_label"] = f"public-{run['index_type'].lower()}-{index:03d}"
    sanitized["db_label"] = "Milvus CPU Cluster"
    if sanitized.get("monitoring_error"):
        sanitized["monitoring_error"] = "指标采集不完整"
    sanitized["stages"] = [
        {
            **stage,
            "monitoring_error": (
                "指标采集不完整" if stage.get("monitoring_error") else None
            ),
        }
        for stage in sanitized.get("stages", [])
    ]
    return sanitized


def sanitize_aggregate(aggregate: dict[str, Any]) -> dict[str, Any]:
    return {
        **aggregate,
        "db_label": "Milvus CPU Cluster",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", default="http://127.0.0.1:8765")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    api_base = args.api_base.rstrip("/")
    runs = fetch_json(f"{api_base}/api/benchmarks?limit=100&offset=0")
    aggregates = fetch_json(
        f"{api_base}/api/benchmark-aggregates?limit=100&offset=0"
    )
    profiles = fetch_json(f"{api_base}/api/benchmark-profiles")

    sanitized_runs = [
        sanitize_run(run, index)
        for index, run in enumerate(runs["items"], start=1)
    ]
    sanitized_aggregates = [
        sanitize_aggregate(aggregate)
        for aggregate in aggregates["items"]
    ]
    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs": {
            **runs,
            "items": sanitized_runs,
            "total": len(sanitized_runs),
        },
        "aggregates": {
            **aggregates,
            "items": sanitized_aggregates,
            "total": len(sanitized_aggregates),
        },
        "profiles": profiles,
    }

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Exported {len(sanitized_runs)} runs and "
        f"{len(sanitized_aggregates)} aggregates to {output}"
    )


if __name__ == "__main__":
    main()
