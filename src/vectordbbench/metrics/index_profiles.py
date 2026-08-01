"""Supported CPU index profiles for local VectorDBBench experiments."""

from __future__ import annotations

from itertools import product
from typing import Any


IndexParameterValue = int | float | bool | str

HNSW_PARAMETERS: list[dict[str, Any]] = [
    {
        "name": "m",
        "label": "M",
        "kind": "integer",
        "default": 16,
        "minimum": 4,
        "maximum": 128,
        "description": "图中每个节点的最大邻接数",
    },
    {
        "name": "ef_construction",
        "label": "efConstruction",
        "kind": "integer",
        "default": 128,
        "minimum": 4,
        "maximum": 1024,
        "description": "构建索引时的候选集合大小",
    },
    {
        "name": "ef_search",
        "label": "efSearch",
        "kind": "integer",
        "default": 128,
        "minimum": 1,
        "maximum": 2048,
        "description": "查询时的候选集合大小，必须不小于 TopK",
    },
]

REFINE_PARAMETERS: list[dict[str, Any]] = [
    {
        "name": "refine",
        "label": "refine",
        "kind": "boolean",
        "default": True,
        "options": [True, False],
        "description": "是否保留原始数据用于精排",
    },
    {
        "name": "refine_type",
        "label": "refineType",
        "kind": "choice",
        "default": "FP32",
        "options": ["SQ6", "SQ8", "BF16", "FP16", "FP32"],
        "description": "精排数据的存储类型",
    },
    {
        "name": "refine_k",
        "label": "refineK",
        "kind": "number",
        "default": 1.0,
        "minimum": 1.0,
        "maximum": 10000.0,
        "description": "精排候选数相对 TopK 的放大倍数",
    },
]


CPU_INDEX_PROFILES: dict[str, dict[str, Any]] = {
    "milvushnsw": {
        "label": "HNSW",
        "index_type": "HNSW",
        "task_prefix": "hnsw",
        "parameters": [*HNSW_PARAMETERS],
    },
    "milvushnswsq": {
        "label": "HNSW_SQ",
        "index_type": "HNSW_SQ",
        "task_prefix": "hnswsq",
        "parameters": [
            *HNSW_PARAMETERS,
            {
                "name": "sq_type",
                "label": "sqType",
                "kind": "choice",
                "default": "SQ8",
                "options": ["SQ4U", "SQ6", "SQ8", "BF16", "FP16", "FP32"],
                "description": "标量量化的数据类型",
            },
            *REFINE_PARAMETERS,
        ],
    },
    "milvushnswpq": {
        "label": "HNSW_PQ",
        "index_type": "HNSW_PQ",
        "task_prefix": "hnswpq",
        "parameters": [
            *HNSW_PARAMETERS,
            {
                "name": "nbits",
                "label": "nbits",
                "kind": "integer",
                "default": 8,
                "minimum": 1,
                "maximum": 65536,
                "description": "PQ 编码使用的位数",
            },
            *REFINE_PARAMETERS,
        ],
    },
    "milvushnswprq": {
        "label": "HNSW_PRQ",
        "index_type": "HNSW_PRQ",
        "task_prefix": "hnswprq",
        "parameters": [
            *HNSW_PARAMETERS,
            {
                "name": "nbits",
                "label": "nbits",
                "kind": "integer",
                "default": 8,
                "minimum": 1,
                "maximum": 65536,
                "description": "PQ 编码使用的位数",
            },
            {
                "name": "nrq",
                "label": "nrq",
                "kind": "integer",
                "default": 2,
                "minimum": 1,
                "maximum": 16,
                "description": "残差子量化器数量",
            },
            *REFINE_PARAMETERS,
        ],
    },
    "milvusivfflat": {
        "label": "IVF_FLAT",
        "index_type": "IVF_FLAT",
        "task_prefix": "ivfflat",
        "parameters": [
            {
                "name": "nlist",
                "label": "nlist",
                "kind": "integer",
                "default": 128,
                "minimum": 1,
                "maximum": 65536,
                "description": "倒排索引的聚类中心/分桶数量",
            },
            {
                "name": "nprobe",
                "label": "nprobe",
                "kind": "integer",
                "default": 16,
                "minimum": 1,
                "maximum": 65536,
                "description": "每次查询探测的桶数量，不能大于 nlist",
            },
        ],
    },
    "milvusivfsq8": {
        "label": "IVF_SQ8",
        "index_type": "IVF_SQ8",
        "task_prefix": "ivfsq8",
        "parameters": [
            {
                "name": "nlist",
                "label": "nlist",
                "kind": "integer",
                "default": 128,
                "minimum": 1,
                "maximum": 65536,
                "description": "倒排索引的聚类中心/分桶数量",
            },
            {
                "name": "nprobe",
                "label": "nprobe",
                "kind": "integer",
                "default": 16,
                "minimum": 1,
                "maximum": 65536,
                "description": "每次查询探测的桶数量，不能大于 nlist",
            },
        ],
    },
    "milvusautoindex": {
        "label": "AUTOINDEX",
        "index_type": "AUTOINDEX",
        "task_prefix": "autoindex",
        "parameters": [],
    },
    "milvusflat": {
        "label": "FLAT",
        "index_type": "FLAT",
        "task_prefix": "flat",
        "parameters": [],
    },
}


def profile_for(command: str) -> dict[str, Any]:
    try:
        return CPU_INDEX_PROFILES[command]
    except KeyError as error:
        raise ValueError(f"unsupported CPU index command: {command}") from error


def _validate_parameter(
    name: str,
    definition: dict[str, Any],
    value: Any,
) -> IndexParameterValue:
    kind = definition["kind"]
    if kind == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    elif kind == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number")
        value = float(value)
    elif kind == "boolean":
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be true or false")
    elif kind == "choice":
        if not isinstance(value, str) or value not in definition["options"]:
            choices = ", ".join(str(option) for option in definition["options"])
            raise ValueError(f"{name} must be one of {choices}")
    else:
        raise ValueError(f"unsupported parameter kind for {name}: {kind}")

    if "minimum" in definition and value < definition["minimum"]:
        raise ValueError(f"{name} must be at least {definition['minimum']}")
    if "maximum" in definition and value > definition["maximum"]:
        raise ValueError(f"{name} must be at most {definition['maximum']}")
    return value


def validate_index_parameters(
    command: str,
    values: dict[str, Any],
    top_k: int,
) -> dict[str, IndexParameterValue]:
    profile = profile_for(command)
    definitions = {
        definition["name"]: definition
        for definition in profile["parameters"]
    }
    if set(values) != set(definitions):
        missing = sorted(set(definitions) - set(values))
        extra = sorted(set(values) - set(definitions))
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if extra:
            details.append(f"unknown {', '.join(extra)}")
        raise ValueError("invalid index parameters: " + "; ".join(details))

    normalized = {
        name: _validate_parameter(name, definition, values[name])
        for name, definition in definitions.items()
    }

    if command.startswith("milvushnsw"):
        if normalized["ef_construction"] < normalized["m"]:
            raise ValueError("ef_construction must be greater than or equal to m")
        if normalized["ef_search"] < top_k:
            raise ValueError("ef_search must be greater than or equal to k")
    elif command in {"milvusivfflat", "milvusivfsq8"}:
        if normalized["nprobe"] > normalized["nlist"]:
            raise ValueError("nprobe must be less than or equal to nlist")
    return normalized


def expand_index_matrix(
    command: str,
    matrix: dict[str, list[Any]],
    top_k: int,
) -> list[dict[str, IndexParameterValue]]:
    profile = profile_for(command)
    names = [definition["name"] for definition in profile["parameters"]]
    if set(matrix) != set(names):
        validate_index_parameters(command, {name: None for name in matrix}, top_k)
    if not names:
        return [{}]

    unique_values: list[list[IndexParameterValue]] = []
    definitions = {
        definition["name"]: definition
        for definition in profile["parameters"]
    }
    for name in names:
        if len(matrix[name]) > 8:
            raise ValueError(f"{name} is limited to 8 values")
        values: list[IndexParameterValue] = []
        for raw_value in matrix[name]:
            value = _validate_parameter(name, definitions[name], raw_value)
            if value not in values:
                values.append(value)
        if not values:
            raise ValueError(f"{name} must contain at least one value")
        unique_values.append(values)
    return [
        validate_index_parameters(
            command,
            dict(zip(names, combination)),
            top_k,
        )
        for combination in product(*unique_values)
    ]


def public_profiles() -> list[dict[str, Any]]:
    return [
        {
            "command": command,
            "label": profile["label"],
            "index_type": profile["index_type"],
            "parameters": profile["parameters"],
        }
        for command, profile in CPU_INDEX_PROFILES.items()
    ]
