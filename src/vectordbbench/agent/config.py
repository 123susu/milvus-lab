"""Agent model and LangSmith configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_BASE_URL = (
    "https://ws-ulvbraz6azwp77bu.cn-beijing.maas.aliyuncs.com"
    "/compatible-mode/v1"
)
DEFAULT_MODEL = "qwen-plus"
DEFAULT_API_KEY_ENV = "DASHSCOPE_API_KEY"
RUN_TOOL_NAME = "run_benchmark"
MAX_BENCHMARK_CALLS = 3

IndexParameterValue = int | float | bool | str


@dataclass(frozen=True)
class LangSmithSettings:
    """Server-side LangSmith tracing settings for one Agent request."""

    enabled: bool
    project: str
    endpoint: str | None


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def configure_langsmith(config_path: Path | None = None) -> LangSmithSettings:
    """Load LangSmith settings, with environment variables taking precedence."""

    file_settings: dict[str, Any] = {}
    if config_path and config_path.is_file():
        try:
            with config_path.open("r", encoding="utf-8") as file:
                document = yaml.safe_load(file) or {}
            if isinstance(document, dict):
                values = document.get("langsmith", document)
                if isinstance(values, dict):
                    file_settings = values
        except (OSError, yaml.YAMLError):
            file_settings = {}

    file_api_key = str(file_settings.get("api_key") or "").strip()
    file_endpoint = str(file_settings.get("endpoint") or "").strip()
    file_project = str(file_settings.get("project") or "").strip()
    file_tracing = file_settings.get("tracing")

    custom_key = os.getenv("MILVUS_LANGSMITH_API_KEY") or file_api_key
    if custom_key and not os.getenv("LANGSMITH_API_KEY"):
        os.environ["LANGSMITH_API_KEY"] = custom_key

    custom_endpoint = os.getenv("MILVUS_LANGSMITH_ENDPOINT") or file_endpoint
    if custom_endpoint and not os.getenv("LANGSMITH_ENDPOINT"):
        os.environ["LANGSMITH_ENDPOINT"] = custom_endpoint

    project = (
        os.getenv("MILVUS_LANGSMITH_PROJECT")
        or os.getenv("LANGSMITH_PROJECT")
        or file_project
        or "milvus-tune-agent"
    ).strip()
    if project and not os.getenv("LANGSMITH_PROJECT"):
        os.environ["LANGSMITH_PROJECT"] = project

    if os.getenv("MILVUS_LANGSMITH_TRACING") is not None:
        requested = _env_flag("MILVUS_LANGSMITH_TRACING")
    elif os.getenv("LANGSMITH_TRACING") is not None:
        requested = _env_flag("LANGSMITH_TRACING")
    elif file_tracing is not None:
        requested = str(file_tracing).strip().lower() in {
            "1", "true", "yes", "on"
        }
    else:
        requested = False
    enabled = requested and bool(os.getenv("LANGSMITH_API_KEY"))
    return LangSmithSettings(
        enabled=enabled,
        project=project,
        endpoint=os.getenv("LANGSMITH_ENDPOINT"),
    )
