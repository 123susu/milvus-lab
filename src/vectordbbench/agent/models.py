"""Agent errors, tool inputs, and workflow state."""

from __future__ import annotations

from typing import Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field

IndexParameterValue = int | float | bool | str

class TuningAgentError(RuntimeError):
    """Base error raised by the tuning workflow."""


class TuningAgentConfigurationError(TuningAgentError):
    """Raised when the configured chat model cannot be used."""


class TuningAgentDataError(TuningAgentError):
    """Raised when benchmark data cannot be queried."""


class TuningAgentBenchmarkConflictError(TuningAgentError):
    """Raised when another benchmark is already using VDBBench."""


class RunBenchmarkInput(BaseModel):
    """Strict input exposed to the model for one benchmark run."""

    model_config = ConfigDict(extra="forbid")

    search_parameters: dict[str, IndexParameterValue] = Field(
        description=(
            "当前索引的完整搜索参数，例如 HNSW 的 ef_search，"
            "或 IVF 的 nprobe"
        )
    )
    reason: str = Field(
        min_length=1,
        max_length=500,
        description="为什么这组配置能为当前 Recall 调优提供信息增益",
    )


class TuningWorkflowState(TypedDict, total=False):
    recall_target: float
    benchmark_history: dict[str, Any]
    current_collection_config: dict[str, Any]
    supported_profiles: list[dict[str, Any]]
    history_configuration_count: int
    report: str
    tools_used: list[str]
    benchmark_runs: list[dict[str, Any]]
    benchmark_tool_call_count: int

