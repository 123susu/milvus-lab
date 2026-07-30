"""LangGraph benchmark-tuning workflow with a bounded Deep Agent."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import monotonic, sleep
from typing import Any, TypedDict
from urllib.parse import quote

import yaml
from deepagents import create_deep_agent
import langsmith as ls
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from .index_profiles import public_profiles, validate_index_parameters
from .jobs import (
    ACTIVE_STATUSES,
    BenchmarkJobConflictError,
    BenchmarkJobManager,
    BenchmarkJobParameters,
)


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

SYSTEM_PROMPT = """
你是 Milvus 向量索引实验调优 Agent。外层 LangGraph 已经通过前置节点读取 SQLite，
并把历史聚合数据、固定工作负载和允许的索引参数传给你。不要再查询 SQLite。

你唯一的业务工具是 run_benchmark。它会：
- 固定使用当前 VDBBench Collection 已构建的索引类型和构建参数；
- 只允许修改当前索引支持的搜索参数；
- 使用固定数据集、TopK 和 Milvus 环境，以并发 1 运行查询；
- 先执行一次 serial search 计算 Recall，再以并发 1 运行 concurrent search 测 P99；
- 设置 drop_old=false、load=false，不重建 Collection、不导入数据；
- 在同一个 VectorDBBench 任务中依次执行 serial search（计算 Recall）和
  concurrent search（并发 1 测 P99）；
- 等待指标写入 SQLite 后返回 Recall、P99 和索引内存；本轮不会产生新的构建指标。

执行规则：
1. 一次请求最多调用 run_benchmark 3 次，绝对不能超过。
2. 每次只调用一组配置；必须等待结果后再决定下一组，禁止并行工具调用。
3. 压测成本较高。先利用历史数据选择最有信息增益的配置，不要重复已有配置。
4. 只能修改前置节点声明的 search_parameters。M、efConstruction、nlist、
   量化类型等构建参数不可修改，也不能切换索引类型。
5. 目标是在 Recall 达标的前提下优先降低 P99，再比较索引内存。
6. 如果历史数据已经足够，可以少于 3 次甚至不压测，但必须解释原因。
7. Recall 只有在结果的 executed_stages 包含 search_serial 时才算已测量；
   没有该 Stage 时必须标记为“不可用”，不能把 0 当成真实 Recall，也不能
   声称是采集故障，除非有明确日志证据。
8. 不得根据 IVF 的 nprobe 单调性或“全桶扫描”推导 Recall 必然达到目标；
   没有本轮有效 Recall 时，只能给出待验证候选，不能给出已达标结论。
9. 工具失败时如实记录，不得声称实验成功，不得编造任何指标。
10. 不同数据集、TopK、并发和资源环境的结果不可直接比较。

最后必须输出中文调优报告，固定包含：
目标与历史基线
压测计划与执行结果
最终推荐配置
推荐依据
后续调优建议
局限性

使用短段落或项目符号，不要输出 Markdown 表格。
""".strip()


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


def _benchmark_profile(
    base_config_path: Path,
    command: str | None = None,
) -> dict[str, Any]:
    config_path = (
        base_config_path.parent / f"{command}.yml"
        if command
        else base_config_path
    )
    try:
        with config_path.open("r", encoding="utf-8") as file:
            document = yaml.safe_load(file) or {}
    except (OSError, yaml.YAMLError) as error:
        raise TuningAgentConfigurationError(
            f"无法读取 Agent Benchmark 基础配置：{error}"
        ) from error
    profile = document.get(command) if command else next(
        (
            value
            for key, value in document.items()
            if not str(key).startswith("_") and isinstance(value, dict)
        ),
        None,
    )
    if profile is None:
        raise TuningAgentConfigurationError(
            "Agent Benchmark 基础配置中没有 command 配置"
        )
    return profile


def build_benchmark_parameters(
    job_manager: BenchmarkJobManager,
    current_config: dict[str, Any],
    search_parameters: dict[str, IndexParameterValue],
) -> BenchmarkJobParameters:
    """Build one search-only benchmark against the retained Collection."""

    command = str(current_config["command"])
    profile = _benchmark_profile(job_manager.base_config_path, command)
    top_k = int(current_config["top_k"])
    allowed_names = set(current_config["allowed_search_parameters"])
    if set(search_parameters) != allowed_names:
        missing = sorted(allowed_names - set(search_parameters))
        extra = sorted(set(search_parameters) - allowed_names)
        details = []
        if missing:
            details.append("缺少 " + ", ".join(missing))
        if extra:
            details.append("不允许 " + ", ".join(extra))
        raise ValueError("搜索参数不完整：" + "；".join(details))
    complete_parameters = {
        **current_config["index_parameters"],
        **search_parameters,
    }
    normalized = validate_index_parameters(
        command,
        complete_parameters,
        top_k,
    )
    return BenchmarkJobParameters(
        command=command,
        uri=str(profile.get("uri", "http://localhost:19530")),
        num_shards=int(current_config["num_shards"]),
        replica_number=int(current_config["replica_number"]),
        case_type=str(current_config["case_type"]),
        drop_old=False,
        load=False,
        load_concurrency=int(current_config["load_concurrency"]),
        search_serial=True,
        search_concurrent=True,
        k=top_k,
        concurrency_duration=int(current_config["concurrency_duration"]),
        num_concurrency=(1,),
        concurrency_timeout=int(current_config["concurrency_timeout"]),
        index_parameters=normalized,
        db_label=str(current_config["db_label"]),
    )


def query_run_result(
    database_path: Path,
    run_id: str,
    case_index: int,
) -> dict[str, Any]:
    """Read the newly completed run that the existing collector saved."""

    sql = """
        SELECT
            br.run_id,
            br.case_index,
            br.command,
            br.index_type,
            br.case_type,
            br.metric_type,
            br.index_parameters_json,
            br.search_parameters_json,
            br.executed_stages_json,
            br.top_k,
            br.recall,
            br.ndcg,
            br.insert_duration_seconds,
            br.optimize_duration_seconds,
            br.load_duration_seconds,
            br.vector_index_memory_bytes,
            br.serial_latency_p99_ms,
            br.serial_latency_p95_ms,
            cs.concurrency,
            cs.qps,
            cs.latency_avg_ms,
            cs.latency_p95_ms,
            cs.latency_p99_ms
        FROM benchmark_runs AS br
        LEFT JOIN concurrency_stage_metrics AS cs
            ON cs.run_id = br.run_id
            AND cs.case_index = br.case_index
        WHERE br.run_id = ? AND br.case_index = ?
        ORDER BY cs.concurrency DESC, cs.stage_index DESC
        LIMIT 1
    """
    try:
        with closing(_readonly_connection(database_path)) as connection:
            row = connection.execute(sql, (run_id, case_index)).fetchone()
    except sqlite3.Error as error:
        raise TuningAgentDataError(f"读取新 Benchmark 结果失败：{error}") from error
    if row is None:
        raise TuningAgentDataError(f"新 Benchmark 结果不存在：{run_id}/{case_index}")
    memory_bytes = row["vector_index_memory_bytes"]
    return {
        "run_id": row["run_id"],
        "case_index": row["case_index"],
        "command": row["command"],
        "index_type": row["index_type"],
        "case_type": row["case_type"],
        "metric_type": row["metric_type"],
        "index_parameters": _parse_json_object(row["index_parameters_json"]),
        "search_parameters": _parse_json_object(row["search_parameters_json"]),
        "top_k": row["top_k"],
        "concurrency": row["concurrency"],
        "executed_stages": json.loads(row["executed_stages_json"] or "[]"),
        "recall": (
            _optional_float(row["recall"])
            if "search_serial"
            in json.loads(row["executed_stages_json"] or "[]")
            else None
        ),
        "ndcg": _optional_float(row["ndcg"]),
        "p99_ms": (
            _optional_float(row["latency_p99_ms"])
            if row["latency_p99_ms"] is not None
            else _optional_float(row["serial_latency_p99_ms"])
        ),
        "latency_avg_ms": _optional_float(row["latency_avg_ms"]),
        "qps": _optional_float(row["qps"]),
        "vector_index_memory_mib": (
            float(memory_bytes) / (1024 * 1024)
            if memory_bytes is not None and memory_bytes > 0
            else None
        ),
        "insert_seconds": _optional_float(row["insert_duration_seconds"]),
        "optimize_seconds": _optional_float(row["optimize_duration_seconds"]),
        "load_seconds": _optional_float(row["load_duration_seconds"]),
    }


class AgentBenchmarkExecutor:
    """Run serial and concurrent stages in one job against the retained Collection."""

    def __init__(
        self,
        job_manager: BenchmarkJobManager,
        database_path: Path,
        *,
        timeout_seconds: int | None = None,
        poll_seconds: float = 1.0,
    ) -> None:
        self.job_manager = job_manager
        self.database_path = database_path
        self.timeout_seconds = timeout_seconds or int(
            os.getenv("MILVUS_TUNING_AGENT_BENCHMARK_TIMEOUT", "7200")
        )
        self.poll_seconds = poll_seconds

    def _run_one(
        self,
        current_config: dict[str, Any],
        search_parameters: dict[str, IndexParameterValue],
        parameters: BenchmarkJobParameters,
        reason: str,
    ) -> dict[str, Any]:
        try:
            job = self.job_manager.submit(parameters, repetitions=1)
        except BenchmarkJobConflictError as error:
            raise TuningAgentBenchmarkConflictError(str(error)) from error

        deadline = monotonic() + self.timeout_seconds
        while job.status in ACTIVE_STATUSES:
            if monotonic() >= deadline:
                self.job_manager.cancel(job.job_id)
                raise TuningAgentError(
                    f"Agent Benchmark 超过 {self.timeout_seconds} 秒，已请求取消"
                )
            sleep(self.poll_seconds)
            job = self.job_manager.get(job.job_id)

        base_result: dict[str, Any] = {
            "job_id": job.job_id,
            "status": job.status,
            "reason": reason,
            "command": current_config["command"],
            "requested_search_parameters": search_parameters,
            "elapsed_seconds": self.job_manager.elapsed_seconds(job),
        }
        if job.status != "succeeded":
            return {
                **base_result,
                "error": job.error or f"Benchmark {job.status}",
            }
        if job.result_run_id is None or job.result_case_index is None:
            return {
                **base_result,
                "status": "failed",
                "error": "Benchmark 完成但没有关联到 SQLite 结果",
            }
        return {
            **base_result,
            **query_run_result(
                self.database_path,
                job.result_run_id,
                job.result_case_index,
            ),
        }

    def run(
        self,
        current_config: dict[str, Any],
        search_parameters: dict[str, IndexParameterValue],
        reason: str,
    ) -> dict[str, Any]:
        return self._run_one(
            current_config,
            search_parameters,
            build_benchmark_parameters(
                self.job_manager,
                current_config,
                search_parameters,
            ),
            reason,
        )


def _message_text(message: object) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    return str(content).strip()


class BenchmarkTuningAgent:
    """Run SQLite preprocessing, bounded experiments, and final reporting."""

    def __init__(
        self,
        database_path: Path,
        job_manager: BenchmarkJobManager,
        model_name: str | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
    ) -> None:
        self.database_path = database_path
        self.job_manager = job_manager
        self.model_name = (
            model_name
            or os.getenv("MILVUS_TUNING_AGENT_MODEL")
            or os.getenv("VDBBENCH_LLM_MODEL")
            or DEFAULT_MODEL
        ).strip()
        self.base_url = (
            base_url
            or os.getenv("MILVUS_TUNING_AGENT_BASE_URL")
            or os.getenv("VDBBENCH_LLM_BASE_URL")
            or DEFAULT_BASE_URL
        ).strip()
        self.api_key_env = (
            api_key_env
            or os.getenv("MILVUS_TUNING_AGENT_API_KEY_ENV")
            or DEFAULT_API_KEY_ENV
        ).strip()
        self.langsmith = configure_langsmith(
            self.job_manager.base_config_path.parent / "langsmith.yml"
        )

    def _chat_model(self) -> ChatOpenAI:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise TuningAgentConfigurationError(
                f"Agent 模型为 {self.model_name}，请先设置环境变量 "
                f"{self.api_key_env}。"
            )
        if not self.base_url:
            raise TuningAgentConfigurationError("Agent 模型 base_url 不能为空")
        return ChatOpenAI(
            model=self.model_name,
            base_url=self.base_url,
            api_key=api_key,
            temperature=0.1,
            timeout=120,
            max_retries=2,
        )

    def _load_history_node(
        self,
        state: TuningWorkflowState,
    ) -> TuningWorkflowState:
        history = query_candidate_data(
            self.database_path,
            state["recall_target"],
        )
        current_config = query_current_collection_config(self.database_path)

        def belongs_to_retained_index(candidate: dict[str, Any]) -> bool:
            return (
                candidate["command"] == current_config["command"]
                and candidate["case_type"] == current_config["case_type"]
                and candidate["top_k"] == current_config["top_k"]
                and candidate["concurrency"] == 1
                and candidate["index_parameters"]
                == current_config["index_parameters"]
            )

        qualified = [
            candidate
            for candidate in history["qualified_candidates"]
            if belongs_to_retained_index(candidate)
        ]
        near_misses = [
            candidate
            for candidate in history["near_misses"]
            if belongs_to_retained_index(candidate)
        ]
        relevant_history = {
            **history,
            "comparison_scope": (
                "仅比较当前保留索引的相同 command、构建参数、数据集、"
                "TopK 和并发 1 历史结果"
            ),
            "configuration_count": len(qualified) + len(near_misses),
            "qualified_count": len(qualified),
            "qualified_candidates": qualified,
            "near_misses": near_misses,
        }
        current_profiles = [
            profile
            for profile in public_profiles()
            if profile["command"] == current_config["command"]
        ]
        return {
            "benchmark_history": relevant_history,
            "current_collection_config": current_config,
            "supported_profiles": current_profiles,
            "history_configuration_count": int(
                relevant_history["configuration_count"]
            ),
        }

    def _build_workflow(self) -> Any:
        executor = AgentBenchmarkExecutor(
            self.job_manager,
            self.database_path,
        )
        execution_lock = Lock()

        def tune_with_agent(
            state: TuningWorkflowState,
        ) -> TuningWorkflowState:
            benchmark_runs: list[dict[str, Any]] = []

            @tool(RUN_TOOL_NAME, args_schema=RunBenchmarkInput)
            def run_benchmark(
                search_parameters: dict[str, IndexParameterValue],
                reason: str,
            ) -> str:
                """在当前 Collection 上以并发 1 运行一次只改搜索参数的查询压测。"""

                with execution_lock:
                    if len(benchmark_runs) >= MAX_BENCHMARK_CALLS:
                        return json.dumps(
                            {
                                "status": "rejected",
                                "error": "本次 Agent 已达到 3 次 Benchmark 上限",
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    try:
                        result = executor.run(
                            state["current_collection_config"],
                            search_parameters,
                            reason,
                        )
                    except TuningAgentBenchmarkConflictError:
                        raise
                    except Exception as error:
                        result = {
                            "status": "failed",
                            "reason": reason,
                            "requested_search_parameters": search_parameters,
                            "error": str(error),
                        }
                    benchmark_runs.append(result)
                return json.dumps(
                    result,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )

            history_context = {
                "recall_target": state["recall_target"],
                "current_collection_config": state[
                    "current_collection_config"
                ],
                "history": state["benchmark_history"],
                "supported_profiles": state["supported_profiles"],
                "max_benchmark_calls": MAX_BENCHMARK_CALLS,
                "benchmark_constraints": {
                    "drop_old": False,
                    "load": False,
                    "search_serial": True,
                    "search_concurrent": True,
                    "num_concurrency": [1],
                    "mutable_fields": state["current_collection_config"][
                        "allowed_search_parameters"
                    ],
                },
            }
            deep_agent = create_deep_agent(
                model=self._chat_model(),
                tools=[run_benchmark],
                system_prompt=SYSTEM_PROMPT,
                middleware=[
                    ToolCallLimitMiddleware(
                        tool_name=RUN_TOOL_NAME,
                        run_limit=MAX_BENCHMARK_CALLS,
                        exit_behavior="continue",
                    )
                ],
            )
            deep_result = deep_agent.invoke(
                {
                    "messages": [
                        HumanMessage(
                            content=(
                                "请根据下面的前置节点数据完成自动调优。"
                                "你可以决定是否调用 run_benchmark，最多 3 次。"
                                "每次获得结果后再决定下一步，最后输出完整报告。\n\n"
                                + json.dumps(
                                    history_context,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                )
                            )
                        )
                    ]
                },
                config={"recursion_limit": 30},
            )
            messages = list(deep_result.get("messages", []))
            report = _message_text(messages[-1]) if messages else ""
            if not report:
                raise TuningAgentError("Agent 没有返回调优报告")
            return {
                "report": report,
                "tools_used": [RUN_TOOL_NAME] if benchmark_runs else [],
                "benchmark_runs": benchmark_runs,
                "benchmark_tool_call_count": len(benchmark_runs),
            }

        builder = StateGraph(TuningWorkflowState)
        builder.add_node("load_benchmark_history", self._load_history_node)
        builder.add_node("tune_with_agent", tune_with_agent)
        builder.add_edge(START, "load_benchmark_history")
        builder.add_edge("load_benchmark_history", "tune_with_agent")
        builder.add_edge("tune_with_agent", END)
        return builder.compile()

    def recommend(self, recall_target: float) -> dict[str, Any]:
        if self.job_manager.active_job_id is not None:
            raise TuningAgentBenchmarkConflictError(
                f"benchmark job {self.job_manager.active_job_id} "
                "正在运行，请等待完成后再启动 Agent"
            )
        try:
            with ls.tracing_context(
                enabled=self.langsmith.enabled,
                project_name=self.langsmith.project,
                tags=["milvus-tuning", "recall-optimization"],
                metadata={
                    "recall_target": recall_target,
                    "agent_model": self.model_name,
                    "benchmark_tool": RUN_TOOL_NAME,
                    "max_benchmark_calls": MAX_BENCHMARK_CALLS,
                },
            ):
                result = self._build_workflow().invoke(
                    {"recall_target": recall_target},
                    config={
                        "recursion_limit": 5,
                        "tags": ["langgraph", "deep-agent"],
                        "metadata": {
                            "recall_target": recall_target,
                            "agent_model": self.model_name,
                        },
                    },
                )
        except TuningAgentError:
            raise
        except Exception as error:
            raise TuningAgentError(f"Agent 执行失败：{error}") from error
        benchmark_runs = list(result.get("benchmark_runs", []))
        return {
            "model": self.model_name,
            "answer": str(result["report"]),
            "tools_used": list(result.get("tools_used", [])),
            "history_configuration_count": int(
                result.get("history_configuration_count", 0)
            ),
            "benchmark_tool_call_count": int(
                result.get("benchmark_tool_call_count", 0)
            ),
            "benchmark_run_count": sum(
                1 for run in benchmark_runs if run.get("status") == "succeeded"
            ),
            "benchmark_runs": benchmark_runs,
        }
