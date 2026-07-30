"""Recall-oriented benchmark tuning agent backed by read-only SQLite data."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any
from urllib.parse import quote

from deepagents import create_deep_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI


DEFAULT_BASE_URL = (
    "https://ws-ulvbraz6azwp77bu.cn-beijing.maas.aliyuncs.com"
    "/compatible-mode/v1"
)
DEFAULT_MODEL = "qwen-plus"
DEFAULT_API_KEY_ENV = "DASHSCOPE_API_KEY"
QUERY_TOOL_NAME = "query_benchmark_candidates"

SYSTEM_PROMPT = """
你是 Milvus 向量索引调优 Agent。用户会提供一个 Recall 目标。

约束：
1. 必须先且只需调用一次 query_benchmark_candidates，SQLite 返回的数据是唯一事实来源。
2. 不得编造不存在的实验、指标或参数，不得声称已经执行新的压测。
3. 推荐时先判断是否有配置达到 Recall 目标。达到目标时，优先考虑较低 P99，
   再比较向量索引内存；没有达到目标时，明确说明，并使用最接近目标的配置作为起点。
4. 索引内存为 0 或 null 时必须说明数据不可用，不能把它当成真正的零内存。
5. 样本少时可以比较，但要用“当前样本中”限定结论，不要宣称存在因果关系。
6. 后续调优建议必须针对推荐配置的索引类型和现有参数，只描述下一轮应该改变什么、
   保持什么，以及预期观察 Recall/P99/内存中的哪些变化。
7. 当前不能启动压测、修改数据库或访问其他系统。

请使用简洁中文纯文本回答，固定包含：
目标判断
推荐配置
推荐依据
后续调优建议
每个部分可以使用短横线，不要输出 Markdown 表格。
""".strip()


class TuningAgentError(RuntimeError):
    """Base error raised by the tuning agent."""


class TuningAgentConfigurationError(TuningAgentError):
    """Raised when the configured chat model cannot be used."""


class TuningAgentDataError(TuningAgentError):
    """Raised when benchmark data cannot be queried."""


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
    qualified_limit: int = 6,
    near_miss_limit: int = 4,
) -> dict[str, Any]:
    """Return aggregate benchmark candidates without exposing arbitrary SQL."""

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
    """Build and invoke one Deep Agents/LangGraph graph per recommendation."""

    def __init__(
        self,
        database_path: Path,
        model_name: str | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
    ) -> None:
        self.database_path = database_path
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

    def recommend(self, recall_target: float) -> dict[str, Any]:
        database_path = self.database_path

        @tool(QUERY_TOOL_NAME)
        def query_benchmark_candidates(recall_target: float) -> str:
            """查询达到 Recall 目标的聚合配置及最接近目标的未达标配置。"""

            payload = query_candidate_data(database_path, recall_target)
            return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

        try:
            graph = create_deep_agent(
                model=self._chat_model(),
                tools=[query_benchmark_candidates],
                system_prompt=SYSTEM_PROMPT,
            )
            result = graph.invoke(
                {
                    "messages": [
                        HumanMessage(
                            content=(
                                f"我的 Recall 目标是 {recall_target:.4f}"
                                f"（{recall_target * 100:.2f}%）。"
                                "请查询当前 SQLite 实验数据并推荐配置，"
                                "然后给出下一轮调优建议。"
                            )
                        )
                    ]
                }
            )
        except TuningAgentError:
            raise
        except Exception as error:
            raise TuningAgentError(f"Agent 执行失败：{error}") from error

        messages = list(result.get("messages", []))
        all_tool_messages = [
            message.name
            for message in messages
            if isinstance(message, ToolMessage) and message.name
        ]
        if all_tool_messages.count(QUERY_TOOL_NAME) != 1:
            raise TuningAgentError(
                "Agent 必须且只能查询一次 SQLite，已拒绝返回不合规建议"
            )
        answer = _message_text(messages[-1]) if messages else ""
        if not answer:
            raise TuningAgentError("Agent 没有返回调优建议")
        return {
            "model": self.model_name,
            "answer": answer,
            "tools_used": [QUERY_TOOL_NAME],
        }
