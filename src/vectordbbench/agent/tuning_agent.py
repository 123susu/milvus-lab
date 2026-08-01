"""LangGraph orchestration for the bounded Milvus tuning agent."""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from typing import Any

from deepagents import create_deep_agent
import langsmith as ls
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from metrics.index_profiles import public_profiles
from metrics.jobs import BenchmarkJobManager
from .tools.benchmark import (
    AgentBenchmarkExecutor,
    build_benchmark_parameters,
    query_run_result,
)
from .config import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    MAX_BENCHMARK_CALLS,
    RUN_TOOL_NAME,
    configure_langsmith,
)
from .history import query_candidate_data, query_current_collection_config
from .models import (
    IndexParameterValue,
    RunBenchmarkInput,
    TuningAgentBenchmarkConflictError,
    TuningAgentConfigurationError,
    TuningAgentDataError,
    TuningAgentError,
    TuningWorkflowState,
)
from .prompts import SYSTEM_PROMPT

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
                """重建 VDBBench 并以并发 1 运行一次仅改变搜索参数的完整压测。"""

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
                    "drop_old": True,
                    "load": True,
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
