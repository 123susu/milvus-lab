"""Generate a Markdown report from a VectorDBBench result via an LLM API."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def required_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def load_report_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as config_file:
        document = yaml.safe_load(config_file) or {}

    report_config = document.get("_report", {})
    if not isinstance(report_config, dict):
        raise ValueError("_report must be a YAML object")
    return report_config


def build_endpoint(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def build_prompt(result: dict[str, Any], language: str) -> str:
    result_json = json.dumps(result, ensure_ascii=False, indent=2)
    return f"""请根据下面的 VectorDBBench 原始结果生成一份 {language} Markdown 性能报告。

要求：
1. 只使用原始结果中存在的数据，不得编造 CPU、内存、磁盘、网络或服务端监控指标。
2. 说明测试配置：数据库、索引、距离类型、索引参数、TopK、并发和持续时间。
3. 分析数据导入、索引构建、Recall、nDCG、串行延迟和并发性能。
4. JSON 中延迟单位为秒，报告表格统一换算成毫秒。
5. 用表格列出每个并发档位的 QPS、平均延迟、P95 和 P99。
6. 找出最大 QPS 对应的并发，并分析吞吐下降或尾延迟上升的拐点。
7. 明确区分“测量事实”和“可能原因”；原因只能表述为需要监控数据验证的推测。
8. 给出下一轮可执行的测试建议和必要的局限性说明。
9. 输出必须以一级标题开始，不要使用 Markdown 代码围栏包裹整份报告。

VectorDBBench 原始结果：
{result_json}
"""


def request_completion(
    endpoint: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    timeout_seconds: int,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是一名严谨的向量数据库性能工程师。"
                    "你必须忠于原始数据，避免把单次本机测试夸大为生产结论。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "stream": False,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )

    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                response_data = json.loads(response.read().decode("utf-8"))
            return response_data["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as error:
            error_body = error.read().decode("utf-8", errors="replace")[:2000]
            if error.code != 429 and error.code < 500:
                raise RuntimeError(f"LLM HTTP {error.code}: {error_body}") from error
            if attempt == 2:
                raise RuntimeError(f"LLM HTTP {error.code}: {error_body}") from error
        except urllib.error.URLError as error:
            if attempt == 2:
                raise RuntimeError(f"Cannot reach LLM endpoint: {error.reason}") from error

        time.sleep(2**attempt)

    raise RuntimeError("LLM request failed")


def strip_outer_code_fence(markdown: str) -> str:
    lines = markdown.strip().splitlines()
    if len(lines) >= 3 and lines[0].strip().lower() in {"```", "```md", "```markdown"}:
        if lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return markdown.strip()


def main() -> int:
    args = parse_args()
    report_config = load_report_config(args.config)

    if not report_config.get("enabled", True):
        print("LLM report generation is disabled")
        return 0

    base_url = required_text(os.getenv("VDBBENCH_LLM_BASE_URL")) or required_text(
        report_config.get("base_url"),
    )
    model = required_text(os.getenv("VDBBENCH_LLM_MODEL")) or required_text(
        report_config.get("model"),
    )
    api_key_env = required_text(report_config.get("api_key_env")) or "DASHSCOPE_API_KEY"
    api_key = required_text(os.getenv(api_key_env))

    missing = []
    if not base_url:
        missing.append("_report.base_url or VDBBENCH_LLM_BASE_URL")
    if not model:
        missing.append("_report.model or VDBBENCH_LLM_MODEL")
    if not api_key:
        missing.append(f"environment variable {api_key_env}")
    if missing:
        raise ValueError("Missing LLM configuration: " + ", ".join(missing))

    with args.result.open("r", encoding="utf-8") as result_file:
        result = json.load(result_file)

    language = required_text(report_config.get("language")) or "简体中文"
    temperature = float(report_config.get("temperature", 0.2))
    timeout_seconds = int(report_config.get("timeout_seconds", 120))
    endpoint = build_endpoint(base_url)
    prompt = build_prompt(result, language)
    markdown = request_completion(
        endpoint=endpoint,
        api_key=api_key,
        model=model,
        prompt=prompt,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
    )
    markdown = strip_outer_code_fence(markdown)

    output_path = args.result.with_suffix(".md")
    temporary_path = output_path.with_suffix(".md.tmp")
    temporary_path.write_text(markdown + "\n", encoding="utf-8")
    temporary_path.replace(output_path)
    print(f"Generated LLM report: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, TypeError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"LLM report generation failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
