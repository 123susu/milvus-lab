"""Execute one real Milvus search and resolve its distributed Jaeger trace."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from pymilvus import DataType, MilvusClient
from sklearn.feature_extraction.text import HashingVectorizer


TRACE_DEMO_COLLECTION = "TraceDemo"
TRACE_DEMO_DIMENSION = 384
TRACE_DEMO_DOCUMENTS = [
    "Milvus Proxy receives SDK requests and routes them to cluster services.",
    "QueryNode loads sealed segments and executes vector similarity search.",
    "StreamingNode handles write-ahead logging and growing segment queries.",
    "DataNode processes ingestion, flushing, compaction, and index files.",
    "MixCoord coordinates metadata, query placement, and data management.",
    "HNSW uses a navigable small-world graph for approximate nearest neighbors.",
    "IVF partitions vectors into clusters and searches selected probe lists.",
    "Recall measures how many true nearest neighbors are returned by search.",
    "P99 latency describes the slowest one percent of observed requests.",
    "Jaeger visualizes distributed traces as nested spans on a shared timeline.",
    "OpenTelemetry exports traces, metrics, and logs using vendor-neutral APIs.",
    "VectorDBBench compares vector database latency, throughput, and recall.",
]


@dataclass(frozen=True)
class TraceSearchRequest:
    uri: str
    database: str
    collection_name: str
    top_k: int


class TraceSearchError(RuntimeError):
    """A user-facing trace search failure."""


class MilvusTraceSearchService:
    def __init__(
        self,
        *,
        jaeger_query_url: str,
    ) -> None:
        self.jaeger_query_url = jaeger_query_url.rstrip("/")
        self._demo_lock = threading.Lock()

    def search(self, request: TraceSearchRequest) -> dict[str, Any]:
        client: MilvusClient | None = None
        try:
            client = MilvusClient(
                uri=request.uri,
                db_name=request.database,
                timeout=30,
            )
            if request.collection_name == TRACE_DEMO_COLLECTION:
                self._ensure_trace_demo(client)
            if not client.has_collection(request.collection_name):
                raise TraceSearchError(
                    f"Collection {request.collection_name!r} 不存在。"
                    "请先保留或加载一个可搜索的 Collection。"
                )

            description = client.describe_collection(request.collection_name)
            fields = description.get("fields") or []
            vector_field = next(
                (
                    str(field["name"])
                    for field in fields
                    if field.get("type") == DataType.FLOAT_VECTOR
                    or str(field.get("type", "")).endswith("FLOAT_VECTOR")
                ),
                None,
            )
            if not vector_field:
                raise TraceSearchError("Collection 中没有 FLOAT_VECTOR 字段。")

            rows = client.query(
                collection_name=request.collection_name,
                filter="",
                output_fields=[vector_field],
                limit=1,
            )
            if not rows or vector_field not in rows[0]:
                raise TraceSearchError("Collection 中没有可用于测试的向量数据。")

            query_vector = rows[0][vector_field]
            search_started_us = time.time_ns() // 1_000
            started_ns = time.perf_counter_ns()
            results = client.search(
                collection_name=request.collection_name,
                data=[query_vector],
                anns_field=vector_field,
                limit=request.top_k,
            )
            client_latency_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
            search_finished_us = time.time_ns() // 1_000

            trace_data = self._wait_for_search_trace(
                search_started_us,
                search_finished_us,
            )
            parsed = self._parse_trace(trace_data)
            parsed.update(
                {
                    "collection_name": request.collection_name,
                    "vector_field": vector_field,
                    "top_k": request.top_k,
                    "hit_count": len(results[0]) if results else 0,
                    "client_latency_ms": round(client_latency_ms, 3),
                    "jaeger_url": (
                        f"{self.jaeger_query_url}/trace/{parsed['trace_id']}"
                    ),
                }
            )
            return parsed
        except TraceSearchError:
            raise
        except Exception as error:
            raise TraceSearchError(f"Milvus Trace 测试失败：{error}") from error
        finally:
            if client is not None:
                client.close()

    def _ensure_trace_demo(self, client: MilvusClient) -> None:
        if client.has_collection(TRACE_DEMO_COLLECTION):
            return
        with self._demo_lock:
            if client.has_collection(TRACE_DEMO_COLLECTION):
                return
            vectorizer = HashingVectorizer(
                n_features=TRACE_DEMO_DIMENSION,
                alternate_sign=False,
                norm="l2",
            )
            vectors = vectorizer.transform(TRACE_DEMO_DOCUMENTS).toarray()

            schema = MilvusClient.create_schema(auto_id=False)
            schema.add_field("id", DataType.INT64, is_primary=True)
            schema.add_field("text", DataType.VARCHAR, max_length=512)
            schema.add_field(
                "vector",
                DataType.FLOAT_VECTOR,
                dim=TRACE_DEMO_DIMENSION,
            )
            index_params = MilvusClient.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_name="vector_idx",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 16, "efConstruction": 64},
            )
            client.create_collection(
                collection_name=TRACE_DEMO_COLLECTION,
                schema=schema,
                index_params=index_params,
            )
            client.insert(
                collection_name=TRACE_DEMO_COLLECTION,
                data=[
                    {
                        "id": index,
                        "text": document,
                        "vector": vector.astype("float32").tolist(),
                    }
                    for index, (document, vector) in enumerate(
                        zip(TRACE_DEMO_DOCUMENTS, vectors, strict=True)
                    )
                ],
            )
            client.flush(TRACE_DEMO_COLLECTION)
            client.load_collection(TRACE_DEMO_COLLECTION)

    def _wait_for_search_trace(
        self,
        search_started_us: int,
        search_finished_us: int,
    ) -> dict[str, Any]:
        query = urlencode(
            {
                "service": "proxy",
                "operation": "milvus.proto.milvus.MilvusService/Search",
                "start": search_started_us - 1_000_000,
                "end": search_finished_us + 2_000_000,
                "limit": 20,
            }
        )
        url = f"{self.jaeger_query_url}/api/traces?{query}"
        last_error: Exception | None = None
        for _ in range(20):
            try:
                with urlopen(url, timeout=3) as response:
                    payload = json.load(response)
                traces = payload.get("data") or []
                if traces:
                    candidates: list[tuple[int, dict[str, Any]]] = []
                    for candidate in traces:
                        processes = candidate.get("processes") or {}
                        for span in candidate.get("spans") or []:
                            process = processes.get(span.get("processID"), {})
                            if (
                                process.get("serviceName") == "proxy"
                                and span.get("operationName")
                                == "milvus.proto.milvus.MilvusService/Search"
                            ):
                                candidates.append(
                                    (
                                        abs(
                                            int(span["startTime"])
                                            - search_started_us
                                        ),
                                        candidate,
                                    )
                                )
                                break
                    if candidates:
                        candidates.sort(key=lambda item: item[0])
                        return candidates[0][1]
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
                last_error = error
            time.sleep(0.25)
        message = "Jaeger 尚未返回本次请求的 Trace，请确认 Jaeger 正在运行。"
        if last_error:
            message = f"{message} {last_error}"
        raise TraceSearchError(message)

    @staticmethod
    def _parse_trace(trace_data: dict[str, Any]) -> dict[str, Any]:
        raw_spans = trace_data.get("spans") or []
        processes = trace_data.get("processes") or {}
        if not raw_spans:
            raise TraceSearchError("Jaeger 返回的 Trace 中没有 Span。")

        trace_start_us = min(int(span["startTime"]) for span in raw_spans)
        trace_end_us = max(
            int(span["startTime"]) + int(span["duration"])
            for span in raw_spans
        )
        parent_by_span: dict[str, str | None] = {}
        for span in raw_spans:
            parent_by_span[str(span["spanID"])] = next(
                (
                    str(reference["spanID"])
                    for reference in span.get("references", [])
                    if reference.get("refType") == "CHILD_OF"
                ),
                None,
            )

        depth_cache: dict[str, int] = {}

        def depth(span_id: str, seen: set[str] | None = None) -> int:
            if span_id in depth_cache:
                return depth_cache[span_id]
            seen = set() if seen is None else seen
            parent_id = parent_by_span.get(span_id)
            if not parent_id or parent_id not in parent_by_span or parent_id in seen:
                depth_cache[span_id] = 0
                return 0
            value = min(depth(parent_id, seen | {span_id}) + 1, 12)
            depth_cache[span_id] = value
            return value

        spans: list[dict[str, Any]] = []
        for span in raw_spans:
            tags = {
                str(tag.get("key")): tag.get("value")
                for tag in span.get("tags", [])
            }
            process = processes.get(span.get("processID"), {})
            span_id = str(span["spanID"])
            spans.append(
                {
                    "span_id": span_id,
                    "parent_span_id": parent_by_span.get(span_id),
                    "service": process.get("serviceName", "unknown"),
                    "operation": span.get("operationName", "unknown"),
                    "start_offset_ms": round(
                        (int(span["startTime"]) - trace_start_us) / 1_000, 3
                    ),
                    "duration_ms": round(int(span["duration"]) / 1_000, 3),
                    "depth": depth(span_id),
                    "error": bool(tags.get("error"))
                    or tags.get("otel.status_code") == "ERROR",
                }
            )
        spans.sort(key=lambda item: (item["start_offset_ms"], -item["duration_ms"]))
        return {
            "trace_id": str(trace_data["traceID"]),
            "total_duration_ms": round((trace_end_us - trace_start_us) / 1_000, 3),
            "spans": spans,
        }
