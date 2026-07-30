"use client";

import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react";

type ConcurrencyStage = {
  stage_index: number;
  concurrency: number;
  started_at: string | null;
  finished_at: string | null;
  duration_seconds: number | null;
  qps: number | null;
  latency_avg_ms: number | null;
  latency_p95_ms: number | null;
  latency_p99_ms: number | null;
  querynode_cpu_avg_cores: number | null;
  querynode_cpu_peak_cores: number | null;
  querynode_cpu_avg_percent: number | null;
  querynode_cpu_peak_percent: number | null;
  querynode_cpu_sample_count: number | null;
  monitoring_error: string | null;
};

type IndexParameterValue = number | string | boolean | null;

type BenchmarkRun = {
  configuration_key: string | null;
  run_id: string;
  case_index: number;
  task_label: string;
  status: string;
  created_at: string;
  case_type: string;
  database: string;
  db_label: string | null;
  command: string;
  index_type: string;
  metric_type: string;
  index_parameters: Record<string, IndexParameterValue>;
  search_parameters: Record<string, IndexParameterValue>;
  hnsw_m: number | null;
  hnsw_ef_construction: number | null;
  hnsw_ef_search: number | null;
  top_k: number;
  num_shards: number | null;
  replica_number: number | null;
  load_concurrency: number | null;
  concurrency_duration_seconds: number | null;
  concurrency_timeout_seconds: number | null;
  insert_duration_seconds: number | null;
  optimize_duration_seconds: number | null;
  load_duration_seconds: number | null;
  recall: number | null;
  ndcg: number | null;
  serial_latency_p95_ms: number | null;
  serial_latency_p99_ms: number | null;
  max_qps: number | null;
  vector_index_memory_bytes: number | null;
  vector_index_memory_mib: number | null;
  vector_index_memory_collected_at: string | null;
  monitoring_error: string | null;
  stages: ConcurrencyStage[];
};

type BenchmarkListResponse = {
  items: BenchmarkRun[];
  total: number;
  limit: number;
  offset: number;
};

type MetricSummary = {
  mean: number | null;
  stddev: number | null;
  minimum: number | null;
  maximum: number | null;
  sample_count: number;
};

type BenchmarkAggregate = {
  configuration_key: string;
  sample_count: number;
  latest_created_at: string;
  latest_run_id: string;
  case_type: string;
  db_label: string | null;
  command: string;
  index_type: string;
  metric_type: string;
  index_parameters: Record<string, IndexParameterValue>;
  search_parameters: Record<string, IndexParameterValue>;
  hnsw_m: number | null;
  hnsw_ef_construction: number | null;
  hnsw_ef_search: number | null;
  top_k: number;
  num_shards: number | null;
  replica_number: number | null;
  load_concurrency: number | null;
  concurrency_duration_seconds: number | null;
  concurrency_timeout_seconds: number | null;
  executed_stages: string[];
  stage_index: number;
  concurrency: number;
  qps: MetricSummary;
  latency_avg_ms: MetricSummary;
  latency_p95_ms: MetricSummary;
  latency_p99_ms: MetricSummary;
  recall: MetricSummary;
  ndcg: MetricSummary;
  serial_latency_p99_ms: MetricSummary;
  insert_duration_seconds: MetricSummary;
  optimize_duration_seconds: MetricSummary;
  load_duration_seconds: MetricSummary;
  querynode_cpu_avg_percent: MetricSummary;
  querynode_cpu_peak_percent: MetricSummary;
  vector_index_memory_mib: MetricSummary;
};

type BenchmarkAggregateListResponse = {
  items: BenchmarkAggregate[];
  total: number;
  limit: number;
  offset: number;
};

type PublicBenchmarkSnapshot = {
  generated_at: string;
  runs: BenchmarkListResponse;
  aggregates: BenchmarkAggregateListResponse;
  profiles: IndexProfile[];
};

type BenchmarkRow = {
  run: BenchmarkRun;
  stage: ConcurrencyStage;
};

type ResultViewMode = "index" | "aggregate" | "raw";

type IndexSummary = {
  key: string;
  indexType: string;
  aggregates: BenchmarkAggregate[];
  configurationCount: number;
  concurrency: number;
  concurrencyDuration: number | null;
};

type SortKey =
  | "indexType"
  | "indexParams"
  | "insert"
  | "optimize"
  | "concurrency"
  | "p99"
  | "recall"
  | "vectorIndex";

type SortDirection = "asc" | "desc";
type SortValue = number | string | null;

type AnalysisPoint = {
  key: string;
  indexType: string;
  xValue: Exclude<IndexParameterValue, null>;
  xLabel: string;
  parameterLabel: string;
  series: string;
  p99: MetricSummary;
  recall: MetricSummary;
  memory: MetricSummary;
};

type AnalysisMetric = "p99" | "recall" | "memory";

type AnalysisRecommendation = {
  key: "p99" | "recall" | "memory" | "balanced";
  label: string;
  description: string;
  point: AnalysisPoint;
};

type RecommendationAnalysis = {
  recommendations: AnalysisRecommendation[];
  configurationCount: number;
};

type ChartHitPoint = {
  x: number;
  y: number;
  radius: number;
  point: AnalysisPoint;
};

type ChartTooltip = {
  left: number;
  top: number;
  placeBelow: boolean;
  point: AnalysisPoint;
};

const CHART_COLORS = [
  "#008f5d",
  "#b94c3c",
  "#2e6fb0",
  "#9b6a13",
  "#6c4ab6",
  "#16858c",
];

const TABLE_COLUMNS: { key: SortKey; label: string }[] = [
  { key: "indexType", label: "索引类型" },
  { key: "indexParams", label: "索引参数" },
  { key: "p99", label: "P99" },
  { key: "recall", label: "Recall" },
  { key: "vectorIndex", label: "Vector Index" },
  { key: "insert", label: "Insert" },
  { key: "optimize", label: "Optimize" },
  { key: "concurrency", label: "并发" },
];

type BenchmarkParameters = {
  command: string;
  uri: string;
  num_shards: number;
  replica_number: number;
  case_type: string;
  drop_old: boolean;
  load: boolean;
  load_concurrency: number;
  search_serial: boolean;
  search_concurrent: boolean;
  k: number;
  concurrency_duration: number;
  num_concurrency: number[];
  concurrency_timeout: number;
  index_parameters: Record<string, Exclude<IndexParameterValue, null>>;
  db_label: string;
};

type IndexParameterDefinition = {
  name: string;
  label: string;
  kind: "integer" | "number" | "boolean" | "choice";
  default: Exclude<IndexParameterValue, null>;
  minimum?: number | null;
  maximum?: number | null;
  options?: Exclude<IndexParameterValue, null>[] | null;
  description: string;
};

type IndexProfile = {
  command: string;
  label: string;
  index_type: string;
  parameters: IndexParameterDefinition[];
};

type BenchmarkJob = {
  job_id: string;
  status: "queued" | "running" | "cancelling" | "succeeded" | "failed" | "cancelled";
  phase: string;
  parameters: BenchmarkParameters;
  task_label: string;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  elapsed_seconds: number | null;
  exit_code: number | null;
  error: string | null;
  result_run_id: string | null;
  result_case_index: number | null;
  repetitions: number;
  configuration_count: number;
  total_runs: number;
  completed_runs: number;
  current_run_number: number;
  result_run_ids: string[];
  log_tail: string[];
};

type TraceSpan = {
  span_id: string;
  parent_span_id: string | null;
  service: string;
  operation: string;
  start_offset_ms: number;
  duration_ms: number;
  depth: number;
  error: boolean;
};

type TraceSearchResult = {
  trace_id: string;
  jaeger_url: string;
  collection_name: string;
  vector_field: string;
  top_k: number;
  hit_count: number;
  client_latency_ms: number;
  total_duration_ms: number;
  spans: TraceSpan[];
};

type TuningAgentResult = {
  recall_target: number;
  model: string;
  answer: string;
  tools_used: string[];
  history_configuration_count: number;
  benchmark_tool_call_count: number;
  benchmark_run_count: number;
  benchmark_runs: Array<Record<string, unknown>>;
};

const API_BASE_URL = (
  process.env.NEXT_PUBLIC_BENCHMARK_API_URL ?? "http://127.0.0.1:8765"
).replace(/\/$/, "");
const READ_ONLY_DEMO =
  process.env.NEXT_PUBLIC_READ_ONLY_DEMO === "true";
const SITE_BASE_PATH = (
  process.env.NEXT_PUBLIC_SITE_BASE_PATH ?? "/"
).replace(/\/?$/, "/");
const PUBLIC_SNAPSHOT_URL = `${SITE_BASE_PATH}benchmark-snapshot.json`;
const PUBLIC_AGENT_DEMO: TuningAgentResult = {
  recall_target: 0.95,
  model: "qwen-plus",
  tools_used: ["run_benchmark"],
  history_configuration_count: 33,
  benchmark_tool_call_count: 3,
  benchmark_run_count: 3,
  benchmark_runs: [
    {
      index_type: "IVF_SQ8",
      search_parameters: { nprobe: 32 },
      recall: 0.9208,
      p99_ms: 10.3,
      qps: 96.8,
    },
    {
      index_type: "IVF_SQ8",
      search_parameters: { nprobe: 64 },
      recall: 0.9637,
      p99_ms: 12.84,
      qps: 78.5,
    },
    {
      index_type: "IVF_SQ8",
      search_parameters: { nprobe: 128 },
      recall: 0.9814,
      p99_ms: 18.62,
      qps: 54.1,
    },
  ],
  answer: `目标与历史基线
- Recall 目标为 95%，当前保留索引为 IVF_SQ8，构建参数 nlist=256。
- 历史配置 nprobe=32 的 Recall 为 92.08%，尚未达到目标。

压测计划与执行结果
- 第 1 次：nprobe=32，Recall 92.08%，P99 10.30 ms，QPS 96.8，未达标。
- 第 2 次：nprobe=64，Recall 96.37%，P99 12.84 ms，QPS 78.5，达到目标。
- 第 3 次：nprobe=128，Recall 98.14%，P99 18.62 ms，QPS 54.1，达到目标。

最终推荐配置
- 索引类型：IVF_SQ8
- 构建参数：nlist=256
- 搜索参数：nprobe=64
- 实验结果：Recall 96.37%，P99 12.84 ms，QPS 78.5

推荐依据
- nprobe=64 是本轮达到 Recall 目标的最小搜索范围。
- 相比 nprobe=128，Recall 仅降低 1.77 个百分点，但 P99 降低 5.78 ms，QPS 提升约 45%。
- 搜索参数不会改变已构建索引的内存占用，本轮重点比较 Recall 与查询性能。

后续调优建议
- 固定 nlist=256，在 nprobe=48、56、64 之间补充实验，寻找满足 Recall 目标的更低延迟边界。
- 对候选配置重复运行 3 次，使用均值和标准差验证结果稳定性。`,
};

const DEFAULT_PARAMETERS: BenchmarkParameters = {
  command: "milvushnsw",
  uri: "http://localhost:19530",
  num_shards: 1,
  replica_number: 1,
  case_type: "Performance1536D50K",
  drop_old: true,
  load: true,
  load_concurrency: 4,
  search_serial: true,
  search_concurrent: true,
  k: 100,
  concurrency_duration: 30,
  num_concurrency: [1],
  concurrency_timeout: 3600,
  index_parameters: {
    m: 16,
    ef_construction: 128,
    ef_search: 128,
  },
  db_label: "local-cluster-8c10_5g-2_6_21",
};

const CASE_TYPE_METRIC_TYPES: Record<string, string> = {
  Performance1536D50K: "COSINE",
};

const HNSW_PARAMETER_DEFINITIONS: IndexParameterDefinition[] = [
  { name: "m", label: "M", kind: "integer", default: 16, minimum: 4, maximum: 128, description: "图中每个节点的最大邻接数" },
  { name: "ef_construction", label: "efConstruction", kind: "integer", default: 128, minimum: 4, maximum: 1024, description: "构建索引时的候选集合大小" },
  { name: "ef_search", label: "efSearch", kind: "integer", default: 128, minimum: 1, maximum: 2048, description: "查询候选集合，必须不小于 TopK" },
];

const REFINE_PARAMETER_DEFINITIONS: IndexParameterDefinition[] = [
  { name: "refine", label: "refine", kind: "boolean", default: true, options: [true, false], description: "是否保留原始数据用于精排" },
  { name: "refine_type", label: "refineType", kind: "choice", default: "FP32", options: ["SQ6", "SQ8", "BF16", "FP16", "FP32"], description: "精排数据的存储类型" },
  { name: "refine_k", label: "refineK", kind: "number", default: 1, minimum: 1, maximum: 10000, description: "精排候选数相对 TopK 的放大倍数" },
];

const FALLBACK_INDEX_PROFILES: IndexProfile[] = [
  {
    command: "milvushnsw",
    label: "HNSW",
    index_type: "HNSW",
    parameters: [...HNSW_PARAMETER_DEFINITIONS],
  },
  {
    command: "milvushnswsq",
    label: "HNSW_SQ",
    index_type: "HNSW_SQ",
    parameters: [
      ...HNSW_PARAMETER_DEFINITIONS,
      { name: "sq_type", label: "sqType", kind: "choice", default: "SQ8", options: ["SQ4U", "SQ6", "SQ8", "BF16", "FP16", "FP32"], description: "标量量化的数据类型" },
      ...REFINE_PARAMETER_DEFINITIONS,
    ],
  },
  {
    command: "milvushnswpq",
    label: "HNSW_PQ",
    index_type: "HNSW_PQ",
    parameters: [
      ...HNSW_PARAMETER_DEFINITIONS,
      { name: "nbits", label: "nbits", kind: "integer", default: 8, minimum: 1, maximum: 65536, description: "PQ 编码使用的位数" },
      ...REFINE_PARAMETER_DEFINITIONS,
    ],
  },
  {
    command: "milvushnswprq",
    label: "HNSW_PRQ",
    index_type: "HNSW_PRQ",
    parameters: [
      ...HNSW_PARAMETER_DEFINITIONS,
      { name: "nbits", label: "nbits", kind: "integer", default: 8, minimum: 1, maximum: 65536, description: "PQ 编码使用的位数" },
      { name: "nrq", label: "nrq", kind: "integer", default: 2, minimum: 1, maximum: 16, description: "残差子量化器数量" },
      ...REFINE_PARAMETER_DEFINITIONS,
    ],
  },
  {
    command: "milvusivfflat",
    label: "IVF_FLAT",
    index_type: "IVF_FLAT",
    parameters: [
      { name: "nlist", label: "nlist", kind: "integer", default: 128, minimum: 1, maximum: 65536, description: "聚类中心/分桶数量" },
      { name: "nprobe", label: "nprobe", kind: "integer", default: 16, minimum: 1, maximum: 65536, description: "查询探测桶数，不能大于 nlist" },
    ],
  },
  {
    command: "milvusivfsq8",
    label: "IVF_SQ8",
    index_type: "IVF_SQ8",
    parameters: [
      { name: "nlist", label: "nlist", kind: "integer", default: 128, minimum: 1, maximum: 65536, description: "聚类中心/分桶数量" },
      { name: "nprobe", label: "nprobe", kind: "integer", default: 16, minimum: 1, maximum: 65536, description: "查询探测桶数，不能大于 nlist" },
    ],
  },
  {
    command: "milvusautoindex",
    label: "AUTOINDEX",
    index_type: "AUTOINDEX",
    parameters: [],
  },
  {
    command: "milvusflat",
    label: "FLAT",
    index_type: "FLAT",
    parameters: [],
  },
];

const PHASE_LABELS: Record<string, string> = {
  queued: "等待启动",
  starting: "准备压测环境",
  loading: "导入 50K 向量",
  optimizing: "Flush / Compaction / 构建索引",
  concurrent_search: "并发搜索",
  serial_search: "串行精度搜索",
  saving: "保存 VectorDBBench 结果",
  collecting_metrics: "采集监控并写入 SQLite",
  completed: "已完成",
  cancelling: "正在取消",
  cancelled: "已取消",
  failed: "运行失败",
};

function number(value: number | null, digits = 2) {
  return value === null ? "—" : value.toFixed(digits);
}

function recallPercent(value: number | null) {
  return value === null ? "—" : `${(value * 100).toFixed(2)}%`;
}

function meanWithStddev(
  metric: MetricSummary,
  digits = 2,
  suffix = "",
) {
  if (metric.mean === null) return "—";
  const mean = metric.mean.toFixed(digits);
  if (metric.sample_count < 2 || metric.stddev === null) {
    return `${mean}${suffix}`;
  }
  return `${mean} ± ${metric.stddev.toFixed(digits)}${suffix}`;
}

function metricFor(point: AnalysisPoint, metric: AnalysisMetric) {
  if (metric === "p99") return point.p99;
  if (metric === "recall") return point.recall;
  return point.memory;
}

function formatChartValue(
  value: number,
  metric: AnalysisMetric,
  axis = false,
) {
  if (metric === "recall") return `${(value * 100).toFixed(axis ? 0 : 2)}%`;
  if (metric === "p99") return `${value.toFixed(axis ? 1 : 2)} ms`;
  return `${value.toFixed(axis ? 0 : 2)} MiB`;
}

function formatChartSummary(summary: MetricSummary, metric: AnalysisMetric) {
  if (summary.mean === null) return "—";
  const mean = formatChartValue(summary.mean, metric);
  if (summary.sample_count < 2 || summary.stddev === null) return mean;
  return `${mean} ± ${formatChartValue(summary.stddev, metric)}`;
}

function nearestChartPoint(
  canvas: HTMLCanvasElement,
  clientX: number,
  clientY: number,
  points: ChartHitPoint[],
) {
  const bounds = canvas.getBoundingClientRect();
  const x = clientX - bounds.left;
  const y = clientY - bounds.top;
  let nearest: ChartHitPoint | null = null;
  let nearestDistance = Number.POSITIVE_INFINITY;
  points.forEach((point) => {
    const distance = Math.hypot(point.x - x, point.y - y);
    if (distance <= point.radius + 7 && distance < nearestDistance) {
      nearest = point;
      nearestDistance = distance;
    }
  });
  return { hit: nearest, bounds };
}

function hasCompleteRecommendationMetrics(point: AnalysisPoint) {
  return point.p99.mean !== null
    && point.recall.mean !== null
    && point.memory.mean !== null;
}

function recommendationMetricRange(
  points: AnalysisPoint[],
  value: (point: AnalysisPoint) => number,
) {
  const values = points.map(value);
  return {
    minimum: Math.min(...values),
    maximum: Math.max(...values),
    range: Math.max(Math.max(...values) - Math.min(...values), 0.000001),
  };
}

function analyzeRecommendations(points: AnalysisPoint[]): RecommendationAnalysis {
  const complete = points.filter(hasCompleteRecommendationMetrics);
  if (complete.length === 0) {
    return {
      recommendations: [],
      configurationCount: 0,
    };
  }

  const lowestP99 = complete.reduce((best, point) =>
    (point.p99.mean as number) < (best.p99.mean as number) ? point : best
  );
  const highestRecall = complete.reduce((best, point) =>
    (point.recall.mean as number) > (best.recall.mean as number) ? point : best
  );
  const lowestMemory = complete.reduce((best, point) =>
    (point.memory.mean as number) < (best.memory.mean as number) ? point : best
  );
  const p99Range = recommendationMetricRange(
    complete,
    (point) => point.p99.mean as number,
  );
  const recallRange = recommendationMetricRange(
    complete,
    (point) => point.recall.mean as number,
  );
  const memoryRange = recommendationMetricRange(
    complete,
    (point) => point.memory.mean as number,
  );
  const balancedScore = (point: AnalysisPoint) => (
    (((point.p99.mean as number) - p99Range.minimum) / p99Range.range)
    + ((recallRange.maximum - (point.recall.mean as number)) / recallRange.range)
    + (((point.memory.mean as number) - memoryRange.minimum) / memoryRange.range)
  ) / 3;
  const balanced = complete.reduce((best, point) =>
    balancedScore(point) < balancedScore(best) ? point : best
  );

  return {
    recommendations: [
      {
        key: "p99",
        label: "P99 推荐",
        description: "当前索引配置中 P99 最低",
        point: lowestP99,
      },
      {
        key: "recall",
        label: "Recall 推荐",
        description: "当前索引配置中 Recall 最高",
        point: highestRecall,
      },
      {
        key: "memory",
        label: "内存推荐",
        description: "当前索引配置中 Vector Index 内存最低",
        point: lowestMemory,
      },
      {
        key: "balanced",
        label: "综合推荐",
        description: "当前索引配置归一化后等权得分最低",
        point: balanced,
      },
    ],
    configurationCount: complete.length,
  };
}

function categoryKey(value: Exclude<IndexParameterValue, null>) {
  return `${typeof value}:${String(value)}`;
}

function sortedCategories(points: AnalysisPoint[]) {
  const values = points.reduce<Exclude<IndexParameterValue, null>[]>(
    (result, point) => (
      result.some((value) => categoryKey(value) === categoryKey(point.xValue))
        ? result
        : [...result, point.xValue]
    ),
    [],
  );
  return values.sort((left, right) => {
    if (typeof left === "number" && typeof right === "number") {
      return left - right;
    }
    return String(left).localeCompare(String(right), "zh-CN");
  });
}

function prepareCanvas(canvas: HTMLCanvasElement, height: number) {
  const width = Math.max(canvas.parentElement?.clientWidth ?? 0, 320);
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  canvas.style.height = `${height}px`;
  const context = canvas.getContext("2d");
  if (!context) return null;
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  return { context, width };
}

function MetricTrendChart({
  points,
  metric,
  title,
}: {
  points: AnalysisPoint[];
  metric: AnalysisMetric;
  title: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const hitPointsRef = useRef<ChartHitPoint[]>([]);
  const [tooltip, setTooltip] = useState<ChartTooltip | null>(null);
  const validPoints = points.filter(
    (point) => metricFor(point, metric).mean !== null,
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const draw = () => {
      const prepared = prepareCanvas(canvas, 250);
      if (!prepared) return;
      const { context, width } = prepared;
      const height = 250;
      const padding = { left: 58, right: 18, top: 20, bottom: 48 };
      const plotWidth = width - padding.left - padding.right;
      const plotHeight = height - padding.top - padding.bottom;
      hitPointsRef.current = [];
      const categories = sortedCategories(validPoints);
      const values = validPoints.flatMap((point) => {
        const summary = metricFor(point, metric);
        if (summary.mean === null) return [];
        const deviation = summary.stddev ?? 0;
        return [summary.mean - deviation, summary.mean + deviation];
      });

      context.font = "11px sans-serif";
      context.fillStyle = "#69766f";
      if (categories.length === 0 || values.length === 0) {
        context.textAlign = "center";
        context.fillText("暂无可绘制数据", width / 2, height / 2);
        return;
      }

      let minimum = Math.min(...values);
      let maximum = Math.max(...values);
      const initialRange = maximum - minimum;
      const paddingValue = initialRange > 0
        ? initialRange * 0.15
        : Math.max(Math.abs(maximum) * 0.1, metric === "recall" ? 0.01 : 1);
      minimum -= paddingValue;
      maximum += paddingValue;
      if (metric === "recall") {
        minimum = Math.max(0, minimum);
        maximum = Math.min(1, maximum);
      } else {
        minimum = Math.max(0, minimum);
      }
      if (maximum <= minimum) maximum = minimum + 1;

      const xPosition = (value: Exclude<IndexParameterValue, null>) => {
        const index = categories.findIndex(
          (candidate) => categoryKey(candidate) === categoryKey(value),
        );
        return padding.left + (
          categories.length === 1
            ? plotWidth / 2
            : (index / (categories.length - 1)) * plotWidth
        );
      };
      const yPosition = (value: number) =>
        padding.top + ((maximum - value) / (maximum - minimum)) * plotHeight;

      context.strokeStyle = "#e3e9e5";
      context.lineWidth = 1;
      context.textAlign = "right";
      context.textBaseline = "middle";
      for (let tick = 0; tick <= 4; tick += 1) {
        const value = minimum + ((maximum - minimum) * tick) / 4;
        const y = yPosition(value);
        context.beginPath();
        context.moveTo(padding.left, y);
        context.lineTo(width - padding.right, y);
        context.stroke();
        context.fillStyle = "#69766f";
        context.fillText(
          formatChartValue(value, metric, true),
          padding.left - 8,
          y,
        );
      }

      context.textAlign = "center";
      context.textBaseline = "top";
      categories.forEach((category) => {
        context.fillStyle = "#69766f";
        context.fillText(
          String(category),
          xPosition(category),
          height - padding.bottom + 12,
        );
      });

      const seriesNames = [...new Set(validPoints.map((point) => point.series))];
      seriesNames.forEach((seriesName, seriesIndex) => {
        const color = CHART_COLORS[seriesIndex % CHART_COLORS.length];
        const seriesPoints = validPoints
          .filter((point) => point.series === seriesName)
          .sort(
            (left, right) =>
              categories.findIndex(
                (category) => categoryKey(category) === categoryKey(left.xValue),
              )
              - categories.findIndex(
                (category) => categoryKey(category) === categoryKey(right.xValue),
              ),
          );

        context.strokeStyle = color;
        context.lineWidth = 2;
        if (seriesPoints.length > 1) {
          context.beginPath();
          seriesPoints.forEach((point, index) => {
            const mean = metricFor(point, metric).mean;
            if (mean === null) return;
            const x = xPosition(point.xValue);
            const y = yPosition(mean);
            if (index === 0) context.moveTo(x, y);
            else context.lineTo(x, y);
          });
          context.stroke();
        }

        seriesPoints.forEach((point) => {
          const summary = metricFor(point, metric);
          if (summary.mean === null) return;
          const x = xPosition(point.xValue);
          const y = yPosition(summary.mean);
          const deviation = summary.stddev ?? 0;
          if (deviation > 0) {
            const top = yPosition(Math.min(maximum, summary.mean + deviation));
            const bottom = yPosition(Math.max(minimum, summary.mean - deviation));
            context.strokeStyle = color;
            context.lineWidth = 1;
            context.beginPath();
            context.moveTo(x, top);
            context.lineTo(x, bottom);
            context.moveTo(x - 4, top);
            context.lineTo(x + 4, top);
            context.moveTo(x - 4, bottom);
            context.lineTo(x + 4, bottom);
            context.stroke();
          }
          context.fillStyle = color;
          context.beginPath();
          context.arc(x, y, 4.5, 0, Math.PI * 2);
          context.fill();
          context.strokeStyle = "#ffffff";
          context.lineWidth = 1.5;
          context.stroke();
          hitPointsRef.current.push({
            x,
            y,
            radius: 4.5,
            point,
          });
        });
      });
    };

    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [metric, validPoints]);

  const seriesNames = [...new Set(validPoints.map((point) => point.series))];
  const handlePointerMove = (clientX: number, clientY: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const { hit, bounds } = nearestChartPoint(
      canvas,
      clientX,
      clientY,
      hitPointsRef.current,
    );
    if (!hit) {
      setTooltip(null);
      return;
    }
    setTooltip({
      left: Math.min(Math.max(hit.x, 105), Math.max(bounds.width - 105, 105)),
      top: hit.y,
      placeBelow: hit.y < 90,
      point: hit.point,
    });
  };
  return (
    <article className="analysis-chart-card">
      <div className="analysis-chart-title">
        <strong>{title}</strong>
        <span>均值 ± 标准差</span>
      </div>
      <div className="analysis-canvas-wrap">
        <canvas
          ref={canvasRef}
          role="img"
          aria-label={`${title}参数趋势图`}
          onMouseMove={(event) => handlePointerMove(event.clientX, event.clientY)}
          onMouseLeave={() => setTooltip(null)}
          onTouchStart={(event) => {
            const touch = event.touches[0];
            if (touch) handlePointerMove(touch.clientX, touch.clientY);
          }}
        />
        {tooltip ? (
          <div
            className={`analysis-chart-tooltip${tooltip.placeBelow ? " tooltip-below" : ""}`}
            style={{ left: tooltip.left, top: tooltip.top }}
          >
            <strong>{tooltip.point.series}</strong>
            <span>{tooltip.point.xLabel}</span>
            <span>
              {metric === "p99" ? "P99" : metric === "recall" ? "Recall" : "Vector Index"}
              {" "}
              <b>{formatChartSummary(metricFor(tooltip.point, metric), metric)}</b>
            </span>
            <small>样本数 n={metricFor(tooltip.point, metric).sample_count}</small>
          </div>
        ) : null}
      </div>
      <div className="chart-legend">
        {seriesNames.map((series, index) => (
          <span key={series}>
            <i style={{ background: CHART_COLORS[index % CHART_COLORS.length] }} />
            {series}
          </span>
        ))}
      </div>
    </article>
  );
}

function TradeoffChart({ points }: { points: AnalysisPoint[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const hitPointsRef = useRef<ChartHitPoint[]>([]);
  const [tooltip, setTooltip] = useState<ChartTooltip | null>(null);
  const validPoints = points.filter(
    (point) => point.p99.mean !== null && point.recall.mean !== null,
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const draw = () => {
      const prepared = prepareCanvas(canvas, 250);
      if (!prepared) return;
      const { context, width } = prepared;
      const height = 250;
      const padding = { left: 58, right: 24, top: 20, bottom: 48 };
      const plotWidth = width - padding.left - padding.right;
      const plotHeight = height - padding.top - padding.bottom;
      hitPointsRef.current = [];
      if (validPoints.length === 0) {
        context.fillStyle = "#69766f";
        context.font = "11px sans-serif";
        context.textAlign = "center";
        context.fillText("暂无可绘制数据", width / 2, height / 2);
        return;
      }

      const xValues = validPoints.map((point) => point.p99.mean as number);
      const yValues = validPoints.map((point) => point.recall.mean as number);
      const memoryValues = validPoints
        .map((point) => point.memory.mean)
        .filter((value): value is number => value !== null);
      const expand = (minimum: number, maximum: number, floor: number, ceil?: number) => {
        const range = maximum - minimum;
        const margin = range > 0 ? range * 0.15 : Math.max(maximum * 0.1, 0.01);
        return [
          Math.max(floor, minimum - margin),
          ceil === undefined ? maximum + margin : Math.min(ceil, maximum + margin),
        ] as const;
      };
      const [xMinimum, xMaximum] = expand(
        Math.min(...xValues),
        Math.max(...xValues),
        0,
      );
      const [yMinimum, yMaximum] = expand(
        Math.min(...yValues),
        Math.max(...yValues),
        0,
        1,
      );
      const xRange = Math.max(xMaximum - xMinimum, 0.001);
      const yRange = Math.max(yMaximum - yMinimum, 0.001);
      const memoryMinimum = memoryValues.length ? Math.min(...memoryValues) : 0;
      const memoryMaximum = memoryValues.length ? Math.max(...memoryValues) : 1;
      const memoryRange = Math.max(memoryMaximum - memoryMinimum, 1);
      const xPosition = (value: number) =>
        padding.left + ((value - xMinimum) / xRange) * plotWidth;
      const yPosition = (value: number) =>
        padding.top + ((yMaximum - value) / yRange) * plotHeight;

      context.font = "11px sans-serif";
      context.strokeStyle = "#e3e9e5";
      context.lineWidth = 1;
      for (let tick = 0; tick <= 4; tick += 1) {
        const xValue = xMinimum + (xRange * tick) / 4;
        const yValue = yMinimum + (yRange * tick) / 4;
        const x = xPosition(xValue);
        const y = yPosition(yValue);
        context.beginPath();
        context.moveTo(x, padding.top);
        context.lineTo(x, height - padding.bottom);
        context.moveTo(padding.left, y);
        context.lineTo(width - padding.right, y);
        context.stroke();
        context.fillStyle = "#69766f";
        context.textAlign = "center";
        context.textBaseline = "top";
        context.fillText(`${xValue.toFixed(1)}`, x, height - padding.bottom + 12);
        context.textAlign = "right";
        context.textBaseline = "middle";
        context.fillText(`${(yValue * 100).toFixed(0)}%`, padding.left - 8, y);
      }

      const seriesNames = [...new Set(validPoints.map((point) => point.series))];
      validPoints.forEach((point) => {
        const p99 = point.p99.mean as number;
        const recall = point.recall.mean as number;
        const memory = point.memory.mean;
        const radius = memory === null
          ? 6
          : 6 + ((memory - memoryMinimum) / memoryRange) * 10;
        const seriesIndex = seriesNames.indexOf(point.series);
        context.globalAlpha = 0.78;
        context.fillStyle = CHART_COLORS[seriesIndex % CHART_COLORS.length];
        context.beginPath();
        context.arc(xPosition(p99), yPosition(recall), radius, 0, Math.PI * 2);
        context.fill();
        context.globalAlpha = 1;
        context.strokeStyle = "#ffffff";
        context.lineWidth = 1.5;
        context.stroke();
        hitPointsRef.current.push({
          x: xPosition(p99),
          y: yPosition(recall),
          radius,
          point,
        });
      });
    };

    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [validPoints]);

  const handlePointerMove = (clientX: number, clientY: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const { hit, bounds } = nearestChartPoint(
      canvas,
      clientX,
      clientY,
      hitPointsRef.current,
    );
    if (!hit) {
      setTooltip(null);
      return;
    }
    setTooltip({
      left: Math.min(Math.max(hit.x, 105), Math.max(bounds.width - 105, 105)),
      top: hit.y,
      placeBelow: hit.y < 105,
      point: hit.point,
    });
  };
  return (
    <article className="analysis-chart-card">
      <div className="analysis-chart-title">
        <strong>P99–Recall–内存权衡</strong>
        <span>越靠左上越好，气泡越小内存越低</span>
      </div>
      <div className="analysis-canvas-wrap">
        <canvas
          ref={canvasRef}
          role="img"
          aria-label="P99、Recall 与索引内存权衡气泡图"
          onMouseMove={(event) => handlePointerMove(event.clientX, event.clientY)}
          onMouseLeave={() => setTooltip(null)}
          onTouchStart={(event) => {
            const touch = event.touches[0];
            if (touch) handlePointerMove(touch.clientX, touch.clientY);
          }}
        />
        {tooltip ? (
          <div
            className={`analysis-chart-tooltip tradeoff-tooltip${tooltip.placeBelow ? " tooltip-below" : ""}`}
            style={{ left: tooltip.left, top: tooltip.top }}
          >
            <strong>{tooltip.point.indexType}</strong>
            <span className="tooltip-parameters">{tooltip.point.parameterLabel}</span>
            <span>P99 <b>{formatChartSummary(tooltip.point.p99, "p99")}</b></span>
            <span>Recall <b>{formatChartSummary(tooltip.point.recall, "recall")}</b></span>
            <span>Vector Index <b>{formatChartSummary(tooltip.point.memory, "memory")}</b></span>
            <small>
              样本数 n={Math.max(
                tooltip.point.p99.sample_count,
                tooltip.point.recall.sample_count,
                tooltip.point.memory.sample_count,
              )}
            </small>
          </div>
        ) : null}
      </div>
      <div className="tradeoff-axis-note">
        <span>横轴 P99（ms）</span>
        <span>纵轴 Recall</span>
      </div>
    </article>
  );
}

function RecommendationSection({
  analysis,
  indexType,
}: {
  analysis: RecommendationAnalysis;
  indexType?: string;
}) {
  if (analysis.recommendations.length === 0) return null;
  return (
    <div className="analysis-recommendations within-index-recommendations">
      <div className="recommendation-heading">
        <div>
          <span>INDEX PARAMETER RECOMMENDATIONS</span>
          <strong>{indexType ?? "当前索引"} 的参数配置建议</strong>
        </div>
        <p>
          所有指标完整的配置均参与，不设置样本数门槛。
          当前索引共比较 {analysis.configurationCount} 组完整配置。
          单项建议直接比较数值大小，综合建议按当前范围归一化后等权计算。
        </p>
      </div>
      <div className="recommendation-grid">
        {analysis.recommendations.map((recommendation) => (
          <article
            className={`recommendation-card recommendation-${recommendation.key}`}
            key={recommendation.key}
          >
            <div className="recommendation-card-title">
              <span>{recommendation.label}</span>
              <small>{recommendation.description}</small>
            </div>
            <strong>{recommendation.point.indexType}</strong>
            <p>{recommendation.point.parameterLabel}</p>
            <dl>
              <div>
                <dt>P99</dt>
                <dd>{formatChartSummary(recommendation.point.p99, "p99")}</dd>
              </div>
              <div>
                <dt>Recall</dt>
                <dd>{formatChartSummary(recommendation.point.recall, "recall")}</dd>
              </div>
              <div>
                <dt>Vector Index</dt>
                <dd>{formatChartSummary(recommendation.point.memory, "memory")}</dd>
              </div>
            </dl>
          </article>
        ))}
      </div>
    </div>
  );
}

function parseIndexParameterList(
  value: string,
  definition: IndexParameterDefinition,
): Exclude<IndexParameterValue, null>[] {
  const values = value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean)
    .map((item) => {
      if (definition.kind === "integer") {
        const parsed = Number(item);
        return Number.isInteger(parsed) ? parsed : null;
      }
      if (definition.kind === "number") {
        const parsed = Number(item);
        return Number.isFinite(parsed) ? parsed : null;
      }
      if (definition.kind === "boolean") {
        if (item.toLowerCase() === "true") return true;
        if (item.toLowerCase() === "false") return false;
        return null;
      }
      return definition.options?.find(
        (option) => String(option).toLowerCase() === item.toLowerCase(),
      ) ?? null;
    })
    .filter((item): item is Exclude<IndexParameterValue, null> => item !== null);
  return values.filter(
    (item, index) => values.findIndex((candidate) => candidate === item) === index,
  );
}

function parameterHelp(definition: IndexParameterDefinition) {
  if (definition.options?.length) {
    return `可选 ${definition.options.map(String).join(" / ")}`;
  }
  if (definition.minimum !== null && definition.minimum !== undefined
    && definition.maximum !== null && definition.maximum !== undefined) {
    return `范围 ${definition.minimum}–${definition.maximum}`;
  }
  return "支持逗号分隔多个值";
}

function indexParameterValues(
  indexParameters: Record<string, IndexParameterValue>,
  searchParameters: Record<string, IndexParameterValue>,
) {
  return [...Object.values(indexParameters), ...Object.values(searchParameters)]
    .filter((value): value is number => typeof value === "number");
}

function formatIndexParameters(
  indexParameters: Record<string, IndexParameterValue>,
  searchParameters: Record<string, IndexParameterValue>,
) {
  const values = { ...indexParameters, ...searchParameters };
  const labels: Record<string, string> = {
    m: "M",
    ef_construction: "efC",
    ef_search: "ef",
    nlist: "nlist",
    nprobe: "nprobe",
    sq_type: "sq",
    nbits: "nbits",
    nrq: "nrq",
    refine: "refine",
    refine_type: "refineType",
    refine_k: "refineK",
  };
  const entries = Object.entries(values)
    .filter(([, value]) => value !== null)
    .map(([name, value]) => `${labels[name] ?? name} ${value}`);
  return entries.length > 0 ? entries : ["无专属参数"];
}

function aggregateAnalysisPoint(
  aggregate: BenchmarkAggregate,
  parameterName: string,
) {
  const indexParameters = { ...aggregate.index_parameters };
  const searchParameters = { ...aggregate.search_parameters };
  const allParameters = { ...indexParameters, ...searchParameters };
  const xValue = allParameters[parameterName];
  if (xValue === null || xValue === undefined) return null;
  return {
    key: `${aggregate.configuration_key}-${aggregate.stage_index}`,
    indexType: aggregate.index_type,
    xValue,
    xLabel: `${parameterName} = ${String(xValue)}`,
    parameterLabel: formatIndexParameters(
      indexParameters,
      searchParameters,
    ).join(" · "),
    series: "五档配置轨迹",
    p99: aggregate.latency_p99_ms,
    recall: aggregate.recall,
    memory: aggregate.vector_index_memory_mib,
  } satisfies AnalysisPoint;
}

function aggregateRecommendationPoint(aggregate: BenchmarkAggregate) {
  return {
    key: `${aggregate.configuration_key}-${aggregate.stage_index}`,
    indexType: aggregate.index_type,
    xValue: aggregate.index_type,
    xLabel: aggregate.index_type,
    parameterLabel: formatIndexParameters(
      aggregate.index_parameters,
      aggregate.search_parameters,
    ).join(" · "),
    series: aggregate.index_type,
    p99: aggregate.latency_p99_ms,
    recall: aggregate.recall,
    memory: aggregate.vector_index_memory_mib,
  } satisfies AnalysisPoint;
}

function sortValues(row: BenchmarkRow, key: SortKey): SortValue[] {
  switch (key) {
    case "indexType":
      return [row.run.index_type];
    case "indexParams":
      return indexParameterValues(
        row.run.index_parameters,
        row.run.search_parameters,
      );
    case "insert":
      return [row.run.insert_duration_seconds];
    case "optimize":
      return [row.run.optimize_duration_seconds];
    case "concurrency":
      return [row.stage.concurrency];
    case "p99":
      return [row.stage.latency_p99_ms];
    case "recall":
      return [row.run.recall];
    case "vectorIndex":
      return [row.run.vector_index_memory_mib];
  }
}

function compareRows(
  left: BenchmarkRow,
  right: BenchmarkRow,
  key: SortKey,
  direction: SortDirection,
) {
  const leftValues = sortValues(left, key);
  const rightValues = sortValues(right, key);

  for (let index = 0; index < leftValues.length; index += 1) {
    const leftValue = leftValues[index];
    const rightValue = rightValues[index];
    if (leftValue === rightValue) continue;
    if (leftValue === null || (typeof leftValue === "number" && Number.isNaN(leftValue))) return 1;
    if (rightValue === null || (typeof rightValue === "number" && Number.isNaN(rightValue))) return -1;
    const comparison =
      typeof leftValue === "number" && typeof rightValue === "number"
        ? leftValue - rightValue
        : String(leftValue).localeCompare(String(rightValue), "zh-CN");
    return direction === "asc" ? comparison : -comparison;
  }
  return 0;
}

function aggregateSortValues(
  aggregate: BenchmarkAggregate,
  key: SortKey,
): SortValue[] {
  switch (key) {
    case "indexType":
      return [aggregate.index_type];
    case "indexParams":
      return indexParameterValues(
        aggregate.index_parameters,
        aggregate.search_parameters,
      );
    case "insert":
      return [aggregate.insert_duration_seconds.mean];
    case "optimize":
      return [aggregate.optimize_duration_seconds.mean];
    case "concurrency":
      return [aggregate.concurrency];
    case "p99":
      return [aggregate.latency_p99_ms.mean];
    case "recall":
      return [aggregate.recall.mean];
    case "vectorIndex":
      return [aggregate.vector_index_memory_mib.mean];
  }
}

function compareAggregates(
  left: BenchmarkAggregate,
  right: BenchmarkAggregate,
  key: SortKey,
  direction: SortDirection,
) {
  const leftValues = aggregateSortValues(left, key);
  const rightValues = aggregateSortValues(right, key);
  for (let index = 0; index < leftValues.length; index += 1) {
    const leftValue = leftValues[index];
    const rightValue = rightValues[index];
    if (leftValue === rightValue) continue;
    if (leftValue === null || (typeof leftValue === "number" && Number.isNaN(leftValue))) return 1;
    if (rightValue === null || (typeof rightValue === "number" && Number.isNaN(rightValue))) return -1;
    const comparison =
      typeof leftValue === "number" && typeof rightValue === "number"
        ? leftValue - rightValue
        : String(leftValue).localeCompare(String(rightValue), "zh-CN");
    return direction === "asc" ? comparison : -comparison;
  }
  return 0;
}

function metricMeanRange(
  aggregates: BenchmarkAggregate[],
  selector: (aggregate: BenchmarkAggregate) => MetricSummary,
): [number, number] | null {
  const values = aggregates
    .map((aggregate) => selector(aggregate).mean)
    .filter((value): value is number => value !== null && Number.isFinite(value));
  return values.length === 0
    ? null
    : [Math.min(...values), Math.max(...values)];
}

function formatNumberRange(
  range: [number, number] | null,
  digits: number,
  suffix = "",
) {
  if (range === null) return "—";
  const [minimum, maximum] = range;
  if (Math.abs(maximum - minimum) < 10 ** -(digits + 1)) {
    return `${minimum.toFixed(digits)}${suffix}`;
  }
  return `${minimum.toFixed(digits)}～${maximum.toFixed(digits)}${suffix}`;
}

function formatPercentRange(range: [number, number] | null) {
  if (range === null) return "—";
  return formatNumberRange(
    [range[0] * 100, range[1] * 100],
    2,
    "%",
  );
}

function indexSummarySortValues(
  summary: IndexSummary,
  key: SortKey,
): SortValue[] {
  switch (key) {
    case "indexType":
      return [summary.indexType];
    case "indexParams":
      return [summary.configurationCount];
    case "insert":
      return [metricMeanRange(
        summary.aggregates,
        (aggregate) => aggregate.insert_duration_seconds,
      )?.[0] ?? null];
    case "optimize":
      return [metricMeanRange(
        summary.aggregates,
        (aggregate) => aggregate.optimize_duration_seconds,
      )?.[0] ?? null];
    case "concurrency":
      return [summary.concurrency];
    case "p99":
      return [metricMeanRange(
        summary.aggregates,
        (aggregate) => aggregate.latency_p99_ms,
      )?.[0] ?? null];
    case "recall":
      return [metricMeanRange(
        summary.aggregates,
        (aggregate) => aggregate.recall,
      )?.[0] ?? null];
    case "vectorIndex":
      return [metricMeanRange(
        summary.aggregates,
        (aggregate) => aggregate.vector_index_memory_mib,
      )?.[0] ?? null];
  }
}

function compareIndexSummaries(
  left: IndexSummary,
  right: IndexSummary,
  key: SortKey,
  direction: SortDirection,
) {
  const leftValues = indexSummarySortValues(left, key);
  const rightValues = indexSummarySortValues(right, key);
  for (let index = 0; index < leftValues.length; index += 1) {
    const leftValue = leftValues[index];
    const rightValue = rightValues[index];
    if (leftValue === rightValue) continue;
    if (leftValue === null || (typeof leftValue === "number" && Number.isNaN(leftValue))) return 1;
    if (rightValue === null || (typeof rightValue === "number" && Number.isNaN(rightValue))) return -1;
    const comparison =
      typeof leftValue === "number" && typeof rightValue === "number"
        ? leftValue - rightValue
        : String(leftValue).localeCompare(String(rightValue), "zh-CN");
    return direction === "asc" ? comparison : -comparison;
  }
  return 0;
}

function AggregateValueCells({
  aggregate,
}: {
  aggregate: BenchmarkAggregate;
}) {
  return (
    <>
      <td>
        <div className="params">
          {formatIndexParameters(
            aggregate.index_parameters,
            aggregate.search_parameters,
          ).map((parameter) => (
            <span key={parameter}>{parameter}</span>
          ))}
        </div>
      </td>
      <td>
        <strong>{meanWithStddev(aggregate.latency_p99_ms, 2, " ms")}</strong>
        <small className="submetric">
          Avg {meanWithStddev(aggregate.latency_avg_ms, 2, " ms")}
        </small>
      </td>
      <td>
        <strong>
          {aggregate.recall.mean === null
            ? "—"
            : `${(aggregate.recall.mean * 100).toFixed(2)}%`}
        </strong>
        <small className="submetric">
          σ {aggregate.recall.stddev === null
            ? "—"
            : (aggregate.recall.stddev * 100).toFixed(2)}%
        </small>
      </td>
      <td>
        <strong>
          {meanWithStddev(aggregate.vector_index_memory_mib, 2, " MiB")}
        </strong>
        <small className="submetric">均值 ± 标准差</small>
      </td>
      <td>
        <strong>
          {meanWithStddev(aggregate.insert_duration_seconds, 2, " s")}
        </strong>
      </td>
      <td>
        <strong>
          {meanWithStddev(aggregate.optimize_duration_seconds, 2, " s")}
        </strong>
      </td>
      <td>
        <strong>{aggregate.concurrency}</strong>
        <small className="submetric">
          {aggregate.concurrency_duration_seconds ?? "—"}s
        </small>
      </td>
    </>
  );
}

function apiErrorMessage(payload: unknown, fallback: string) {
  if (!payload || typeof payload !== "object" || !("detail" in payload)) {
    return fallback;
  }
  const detail = payload.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => {
        if (!item || typeof item !== "object" || !("msg" in item)) return null;
        const field = "loc" in item && Array.isArray(item.loc)
          ? String(item.loc.at(-1))
          : "参数";
        return `${field}: ${String(item.msg)}`;
      })
      .filter(Boolean);
    if (messages.length > 0) return messages.join("；");
  }
  return fallback;
}

const TRACE_SERVICE_COLORS: Record<string, string> = {
  "milvus-trace-tester": "#596860",
  proxy: "#008f5d",
  querynode: "#2e6fb0",
  streamingnode: "#6c4ab6",
  mixcoord: "#9b6a13",
  datanode: "#b94c3c",
};

const PUBLIC_TRACE_DEMO: TraceSearchResult = {
  trace_id: "61a10cf13ea97b69e7b855159210dde7",
  jaeger_url: "",
  collection_name: "TraceDemo",
  vector_field: "vector",
  top_k: 10,
  hit_count: 10,
  client_latency_ms: 3.662,
  total_duration_ms: 2.489,
  spans: [
    { span_id: "17e3635376f0f904", parent_span_id: null, service: "proxy", operation: "milvus.proto.milvus.MilvusService/Search", start_offset_ms: 0, duration_ms: 2.489, depth: 0, error: false },
    { span_id: "ff692cc18904ac54", parent_span_id: "17e3635376f0f904", service: "proxy", operation: "Proxy-Search", start_offset_ms: 0.113, duration_ms: 2.24, depth: 1, error: false },
    { span_id: "8df39ef37f948386", parent_span_id: "ff692cc18904ac54", service: "proxy", operation: "SearchTask", start_offset_ms: 0.153, duration_ms: 2.168, depth: 2, error: false },
    { span_id: "1805ad3101ff9444", parent_span_id: "8df39ef37f948386", service: "proxy", operation: "Proxy-Search-PreExecute", start_offset_ms: 0.157, duration_ms: 0.11, depth: 3, error: false },
    { span_id: "7ed5a0a253f820db", parent_span_id: "1805ad3101ff9444", service: "proxy", operation: "init search request", start_offset_ms: 0.185, duration_ms: 0.073, depth: 4, error: false },
    { span_id: "a0fa03ca6e8ebf3c", parent_span_id: "8df39ef37f948386", service: "proxy", operation: "Proxy-Search-Execute", start_offset_ms: 0.269, duration_ms: 2.001, depth: 3, error: false },
    { span_id: "cda53a5dacca7bd1", parent_span_id: "a0fa03ca6e8ebf3c", service: "proxy", operation: "milvus.proto.query.QueryNode/Search", start_offset_ms: 0.316, duration_ms: 1.937, depth: 4, error: false },
    { span_id: "29b48c4d0edfa1f3", parent_span_id: "cda53a5dacca7bd1", service: "streamingnode", operation: "milvus.proto.query.QueryNode/Search", start_offset_ms: 0.44, duration_ms: 1.645, depth: 5, error: false },
    { span_id: "d76a3a5c10391d34", parent_span_id: "29b48c4d0edfa1f3", service: "streamingnode", operation: "Delegator-waitTSafe", start_offset_ms: 0.557, duration_ms: 0, depth: 6, error: false },
    { span_id: "ed4c93cc25f9ab49", parent_span_id: "29b48c4d0edfa1f3", service: "streamingnode", operation: "schedule", start_offset_ms: 0.601, duration_ms: 0.018, depth: 6, error: false },
    { span_id: "daf9766fe70f9aa1", parent_span_id: "8df39ef37f948386", service: "proxy", operation: "Proxy-Search-PostExecute", start_offset_ms: 2.274, duration_ms: 0.044, depth: 3, error: false },
    { span_id: "f451b7bde60c7c66", parent_span_id: "daf9766fe70f9aa1", service: "proxy", operation: "searchReduceOperator", start_offset_ms: 2.288, duration_ms: 0.01, depth: 4, error: false },
    { span_id: "b07b77ad360eb33b", parent_span_id: "ff692cc18904ac54", service: "proxy", operation: "reduceResults", start_offset_ms: 2.288, duration_ms: 0.006, depth: 2, error: false },
    { span_id: "895a0f2e9daf9016", parent_span_id: "b07b77ad360eb33b", service: "proxy", operation: "decodeSearchResults", start_offset_ms: 2.289, duration_ms: 0.002, depth: 3, error: false },
  ],
};

function TraceWaterfall({ result }: { result: TraceSearchResult }) {
  const total = Math.max(result.total_duration_ms, 0.001);
  return (
    <div className="trace-waterfall">
      <div className="trace-axis" aria-hidden="true">
        {[0, 25, 50, 75, 100].map((percent) => (
          <span key={percent} style={{ left: `${percent}%` }}>
            {((total * percent) / 100).toFixed(total < 10 ? 2 : 1)} ms
          </span>
        ))}
      </div>
      <div
        className="trace-rows"
        role="img"
        aria-label={`Trace ${result.trace_id}，共 ${result.spans.length} 个 Span，总耗时 ${result.total_duration_ms} 毫秒`}
      >
        {result.spans.map((span) => {
          const left = Math.min(100, (span.start_offset_ms / total) * 100);
          const width = Math.max(0.35, Math.min(
            100 - left,
            (span.duration_ms / total) * 100,
          ));
          const color = TRACE_SERVICE_COLORS[span.service] ?? "#16858c";
          return (
            <div className="trace-row" key={span.span_id}>
              <div
                className="trace-label"
                style={{ paddingLeft: `${Math.min(span.depth, 8) * 10}px` }}
                title={`${span.service} · ${span.operation}`}
              >
                <strong>{span.service}</strong>
                <span>{span.operation}</span>
              </div>
              <div className="trace-track">
                <i
                  className={span.error ? "trace-bar trace-bar-error" : "trace-bar"}
                  style={{
                    left: `${left}%`,
                    width: `${width}%`,
                    background: span.error ? undefined : color,
                  }}
                  title={`${span.operation}：${span.duration_ms.toFixed(3)} ms，开始于 ${span.start_offset_ms.toFixed(3)} ms`}
                />
                <b
                  className="trace-duration"
                  style={{ left: `${Math.min(94, left + width)}%` }}
                >
                  {span.duration_ms.toFixed(span.duration_ms < 10 ? 3 : 2)} ms
                </b>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function Home() {
  const [runs, setRuns] = useState<BenchmarkRun[]>([]);
  const [aggregates, setAggregates] = useState<BenchmarkAggregate[]>([]);
  const [total, setTotal] = useState(0);
  const [aggregateTotal, setAggregateTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [topK, setTopK] = useState(DEFAULT_PARAMETERS.k);
  const [profiles, setProfiles] = useState(FALLBACK_INDEX_PROFILES);
  const [selectedCommand, setSelectedCommand] = useState("milvushnsw");
  const [indexMatrixTexts, setIndexMatrixTexts] = useState<
    Record<string, string>
  >({
    m: "16",
    ef_construction: "128",
    ef_search: "128",
  });
  const parameters = {
    ...DEFAULT_PARAMETERS,
    command: selectedCommand,
    index_parameters: {},
    k: topK,
  };
  const caseMetricType = CASE_TYPE_METRIC_TYPES[parameters.case_type] ?? "—";
  const numConcurrencyText = parameters.num_concurrency.join(",");
  const [repetitions, setRepetitions] = useState(3);
  const [viewMode, setViewMode] = useState<ResultViewMode>("index");
  const [job, setJob] = useState<BenchmarkJob | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [jobError, setJobError] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>("indexType");
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");
  const [expandedIndexGroups, setExpandedIndexGroups] = useState<string[]>([]);
  const [analysisIndexType, setAnalysisIndexType] = useState("");
  const [analysisParameter, setAnalysisParameter] = useState("");
  const [snapshotGeneratedAt, setSnapshotGeneratedAt] = useState<string | null>(
    null,
  );
  const [traceCollection, setTraceCollection] = useState("TraceDemo");
  const [traceTopK, setTraceTopK] = useState(10);
  const [traceRunning, setTraceRunning] = useState(false);
  const [traceError, setTraceError] = useState<string | null>(null);
  const [traceResult, setTraceResult] = useState<TraceSearchResult | null>(null);
  const [agentRecallTarget, setAgentRecallTarget] = useState(95);
  const [agentRunning, setAgentRunning] = useState(false);
  const [agentError, setAgentError] = useState<string | null>(null);
  const [agentResult, setAgentResult] = useState<TuningAgentResult | null>(null);
  const displayedAgentResult = READ_ONLY_DEMO
    ? PUBLIC_AGENT_DEMO
    : agentResult;
  const displayedTraceResult = READ_ONLY_DEMO ? PUBLIC_TRACE_DEMO : traceResult;

  const loadBenchmarks = useCallback(async (signal?: AbortSignal) => {
    try {
      if (READ_ONLY_DEMO) {
        const response = await fetch(PUBLIC_SNAPSHOT_URL, {
          cache: "no-store",
          signal,
        });
        if (!response.ok) {
          throw new Error(`公开快照返回 HTTP ${response.status}`);
        }
        const snapshot = (await response.json()) as PublicBenchmarkSnapshot;
        setRuns(snapshot.runs.items);
        setTotal(snapshot.runs.total);
        setAggregates(snapshot.aggregates.items);
        setAggregateTotal(snapshot.aggregates.total);
        setProfiles(snapshot.profiles);
        setSnapshotGeneratedAt(snapshot.generated_at);
        setError(null);
        return;
      }
      const [rawResponse, aggregateResponse, profileResponse] = await Promise.all([
        fetch(`${API_BASE_URL}/api/benchmarks?limit=100&offset=0`, {
          cache: "no-store",
          signal,
        }),
        fetch(`${API_BASE_URL}/api/benchmark-aggregates?limit=100&offset=0`, {
          cache: "no-store",
          signal,
        }),
        fetch(`${API_BASE_URL}/api/benchmark-profiles`, {
          cache: "no-store",
          signal,
        }),
      ]);
      if (!rawResponse.ok || !aggregateResponse.ok || !profileResponse.ok) {
        throw new Error(
          `指标 API 返回 HTTP ${rawResponse.status}/${aggregateResponse.status}/${profileResponse.status}`,
        );
      }
      const rawPayload = (await rawResponse.json()) as BenchmarkListResponse;
      const aggregatePayload =
        (await aggregateResponse.json()) as BenchmarkAggregateListResponse;
      const profilePayload = (await profileResponse.json()) as IndexProfile[];
      setRuns(rawPayload.items);
      setTotal(rawPayload.total);
      setAggregates(aggregatePayload.items);
      setAggregateTotal(aggregatePayload.total);
      setProfiles(profilePayload);
      setSnapshotGeneratedAt(null);
      setError(null);
    } catch (reason) {
      if (reason instanceof DOMException && reason.name === "AbortError") {
        return;
      }
      setError(
        reason instanceof Error
          ? reason.message
          : "无法读取本地 benchmark 指标",
      );
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => {
      void loadBenchmarks(controller.signal);
      if (READ_ONLY_DEMO) return;
      void fetch(`${API_BASE_URL}/api/benchmark-jobs`, {
        cache: "no-store",
        signal: controller.signal,
      })
        .then((response) => response.ok ? response.json() : [])
        .then((jobs: BenchmarkJob[]) => {
          if (jobs.length > 0) setJob(jobs[0]);
        })
        .catch(() => undefined);
    }, 0);
    return () => {
      window.clearTimeout(timeoutId);
      controller.abort();
    };
  }, [loadBenchmarks]);

  const refreshBenchmarks = useCallback(() => {
    setRefreshing(true);
    void loadBenchmarks();
  }, [loadBenchmarks]);

  useEffect(() => {
    if (READ_ONLY_DEMO) return;
    if (!job || !["queued", "running", "cancelling"].includes(job.status)) {
      return;
    }
    const controller = new AbortController();
    const intervalId = window.setInterval(() => {
      void fetch(`${API_BASE_URL}/api/benchmark-jobs/${job.job_id}`, {
        cache: "no-store",
        signal: controller.signal,
      })
        .then(async (response) => {
          if (!response.ok) throw new Error(`任务状态返回 HTTP ${response.status}`);
          return response.json() as Promise<BenchmarkJob>;
        })
        .then((updated) => {
          setJob(updated);
          if (updated.status === "succeeded") {
            void loadBenchmarks();
          }
        })
        .catch((reason) => {
          if (!(reason instanceof DOMException && reason.name === "AbortError")) {
            setJobError(reason instanceof Error ? reason.message : "无法刷新任务状态");
          }
        });
    }, 1000);
    return () => {
      controller.abort();
      window.clearInterval(intervalId);
    };
  }, [job, loadBenchmarks]);

  async function startBenchmark(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setJobError(null);
    try {
      const numConcurrency = numConcurrencyText
        .split(",")
        .map((value) => Number(value.trim()))
        .filter((value) => Number.isFinite(value));
      if (numConcurrency.length === 0) {
        throw new Error("并发列表至少需要一个整数");
      }
      if (selectedProfile.parameters.some(
        (definition) => indexMatrixValues[definition.name].length === 0,
      )) {
        throw new Error("每个索引参数都至少需要一个有效值");
      }
      const confirmed = window.confirm(
        [
          `即将开始 ${plannedRunCount} 次 Benchmark`,
          `索引：${selectedProfile.label}（${selectedCommand}）`,
          `${configurationCount} 组配置 × 每组 ${repetitions} 次，串行执行。`,
          "",
          parameters.drop_old
            ? "本操作会删除并重建 VDBBench Collection。"
            : "本操作将使用现有 VDBBench Collection。",
          "",
          "确认继续运行吗？",
        ].join("\n"),
      );
      if (!confirmed) return;

      setSubmitting(true);
      const commonParameters = {
        uri: parameters.uri,
        num_shards: parameters.num_shards,
        replica_number: parameters.replica_number,
        case_type: parameters.case_type,
        drop_old: parameters.drop_old,
        load: parameters.load,
        load_concurrency: parameters.load_concurrency,
        search_serial: parameters.search_serial,
        search_concurrent: parameters.search_concurrent,
        k: parameters.k,
        concurrency_duration: parameters.concurrency_duration,
        num_concurrency: numConcurrency,
        concurrency_timeout: parameters.concurrency_timeout,
        db_label: parameters.db_label,
      };
      const response = await fetch(`${API_BASE_URL}/api/benchmark-jobs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          command: selectedCommand,
          parameters: commonParameters,
          index_matrix: indexMatrixValues,
          repetitions,
        }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(apiErrorMessage(payload, "无法启动 Benchmark"));
      }
      setJob(payload as BenchmarkJob);
    } catch (reason) {
      setJobError(reason instanceof Error ? reason.message : "无法启动 Benchmark");
    } finally {
      setSubmitting(false);
    }
  }

  async function runTraceSearch(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setTraceRunning(true);
    setTraceError(null);
    setTraceResult(null);
    try {
      const response = await fetch(`${API_BASE_URL}/api/trace-search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          uri: parameters.uri,
          database: "default",
          collection_name: traceCollection,
          top_k: traceTopK,
        }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(apiErrorMessage(payload, `Trace API 返回 HTTP ${response.status}`));
      }
      setTraceResult(payload as TraceSearchResult);
    } catch (reason) {
      setTraceError(
        reason instanceof Error ? reason.message : "无法完成 Trace 查询测试",
      );
    } finally {
      setTraceRunning(false);
    }
  }

  async function runTuningAgent(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (
      !Number.isFinite(agentRecallTarget)
      || agentRecallTarget <= 0
      || agentRecallTarget > 100
    ) {
      setAgentError("Recall 目标必须大于 0%，且不超过 100%");
      return;
    }
    const confirmed = window.confirm(
      "Agent 会先读取 SQLite 历史数据，并可能在当前 VDBBench Collection 上测试最多 3 组搜索参数。每组只提交一个 Benchmark 任务，在同一任务中依次执行 Recall serial search 和并发 1 的 P99 查询，不会重建 Collection 或重新导入数据。是否继续？",
    );
    if (!confirmed) return;

    setAgentRunning(true);
    setAgentError(null);
    setAgentResult(null);
    try {
      const response = await fetch(`${API_BASE_URL}/api/tuning-agent/recommend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          recall_target: agentRecallTarget / 100,
        }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(
          apiErrorMessage(payload, `Agent API 返回 HTTP ${response.status}`),
        );
      }
      setAgentResult(payload as TuningAgentResult);
    } catch (reason) {
      setAgentError(
        reason instanceof Error ? reason.message : "无法生成索引调优建议",
      );
    } finally {
      setAgentRunning(false);
    }
  }

  async function cancelBenchmark() {
    if (!job) return;
    setJobError(null);
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/benchmark-jobs/${job.job_id}/cancel`,
        { method: "POST" },
      );
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(apiErrorMessage(payload, "无法取消任务"));
      }
      setJob(payload as BenchmarkJob);
    } catch (reason) {
      setJobError(reason instanceof Error ? reason.message : "无法取消任务");
    }
  }

  const rows = useMemo<BenchmarkRow[]>(
    () =>
      runs.flatMap((run) =>
        run.stages.map((stage) => ({ run, stage })),
      ),
    [runs],
  );
  const sortedRows = useMemo(
    () =>
      rows
        .map((row, originalIndex) => ({ row, originalIndex }))
        .sort(
          (left, right) =>
            compareRows(left.row, right.row, sortKey, sortDirection)
            || left.originalIndex - right.originalIndex,
        )
        .map(({ row }) => row),
    [rows, sortDirection, sortKey],
  );
  const sortedAggregates = useMemo(
    () =>
      aggregates
        .map((aggregate, originalIndex) => ({ aggregate, originalIndex }))
        .sort(
          (left, right) =>
            compareAggregates(
              left.aggregate,
              right.aggregate,
              sortKey,
              sortDirection,
            ) || left.originalIndex - right.originalIndex,
        )
        .map(({ aggregate }) => aggregate),
    [aggregates, sortDirection, sortKey],
  );
  const indexSummaries = useMemo<IndexSummary[]>(() => {
    const groups = new Map<string, BenchmarkAggregate[]>();
    aggregates.forEach((aggregate) => {
      const key = [
        aggregate.index_type,
        aggregate.case_type,
        aggregate.metric_type,
        aggregate.db_label ?? "",
        aggregate.top_k,
        aggregate.concurrency,
        aggregate.num_shards ?? "",
        aggregate.replica_number ?? "",
        aggregate.concurrency_duration_seconds ?? "",
      ].join("|");
      const group = groups.get(key) ?? [];
      group.push(aggregate);
      groups.set(key, group);
    });
    return [...groups].map(([key, group]) => ({
      key,
      indexType: group[0].index_type,
      aggregates: [...group].sort((left, right) =>
        compareAggregates(left, right, "indexParams", "asc")
      ),
      configurationCount: new Set(
        group.map((aggregate) => aggregate.configuration_key),
      ).size,
      concurrency: group[0].concurrency,
      concurrencyDuration: group[0].concurrency_duration_seconds,
    }));
  }, [aggregates]);
  const sortedIndexSummaries = useMemo(
    () =>
      indexSummaries
        .map((summary, originalIndex) => ({ summary, originalIndex }))
        .sort(
          (left, right) =>
            compareIndexSummaries(
              left.summary,
              right.summary,
              sortKey,
              sortDirection,
            ) || left.originalIndex - right.originalIndex,
        )
        .map(({ summary }) => summary),
    [indexSummaries, sortDirection, sortKey],
  );
  const analysisIndexOptions = useMemo(() => {
    const counts = new Map<string, number>();
    aggregates.forEach((aggregate) => {
      counts.set(
        aggregate.index_type,
        (counts.get(aggregate.index_type) ?? 0) + 1,
      );
    });
    return [...counts].sort(
      (left, right) => right[1] - left[1] || left[0].localeCompare(right[0]),
    ).map(([indexType]) => indexType);
  }, [aggregates]);
  const effectiveAnalysisIndex = analysisIndexOptions.includes(analysisIndexType)
    ? analysisIndexType
    : analysisIndexOptions[0] ?? "";
  const analysisBaseline = aggregates
    .filter((aggregate) => aggregate.index_type === effectiveAnalysisIndex)
    .sort(
      (left, right) =>
        Date.parse(right.latest_created_at) - Date.parse(left.latest_created_at),
    )[0] ?? null;
  const analysisRows = useMemo(() => {
    if (!analysisBaseline) return [];
    return aggregates.filter((aggregate) =>
      aggregate.index_type === analysisBaseline.index_type
      && aggregate.case_type === analysisBaseline.case_type
      && aggregate.metric_type === analysisBaseline.metric_type
      && aggregate.db_label === analysisBaseline.db_label
      && aggregate.top_k === analysisBaseline.top_k
      && aggregate.concurrency === analysisBaseline.concurrency
      && aggregate.num_shards === analysisBaseline.num_shards
      && aggregate.replica_number === analysisBaseline.replica_number
      && aggregate.concurrency_duration_seconds
        === analysisBaseline.concurrency_duration_seconds
    );
  }, [aggregates, analysisBaseline]);
  const analysisParameterOptions = useMemo(() => {
    const valuesByName = new Map<string, Set<string>>();
    analysisRows.forEach((aggregate) => {
      const values = {
        ...aggregate.index_parameters,
        ...aggregate.search_parameters,
      };
      Object.entries(values).forEach(([name, value]) => {
        if (value === null) return;
        const valuesForName = valuesByName.get(name) ?? new Set<string>();
        valuesForName.add(categoryKey(value));
        valuesByName.set(name, valuesForName);
      });
    });
    return [...valuesByName]
      .filter(([, values]) => values.size > 1)
      .sort(
        (left, right) =>
          right[1].size - left[1].size || left[0].localeCompare(right[0]),
      )
      .map(([name, values]) => ({
        name,
        distinctCount: values.size,
      }));
  }, [analysisRows]);
  const effectiveAnalysisParameter = analysisParameterOptions.some(
    (parameter) => parameter.name === analysisParameter,
  )
    ? analysisParameter
    : analysisParameterOptions[0]?.name ?? "";
  const analysisPoints = useMemo<AnalysisPoint[]>(() => {
    if (!effectiveAnalysisParameter) return [];
    const latestByParameterValue = new Map<string, BenchmarkAggregate>();
    analysisRows.forEach((aggregate) => {
      const allParameters = {
        ...aggregate.index_parameters,
        ...aggregate.search_parameters,
      };
      const xValue = allParameters[effectiveAnalysisParameter];
      if (xValue === null || xValue === undefined) return;
      const key = categoryKey(xValue);
      const current = latestByParameterValue.get(key);
      if (
        !current
        || Date.parse(aggregate.latest_created_at)
          > Date.parse(current.latest_created_at)
      ) {
        latestByParameterValue.set(key, aggregate);
      }
    });
    return [...latestByParameterValue.values()].flatMap((aggregate) => {
      const point = aggregateAnalysisPoint(
        aggregate,
        effectiveAnalysisParameter,
      );
      return point ? [point] : [];
    });
  }, [analysisRows, effectiveAnalysisParameter]);
  const tradeoffPoints = useMemo<AnalysisPoint[]>(
    () => analysisRows.flatMap((aggregate) => {
      const point = aggregateAnalysisPoint(
        aggregate,
        effectiveAnalysisParameter,
      );
      return point ? [point] : [];
    }),
    [analysisRows, effectiveAnalysisParameter],
  );
  const withinIndexRecommendationPoints = useMemo(
    () => analysisRows.map(aggregateRecommendationPoint),
    [analysisRows],
  );
  const withinIndexRecommendations = useMemo(
    () => analyzeRecommendations(withinIndexRecommendationPoints),
    [withinIndexRecommendationPoints],
  );
  const jobIsActive = job
    ? ["queued", "running", "cancelling"].includes(job.status)
    : false;
  const formDisabled = jobIsActive || submitting || agentRunning;
  const selectedProfile =
    profiles.find((profile) => profile.command === selectedCommand)
    ?? FALLBACK_INDEX_PROFILES[0];
  const indexMatrixValues = Object.fromEntries(
    selectedProfile.parameters.map((definition) => [
      definition.name,
      parseIndexParameterList(
        indexMatrixTexts[definition.name] ?? "",
        definition,
      ),
    ]),
  );
  const configurationCount = selectedProfile.parameters.length === 0
    ? 1
    : Object.values(indexMatrixValues).reduce(
        (count, values) => count * values.length,
        1,
      );
  const plannedRunCount = configurationCount * repetitions;
  const parameterOutOfRange = selectedProfile.parameters.some((definition) =>
    indexMatrixValues[definition.name].some(
      (value) => typeof value === "number" && (
        (definition.minimum !== null && definition.minimum !== undefined
          && value < definition.minimum)
        || (definition.maximum !== null && definition.maximum !== undefined
          && value > definition.maximum)
      ),
    )
  );
  const numericIndexValues = (name: string) =>
    (indexMatrixValues[name] ?? [])
      .filter((value): value is number => typeof value === "number");
  const hnswRelationshipInvalid = selectedCommand.startsWith("milvushnsw") && (
    Math.min(...numericIndexValues("ef_construction"))
      < Math.max(...numericIndexValues("m"))
    || Math.min(...numericIndexValues("ef_search")) < topK
  );
  const ivfRelationshipInvalid =
    ["milvusivfflat", "milvusivfsq8"].includes(selectedCommand)
    && Math.max(...numericIndexValues("nprobe"))
      > Math.min(...numericIndexValues("nlist"));
  const parameterRelationshipInvalid =
    configurationCount === 0
    || !Number.isInteger(topK)
    || topK < 1
    || topK > 2048
    || parameterOutOfRange
    || hnswRelationshipInvalid
    || ivfRelationshipInvalid
    || plannedRunCount > 30
    || !(parameters.load || parameters.search_serial || parameters.search_concurrent);

  function changeIndexCommand(command: string) {
    const profile = profiles.find((candidate) => candidate.command === command);
    if (!profile) return;
    setSelectedCommand(command);
    setIndexMatrixTexts(Object.fromEntries(
      profile.parameters.map((definition) => [
        definition.name,
        String(definition.default),
      ]),
    ));
  }

  function changeSort(nextKey: SortKey) {
    if (nextKey === sortKey) {
      setSortDirection((current) => current === "asc" ? "desc" : "asc");
      return;
    }
    setSortKey(nextKey);
    setSortDirection("desc");
  }

  function toggleIndexGroup(key: string) {
    setExpandedIndexGroups((current) =>
      current.includes(key)
        ? current.filter((item) => item !== key)
        : [...current, key]
    );
  }

  return (
    <main>
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">M</span>
          <div>
            <strong>MilvusTune</strong>
            <small>LOCAL CPU INDEX LAB</small>
          </div>
        </div>
        <div className={`status ${error ? "status-error" : ""}`}>
          <i />
          {error
            ? READ_ONLY_DEMO ? "公开快照读取失败" : "本地 API 未连接"
            : loading
              ? READ_ONLY_DEMO ? "正在读取公开快照" : "正在连接本地 API"
              : READ_ONLY_DEMO
                ? `${total} 次实验 · 公开只读`
                : `${total} 次实验 · SQLite`}
        </div>
      </header>

      <div className="shell">
        <section className="hero">
          <div>
            <p className="eyebrow">VECTOR INDEX BENCHMARK</p>
            <h1>Milvus CPU 索引实验台</h1>
            <p className="subtitle">
              {READ_ONLY_DEMO ? "查看" : "配置并运行"} VectorDBBench，
              对比不同索引参数下的构建耗时、
              P99、Recall 与 Vector Index 内存。
            </p>
          </div>
          <div className="scope">
            <span>当前数据源</span>
            <strong>
              {READ_ONLY_DEMO
                ? "VectorDBBench 公开实验快照"
                : "VectorDBBench + Prometheus"}
            </strong>
            <small>
              {READ_ONLY_DEMO
                ? snapshotGeneratedAt
                  ? `更新于 ${new Date(snapshotGeneratedAt).toLocaleString("zh-CN")}`
                  : "只读数据，不连接实验集群"
                : API_BASE_URL}
            </small>
          </div>
        </section>

        {!READ_ONLY_DEMO && (
          <section className="runner-panel">
          <div className="runner-copy">
            <p className="eyebrow">RUN CPU INDEX BENCHMARK</p>
            <h2>发起一组新的向量索引实验</h2>
            <p>
              这里列出临时 YAML 的全部参数。运行时会生成独立配置文件，
              并使用所选 CPU 索引子命令调用 VectorDBBench。
            </p>
          </div>
          <form className="runner-form" onSubmit={startBenchmark}>
            <fieldset>
              <legend>
                数据集与执行 Stage <span className="fixed-badge">固定</span>
              </legend>
              <div className="parameter-grid">
                <label className="field-wide">
                  <span>case_type</span>
                  <input
                    type="text"
                    value={parameters.case_type}
                    readOnly
                    disabled={formDisabled}
                    pattern="[A-Za-z][A-Za-z0-9_]*"
                    required
                  />
                  <small>当前数据集：50K × 1536D</small>
                </label>
                <label>
                  <span>metric_type</span>
                  <input
                    type="text"
                    value={caseMetricType}
                    readOnly
                    disabled={formDisabled}
                  />
                  <small>由 case_type 对应的标准数据集固定</small>
                </label>
                <label>
                  <span>load_concurrency</span>
                  <input
                    type="number"
                    min={0}
                    max={256}
                    value={parameters.load_concurrency}
                    readOnly
                    disabled={formDisabled}
                    required
                  />
                  <small>0 表示使用 CPU 核数</small>
                </label>
                <div className="stage-switches field-full">
                  {([
                    ["drop_old", "drop_old", "删除旧 VDBBench"],
                    ["load", "load", "导入数据"],
                    ["search_serial", "search_serial", "串行搜索"],
                    ["search_concurrent", "search_concurrent", "并发搜索"],
                  ] as const).map(([key, label, description]) => (
                    <label className="switch-field" key={key}>
                      <input
                        type="checkbox"
                        checked={parameters[key]}
                        readOnly
                        disabled
                      />
                      <span>
                        <b>{label}</b>
                        <small>{description}</small>
                      </span>
                    </label>
                  ))}
                </div>
              </div>
            </fieldset>

            <fieldset>
              <legend>
                Milvus 连接与部署 <span className="fixed-badge">固定</span>
              </legend>
              <div className="parameter-grid parameter-grid-connection">
                <label className="field-wide">
                  <span>uri</span>
                  <input
                    type="url"
                    value={parameters.uri}
                    readOnly
                    disabled={formDisabled}
                    required
                  />
                  <small>Milvus 服务地址</small>
                </label>
                <label>
                  <span>num_shards</span>
                  <input
                    type="number"
                    min={1}
                    max={64}
                    value={parameters.num_shards}
                    readOnly
                    disabled={formDisabled}
                    required
                  />
                  <small>Collection 分片数</small>
                </label>
                <label>
                  <span>replica_number</span>
                  <input
                    type="number"
                    min={1}
                    max={16}
                    value={parameters.replica_number}
                    readOnly
                    disabled={formDisabled}
                    required
                  />
                  <small>加载副本数</small>
                </label>
                <label className="field-wide">
                  <span>db_label</span>
                  <input
                    type="text"
                    value={parameters.db_label}
                    readOnly
                    disabled={formDisabled}
                    pattern="[A-Za-z0-9][A-Za-z0-9_.-]*"
                    required
                  />
                  <small>结果中用于区分 Milvus 环境</small>
                </label>
              </div>
            </fieldset>

            <fieldset>
              <legend>
                搜索负载
                <span className="editable-badge">TopK 可调，其余固定</span>
              </legend>
              <div className="parameter-grid parameter-grid-five">
                <label>
                  <span>k（TopK）</span>
                  <input
                    type="number"
                    min={1}
                    max={2048}
                    step={1}
                    value={topK}
                    onChange={(event) => setTopK(Number(event.target.value))}
                    disabled={formDisabled}
                    required
                  />
                  <small>每条查询返回的最近邻数量</small>
                </label>
                <label>
                  <span>concurrency_duration</span>
                  <input
                    type="number"
                    min={1}
                    max={86400}
                    value={parameters.concurrency_duration}
                    readOnly
                    disabled={formDisabled}
                    required
                  />
                  <small>每档并发秒数</small>
                </label>
                <label>
                  <span>num_concurrency</span>
                  <input
                    type="text"
                    value={numConcurrencyText}
                    readOnly
                    disabled={formDisabled}
                    pattern="\s*\d+\s*(,\s*\d+\s*)*"
                    required
                  />
                  <small>逗号分隔，如 1,5,10</small>
                </label>
                <label>
                  <span>concurrency_timeout</span>
                  <input
                    type="number"
                    min={-1}
                    max={86400}
                    value={parameters.concurrency_timeout}
                    readOnly
                    disabled={formDisabled}
                    required
                  />
                  <small>-1 表示无限等待</small>
                </label>
              </div>
            </fieldset>

            <fieldset className="hnsw-fieldset">
              <legend>
                CPU 索引与参数 <span className="editable-badge">可调整</span>
              </legend>
              <div className="parameter-grid parameter-grid-four">
                <label>
                  <span>索引类型</span>
                  <select
                    value={selectedCommand}
                    onChange={(event) => changeIndexCommand(event.target.value)}
                    disabled={formDisabled}
                    required
                  >
                    {profiles.map((profile) => (
                      <option key={profile.command} value={profile.command}>
                        {profile.label} · {profile.command}
                      </option>
                    ))}
                  </select>
                  <small>当前只开放适合本地环境的 CPU 索引</small>
                </label>
                {selectedProfile.parameters.map((definition) => (
                  <label key={definition.name}>
                    <span>{definition.label} 参数列表</span>
                    <input
                      type="text"
                      value={indexMatrixTexts[definition.name] ?? ""}
                      onChange={(event) => setIndexMatrixTexts((current) => ({
                        ...current,
                        [definition.name]: event.target.value,
                      }))}
                      disabled={formDisabled}
                      pattern={
                        definition.kind === "integer"
                          ? String.raw`\s*\d+\s*(,\s*\d+\s*)*`
                          : undefined
                      }
                      placeholder={
                        definition.options?.map(String).join(",") ?? undefined
                      }
                      required
                    />
                    <small>
                      {definition.description}；{parameterHelp(definition)}
                    </small>
                  </label>
                ))}
                <label>
                  <span>每组重复次数</span>
                  <input
                    type="number"
                    min={1}
                    max={5}
                    value={repetitions}
                    onChange={(event) => setRepetitions(Number(event.target.value))}
                    disabled={formDisabled}
                    required
                  />
                  <small>建议 3–5 次，用于计算标准差</small>
                </label>
                <div className="matrix-summary field-full">
                  <strong>{configurationCount} 组配置 × {repetitions} 次</strong>
                  <span>计划执行 {plannedRunCount} 次 Benchmark，最多 30 次</span>
                </div>
              </div>
            </fieldset>

            {parameterRelationshipInvalid && (
              <p className="form-warning">
                请检查参数值、范围和关系：所有 HNSW 变体要求
                efConstruction ≥ M 且 efSearch ≥ TopK；IVF 要求
                nprobe ≤ nlist；计划总次数最多 30。
              </p>
            )}
            <div className="runner-submit">
              <p>
                {parameters.drop_old && parameters.load
                  ? "本次会删除并重建 VDBBench Collection。"
                  : "本次不会执行 drop_old；请确认现有 VDBBench 可供搜索。"}
              </p>
              <button
                className="run-button"
                type="submit"
                disabled={formDisabled || parameterRelationshipInvalid}
              >
                {submitting ? "提交中…" : jobIsActive ? "实验运行中" : "运行 Benchmark"}
              </button>
            </div>
          </form>

          {(job || jobError) && (
            <div className={`job-status ${job?.status === "failed" ? "job-failed" : ""}`}>
              <div className="job-status-main">
                <div>
                  <span className={`job-dot ${jobIsActive ? "job-dot-active" : ""}`} />
                  <strong>
                    {job
                      ? PHASE_LABELS[job.phase] ?? job.phase
                      : "任务提交失败"}
                  </strong>
                  {job && (
                    <small>
                      第 {Math.max(job.current_run_number, 1)}/{job.total_runs} 次
                      {" · "}{job.parameters.command}
                      {" · "}{formatIndexParameters(
                        job.parameters.index_parameters,
                        {},
                      ).join(" · ")}
                      {job.elapsed_seconds !== null
                        ? ` · ${Math.round(job.elapsed_seconds)}s`
                        : ""}
                    </small>
                  )}
                </div>
                {jobIsActive && (
                  <button
                    type="button"
                    className="cancel-button"
                    onClick={() => void cancelBenchmark()}
                    disabled={job.status === "cancelling"}
                  >
                    {job.status === "cancelling" ? "取消中…" : "取消任务"}
                  </button>
                )}
              </div>
              {job && job.total_runs > 1 && (
                <div className="job-progress" aria-label="实验执行进度">
                  <span
                    style={{
                      width: `${Math.min(
                        100,
                        (job.completed_runs / job.total_runs) * 100,
                      )}%`,
                    }}
                  />
                  <small>
                    已完成 {job.completed_runs}/{job.total_runs}
                  </small>
                </div>
              )}
              {(jobError || job?.error) && (
                <p className="job-error">{jobError ?? job?.error}</p>
              )}
              {job?.status === "succeeded" && (
                <p className="job-success">
                  {job.total_runs} 次实验已完成，原始 Run 已保留，聚合结果已刷新。
                </p>
              )}
              {job && job.log_tail.length > 0 && (
                <details className="job-log">
                  <summary>查看最近日志</summary>
                  <pre>{job.log_tail.slice(-12).join("\n")}</pre>
                </details>
              )}
            </div>
          )}
          </section>
        )}

        <section className="agent-panel">
            <div className="agent-heading">
              <div>
                <p className="eyebrow">LANGGRAPH · DEEP AGENTS</p>
                <h2>
                  {READ_ONLY_DEMO
                    ? "Recall 目标调优 Agent 演示"
                    : "Recall 目标调优 Agent"}
                </h2>
                <p>
                  {READ_ONLY_DEMO
                    ? "展示 Agent 基于公开实验快照生成的建议；公开页面不会连接本地 SQLite、调用模型或执行压测。"
                    : "前置节点先读取 SQLite 历史和当前索引配置，Agent 再按 Recall 目标决定是否调用查询压测。每次包含 Recall 所需的 serial search，并发查询固定为 1，最多 3 次，不重建 Collection。"}
                </p>
              </div>
              {READ_ONLY_DEMO ? (
                <div className="agent-demo-badge">
                  <span>交互流程 · 静态 Mock</span>
                  <strong>Recall 目标 95% · qwen-plus</strong>
                </div>
              ) : (
                <form className="agent-form" onSubmit={runTuningAgent}>
                  <label>
                    <span>Recall 目标</span>
                    <div className="agent-target-input">
                      <input
                        type="number"
                        min={0.01}
                        max={100}
                        step={0.01}
                        value={agentRecallTarget}
                        onChange={(event) =>
                          setAgentRecallTarget(Number(event.target.value))
                        }
                        disabled={agentRunning || jobIsActive}
                        required
                      />
                      <b>%</b>
                    </div>
                  </label>
                  <button
                    type="submit"
                    className="run-button"
                    disabled={agentRunning || jobIsActive}
                  >
                    {agentRunning
                      ? "Agent 调优中…"
                      : jobIsActive
                        ? "Benchmark 运行中"
                        : "生成调优建议"}
                  </button>
                </form>
              )}
            </div>

            {!READ_ONLY_DEMO && agentError && (
              <div className="agent-message agent-message-error" role="alert">
                <strong>Agent 未完成</strong>
                <span>{agentError}</span>
              </div>
            )}

            {displayedAgentResult && (
              <div className="agent-result" aria-live="polite">
                <div className="agent-result-meta">
                  <span>
                    目标 <b>{(displayedAgentResult.recall_target * 100).toFixed(2)}%</b>
                  </span>
                  <span>
                    模型 <b>{displayedAgentResult.model}</b>
                  </span>
                  <span>
                    工具 <b>{displayedAgentResult.tools_used.join(", ") || "未调用"}</b>
                  </span>
                  <span>
                    历史配置 <b>{displayedAgentResult.history_configuration_count}</b>
                  </span>
                  <span>
                    查询压测 <b>{displayedAgentResult.benchmark_run_count}/{displayedAgentResult.benchmark_tool_call_count}</b>
                  </span>
                </div>
                <div className="agent-answer">{displayedAgentResult.answer}</div>
              </div>
            )}
          </section>

        <section className="trace-panel">
            <div className="trace-heading">
              <div>
                <p className="eyebrow">DISTRIBUTED TRACE</p>
                <h2>
                  {READ_ONLY_DEMO ? "单次查询链路演示" : "单次查询链路测试"}
                </h2>
                <p>
                  {READ_ONLY_DEMO
                    ? "展示从本地 Milvus Cluster 真实采集的 Search Trace，按统一时间轴观察 Proxy、StreamingNode 与内部操作的调用关系。"
                    : "从现有 Collection 读取一条真实向量执行 Search，并把本次请求在 Client、Proxy、QueryNode 等节点中的 Span 按时间轴展开。首次测试会自动初始化独立的 TraceDemo，不影响 Benchmark 数据。"}
                </p>
              </div>
              {READ_ONLY_DEMO ? (
                <div className="trace-demo-badge">
                  <span>真实采集 · 静态快照</span>
                  <strong>TraceDemo · TopK 10</strong>
                </div>
              ) : (
                <form className="trace-form" onSubmit={runTraceSearch}>
                  <label>
                    <span>Collection</span>
                    <input
                      type="text"
                      value={traceCollection}
                      onChange={(event) => setTraceCollection(event.target.value)}
                      pattern="[A-Za-z0-9_]+"
                      disabled={traceRunning || agentRunning}
                      required
                    />
                  </label>
                  <label>
                    <span>TopK</span>
                    <input
                      type="number"
                      min={1}
                      max={2048}
                      value={traceTopK}
                      onChange={(event) => setTraceTopK(Number(event.target.value))}
                      disabled={traceRunning || agentRunning}
                      required
                    />
                  </label>
                  <button
                    type="submit"
                    className="run-button"
                    disabled={traceRunning || jobIsActive || agentRunning}
                  >
                    {agentRunning
                      ? "Agent 调优中"
                      : traceRunning
                      ? "查询并等待 Trace…"
                      : jobIsActive
                        ? "Benchmark 运行中"
                        : "测试查询"}
                  </button>
                </form>
              )}
            </div>

            <details className="trace-example">
              <summary>查看查询示例</summary>
              <pre><code>{`from pymilvus import MilvusClient

client = MilvusClient(uri="${parameters.uri}")
row = client.query(
    collection_name="${traceCollection}",
    filter="",
    output_fields=["vector"],
    limit=1,
)
result = client.search(
    collection_name="${traceCollection}",
    data=[row[0]["vector"]],
    anns_field="vector",
    limit=${traceTopK},
)`}</code></pre>
            </details>

            {traceError && (
              <div className="trace-message trace-message-error" role="alert">
                <strong>测试未完成</strong>
                <span>{traceError}</span>
              </div>
            )}

            {displayedTraceResult && (
              <div className="trace-result">
                <div className="trace-summary">
                  <div>
                    <span>端到端 Trace</span>
                    <strong>{displayedTraceResult.total_duration_ms.toFixed(3)} ms</strong>
                  </div>
                  <div>
                    <span>客户端 Search</span>
                    <strong>{displayedTraceResult.client_latency_ms.toFixed(3)} ms</strong>
                  </div>
                  <div>
                    <span>Span / 命中</span>
                    <strong>{displayedTraceResult.spans.length} / {displayedTraceResult.hit_count}</strong>
                  </div>
                  {READ_ONLY_DEMO ? (
                    <span className="trace-static-note">本地 Jaeger 真实采集快照</span>
                  ) : (
                    <a
                      href={displayedTraceResult.jaeger_url}
                      target="_blank"
                      rel="noreferrer"
                    >
                      在 Jaeger 打开完整 Trace
                    </a>
                  )}
                </div>
                <TraceWaterfall result={displayedTraceResult} />
              </div>
            )}
          </section>

        {error && (
          <section className="notice notice-error" role="alert">
            <div>
              <strong>
                {READ_ONLY_DEMO
                  ? "无法读取公开实验快照"
                  : "无法连接本地指标 API"}
              </strong>
              <span>
                {READ_ONLY_DEMO
                  ? `${error}。请稍后刷新页面。`
                  : `${error}。请先启动 FastAPI 服务，再刷新数据。`}
              </span>
            </div>
            <button type="button" onClick={refreshBenchmarks}>
              重试
            </button>
          </section>
        )}

        <section className="table-panel">
          <div className="table-heading">
            <div>
              <p className="eyebrow">BENCHMARK RUNS</p>
              <h2>向量索引实验结果</h2>
            </div>
            <div className="table-actions">
              <div className="view-toggle" aria-label="指标视图">
                <button
                  type="button"
                  className={viewMode === "index" ? "active" : ""}
                  onClick={() => setViewMode("index")}
                >
                  索引汇总
                </button>
                <button
                  type="button"
                  className={viewMode === "aggregate" ? "active" : ""}
                  onClick={() => setViewMode("aggregate")}
                >
                  参数明细
                </button>
                <button
                  type="button"
                  className={viewMode === "raw" ? "active" : ""}
                  onClick={() => setViewMode("raw")}
                >
                  原始 Run
                </button>
              </div>
              <div className="method">
                <b>
                  {viewMode === "index"
                    ? `${indexSummaries.length} 类索引汇总`
                    : viewMode === "aggregate"
                      ? `${aggregateTotal} 组参数明细`
                      : `${rows.length} 个并发 Stage`}
                </b>
                <span>
                  {viewMode === "index"
                    ? `${aggregateTotal} 组参数配置`
                    : viewMode === "aggregate"
                      ? `${total} 次原始 Run 已保留`
                      : "按实验入库时间倒序"}
                </span>
              </div>
              <button
                type="button"
                className="refresh-button"
                onClick={refreshBenchmarks}
                disabled={refreshing}
              >
                {refreshing
                  ? "刷新中…"
                  : READ_ONLY_DEMO ? "刷新快照" : "刷新数据"}
              </button>
            </div>
          </div>

          {loading ? (
            <div className="empty-state" role="status">
              <span className="loading-dot" />
              {READ_ONLY_DEMO
                ? "正在读取公开实验快照…"
                : "正在读取本地 SQLite 指标…"}
            </div>
          ) : (
            viewMode === "index"
              ? indexSummaries.length === 0
              : viewMode === "aggregate"
                ? aggregates.length === 0
                : rows.length === 0
          ) ? (
            <div className="empty-state">
              SQLite 中还没有 benchmark 记录，请先完成一次压测。
            </div>
          ) : (
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    {TABLE_COLUMNS.map((column) => {
                      const active = sortKey === column.key;
                      return (
                        <th
                          key={column.key}
                          aria-sort={
                            active
                              ? sortDirection === "asc"
                                ? "ascending"
                                : "descending"
                              : "none"
                          }
                        >
                          <button
                            type="button"
                            className={`sort-button ${active ? "sort-button-active" : ""}`}
                            onClick={() => changeSort(column.key)}
                            title={`${column.label}排序`}
                          >
                            <span>{column.label}</span>
                            <span className="sort-icon" aria-hidden="true">
                              {active
                                ? sortDirection === "asc" ? "↑" : "↓"
                                : "↕"}
                            </span>
                          </button>
                        </th>
                      );
                    })}
                  </tr>
                </thead>
                <tbody>
                  {viewMode === "index"
                    ? sortedIndexSummaries.map((summary) => {
                      const expanded = expandedIndexGroups.includes(summary.key);
                      return (
                        <Fragment key={summary.key}>
                          <tr className="index-summary-row">
                            <td>
                              <strong>{summary.indexType}</strong>
                            </td>
                            <td>
                              <button
                                type="button"
                                className="summary-expand"
                                aria-expanded={expanded}
                                onClick={() => toggleIndexGroup(summary.key)}
                              >
                                <span>{summary.configurationCount} 组参数</span>
                                <i aria-hidden="true">⌄</i>
                              </button>
                            </td>
                            <td>
                              <strong>
                                {formatNumberRange(
                                  metricMeanRange(
                                    summary.aggregates,
                                    (aggregate) => aggregate.latency_p99_ms,
                                  ),
                                  2,
                                  " ms",
                                )}
                              </strong>
                              <small className="submetric">参数配置范围</small>
                            </td>
                            <td>
                              <strong>
                                {formatPercentRange(metricMeanRange(
                                  summary.aggregates,
                                  (aggregate) => aggregate.recall,
                                ))}
                              </strong>
                              <small className="submetric">参数配置范围</small>
                            </td>
                            <td>
                              <strong>
                                {formatNumberRange(
                                  metricMeanRange(
                                    summary.aggregates,
                                    (aggregate) =>
                                      aggregate.vector_index_memory_mib,
                                  ),
                                  2,
                                  " MiB",
                                )}
                              </strong>
                              <small className="submetric">参数配置范围</small>
                            </td>
                            <td>
                              <strong>
                                {formatNumberRange(
                                  metricMeanRange(
                                    summary.aggregates,
                                    (aggregate) =>
                                      aggregate.insert_duration_seconds,
                                  ),
                                  2,
                                  " s",
                                )}
                              </strong>
                            </td>
                            <td>
                              <strong>
                                {formatNumberRange(
                                  metricMeanRange(
                                    summary.aggregates,
                                    (aggregate) =>
                                      aggregate.optimize_duration_seconds,
                                  ),
                                  2,
                                  " s",
                                )}
                              </strong>
                            </td>
                            <td>
                              <strong>{summary.concurrency}</strong>
                              <small className="submetric">
                                {summary.concurrencyDuration ?? "—"}s
                              </small>
                            </td>
                          </tr>
                          {expanded && summary.aggregates.map((aggregate, index) => (
                            <tr
                              className="summary-detail-row"
                              key={`${aggregate.configuration_key}-${aggregate.stage_index}`}
                            >
                              <td>
                                <span className="detail-index-label">
                                  配置 {index + 1}
                                </span>
                              </td>
                              <AggregateValueCells aggregate={aggregate} />
                            </tr>
                          ))}
                        </Fragment>
                      );
                    })
                    : viewMode === "aggregate"
                      ? sortedAggregates.map((aggregate) => (
                        <tr
                          key={`${aggregate.configuration_key}-${aggregate.stage_index}`}
                        >
                          <td>
                            <strong>{aggregate.index_type}</strong>
                          </td>
                          <AggregateValueCells aggregate={aggregate} />
                        </tr>
                      ))
                      : sortedRows.map(({ run, stage }) => (
                    <tr key={`${run.run_id}-${run.case_index}-${stage.stage_index}`}>
                      <td>
                        <strong>{run.index_type}</strong>
                      </td>
                      <td>
                        <div className="params">
                          {formatIndexParameters(
                            run.index_parameters,
                            run.search_parameters,
                          ).map((parameter) => (
                            <span key={parameter}>{parameter}</span>
                          ))}
                        </div>
                      </td>
                      <td>
                        <strong>{number(stage.latency_p99_ms)} ms</strong>
                        <small className="submetric">
                          Avg {number(stage.latency_avg_ms)} ms
                        </small>
                      </td>
                      <td>
                        <strong>{recallPercent(run.recall)}</strong>
                        <small className="submetric">
                          nDCG {recallPercent(run.ndcg)}
                        </small>
                      </td>
                      <td>
                        <strong>{number(run.vector_index_memory_mib)} MiB</strong>
                        <small className="submetric">
                          {run.monitoring_error ? "采集不完整" : "即时快照"}
                        </small>
                      </td>
                      <td>
                        <strong>{number(run.insert_duration_seconds)} s</strong>
                      </td>
                      <td>
                        <strong>{number(run.optimize_duration_seconds)} s</strong>
                      </td>
                      <td>
                        <strong>{stage.concurrency}</strong>
                        <small className="submetric">
                          {number(stage.duration_seconds, 0)}s
                        </small>
                      </td>
                    </tr>
                      ))}
                </tbody>
              </table>
            </div>
          )}

          <div className="comparison">
            <span>数据口径</span>
            <p>
              Insert 为 VectorDBBench 数据写入耗时，Optimize 为索引构建耗时；
              Vector Index 为实验完成、写入 SQLite 时的即时内存快照。
              缺失监控指标显示为“—”，不会用 0 代替。
            </p>
          </div>
        </section>

        <section className="analysis-panel">
          <div className="analysis-heading">
            <div>
              <p className="eyebrow">PARAMETER SENSITIVITY</p>
              <h2>参数影响分析</h2>
              <p>
                只比较相同数据集、TopK、并发和 Milvus 环境下的聚合结果；
                同一索引的参数档位按所选横轴连接。多个参数同步变化时，
                折线表示整体配置轨迹。
              </p>
            </div>
            <div className="analysis-controls">
              <label>
                <span>索引类型</span>
                <select
                  value={effectiveAnalysisIndex}
                  onChange={(event) => {
                    setAnalysisIndexType(event.target.value);
                    setAnalysisParameter("");
                  }}
                  disabled={analysisIndexOptions.length === 0}
                >
                  {analysisIndexOptions.map((indexType) => (
                    <option key={indexType} value={indexType}>
                      {indexType}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                <span>分析参数</span>
                <select
                  value={effectiveAnalysisParameter}
                  onChange={(event) => setAnalysisParameter(event.target.value)}
                  disabled={analysisParameterOptions.length === 0}
                >
                  {analysisParameterOptions.map((parameter) => (
                    <option key={parameter.name} value={parameter.name}>
                      {parameter.name} · {parameter.distinctCount} 个值
                    </option>
                  ))}
                </select>
              </label>
            </div>
          </div>

          {analysisBaseline && effectiveAnalysisParameter ? (
            <>
              <div className="analysis-context">
                <span>{analysisBaseline.case_type}</span>
                <span>{analysisBaseline.metric_type}</span>
                <span>TopK {analysisBaseline.top_k}</span>
                <span>并发 {analysisBaseline.concurrency}</span>
                <span>{analysisBaseline.db_label ?? "未标记环境"}</span>
              </div>
              <div className="analysis-chart-grid">
                <MetricTrendChart
                  points={analysisPoints}
                  metric="p99"
                  title={`P99 随 ${effectiveAnalysisParameter} 变化`}
                />
                <MetricTrendChart
                  points={analysisPoints}
                  metric="recall"
                  title={`Recall 随 ${effectiveAnalysisParameter} 变化`}
                />
                <MetricTrendChart
                  points={analysisPoints}
                  metric="memory"
                  title={`Vector Index 随 ${effectiveAnalysisParameter} 变化`}
                />
                <TradeoffChart points={tradeoffPoints} />
              </div>
            </>
          ) : (
            <div className="analysis-empty">
              {analysisBaseline
                ? "当前索引没有至少两个不同取值的可分析参数。"
                : "暂无聚合结果，完成 benchmark 后即可生成参数趋势图。"}
            </div>
          )}
          {analysisBaseline ? (
            <>
              <RecommendationSection
                analysis={withinIndexRecommendations}
                indexType={analysisBaseline.index_type}
              />
            </>
          ) : null}
        </section>

        <footer>
          <span>
            {READ_ONLY_DEMO
              ? "公开只读数据 · VectorDBBench 实验快照"
              : "本地只读数据源 · FastAPI / SQLite"}
          </span>
          <span>
            {READ_ONLY_DEMO
              ? "不连接 Milvus 集群，不提供实验执行入口"
              : "刷新页面不会重新执行压测"}
          </span>
        </footer>
      </div>
    </main>
  );
}
