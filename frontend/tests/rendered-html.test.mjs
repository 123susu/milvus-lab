import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the MilvusTune local metrics shell", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>MilvusTune · Vector Performance Lab<\/title>/i);
  assert.match(html, /MilvusTune/);
  assert.match(html, /Milvus CPU 索引实验台/);
  assert.match(html, /配置并运行 VectorDBBench/);
  assert.match(html, /正在读取本地 SQLite 指标/);
  assert.doesNotMatch(html, /压测完成|指标合并|本地归档/);
  assert.doesNotMatch(html, /M32 实验|651\.41|Your site is taking shape/);
});

test("wires the page to the local benchmark API with resilient states", async () => {
  const page = await readFile(
    new URL("../app/page.tsx", import.meta.url),
    "utf8",
  );

  assert.match(page, /"use client"/);
  assert.match(page, /NEXT_PUBLIC_BENCHMARK_API_URL/);
  assert.match(page, /\/api\/benchmarks\?limit=100&offset=0/);
  assert.match(page, /\/api\/benchmark-aggregates\?limit=100&offset=0/);
  assert.match(page, /\/api\/benchmark-profiles/);
  assert.match(page, /\/api\/benchmark-jobs/);
  assert.match(page, /method:\s*"POST"/);
  assert.match(page, /parameters:\s*commonParameters/);
  assert.match(page, /index_matrix:\s*indexMatrixValues/);
  assert.match(page, /load_concurrency/);
  assert.match(page, /concurrency_duration/);
  assert.match(page, /task_label/);
  assert.match(page, /value=\{selectedCommand\}/);
  assert.doesNotMatch(page, /VectorDBBench 命令行/);
  assert.doesNotMatch(page, /command-config/);
  assert.match(page, /CASE_TYPE_METRIC_TYPES/);
  assert.match(page, /Performance1536D50K:\s*"COSINE"/);
  assert.match(page, /<span>metric_type<\/span>/);
  assert.match(page, /由 case_type 对应的标准数据集固定/);
  assert.match(page, /Milvus 连接与部署 <span className="fixed-badge">固定/);
  assert.match(page, /TopK 可调，其余固定/);
  assert.match(page, /const \[topK, setTopK\]/);
  assert.match(page, /onChange=\{\(event\) => setTopK\(Number\(event\.target\.value\)\)\}/);
  assert.match(page, /numericIndexValues\("ef_search"\)/);
  assert.match(page, /CPU 索引与参数 <span className="editable-badge">可调整/);
  assert.match(page, /milvusivfflat/);
  assert.match(page, /milvusivfsq8/);
  assert.match(page, /milvushnswsq/);
  assert.match(page, /milvushnswpq/);
  assert.match(page, /milvushnswprq/);
  assert.match(page, /refine_type/);
  assert.match(page, /refine_k/);
  assert.match(page, /sq_type/);
  assert.match(page, /nbits/);
  assert.match(page, /nrq/);
  assert.match(page, /milvusautoindex/);
  assert.match(page, /milvusflat/);
  assert.match(page, /nprobe ≤ nlist/);
  assert.match(page, /repetitions/);
  assert.match(page, /索引汇总/);
  assert.match(page, /参数明细/);
  assert.match(page, /原始 Run/);
  assert.match(page, /useState<ResultViewMode>\("index"\)/);
  assert.match(page, /\$\{indexSummaries\.length\} 类索引汇总/);
  assert.match(page, /\$\{aggregateTotal\} 组参数明细/);
  assert.match(page, /\$\{aggregateTotal\} 组参数配置/);
  assert.match(page, /\$\{rows\.length\} 个并发 Stage/);
  assert.match(page, /meanWithStddev/);
  assert.match(page, /metricMeanRange/);
  assert.match(page, /formatNumberRange/);
  assert.match(page, /formatPercentRange/);
  assert.match(page, /expandedIndexGroups/);
  assert.match(page, /summary-expand/);
  assert.match(page, /\{summary\.configurationCount\} 组参数/);
  assert.match(page, /\{ key: "indexType", label: "索引类型" \}/);
  assert.match(page, /\{ key: "indexParams", label: "索引参数" \}/);
  assert.match(page, /\{ key: "p99", label: "P99" \}/);
  assert.match(page, /\{ key: "recall", label: "Recall" \}/);
  assert.match(page, /\{ key: "vectorIndex", label: "Vector Index" \}/);
  assert.match(page, /\{ key: "insert", label: "Insert" \}/);
  assert.match(page, /\{ key: "optimize", label: "Optimize" \}/);
  assert.ok(
    page.indexOf('{ key: "indexParams", label: "索引参数" }')
      < page.indexOf('{ key: "p99", label: "P99" }'),
  );
  assert.ok(
    page.indexOf('{ key: "p99", label: "P99" }')
      < page.indexOf('{ key: "recall", label: "Recall" }'),
  );
  assert.ok(
    page.indexOf('{ key: "recall", label: "Recall" }')
      < page.indexOf('{ key: "vectorIndex", label: "Vector Index" }'),
  );
  assert.ok(
    page.indexOf('{ key: "vectorIndex", label: "Vector Index" }')
      < page.indexOf('{ key: "insert", label: "Insert" }'),
  );
  assert.match(page, /aggregate\.optimize_duration_seconds/);
  assert.match(page, /aggregate\.insert_duration_seconds/);
  assert.match(page, /run\.insert_duration_seconds/);
  assert.match(page, /Optimize 为索引构建耗时/);
  assert.match(page, /<strong>\{aggregate\.index_type\}<\/strong>/);
  assert.match(page, /<strong>\{run\.index_type\}<\/strong>/);
  assert.doesNotMatch(page, /formatTime/);
  assert.doesNotMatch(page, /\{ key: "qps", label: "QPS" \}/);
  assert.doesNotMatch(page, /\{ key: "cpu", label: "QueryNode CPU" \}/);
  assert.doesNotMatch(page, /metrics-config/);
  assert.doesNotMatch(page, /prometheus_url/);
  assert.match(page, /运行 Benchmark/);
  assert.match(page, /window\.confirm/);
  assert.match(page, /即将开始 \$\{plannedRunCount\} 次 Benchmark/);
  assert.match(page, /本操作会删除并重建 VDBBench Collection/);
  assert.match(page, /if \(!confirmed\) return/);
  assert.match(page, /取消任务/);
  assert.match(page, /TABLE_COLUMNS/);
  assert.match(page, /aria-sort/);
  assert.match(page, /changeSort/);
  assert.match(page, /sortedRows\.map/);
  assert.match(page, /cache:\s*"no-store"/);
  assert.match(page, /无法连接本地指标 API/);
  assert.match(page, /Vector Index/);
  assert.match(page, /缺失监控指标显示为“—”/);
  assert.match(page, /参数影响分析/);
  assert.match(page, /PARAMETER SENSITIVITY/);
  assert.match(page, /MetricTrendChart/);
  assert.match(page, /TradeoffChart/);
  assert.match(page, /P99–Recall–内存权衡/);
  assert.match(page, /ResizeObserver/);
  assert.match(page, /相同数据集、TopK、并发和 Milvus 环境/);
  assert.match(page, /五档配置轨迹/);
  assert.match(page, /latestByParameterValue/);
  assert.match(page, /\.filter\(\(\[, values\]\) => values\.size > 1\)/);
  assert.match(page, /折线表示整体配置轨迹/);
  assert.match(page, /当前索引没有至少两个不同取值的可分析参数/);
  assert.doesNotMatch(page, /const results\s*=/);
});
