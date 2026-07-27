# VectorDBBench：面向 Milvus 的向量数据库基准测试指南

## 1. VectorDBBench 是什么

VectorDBBench（也称 VDBBench）是由 Zilliz 主导维护的开源向量数据库基准测试工具。它不是向量数据库，也不是 Embedding 模型，而是一个统一的测试框架。

它解决的核心问题是：

> 在固定数据集、查询、真值、并发模型和统计方式的前提下，测量向量数据库的写入、索引、检索、过滤及混合读写能力。

VectorDBBench 可以测试 Milvus，也支持多种其他向量数据库。使用统一框架的价值在于减少不同测试脚本带来的口径差异，使不同索引参数、不同数据库版本和不同部署方式之间的结果更容易复现和比较。

本项目使用的版本：

| 组件 | 版本 |
|---|---|
| VectorDBBench | 1.0.22 |
| PyMilvus | 2.6.17 |
| Milvus Standalone | 2.6.21 |

## 2. 它不负责什么

VectorDBBench 主要评测数据库系统，不负责训练 Embedding 模型，也不直接判断一段中文和另一段中文在业务上是否相关。

需要区分两种评测：

| 评测对象 | 关注问题 | 常见真值 |
|---|---|---|
| Embedding 模型 | 语义是否表达准确 | 人工标注的 query-document 相关性 |
| Milvus/ANN 索引 | 数据库能否快速找回精确近邻 | FLAT 暴力搜索得到的精确 TopK |

在 Milvus 索引评测中，向量通常已经固定。VectorDBBench 将 Milvus 返回的 TopK ID 与数据集提供的精确近邻 ID 比较，从而计算 ANN Recall。这样可以尽量把 Embedding 模型质量排除在外，专门观察索引和数据库系统的误差与性能。

## 3. 一次测试是怎样运行的

典型流程如下：

```text
选择测试用例和数据集
        ↓
下载并缓存 train/test/neighbors
        ↓
创建数据库 Collection
        ↓
批量写入 train 向量
        ↓
Flush、Compaction、构建并加载索引
        ↓
串行搜索：计算 Recall、nDCG、单请求延迟
        ↓
并发搜索：计算 QPS、平均延迟、P95、P99
        ↓
生成 JSON 结果和日志
```

VectorDBBench 将流程拆成几个可控制阶段：

- `drop-old`：删除上一次基准 Collection；
- `load`：建表、写入、整理数据、构建并加载索引；
- `search-serial`：逐个执行查询，评估准确率与单请求延迟；
- `search-concurrent`：以不同并发数持续发起查询，评估吞吐量和尾延迟。

加载新数据时，当前版本要求同时执行 `drop-old`。Milvus 适配器默认只管理名为 `VDBBench` 的 Collection，不会删除其他名称的 Collection。

## 4. 数据集包含什么

一个标准 ANN 数据集通常包含三部分：

```text
train/base：写入向量数据库的库内向量
test/query：用于发起搜索的查询向量
neighbors：每个查询对应的精确近邻 ID
```

其中 `neighbors` 是评价索引召回率的真值。假设 TopK 为 100：

```text
Recall@100 =
Milvus 返回结果与精确 Top100 的交集数量 / 100
```

例如，Milvus 找回了 96 个精确近邻，则该查询的 Recall@100 为 0.96。

VectorDBBench 内置或使用的典型数据包括：

| 数据 | 特征 | 主要用途 |
|---|---|---|
| SIFT | 128 维，L2 | 经典 ANN 和容量基线 |
| GIST | 960 维，L2 | 高维计算与内存压力 |
| Cohere | 768 维文本向量，Cosine | 接近常见语义检索负载 |
| OpenAI | 1536 维文本向量，Cosine | 高维文本向量负载 |
| LAION | 大规模 768 维向量 | 千万、亿级容量和扩展性测试 |

当前项目默认使用 `Performance1536D50K`：

- 50,000 条库内向量；
- 1,536 维；
- Cosine 距离；
- 1,000 条查询；
- 每条查询带有精确近邻真值。

VectorDBBench 也支持自定义数据集，主要文件格式为：

```text
train.parquet
test.parquet
neighbors.parquet
```

自定义业务向量也可以接入，但标准索引基线优先使用带精确近邻真值的公共数据。

## 5. 主要测试场景

### 5.1 搜索性能

固定数据和索引后测试：

- Recall 和 nDCG；
- 单请求平均延迟；
- P95/P99 尾延迟；
- 不同并发数下的 QPS；
- QPS 是否随并发增加而饱和或下降。

### 5.2 容量

持续装载向量，观察系统在给定资源下能够承载的数据规模。SIFT 128 维和 GIST 960 维常用于比较维度对容量的影响。

容量用例可能持续写入直到达到限制，不适合作为第一次烟测。

### 5.3 标量过滤

在向量检索之外附加条件，例如：

```text
category == "database"
id >= 10000
```

通过改变过滤选择率，观察标量过滤、候选集大小、Recall 和延迟之间的关系。

### 5.4 持续写入

在保持查询压力的同时持续插入数据，用来观察：

- 写入是否影响查询 P99；
- Growing Segment 和 Sealed Segment 混合查询表现；
- 新数据何时变得可搜索；
- Flush、Seal 和索引构建对在线流量的影响。

这类场景比静态查询更接近真实生产系统。

### 5.5 自定义数据集

可将生产向量数据脱敏后转换成 Parquet，再沿用同一套加载、并发和结果统计框架。

## 6. 主要指标应该怎样理解

### Recall@K

衡量近似索引相对精确搜索丢失了多少近邻。

- 越接近 1 越好；
- 提高 HNSW `ef` 或 IVF `nprobe` 通常能提升 Recall；
- Recall 提升往往会增加计算量和查询延迟。

### nDCG@K

同时考虑结果是否正确以及正确结果的排序位置。相关结果排得越靠前，得分越高。

### QPS

每秒完成的查询数。QPS 必须与并发数、TopK、向量维度、Recall 和延迟一起观察，不能单独比较。

### P95/P99

P99 表示 99% 请求的延迟不超过该值。对在线数据库而言，P99 往往比平均延迟更能暴露锁竞争、调度抖动、缓存失效和资源争用。

### Insert Duration

将向量批量发送并写入数据库所需的时间。需要同时记录批大小、客户端并发和数据可见性要求。

### Optimize Duration

VectorDBBench 在搜索前执行数据整理、Flush、Compaction、索引构建和加载等适配器流程。它是一个组合耗时，不应直接等同于纯索引构建时间。

若要分析 Milvus 内核，需要进一步将它拆分为：

```text
Flush
→ Segment Seal
→ Compaction
→ Index Build
→ Index Load
```

## 7. 与目标岗位的对应关系

VectorDBBench 能覆盖岗位描述中的一部分关键能力：

| 岗位能力 | 可执行实验 |
|---|---|
| 索引构建 | 对比 HNSW、IVF_FLAT、IVF_SQ8、DiskANN |
| 查询优化 | 调整 `ef`、`nprobe`、TopK 和并发 |
| 性能调优 | 分析 Recall、QPS、P99、CPU、内存和磁盘 |
| 网关和服务 | 观察客户端并发、超时、背压与失败率 |
| 高可靠 | 增加重启、故障注入和恢复测试 |
| 高扩展 | 后续在 Milvus Distributed/Kubernetes 上横向扩容 |

但 VectorDBBench 只是评测入口。要体现“内核开发”能力，还需要根据结果继续定位：

- QueryNode 的 CPU 与内存热点；
- Knowhere 索引参数和执行路径；
- Segment 数量与大小；
- Compaction 调度；
- 数据加载、缓存和磁盘 IO；
- Proxy 聚合和查询调度；
- 不同一致性级别的代价。

## 8. 本项目如何运行

项目使用独立的基准测试环境：

```text
.venv-bench  VectorDBBench
```

这样可以隔离 VectorDBBench、PyMilvus 及数据处理依赖。

执行 HNSW 基准：

```powershell
.\benchmark\vectordbbench\run_benchmark.ps1 -Command milvushnsw
```

脚本配置：

```text
Milvus URI       http://localhost:19530
鉴权             无
Milvus 资源      4 CPU / 8 GiB / 无额外 swap
Collection       VDBBench
数据集           Performance1536D50K
索引             HNSW
M                16
efConstruction   128
ef               128
TopK             100
并发             1, 5, 10, 20
每档持续时间     10 秒
```

数据缓存：

```text
data/vectordbbench/datasets
```

结果和日志：

```text
results/vectordbbench
```

## 9. 建议的后续实验顺序

1. HNSW 参数矩阵：
   - `M = 8, 16, 32`
   - `efConstruction = 64, 128, 256`
   - `ef = 100, 128, 256, 512`
2. 绘制 Recall-QPS-P99 三者的权衡曲线。
3. 测试 IVF_FLAT 的 `nlist/nprobe`。
4. 测试 IVF_SQ8 或 IVF_PQ 的内存与精度权衡。
5. 增加标量过滤选择率测试。
6. 增加持续写入下的查询测试。
7. 将 VectorDBBench 指标与 Milvus Prometheus 指标对齐。
8. 迁移到 Milvus Distributed/Kubernetes，测试扩容和故障恢复。

## 10. 参考资料

- VectorDBBench 仓库：<https://github.com/zilliztech/VectorDBBench>
- VectorDBBench Leaderboard：<https://zilliz.com/benchmark>
- Milvus 文档：<https://milvus.io/docs>
- 当前项目运行脚本：`benchmark/vectordbbench/run_benchmark.ps1`
