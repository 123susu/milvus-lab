# VectorDBBench 的 Milvus 命令与 CaseType

> 整理依据：本项目 `.venv-bench` 中安装的 **VectorDBBench 1.0.22**。
> 当前测试环境：Milvus Standalone 2.6.21、无鉴权、容器限制为 4 CPU / 8 GiB。

## 1. 先理解命令结构

VectorDBBench 的命令可以拆成两部分：

```text
vectordbbench <数据库及索引子命令> --case-type <测试用例> [其他参数]
```

- `milvushnsw`、`milvusivfflat` 等子命令：决定使用 Milvus 的哪一种索引。
- `--case-type`：决定使用什么数据集、数据规模，以及执行容量、检索、过滤或流式测试。
- 同一个 CaseType 可以搭配不同索引，用于比较构建时间、召回率、延迟和 QPS。

查看当前安装版本实际支持的命令：

```powershell
& .\.venv-bench\Scripts\vectordbbench.exe --help
& .\.venv-bench\Scripts\vectordbbench.exe milvushnsw --help
```

## 2. Milvus 相关子命令

当前版本一共暴露了 **17 个** `milvus*` 子命令。

### 2.1 常规 CPU 索引

| 子命令 | Milvus 索引 | 主要用途 | 索引专属参数 | 当前 4C8G 环境 |
|---|---|---|---|---|
| `milvusautoindex` | `AUTOINDEX` | 让 Milvus 自动选择和配置索引 | 无 | 可跑 |
| `milvusflat` | `FLAT` | 精确暴力检索，可作为召回率和性能基线 | 无 | 只建议先跑 50K |
| `milvushnsw` | `HNSW` | 高召回、低延迟的内存图索引 | `--m`、`--ef-construction`、`--ef-search` | 最适合当前阶段 |
| `milvusivfflat` | `IVF_FLAT` | 倒排分桶，便于观察 `nlist/nprobe` 权衡 | `--lists`、`--probes` | 可跑 |
| `milvusivfsq8` | `IVF_SQ8` | IVF 加 8 位标量量化，降低内存 | `--lists`、`--probes` | 可跑 |
| `milvusivfrabitq` | `IVF_RABITQ` | IVF 加 RaBitQ 量化和可选精排 | IVF 参数及 RaBitQ/精排参数 | Milvus 支持时可实验 |

关键参数：

- `m`：HNSW 图中每个节点的最大邻接程度。更大通常提高召回率，也增加构建时间和内存。
- `ef-construction`：建图时的候选集合大小。更大通常提高图质量，但构建更慢。
- `ef-search`：查询时的候选集合大小。更大通常提高召回率，但查询更慢。
- `nlist`：IVF 的聚类中心/分桶数量。
- `nprobe`：每次查询探测的桶数量。越大通常召回越高、延迟也越高，并且不能大于 `nlist`。
- VectorDBBench 1.0.22 的直接 CLI 选项名是 `--lists/--probes`，
  但 YAML 字段及内部 Milvus 参数名仍是 `nlist/nprobe`。
- `rbq-bits-query`：RaBitQ 查询向量量化级别；当前 CLI 要求显式提供。
- `refine`、`refine-type`、`refine-k`：是否保留精排数据、精排数据类型、精排候选放大倍数。

### 2.2 HNSW 量化变体

| 子命令 | Milvus 索引 | 作用 | 额外参数 |
|---|---|---|---|
| `milvushnswsq` | `HNSW_SQ` | HNSW 加标量量化 | `--sq-type` 和精排参数 |
| `milvushnswpq` | `HNSW_PQ` | HNSW 加乘积量化 | `--nbits` 和精排参数 |
| `milvushnswprq` | `HNSW_PRQ` | HNSW 加残差乘积量化 | `--nbits`、`--nrq` 和精排参数 |

这些命令也都需要 HNSW 的三个基础参数：

```text
--m
--ef-construction
--ef-search
```

量化和精排参数：

| 参数 | 含义 |
|---|---|
| `--sq-type` | `SQ4U`、`SQ6`、`SQ8`、`BF16`、`FP16` 或 `FP32` |
| `--nbits` | PQ 编码使用的位数 |
| `--nrq` | PRQ 的残差子量化器数量 |
| `--refine` | 是否为精排保留数据；CLI 接收布尔值 |
| `--refine-type` | `SQ6`、`SQ8`、`BF16`、`FP16` 或 `FP32` |
| `--refine-k` | 精排候选数相对 TopK 的放大倍数 |

量化索引主要用于研究“内存占用、召回率、延迟”之间的权衡。建议先完成普通 HNSW 基线，再研究这些变体。

### 2.3 DiskANN

| 子命令 | Milvus 索引 | 专属参数 | 当前环境 |
|---|---|---|---|
| `milvusdiskann` | `DISKANN` | `--search-list` | 需要先配置磁盘索引，不建议现在直接跑 |

DiskANN 把更多索引数据放在磁盘上，适合超出内存容量的数据。它依赖高性能 SSD，并要求 Milvus 启用磁盘索引能力，例如 `queryNode.enableDisk: true`。当前 Standalone 没有确认完成这项配置，因此不能仅把 `milvushnsw` 改成 `milvusdiskann` 就运行。

`search-list` 是搜索阶段访问的候选规模。通常值越大，召回率越高，磁盘读取和查询延迟也越高。

### 2.4 SVS Vamana 系列

| 子命令 | Milvus 索引 | 特点 |
|---|---|---|
| `milvussvsvamana` | `SVS_VAMANA` | Vamana 图索引，支持多种存储格式 |
| `milvussvsvamanalvq` | `SVS_VAMANA_LVQ` | 使用 LVQ 压缩的 Vamana 变体 |
| `milvussvsvamanaleanvec` | `SVS_VAMANA_LEANVEC` | 使用 LeanVec 降维/压缩的 Vamana 变体 |

主要参数：

| 参数 | 含义 |
|---|---|
| `--svs-graph-max-degree` | 图的最大度数，范围 4～256，必填 |
| `--svs-construction-window-size` | 建图窗口，默认 40 |
| `--svs-alpha` | 图剪枝参数；默认值随距离度量变化 |
| `--svs-storage-kind` | 存储格式，默认 `fp32` |
| `--svs-search-window-size` | 查询搜索窗口 |
| `--svs-search-buffer-capacity` | 查询优先队列容量 |
| `--svs-leanvec-dim` | LeanVec 目标维度；仅 LeanVec 子命令使用，0 表示默认取原维度的一半 |

`--svs-storage-kind` 可选：

```text
fp32, fp16, sqi8,
lvq4x0, lvq4x4, lvq4x8,
leanvec4x4, leanvec4x8, leanvec8x8
```

这是较新的高级索引系列。是否能够创建成功，还取决于 Milvus 2.6.21 服务端构建和对应索引支持情况，建议放在 HNSW、IVF 对比之后。

### 2.5 GPU 索引

| 子命令 | Milvus 索引 | 主要参数 |
|---|---|---|
| `milvusgpubruteforce` | `GPU_BRUTE_FORCE` | `--metric-type`、`--limit` |
| `milvusgpuivfflat` | `GPU_IVF_FLAT` | IVF 参数、`--cache-dataset-on-device`、`--refine-ratio` |
| `milvusgpuivfpq` | `GPU_IVF_PQ` | GPU IVF 参数、`--m`、`--nbits` |
| `milvusgpucagra` | `GPU_CAGRA` | 图构建、搜索宽度、迭代次数等参数 |

`milvusgpucagra` 还要求：

```text
--intermediate-graph-degree
--graph-degree
--build_algo
--team-size
--search-width
--itopk-size
--min-iterations
--max-iterations
```

注意：当前 Milvus 是普通 CPU Standalone 容器，不具备 GPU 索引运行条件。客户端电脑有 GPU 并不够，运行 Milvus 的服务端容器必须使用支持 GPU 的 Milvus 部署，并正确挂载 NVIDIA 驱动和运行时。

## 3. 所有 Milvus 命令的公共参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--uri` | 无 | Milvus 地址，必填；当前为 `http://localhost:19530` |
| `--user-name` / `--password` | 空 | 鉴权信息；当前无鉴权，不需要传 |
| `--num-shards` | 1 | Collection 分片数量 |
| `--replica-number` | 1 | 查询副本数；Standalone 当前保持 1 |
| `--case-type` | 无 | 选择测试用例 |
| `--drop-old` / `--skip-drop-old` | `drop-old` | 是否删除同一任务的旧测试数据 |
| `--load` / `--skip-load` | `load` | 是否导入测试数据 |
| `--load-concurrency` | 0 | 数据导入并发；0 表示使用 CPU 数量 |
| `--search-serial` | 开启 | 是否运行串行检索 |
| `--search-concurrent` | 开启 | 是否运行并发检索 |
| `--k` | 100 | TopK |
| `--concurrency-duration` | 30 秒 | 每档并发持续时间 |
| `--num-concurrency` | `1,5,10,20,30,40,60,80` | 要测试的并发数列表 |
| `--concurrency-timeout` | 3600 秒 | 等待并发槽位的超时 |
| `--db-label` | 当前时间 | 标记数据库/环境，便于结果对比 |
| `--task-label` | 空 | 标记本次测试任务 |
| `--dry-run` | 关闭 | 只打印最终配置，不执行测试 |
| `--config-file` | 空 | 从 YAML 读取配置 |

这里的并发是 VectorDBBench 客户端创建的并发查询执行单元，不等于给 Milvus 固定分配同样数量的 CPU 线程。并发测试采用闭环压测：一个执行单元收到响应后马上发送下一次查询，因此实际 QPS 可以远高于并发数。

## 4. CaseType 命名规则

以 `Performance1536D50K` 为例：

```text
Performance | 1536D | 50K
测试类型       维度     数据量
```

它在 VectorDBBench 1.0.22 源码中对应：

```python
Dataset.OPENAI.manager(50_000)
```

也就是使用已经制作好的 OpenAI 向量评测数据：5 万条、1536 维。这里不是现场调用 OpenAI 接口生成 embedding。

常见缩写：

- `K`：千，例如 `50K = 50,000`。
- `M`：百万，例如 `10M = 10,000,000`。
- `D`：向量维度。
- 末尾 `1P` / `99P`：固定整数过滤用例中的 `filter_rate=0.01 / 0.99`。

## 5. 当前版本的全部 CaseType

CLI 一共允许选择 **23 个** CaseType。源码内部还有一个 `Custom` 枚举值，但它没有作为当前 CLI 的 `--case-type` 选项暴露，所以不应直接使用。

### 5.1 无过滤检索性能用例

| CaseType | 数据集 | 数据量 | 维度 | 4C8G 建议 |
|---|---|---:|---:|---|
| `Performance1536D50K` | OpenAI | 50K | 1536 | **当前首选** |
| `Performance1536D500K` | OpenAI | 500K | 1536 | 后续尝试，先检查磁盘和内存 |
| `Performance1536D5M` | OpenAI | 5M | 1536 | 当前不建议 |
| `Performance768D1M` | Cohere | 1M | 768 | 可作为第二阶段目标 |
| `Performance768D10M` | Cohere | 10M | 768 | 当前不建议 |
| `Performance768D100M` | LAION | 100M | 768 | 当前环境不适合 |
| `Performance1024D1M` | BioASQ | 1M | 1024 | 可作为第二阶段目标 |
| `Performance1024D10M` | BioASQ | 10M | 1024 | 当前不建议 |

这些用例通常会报告：

- 数据导入时间；
- 索引构建/优化时间；
- 串行查询平均延迟、P95、P99；
- Recall 和 nDCG；
- 不同并发下的 QPS、平均延迟、P95、P99。

### 5.2 固定整数过滤用例

| CaseType | 数据集 | 数据量 | 维度 | 过滤率 |
|---|---|---:|---:|---:|
| `Performance768D1M1P` | Cohere | 1M | 768 | 1% |
| `Performance768D1M99P` | Cohere | 1M | 768 | 99% |
| `Performance768D10M1P` | Cohere | 10M | 768 | 1% |
| `Performance768D10M99P` | Cohere | 10M | 768 | 99% |
| `Performance1536D500K1P` | OpenAI | 500K | 1536 | 1% |
| `Performance1536D500K99P` | OpenAI | 500K | 1536 | 99% |
| `Performance1536D5M1P` | OpenAI | 5M | 1536 | 1% |
| `Performance1536D5M99P` | OpenAI | 5M | 1536 | 99% |

这些用例用于考察“标量条件过滤 + 向量检索”的组合性能。源码通过数据 ID 阈值构造整数过滤条件；`1P` 和 `99P` 表示用例设置的过滤率，不应仅凭名称推断为“最终一定返回 1% 或 99% 的数据”，应结合生成的表达式理解。

### 5.3 容量用例

| CaseType | 数据集块 | 维度 | 行为 | 当前环境 |
|---|---|---:|---|---|
| `CapacityDim128` | SIFT 500K | 128 | 重复插入直到数据库无法继续承载 | 不要随便运行 |
| `CapacityDim960` | GIST 100K | 960 | 重复插入直到数据库无法继续承载 | 不要随便运行 |

注意：1.0.22 源码中 `CapacityDim128` 实际使用 `SIFT.manager(500_000)`，但它自带的英文描述仍写着 “SIFT 100K”。本表按实际代码填写为 500K。

容量测试不是普通的固定数据量性能测试。它会持续重复插入，目标就是找到容量上限，可能占满磁盘或长时间占用资源。

### 5.4 动态过滤和自定义数据集用例

| CaseType | 用途 | 关键参数 |
|---|---|---|
| `NewIntFilterPerformanceCase` | 自选内置数据集规模和整数过滤率 | `--dataset-with-size-type`、`--filter-rate` |
| `LabelFilterPerformanceCase` | 标签/字符串过滤性能 | `--dataset-with-size-type`、`--label-percentage` |
| `PerformanceCustomDataset` | 使用本地自定义数据集 | `--custom-dataset-*` 系列参数 |

`--dataset-with-size-type` 支持：

```text
Medium Cohere (768dim, 1M)
Large Cohere (768dim, 10M)
Medium Bioasq (1024dim, 1M)
Large Bioasq (1024dim, 10M)
Medium OpenAI (1536dim, 500K)
Large OpenAI (1536dim, 5M)
```

自定义数据集相关参数包括名称、目录、数据量、维度、距离度量、文件数量、是否包含 ground truth 等。若要让 Recall 有意义，必须提供正确的真值近邻数据。

### 5.5 流式用例

| CaseType | 用途 |
|---|---|
| `StreamingPerformanceCase` | 使用内置数据进行持续写入期间的检索测试 |
| `StreamingCustomDataset` | 使用自定义数据进行流式写入和检索测试 |

流式用例关注写入与查询同时发生时的吞吐、延迟和稳定性，比一次性导入后再检索更接近在线业务，但也更难控制变量，建议在基础索引对比完成后再做。

## 6. 当前 4C8G Standalone 的推荐测试顺序

### 使用统一配置文件运行

配置按照数据库及索引子命令拆分，一个子命令对应一个 YAML。当前本地页面支持：

```text
src/vectordbbench/config/milvushnsw.yml
src/vectordbbench/config/milvusivfflat.yml
src/vectordbbench/config/milvusivfsq8.yml
src/vectordbbench/config/milvusautoindex.yml
src/vectordbbench/config/milvusflat.yml
src/vectordbbench/run_benchmark.ps1
```

`-Command <名称>` 默认读取 `config/<名称>.yml`。例如以后增加 `milvusivfflat` 时，应建立独立的 `config/milvusivfflat.yml`。先执行只解析配置、不访问数据库的检查：

```powershell
.\src\vectordbbench\run_benchmark.ps1 -Command milvushnsw -DryRun
```

确认配置后正式运行：

```powershell
.\src\vectordbbench\run_benchmark.ps1 -Command milvushnsw
```

页面可以在 HNSW、HNSW_SQ、HNSW_PQ、HNSW_PRQ、IVF_FLAT、IVF_SQ8、
AUTOINDEX 和 FLAT 之间切换，并为每次实验生成独立 YAML。URI、CaseType、
TopK、并发、索引参数和结果标签都会保存在该次实验配置中。量化变体的
`refine`、`refine_type`、`refine_k` 及量化专属参数同样支持参数矩阵。

### 第一步：运行 HNSW 基准

```powershell
.\src\vectordbbench\run_benchmark.ps1 -Command milvushnsw
```

当前脚本使用：

```text
milvushnsw
Performance1536D50K
M=16
efConstruction=128
efSearch=128
```

### 第二步：用 FLAT 建立精确检索性能基线

先用 `--dry-run` 检查配置：

```powershell
& .\.venv-bench\Scripts\vectordbbench.exe milvusflat `
    --uri "http://localhost:19530" `
    --case-type "Performance1536D50K" `
    --drop-old `
    --search-serial `
    --search-concurrent `
    --concurrency-duration 10 `
    --num-concurrency "1,5,10,20" `
    --db-label "local-standalone-4c8g-2_6_21" `
    --task-label "flat-performance-4c8g" `
    --dry-run
```

FLAT 是精确检索，主要用于回答：“不使用近似索引时性能怎样，以及 HNSW 的召回损失有多大？”

### 第三步：比较 IVF_FLAT

```powershell
& .\.venv-bench\Scripts\vectordbbench.exe milvusivfflat `
    --uri "http://localhost:19530" `
    --case-type "Performance1536D50K" `
    --lists 128 `
    --probes 16 `
    --drop-old `
    --search-serial `
    --search-concurrent `
    --concurrency-duration 10 `
    --num-concurrency "1,5,10,20" `
    --db-label "local-standalone-4c8g-2_6_21" `
    --task-label "ivf-flat-performance-4c8g" `
    --dry-run
```

之后可以只改变 `nprobe`，例如比较 `8 / 16 / 32 / 64`，观察 Recall、延迟和 QPS 的变化。比较索引时，应保持 CaseType、TopK、并发、持续时间和资源限制相同。

### 第四步：再研究压缩索引

建议依次考虑：

1. `milvusivfsq8`
2. `milvushnswsq`
3. `milvusivfrabitq`
4. `milvushnswpq` / `milvushnswprq`

当前先不要运行：

- 两个 Capacity 用例；
- 5M、10M、100M 数据用例；
- 所有 GPU 子命令；
- 尚未配置磁盘索引的 `milvusdiskann`。

## 7. 版本与源码位置

本文不是根据命令名称猜测，而是按当前虚拟环境中的实际代码和 CLI 帮助整理：

```text
.venv-bench\Lib\site-packages\vectordb_bench\backend\clients\milvus\cli.py
.venv-bench\Lib\site-packages\vectordb_bench\backend\clients\milvus\config.py
.venv-bench\Lib\site-packages\vectordb_bench\backend\cases.py
```

VectorDBBench 升级后，子命令、CaseType 或参数可能变化，应重新执行 `vectordbbench --help` 和对应子命令的 `--help` 进行确认。
