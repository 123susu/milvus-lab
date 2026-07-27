# milvus-lab

本地 Milvus 向量数据库实验仓库，用于理解 Milvus 分布式架构、HNSW 检索、VectorDBBench 基准测试，以及 Prometheus + Grafana 可观测性。

当前环境为单机 Docker Compose 的 Milvus `v2.6.21` Cluster。它适合学习、功能验证和轻量压测；组件已经拆分，但每类工作节点只有一个实例，因此**不是高可用生产集群**。

## 当前完成情况

- 使用 Docker Compose 部署 Milvus Cluster：etcd、MinIO、MixCoord、Proxy、StreamingNode、DataNode、QueryNode。
- 保留旧的 `milvus-standalone` 容器作为回退，不与 Cluster 同时运行。
- 配置 VectorDBBench `1.0.22`，当前基线为 Milvus HNSW。
- 使用 `Performance1536D50K`（50,000 条、1536 维真实文本向量）进行检索评测。
- 部署 Prometheus、cAdvisor、Grafana；采集 Cluster 容器资源指标和 Milvus 组件 `/metrics`。
- 已建立 Cluster 总览、查询压测、写入索引、五个组件看板，以及原始指标浏览器。

## 架构与边界

```text
客户端 / VectorDBBench
          |
        Proxy
          |
   StreamingNode <----> QueryNode
          |                 |
        Woodpecker        sealed segments / index
          |                 |
       etcd / MinIO <------+

Prometheus <- /metrics (各 Milvus 组件)
Prometheus <- cAdvisor (Docker cgroup 资源)
Grafana    <- Prometheus
```

Milvus 2.6 中，StreamingNode 不只是写入节点：它带有嵌入式 QueryNode 角色，处理 growing segment，并在查询链路中协调 QueryNode、归并分片结果。因此并发查询期间 StreamingNode 出现 CPU 使用是正常现象。

### 本地资源限制

| 服务 | CPU 上限 | 内存上限 | 主要职责 |
|---|---:|---:|---|
| etcd | 0.5 | 512 MiB | 元数据 |
| MinIO | 0.5 | 1 GiB | 对象存储 |
| MixCoord | 1 | 1.5 GiB | RootCoord、DataCoord、QueryCoord 控制面 |
| Proxy | 1 | 1 GiB | SDK 入口与路由 |
| StreamingNode | 1 | 1.5 GiB | WAL、实时数据、分片级查询协调 |
| DataNode | 2 | 2 GiB | Flush、Compaction、索引相关数据处理 |
| QueryNode | 2 | 3 GiB | sealed segment 加载与向量检索 |
| **合计** | **8** | **10.5 GiB** | |

`replica_number=1`，Collection 当前只有一个查询副本。若扩容独立 QueryNode 后，才适合继续测试 `replica_number=2` 和查询横向扩容。

## Windows 重启后的启动方式

先启动 Docker Desktop，确认 Docker Engine 已就绪，再在仓库根目录执行：

```powershell
.\deployments\milvus-cluster\start.ps1
docker start attu
docker compose -f .\deployments\monitoring\docker-compose.yml up -d
```

检查状态：

```powershell
.\deployments\milvus-cluster\status.ps1
docker ps
Invoke-WebRequest -UseBasicParsing http://localhost:9091/healthz
```

正常情况下，7 个 `milvus-cluster-*` 容器应为 `healthy`。`milvus-standalone` 应保持停止状态，因为它与 Cluster 占用相同的 `19530`、`9091` 端口。

## 常用地址

| 服务 | 地址 |
|---|---|
| Milvus SDK | `http://localhost:19530` |
| Attu | `http://localhost:3000` |
| Milvus WebUI | `http://localhost:9091/webui/` |
| MinIO Console | `http://localhost:9001` |
| Prometheus | `http://localhost:9090` |
| Prometheus Targets | `http://localhost:9090/targets` |
| Grafana | `http://localhost:3001` |

当前 Milvus 未启用鉴权，仅用于本地实验。

## VectorDBBench：当前 HNSW 基线

配置文件：[benchmark/vectordbbench/config/milvushnsw.yml](benchmark/vectordbbench/config/milvushnsw.yml)

```text
CaseType:       Performance1536D50K
Index:          HNSW / COSINE
M:              16
efConstruction: 128
efSearch:       128
TopK:           100
num_shards:     1
replica_number: 1
并发：          1、5、10、20（每阶段 10 秒）
```

先校验配置：

```powershell
.\benchmark\vectordbbench\run_benchmark.ps1 -Command milvushnsw -DryRun
```

执行基准测试：

```powershell
.\benchmark\vectordbbench\run_benchmark.ps1 -Command milvushnsw
```

每次运行会重建 VectorDBBench 使用的 `VDBBench` Collection（`drop_old: true`），不会影响其他 Collection。结果保存在 `results/vectordbbench/Milvus/`：

```text
result_<日期>_<task_label>-<HHmmssfff>_milvus.json
result_<日期>_<task_label>-<HHmmssfff>_milvus.log
```

JSON 会格式化为 UTF-8、两空格缩进；日志也是 UTF-8。LLM 自动报告目前在 YAML 中保持 `enabled: false`，因此默认不会产生 Markdown 总结。

### 最近一次结果

结果文件：[result_20260727_hnsw-performance-cluster-local-093310362_milvus.json](results/vectordbbench/Milvus/result_20260727_hnsw-performance-cluster-local-093310362_milvus.json)

| 并发 | QPS | 平均延迟 | P95 | P99 |
|---:|---:|---:|---:|---:|
| 1 | 265.80 | 3.74 ms | 4.89 ms | 7.04 ms |
| 5 | 892.43 | 5.56 ms | 9.60 ms | 18.35 ms |
| 10 | **952.69** | 10.40 ms | 29.57 ms | 41.15 ms |
| 20 | 924.26 | 21.34 ms | 41.35 ms | 48.74 ms |

Recall 为 `96.07%`，NDCG 为 `96.79%`。当前硬件与参数下，并发约 `10` 是吞吐拐点；提高到 `20` 后，QPS 不再增长而尾延迟继续升高。

## 监控与 Grafana

Grafana 登录地址为 `http://localhost:3001`，默认账号为 `admin` / `admin`（首次登录后应修改）。所有 Dashboard 位于 `Dashboards -> Milvus`。

### 看板列表

| Dashboard | 用途 |
|---|---|
| `Milvus Cluster Overview` | 组件健康、Cluster CPU/内存、Collection、Segment、Query Worker 概览 |
| `Milvus Query & Benchmark` | QPS、P50/P95/P99、QueryNode 队列、HNSW 工作量 |
| `Milvus Write & Index` | 写入、Flush、Compaction、索引构建、对象存储数据量 |
| `Milvus Component - Proxy` | API 请求率、请求延迟、Proxy 队列与限流 |
| `Milvus Component - MixCoord` | 控制面、DDL、Collection、DML Channel、Query Worker |
| `Milvus Component - StreamingNode` | CPU、WAL、嵌入式查询工作线程、实际向量搜索与写入活动 |
| `Milvus Component - DataNode` | 消费、Flush、Compaction、TimeTick lag |
| `Milvus Component - QueryNode` | 读线程池、队列、搜索流水线、Segment、Entity |
| `Milvus Metrics Explorer` | 按组件和指标名查看所有原始 `/metrics` 时序与 Label |

`Cluster Container CPU Cores` 的单位是逻辑核数：`1.0` 表示 Cluster 容器合计持续占用一个逻辑核。它来自 cAdvisor，而不是 Milvus 自身。各 Milvus 组件的业务指标则由对应容器的 `:9091/metrics` 暴露，Prometheus 每秒抓取一次。

Grafana 的 Dashboard 定义位于 [deployments/monitoring/grafana/dashboards](deployments/monitoring/grafana/dashboards)。在 Windows + Docker Desktop 环境下，修改 JSON 后若十几秒内未自动刷新，可执行：

```powershell
docker restart milvus-grafana
```

这只会短暂重启 Grafana，不会影响 Milvus、Prometheus 或已保存的实验数据。

## 停止与回退

停止 Cluster（保留数据卷）：

```powershell
.\deployments\milvus-cluster\stop.ps1
```

停止监控：

```powershell
docker compose -f .\deployments\monitoring\docker-compose.yml down
```

停止 Attu：

```powershell
docker stop attu
```

不要在上述命令中添加 `--volumes`，除非明确要永久删除 Milvus、Prometheus 或 Grafana 数据。

如需临时切回旧 Standalone：

```powershell
.\deployments\milvus-cluster\stop.ps1
docker start milvus-standalone
```

Cluster 与 Standalone 使用不同存储，旧 Collection 不会自动迁移到 Cluster。

## 目录说明

```text
benchmark/vectordbbench/       VectorDBBench 运行器与按索引划分的配置
deployments/milvus-cluster/    Milvus Cluster Compose、资源限制、启动脚本
deployments/monitoring/        Prometheus、cAdvisor、Grafana 与 Dashboard 定义
docs/                          VectorDBBench、Milvus 索引和实验说明
results/vectordbbench/Milvus/  基准结果 JSON 与单次运行日志
```

## 详细文档

- [Cluster 启动与恢复](deployments/README.md)
- [Milvus Cluster 部署细节](deployments/milvus-cluster/README.md)
- [VectorDBBench 基线与运行器](benchmark/vectordbbench/README.md)
- [Prometheus 与 Grafana](deployments/monitoring/README.md)
- [VectorDBBench Milvus 索引与 CaseType 参考](docs/vectordbbench-milvus-reference.md)
