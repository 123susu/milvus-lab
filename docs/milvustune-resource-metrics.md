# MilvusTune CPU 与内存采集方案

## 1. 数据来源

VectorDBBench 负责产生客户端指标：

- 并发数；
- QPS；
- 平均延迟；
- P95、P99；
- 串行 Recall 和 nDCG。

CPU、内存不需要由 VectorDBBench 自己输出。当前项目已经部署：

```text
Milvus containers → cAdvisor → Prometheus
```

MilvusTune 只需要用每个并发阶段的开始和结束时间查询 Prometheus，再把资源指标和 VectorDBBench 的并发结果合并。

## 2. 时间窗口

VectorDBBench 日志已经包含每个并发阶段的有效压测时间：

```text
Syncing all process and start concurrency search, concurrency=5
End search in concurrency 5
```

资源统计必须使用这两个时间点之间的窗口，而不是：

- 从整个 Benchmark 开始到结束；
- `Start search` 到 `End search`；
- JSON 文件的写入时间。

`Start search` 后可能还有 Worker 初始化和同步等待，不属于稳定压测窗口。

建议后续把窗口保存为：

```json
{
  "concurrency": 5,
  "started_at": "2026-07-27T09:34:58.788+08:00",
  "finished_at": "2026-07-27T09:35:08.866+08:00"
}
```

## 3. PromQL

### 3.1 Cluster CPU 使用核数

```promql
sum(
  rate(
    container_cpu_usage_seconds_total{
      job="cadvisor",
      container_label_com_docker_compose_project="milvus-cluster-local"
    }[10s]
  )
)
```

单位是 CPU cores：

```text
1.0 = 持续占用一个逻辑 CPU 核
```

对每个并发窗口保存：

- `cpu_cores_avg`
- `cpu_cores_peak`

### 3.2 Cluster 内存 Working Set

```promql
sum(
  container_memory_working_set_bytes{
    job="cadvisor",
    container_label_com_docker_compose_project="milvus-cluster-local"
  }
)
```

对每个并发窗口保存：

- `memory_bytes_avg`
- `memory_bytes_peak`

展示时换算为 MiB 或 GiB。

### 3.3 按组件采集 CPU

以 QueryNode 为例：

```promql
sum(
  rate(
    container_cpu_usage_seconds_total{
      job="cadvisor",
      container_label_com_docker_compose_project="milvus-cluster-local",
      name=~".*querynode.*"
    }[10s]
  )
)
```

建议分别保存：

- Proxy；
- QueryNode；
- StreamingNode；
- DataNode；
- MixCoord。

### 3.4 按组件采集内存

```promql
sum(
  container_memory_working_set_bytes{
    job="cadvisor",
    container_label_com_docker_compose_project="milvus-cluster-local",
    name=~".*querynode.*"
  }
)
```

### 3.5 CPU 限额使用率

集群总 CPU 限额由多个容器的 Compose 配置组成。最清晰的做法是在任务环境快照中记录总限额，然后计算：

```text
cpu_limit_usage_percent = cpu_cores / configured_cpu_limit_cores × 100
```

例如总限额为 8 cores：

```text
3.2 cores / 8 cores × 100% = 40%
```

但总使用率可能掩盖单个 QueryNode 已达到自身限额，因此还必须检查各组件。

### 3.6 CPU Throttling

判断容器是否因为 CPU quota 被限流：

```promql
sum(
  rate(
    container_cpu_cfs_throttled_seconds_total{
      job="cadvisor",
      container_label_com_docker_compose_project="milvus-cluster-local"
    }[10s]
  )
)
```

如果 cAdvisor 当前版本没有暴露该指标，可以使用：

```promql
sum(
  rate(
    container_cpu_cfs_throttled_periods_total{
      job="cadvisor",
      container_label_com_docker_compose_project="milvus-cluster-local"
    }[10s]
  )
)
/
sum(
  rate(
    container_cpu_cfs_periods_total{
      job="cadvisor",
      container_label_com_docker_compose_project="milvus-cluster-local"
    }[10s]
  )
)
```

这比只看 Cluster CPU 总量更适合判断“并发 5 时 CPU 是否已经超限”。

## 4. Prometheus HTTP API

查询接口：

```text
GET /api/v1/query_range
```

请求参数：

```text
query=<PromQL>
start=<Unix timestamp>
end=<Unix timestamp>
step=1
```

当前 Prometheus 每秒抓取一次，`step=1` 可以保留并发窗口内的短时峰值。

查询结果处理：

1. 过滤 `NaN` 和无穷值；
2. 计算平均值；
3. 计算最大值；
4. 保存采样数量；
5. 采样数量不足时标记资源数据不完整；
6. 不用 `0` 表示查询失败。

## 5. 当前基线回填结果

根据现有日志中的并发窗口，从 Prometheus 历史数据回填得到：

| 并发 | QPS | P99 | Cluster CPU 平均 | Cluster CPU 峰值 | 内存平均 | 内存峰值 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 265.80 | 7.04 ms | 0.54 cores | 1.03 cores | 1.65 GiB | 2.10 GiB |
| 5 | 892.43 | 18.35 ms | 1.60 cores | 3.10 cores | 1.57 GiB | 1.59 GiB |
| 10 | 952.69 | 41.15 ms | 1.49 cores | 3.16 cores | 1.58 GiB | 1.59 GiB |
| 20 | 924.26 | 48.74 ms | 1.46 cores | 3.16 cores | 1.59 GiB | 1.61 GiB |

这里的 CPU 是整个 Milvus Cluster 的 cAdvisor 使用核数，不是 CPU 百分比。

这些数据说明：

- 并发 10 达到当前测试中的最大 QPS；
- 并发 20 的 QPS 下降，P99 继续上升；
- 仅看 Cluster CPU 总量不能证明 CPU quota 已经触顶；
- 需要继续查看 QueryNode、StreamingNode 的单组件 CPU 和 throttling 指标。

## 6. 与 Benchmark 结果合并

最终每个并发点保存：

```json
{
  "concurrency": 5,
  "qps": 892.4343,
  "latency_avg_ms": 5.5575,
  "latency_p95_ms": 9.5952,
  "latency_p99_ms": 18.3499,
  "cpu_cores_avg": 1.60,
  "cpu_cores_peak": 3.10,
  "memory_gib_avg": 1.57,
  "memory_gib_peak": 1.59,
  "resource_sample_count": 12
}
```

Recall 作为本次参数组合的串行指标单独保存，不需要复制到每个并发点：

```json
{
  "recall": 0.9607,
  "ndcg": 0.9679,
  "serial_latency_p95_ms": 4.2,
  "serial_latency_p99_ms": 5.8
}
```

## 7. 推荐实施顺序

1. 先解析 VectorDBBench 日志，提取每个并发的有效开始和结束时间。
2. 查询 Cluster CPU 和内存。
3. 保存平均值、峰值和采样数量。
4. 增加 QueryNode、StreamingNode 的组件级指标。
5. 增加 CPU throttling。
6. 在前端将 QPS、P99、CPU、内存按相同并发点对齐展示。
