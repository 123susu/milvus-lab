# Milvus monitoring

This stack collects both the built-in Milvus Cluster component metrics and
Docker container metrics. cAdvisor reads Docker/cgroup statistics, Prometheus
stores them, and Grafana queries Prometheus.

## Components

- Prometheus `v3.13.1`: `http://localhost:9090`
- Grafana `13.1.1`: `http://localhost:3001`
- cAdvisor `0.55.1`: `http://localhost:8080`
- Milvus metrics: Proxy, MixCoord, StreamingNode, DataNode, and QueryNode

Grafana uses port `3001` because Attu already occupies port `3000`.

## Start

From the project root:

```powershell
docker compose -f .\deployments\monitoring\docker-compose.yml up -d
```

Open Grafana and sign in:

```text
URL:      http://localhost:3001
Username: admin
Password: admin
```

Grafana asks you to change the default password on first sign-in. To choose a
different initial password before the first start:

```powershell
$env:GRAFANA_ADMIN_PASSWORD = "your-password"
docker compose -f .\deployments\monitoring\docker-compose.yml up -d
```

The provisioned dashboards are under:

```text
Dashboards -> Milvus
```

## Verify

Prometheus target status:

```text
http://localhost:9090/targets
```

The five `milvus-cluster` component targets and the `cadvisor` target should
be `UP`.

Prometheus query:

```promql
up{job="milvus-cluster"}
```

The result should be `1`.

Total container working-set memory for the Milvus Cluster:

```promql
sum(container_memory_working_set_bytes{
  job="cadvisor",
  container_label_com_docker_compose_project="milvus-cluster-local"
})
```

Container memory limit:

```promql
sum(container_spec_memory_limit_bytes{
  job="cadvisor",
  container_label_com_docker_compose_project="milvus-cluster-local"
})
```

Container CPU usage in cores:

```promql
sum(rate(container_cpu_usage_seconds_total{
  job="cadvisor",
  container_label_com_docker_compose_project="milvus-cluster-local"
}[1m]))
```

The `name` label depends on the container name reported by Docker. Inspect the
available names with:

```promql
count by (name) (container_last_seen{job="cadvisor"})
```

## Stop

Stop the monitoring containers while retaining Prometheus and Grafana data:

```powershell
docker compose -f .\deployments\monitoring\docker-compose.yml down
```

Named volumes are intentionally retained. Do not add `--volumes` unless the
stored metrics and Grafana state should also be deleted.
