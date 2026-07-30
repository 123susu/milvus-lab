# Local deployment startup guide

This directory contains the reproducible deployment files for the local
Milvus Cluster and its monitoring stack.

```text
deployments/
|-- milvus-cluster/   Milvus v2.6.21 distributed deployment
\-- monitoring/       Prometheus, Grafana, and cAdvisor
```

## What should be running

The complete local environment contains:

| Group | Containers |
|---|---|
| Milvus Cluster | `milvus-cluster-etcd`, `milvus-cluster-minio`, `milvus-cluster-mixcoord`, `milvus-cluster-proxy`, `milvus-cluster-streamingnode`, `milvus-cluster-datanode`, `milvus-cluster-querynode` |
| Monitoring | `milvus-prometheus`, `milvus-grafana`, `milvus-cadvisor` |
| Management UI | `attu` |

The old `milvus-standalone` container is retained only as a backup of the
previous environment. Keep it stopped while the Cluster is running because
both deployments use ports `19530` and `9091`.

## Start after restarting Windows

First, start Docker Desktop and wait until its Docker Engine is ready. Then
open PowerShell and run:

```powershell
Set-Location D:\IdeaProjects\milvus-lab

.\deployments\milvus-cluster\start.ps1

docker start attu

docker compose -f .\deployments\monitoring\docker-compose.yml up -d
```

This order is recommended because:

1. the Cluster startup script creates the shared `milvus-lab` Docker network;
2. it starts etcd and MinIO before starting the Milvus components;
3. Attu connects to the Cluster through `host.docker.internal:19530`;
4. Prometheus starts after the Cluster and scrapes all Milvus components.

These commands are idempotent. It is safe to run them when the containers are
already running.

## Automatic restart behavior

The Milvus Cluster and monitoring containers use the `unless-stopped` restart
policy. If Docker Desktop starts automatically with Windows, they will
normally recover automatically.

The existing `attu` container does not have an automatic restart policy.
Start it manually after a reboot:

```powershell
docker start attu
```

Running the complete startup sequence is still recommended because it verifies
the expected Compose configuration and waits for every Milvus Cluster
container to become healthy.

## Verify the environment

Check the Cluster and its resource usage:

```powershell
.\deployments\milvus-cluster\status.ps1
```

All seven Cluster containers should show `healthy`.

Check the monitoring stack:

```powershell
docker compose -f .\deployments\monitoring\docker-compose.yml ps
```

Expected monitoring containers:

```text
milvus-prometheus
milvus-grafana
milvus-cadvisor
```

Check all currently running containers:

```powershell
docker ps
```

Check the Milvus Proxy health endpoint:

```powershell
Invoke-WebRequest -UseBasicParsing http://localhost:9091/healthz
```

An HTTP `200` response means the Proxy is healthy.

## Local addresses

| Service | Address |
|---|---|
| Milvus SDK | `http://localhost:19530` |
| Attu | `http://localhost:3000` |
| Milvus WebUI | `http://localhost:9091/webui/` |
| Jaeger Trace UI | `http://localhost:16686` |
| MinIO Console | `http://localhost:9001` |
| Prometheus | `http://localhost:9090` |
| Prometheus targets | `http://localhost:9090/targets` |
| Grafana | `http://localhost:3001` |
| Milvus Cluster dashboard | `http://localhost:3001/d/milvus-cluster-local/milvus-cluster-local` |
| cAdvisor | `http://localhost:8080` |

Milvus authentication is disabled in this local environment.

## Stop the environment

Stop the Cluster while retaining all data:

```powershell
.\deployments\milvus-cluster\stop.ps1
```

Stop monitoring while retaining Prometheus and Grafana data:

```powershell
docker compose -f .\deployments\monitoring\docker-compose.yml down
```

Stop Attu:

```powershell
docker stop attu
```

Do not add `--volumes` to either Compose `down` command unless the stored data
should be permanently deleted.

## Common startup problems

### Port 19530 or 9091 is already in use

Confirm that the old Standalone container is stopped:

```powershell
docker stop milvus-standalone
.\deployments\milvus-cluster\start.ps1
```

### Attu is unavailable

```powershell
docker start attu
docker logs attu --tail 50
```

Attu should connect to `host.docker.internal:19530`.

### Prometheus shows Milvus targets as DOWN

Start the Cluster first, then verify its health:

```powershell
.\deployments\milvus-cluster\start.ps1
.\deployments\milvus-cluster\status.ps1
```

Prometheus retries failed targets automatically; no metrics data needs to be
deleted.

## Detailed documentation

- [Milvus Cluster deployment](milvus-cluster/README.md)
- [Monitoring deployment](monitoring/README.md)
