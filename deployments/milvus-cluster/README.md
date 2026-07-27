# Milvus Cluster local deployment

This directory contains the complete, reproducible Docker Compose deployment
for a local Milvus `v2.6.21` cluster.

## Topology

| Service | Purpose | Local resource limit |
|---|---|---:|
| etcd | Metadata storage | 0.5 CPU / 512 MiB |
| MinIO | Object storage | 0.5 CPU / 1 GiB |
| MixCoord | RootCoord, QueryCoord, and DataCoord control plane | 1 CPU / 1.5 GiB |
| Proxy | SDK entry point and request routing | 1 CPU / 1 GiB |
| StreamingNode | Woodpecker WAL and streaming ingestion | 1 CPU / 1.5 GiB |
| DataNode | Data ingestion, flushing, compaction, and index work | 2 CPU / 2 GiB |
| QueryNode | Segment loading and vector search | 2 CPU / 3 GiB |

This is a single-machine, minimum-size distributed deployment for learning,
functional verification, and light benchmarks. It is not a production
high-availability deployment.

Milvus recommends at least 8 CPU cores and 32 GiB RAM for Cluster. Docker
Desktop currently has about 16 GiB, so this Compose intentionally uses one
instance of each worker role and avoids Pulsar.

## Start

From the repository root:

```powershell
.\deployments\milvus-cluster\start.ps1
```

The script:

1. creates the shared `milvus-lab` Docker network when needed;
2. stops `milvus-standalone` without deleting its container or data;
3. starts the Cluster services.

Endpoints:

```text
Milvus SDK:    http://localhost:19530
Milvus WebUI:  http://localhost:9091/webui/
MinIO API:     http://localhost:9000
MinIO Console: http://localhost:9001
```

Milvus authentication remains disabled, matching the previous local setup.

## Status

```powershell
.\deployments\milvus-cluster\status.ps1
```

## Stop

```powershell
.\deployments\milvus-cluster\stop.ps1
```

`docker compose down` removes Cluster containers and its private Compose
network resources but retains all named volumes.

## Restore the previous Standalone temporarily

The switch does not delete the previous Standalone container or its data. To
roll back:

```powershell
.\deployments\milvus-cluster\stop.ps1
docker start milvus-standalone
```

The Standalone and Cluster use different storage. Existing Standalone
collections are not automatically copied into the new Cluster.

## Data

Persistent data is stored in these Docker named volumes:

```text
milvus-cluster-etcd-data
milvus-cluster-minio-data
milvus-cluster-streamingnode-data
milvus-cluster-datanode-data
milvus-cluster-querynode-data
```

Do not add `--volumes` to the stop command unless the Cluster data should be
permanently deleted.

## Configuration

- `docker-compose.yml`: topology, images, ports, health checks, and limits.
- `user.yaml`: Milvus configuration overrides.
- Milvus image: `milvusdb/milvus:v2.6.21`.
- Message queue: Woodpecker.
- Metadata: etcd `v3.5.25`.
- Object storage: MinIO `RELEASE.2024-12-18T13-15-44Z`.

The dependency versions match the official Milvus `v2.6.21` Standalone
Compose, while the Milvus processes are split into Cluster roles.
