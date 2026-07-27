$ErrorActionPreference = "Stop"
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

$deploymentDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$composeFile = Join-Path $deploymentDirectory "docker-compose.yml"

docker compose -f $composeFile ps

Write-Host ""
Write-Host "Resource usage:"
docker stats --no-stream --format "table {{.Name}}`t{{.CPUPerc}}`t{{.MemUsage}}`t{{.MemPerc}}" `
    milvus-cluster-etcd `
    milvus-cluster-minio `
    milvus-cluster-mixcoord `
    milvus-cluster-proxy `
    milvus-cluster-streamingnode `
    milvus-cluster-datanode `
    milvus-cluster-querynode
