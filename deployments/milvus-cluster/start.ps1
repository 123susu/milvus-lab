$ErrorActionPreference = "Stop"
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

$deploymentDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$composeFile = Join-Path $deploymentDirectory "docker-compose.yml"

docker compose -f $composeFile config --quiet

if (-not (docker network ls --filter "name=^milvus-lab$" --format "{{.Name}}")) {
    docker network create milvus-lab | Out-Null
}

$standaloneRunning = docker ps --filter "name=^milvus-standalone$" --format "{{.Names}}"
if ($standaloneRunning -eq "milvus-standalone") {
    Write-Host "Stopping the existing Standalone container without deleting its data..."
    docker stop milvus-standalone | Out-Null
}

Write-Host "Starting Milvus Cluster..."
docker compose -f $composeFile up -d --wait --wait-timeout 180

Write-Host ""
Write-Host "Cluster containers:"
docker compose -f $composeFile ps
