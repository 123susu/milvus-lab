$ErrorActionPreference = "Stop"
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

$deploymentDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$composeFile = Join-Path $deploymentDirectory "docker-compose.yml"

docker compose -f $composeFile down

Write-Host "Milvus Cluster stopped. Docker volumes and data were retained."
