param(
    [ValidateSet(
        "milvusautoindex",
        "milvusdiskann",
        "milvusflat",
        "milvusgpubruteforce",
        "milvusgpucagra",
        "milvusgpuivfflat",
        "milvusgpuivfpq",
        "milvushnsw",
        "milvushnswpq",
        "milvushnswprq",
        "milvushnswsq",
        "milvusivfflat",
        "milvusivfrabitq",
        "milvusivfsq8",
        "milvussvsvamana",
        "milvussvsvamanaleanvec",
        "milvussvsvamanalvq"
    )]
    [string]$Command = "milvushnsw",

    [string]$ConfigFile,

    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# Force both Python log files and native-process console output to UTF-8.
# This must be set before vectordbbench.exe starts because its logger is
# initialized while the Python package is imported.
$utf8Encoding = New-Object System.Text.UTF8Encoding($false)
[Console]::InputEncoding = $utf8Encoding
[Console]::OutputEncoding = $utf8Encoding
$OutputEncoding = $utf8Encoding
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$benchExecutable = Join-Path $projectRoot ".venv-bench\Scripts\vectordbbench.exe"
$pythonExecutable = Join-Path $projectRoot ".venv-bench\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $benchExecutable -PathType Leaf)) {
    throw "VectorDBBench is not installed. Expected: $benchExecutable"
}

if ([string]::IsNullOrWhiteSpace($ConfigFile)) {
    $ConfigFile = Join-Path $PSScriptRoot "config\$Command.yml"
}

$resolvedConfigFile = (Resolve-Path -LiteralPath $ConfigFile -ErrorAction Stop).Path

# Treat task_label in YAML as a stable prefix, then append the current time.
# VectorDBBench otherwise overwrites results with the same date and task label.
$taskLabelPrefix = & $pythonExecutable -c `
    "import sys, yaml; data=yaml.safe_load(open(sys.argv[1], encoding='utf-8')) or {}; print((data.get(sys.argv[2]) or {}).get('task_label') or sys.argv[2])" `
    $resolvedConfigFile `
    $Command

if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($taskLabelPrefix)) {
    throw "Failed to read task_label from configuration: $resolvedConfigFile"
}

$taskLabelPrefix = $taskLabelPrefix.Trim()
$runtimeTaskLabel = "$taskLabelPrefix-$(Get-Date -Format 'HHmmssfff')"
$runDate = Get-Date -Format "yyyyMMdd"

# Keep downloaded datasets, logs, and result databases inside the project.
$env:DATASET_SOURCE = "AliyunOSS"
$env:DATASET_LOCAL_DIR = Join-Path $projectRoot "data\vectordbbench\datasets"
$env:RESULTS_LOCAL_DIR = Join-Path $projectRoot "results\vectordbbench"
$milvusResultDirectory = Join-Path $env:RESULTS_LOCAL_DIR "Milvus"
$runArtifactStem = "result_${runDate}_${runtimeTaskLabel}_milvus"
$env:LOG_FILE = Join-Path $milvusResultDirectory "$runArtifactStem.log"

New-Item -ItemType Directory -Force $env:DATASET_LOCAL_DIR | Out-Null
New-Item -ItemType Directory -Force $env:RESULTS_LOCAL_DIR | Out-Null
New-Item -ItemType Directory -Force $milvusResultDirectory | Out-Null

$benchmarkArguments = @(
    $Command
    "--config-file"
    $resolvedConfigFile
    "--task-label"
    $runtimeTaskLabel
)

if ($DryRun) {
    $benchmarkArguments += "--dry-run"
}

Write-Host "VectorDBBench command : $Command"
Write-Host "Configuration file    : $resolvedConfigFile"
Write-Host "Runtime task label    : $runtimeTaskLabel"
Write-Host "Run log file          : $env:LOG_FILE"
Write-Host "Dry run               : $($DryRun.IsPresent)"

$benchmarkStartedAt = Get-Date
& $benchExecutable @benchmarkArguments

if ($LASTEXITCODE -ne 0) {
    throw "VectorDBBench exited with code $LASTEXITCODE"
}

if ($DryRun) {
    Write-Host "Dry run completed; result formatting and SQLite metric collection were skipped."
    return
}

# VectorDBBench writes compact, single-line JSON. Pretty-print every Milvus
# result produced by this run so it can be read and diffed directly.
$newResultFiles = @()

if (Test-Path -LiteralPath $milvusResultDirectory -PathType Container) {
    $newResultFiles = @(
        Get-ChildItem -LiteralPath $milvusResultDirectory -Filter "*.json" -File |
            Where-Object { $_.LastWriteTime -ge $benchmarkStartedAt.AddSeconds(-2) }
    )

    $newResultFiles | ForEach-Object {
        $resultFile = $_
        $temporaryFile = "$($resultFile.FullName).formatted.tmp"

        & $pythonExecutable -m json.tool `
            --no-ensure-ascii `
            --indent 2 `
            $resultFile.FullName `
            $temporaryFile

        if ($LASTEXITCODE -ne 0) {
            Remove-Item -LiteralPath $temporaryFile -Force -ErrorAction SilentlyContinue
            throw "Failed to format result JSON: $($resultFile.FullName)"
        }

        Move-Item -LiteralPath $temporaryFile -Destination $resultFile.FullName -Force
        Write-Host "Formatted result JSON : $($resultFile.FullName)"
    }
}

if ($newResultFiles.Count -eq 0) {
    Write-Warning "Benchmark completed, but no new Milvus result JSON was found."
    return
}

$latestResultFile = $newResultFiles |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
$metricsCollector = Join-Path $PSScriptRoot "collect_benchmark_metrics.py"
$metricsDatabase = Join-Path $env:RESULTS_LOCAL_DIR "benchmark_metrics.sqlite3"

& $pythonExecutable $metricsCollector `
    --config $resolvedConfigFile `
    --result $latestResultFile.FullName `
    --log $env:LOG_FILE `
    --database $metricsDatabase

if ($LASTEXITCODE -ne 0) {
    Write-Warning (
        "Benchmark succeeded, but its metrics were not saved to SQLite. " +
        "Check the benchmark log and Prometheus metric settings, then run again."
    )
}
