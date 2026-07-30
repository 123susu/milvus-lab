param(
    [string]$ApiBase = "http://127.0.0.1:8765",
    [string]$MilvusUri = "http://localhost:19530",
    [string]$DbLabel = "local-cluster-8c10_5g-2_6_21",
    [ValidateRange(1, 5)]
    [int]$Repetitions = 1,
    [switch]$DryRun,
    [switch]$RequireConfirmation
)

$ErrorActionPreference = "Stop"
$terminalStatuses = @("succeeded", "failed", "cancelled")

# The default configurations have already been collected and act as the
# middle baseline. This script adds four surrounding representative
# configurations, so each tunable index has five configurations in total
# without repeating the defaults.
$experiments = @(
    [pscustomobject]@{
        Command = "milvushnsw"
        Label = "HNSW"
        Variant = "very-low"
        Parameters = @{
            m = 8
            ef_construction = 64
            ef_search = 100
        }
    },
    [pscustomobject]@{
        Command = "milvushnsw"
        Label = "HNSW"
        Variant = "low"
        Parameters = @{
            m = 12
            ef_construction = 96
            ef_search = 112
        }
    },
    [pscustomobject]@{
        Command = "milvushnsw"
        Label = "HNSW"
        Variant = "high"
        Parameters = @{
            m = 24
            ef_construction = 192
            ef_search = 192
        }
    },
    [pscustomobject]@{
        Command = "milvushnsw"
        Label = "HNSW"
        Variant = "very-high"
        Parameters = @{
            m = 32
            ef_construction = 256
            ef_search = 256
        }
    },
    [pscustomobject]@{
        Command = "milvushnswsq"
        Label = "HNSW_SQ"
        Variant = "very-low"
        Parameters = @{
            m = 8
            ef_construction = 64
            ef_search = 100
            sq_type = "SQ8"
            refine = $true
            refine_type = "FP32"
            refine_k = 1.0
        }
    },
    [pscustomobject]@{
        Command = "milvushnswsq"
        Label = "HNSW_SQ"
        Variant = "low"
        Parameters = @{
            m = 12
            ef_construction = 96
            ef_search = 112
            sq_type = "SQ8"
            refine = $true
            refine_type = "FP32"
            refine_k = 1.0
        }
    },
    [pscustomobject]@{
        Command = "milvushnswsq"
        Label = "HNSW_SQ"
        Variant = "high"
        Parameters = @{
            m = 24
            ef_construction = 192
            ef_search = 192
            sq_type = "SQ8"
            refine = $true
            refine_type = "FP32"
            refine_k = 1.0
        }
    },
    [pscustomobject]@{
        Command = "milvushnswsq"
        Label = "HNSW_SQ"
        Variant = "very-high"
        Parameters = @{
            m = 32
            ef_construction = 256
            ef_search = 256
            sq_type = "SQ8"
            refine = $true
            refine_type = "FP32"
            refine_k = 1.0
        }
    },
    [pscustomobject]@{
        Command = "milvushnswpq"
        Label = "HNSW_PQ"
        Variant = "very-low"
        Parameters = @{
            m = 8
            ef_construction = 64
            ef_search = 100
            nbits = 8
            refine = $true
            refine_type = "FP32"
            refine_k = 1.0
        }
    },
    [pscustomobject]@{
        Command = "milvushnswpq"
        Label = "HNSW_PQ"
        Variant = "low"
        Parameters = @{
            m = 12
            ef_construction = 96
            ef_search = 112
            nbits = 8
            refine = $true
            refine_type = "FP32"
            refine_k = 1.0
        }
    },
    [pscustomobject]@{
        Command = "milvushnswpq"
        Label = "HNSW_PQ"
        Variant = "high"
        Parameters = @{
            m = 24
            ef_construction = 192
            ef_search = 192
            nbits = 8
            refine = $true
            refine_type = "FP32"
            refine_k = 1.0
        }
    },
    [pscustomobject]@{
        Command = "milvushnswpq"
        Label = "HNSW_PQ"
        Variant = "very-high"
        Parameters = @{
            m = 32
            ef_construction = 256
            ef_search = 256
            nbits = 8
            refine = $true
            refine_type = "FP32"
            refine_k = 1.0
        }
    },
    [pscustomobject]@{
        Command = "milvushnswprq"
        Label = "HNSW_PRQ"
        Variant = "very-low"
        Parameters = @{
            m = 8
            ef_construction = 64
            ef_search = 100
            nbits = 8
            nrq = 2
            refine = $true
            refine_type = "FP32"
            refine_k = 1.0
        }
    },
    [pscustomobject]@{
        Command = "milvushnswprq"
        Label = "HNSW_PRQ"
        Variant = "low"
        Parameters = @{
            m = 12
            ef_construction = 96
            ef_search = 112
            nbits = 8
            nrq = 2
            refine = $true
            refine_type = "FP32"
            refine_k = 1.0
        }
    },
    [pscustomobject]@{
        Command = "milvushnswprq"
        Label = "HNSW_PRQ"
        Variant = "high"
        Parameters = @{
            m = 24
            ef_construction = 192
            ef_search = 192
            nbits = 8
            nrq = 2
            refine = $true
            refine_type = "FP32"
            refine_k = 1.0
        }
    },
    [pscustomobject]@{
        Command = "milvushnswprq"
        Label = "HNSW_PRQ"
        Variant = "very-high"
        Parameters = @{
            m = 32
            ef_construction = 256
            ef_search = 256
            nbits = 8
            nrq = 2
            refine = $true
            refine_type = "FP32"
            refine_k = 1.0
        }
    },
    [pscustomobject]@{
        Command = "milvusivfflat"
        Label = "IVF_FLAT"
        Variant = "very-low"
        Parameters = @{
            nlist = 64
            nprobe = 8
        }
    },
    [pscustomobject]@{
        Command = "milvusivfflat"
        Label = "IVF_FLAT"
        Variant = "low"
        Parameters = @{
            nlist = 96
            nprobe = 12
        }
    },
    [pscustomobject]@{
        Command = "milvusivfflat"
        Label = "IVF_FLAT"
        Variant = "high"
        Parameters = @{
            nlist = 192
            nprobe = 24
        }
    },
    [pscustomobject]@{
        Command = "milvusivfflat"
        Label = "IVF_FLAT"
        Variant = "very-high"
        Parameters = @{
            nlist = 256
            nprobe = 32
        }
    },
    [pscustomobject]@{
        Command = "milvusivfsq8"
        Label = "IVF_SQ8"
        Variant = "very-low"
        Parameters = @{
            nlist = 64
            nprobe = 8
        }
    },
    [pscustomobject]@{
        Command = "milvusivfsq8"
        Label = "IVF_SQ8"
        Variant = "low"
        Parameters = @{
            nlist = 96
            nprobe = 12
        }
    },
    [pscustomobject]@{
        Command = "milvusivfsq8"
        Label = "IVF_SQ8"
        Variant = "high"
        Parameters = @{
            nlist = 192
            nprobe = 24
        }
    },
    [pscustomobject]@{
        Command = "milvusivfsq8"
        Label = "IVF_SQ8"
        Variant = "very-high"
        Parameters = @{
            nlist = 256
            nprobe = 32
        }
    }
)

function Format-ParameterSet {
    param([hashtable]$Parameters)

    $formatted = $Parameters.GetEnumerator() |
        Sort-Object Name |
        ForEach-Object { "$($_.Name)=$($_.Value)" }
    return $formatted -join ", "
}

function Wait-BenchmarkJob {
    param(
        [Parameter(Mandatory = $true)]
        [string]$JobId,
        [Parameter(Mandatory = $true)]
        [string]$Label
    )

    $lastPhase = ""
    $lastReport = [datetime]::MinValue
    do {
        Start-Sleep -Seconds 5
        $job = Invoke-RestMethod `
            -Uri "$ApiBase/api/benchmark-jobs/$JobId" `
            -Method Get
        $now = Get-Date
        if (($job.phase -ne $lastPhase) -or (($now - $lastReport).TotalSeconds -ge 30)) {
            Write-Host (
                "[{0:HH:mm:ss}] {1}: status={2}, phase={3}, run={4}/{5}, elapsed={6}s" -f
                $now,
                $Label,
                $job.status,
                $job.phase,
                $job.completed_runs,
                $job.total_runs,
                $job.elapsed_seconds
            )
            $lastPhase = $job.phase
            $lastReport = $now
        }
    } while ($job.status -notin $terminalStatuses)

    return $job
}

Write-Host ""
Write-Host "Representative CPU index parameter sweep"
Write-Host "========================================"
Write-Host "TopK=100, concurrency=1, concurrent duration=30s"
Write-Host "Default configurations are reused as the middle baselines."
Write-Host "Each tunable index: 4 new configurations + 1 existing default = 5."
Write-Host "AUTOINDEX and FLAT are omitted because they have no tunable parameters."
Write-Host ""

for ($index = 0; $index -lt $experiments.Count; $index += 1) {
    $experiment = $experiments[$index]
    Write-Host (
        "{0,2}. {1,-10} {2,-4}  {3}" -f
        ($index + 1),
        $experiment.Label,
        $experiment.Variant,
        (Format-ParameterSet $experiment.Parameters)
    )
}

$totalRuns = $experiments.Count * $Repetitions
Write-Host ""
Write-Host "Planned benchmark runs: $totalRuns (strictly serial)"

if ($DryRun) {
    Write-Host "Dry run only; no benchmark was submitted."
    exit 0
}

if ($RequireConfirmation) {
    $answer = Read-Host "Type RUN to start"
    if ($answer -cne "RUN") {
        Write-Host "Cancelled; no benchmark was submitted."
        exit 0
    }
}

$health = Invoke-RestMethod -Uri "$ApiBase/api/health" -Method Get
if ($health.active_job_id) {
    throw "Another benchmark is active: $($health.active_job_id). Wait for it to finish."
}

$resultsDirectory = [System.IO.Path]::GetFullPath(
    (Join-Path $PSScriptRoot "..\..\results\vectordbbench")
)
New-Item -ItemType Directory -Path $resultsDirectory -Force | Out-Null
$transcriptPath = Join-Path $resultsDirectory (
    "representative-index-sweep-{0}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss")
)
Start-Transcript -Path $transcriptPath | Out-Null

$commonParameters = @{
    uri = $MilvusUri
    num_shards = 1
    replica_number = 1
    case_type = "Performance1536D50K"
    drop_old = $true
    load = $true
    load_concurrency = 4
    search_serial = $true
    search_concurrent = $true
    k = 100
    concurrency_duration = 30
    num_concurrency = @(1)
    concurrency_timeout = 3600
    db_label = $DbLabel
}
$completed = @()

try {
    foreach ($experiment in $experiments) {
        # Recheck before every submission. This prevents this sweep from
        # overlapping with a benchmark started from the frontend.
        $health = Invoke-RestMethod -Uri "$ApiBase/api/health" -Method Get
        if ($health.active_job_id) {
            throw "Unexpected active benchmark before submission: $($health.active_job_id)"
        }

        $indexMatrix = @{}
        foreach ($entry in $experiment.Parameters.GetEnumerator()) {
            $indexMatrix[$entry.Name] = @($entry.Value)
        }
        $payload = @{
            command = $experiment.Command
            parameters = $commonParameters
            index_matrix = $indexMatrix
            repetitions = $Repetitions
        } | ConvertTo-Json -Depth 10

        $displayLabel = "$($experiment.Label) [$($experiment.Variant)]"
        Write-Host ""
        Write-Host "Starting $displayLabel"
        $job = Invoke-RestMethod `
            -Uri "$ApiBase/api/benchmark-jobs" `
            -Method Post `
            -ContentType "application/json; charset=utf-8" `
            -Body $payload
        Write-Host "Job: $($job.job_id)"

        $finishedJob = Wait-BenchmarkJob `
            -JobId $job.job_id `
            -Label $displayLabel
        if ($finishedJob.status -ne "succeeded") {
            throw (
                "{0} ended with status={1}: {2}" -f
                $displayLabel,
                $finishedJob.status,
                $finishedJob.error
            )
        }
        $completed += [pscustomobject]@{
            Index = $experiment.Label
            Variant = $experiment.Variant
            JobId = $finishedJob.job_id
            ElapsedSeconds = $finishedJob.elapsed_seconds
            RunIds = ($finishedJob.result_run_ids -join ",")
        }
    }

    Write-Host ""
    Write-Host "All representative configurations completed successfully."
    $completed | Format-Table -AutoSize
    Write-Host "Log: $transcriptPath"
}
finally {
    Stop-Transcript | Out-Null
}
