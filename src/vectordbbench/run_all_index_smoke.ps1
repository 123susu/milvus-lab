param(
    [string]$ApiBase = "http://127.0.0.1:8765",
    [string]$MilvusUri = "http://localhost:19530",
    [string]$DbLabel = "local-cluster-8c10_5g-2_6_21"
)

$ErrorActionPreference = "Stop"
$terminalStatuses = @("succeeded", "failed", "cancelled")
$attemptedCommands = [System.Collections.Generic.HashSet[string]]::new()

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
        $job = Invoke-RestMethod -Uri "$ApiBase/api/benchmark-jobs/$JobId" -Method Get
        $now = Get-Date
        if (($job.phase -ne $lastPhase) -or (($now - $lastReport).TotalSeconds -ge 30)) {
            Write-Output (
                "{0:o} PROGRESS {1} status={2} phase={3} run={4}/{5} elapsed={6}s" -f
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

    Write-Output (
        "{0:o} DONE {1} status={2} elapsed={3}s run_id={4} error={5}" -f
        (Get-Date),
        $Label,
        $job.status,
        $job.elapsed_seconds,
        $job.result_run_id,
        $job.error
    )
}

$health = Invoke-RestMethod -Uri "$ApiBase/api/health" -Method Get
if ($health.active_job_id) {
    $activeJob = Invoke-RestMethod `
        -Uri "$ApiBase/api/benchmark-jobs/$($health.active_job_id)" `
        -Method Get
    [void]$attemptedCommands.Add($activeJob.parameters.command)
    Write-Output (
        "{0:o} ADOPT {1} command={2}" -f
        (Get-Date),
        $activeJob.job_id,
        $activeJob.parameters.command
    )
    Wait-BenchmarkJob `
        -JobId $activeJob.job_id `
        -Label $activeJob.parameters.command
}

$profiles = Invoke-RestMethod -Uri "$ApiBase/api/benchmark-profiles" -Method Get
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

foreach ($profile in $profiles) {
    if ($attemptedCommands.Contains($profile.command)) {
        Write-Output (
            "{0:o} SKIP_ALREADY_ATTEMPTED {1}" -f
            (Get-Date),
            $profile.label
        )
        continue
    }

    $indexMatrix = @{}
    foreach ($definition in $profile.parameters) {
        $indexMatrix[$definition.name] = @($definition.default)
    }
    $body = @{
        command = $profile.command
        parameters = $commonParameters
        index_matrix = $indexMatrix
        repetitions = 1
    } | ConvertTo-Json -Depth 10

    Write-Output (
        "{0:o} START {1} command={2}" -f
        (Get-Date),
        $profile.label,
        $profile.command
    )
    try {
        $job = Invoke-RestMethod `
            -Uri "$ApiBase/api/benchmark-jobs" `
            -Method Post `
            -ContentType "application/json; charset=utf-8" `
            -Body $body
        [void]$attemptedCommands.Add($profile.command)
        Write-Output (
            "{0:o} JOB {1} {2}" -f
            (Get-Date),
            $profile.label,
            $job.job_id
        )
        Wait-BenchmarkJob -JobId $job.job_id -Label $profile.label
    }
    catch {
        Write-Output (
            "{0:o} FAILED_TO_RUN {1} error={2}" -f
            (Get-Date),
            $profile.label,
            $_.Exception.Message
        )
    }
}

Write-Output ("{0:o} ALL_INDEX_SMOKE_FINISHED" -f (Get-Date))
