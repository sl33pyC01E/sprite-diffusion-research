param(
    [Parameter(Mandatory = $true)]
    [int]$WaitForPrimaryFinalizerId,
    [Parameter(Mandatory = $true)]
    [int]$WaitForCoverageFinalizerId
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
$processed = Join-Path $root 'data\processed'
$reports = Join-Path $root 'data\index\reports'
$experiments = Join-Path $root 'data\experiments'
$manifest = Join-Path $processed 'mugen-six-action-broad-autoencoder-v1.json'
$runBase = Join-Path $experiments 'mugen-six-action-rgba-autoencoder-2x-v1-20000'
$auditBase = Join-Path $reports 'mugen-six-action-rgba-autoencoder-2x-step20000-audit-v1'
$latentsBase = Join-Path $processed 'mugen-six-action-broad-rgba-latents-2x-v1'

foreach ($processId in @($WaitForPrimaryFinalizerId, $WaitForCoverageFinalizerId)) {
    if (Get-Process -Id $processId -ErrorAction SilentlyContinue) {
        Wait-Process -Id $processId
    }
}
if (-not (Test-Path -LiteralPath $manifest)) {
    throw "Broad codec manifest is incomplete: $manifest"
}
if (-not (Test-Path -LiteralPath $python)) {
    throw "Global PyTorch Python is absent: $python"
}
function Resolve-ImmutableArtifactDirectory {
    param(
        [Parameter(Mandatory = $true)][string]$Base,
        [Parameter(Mandatory = $true)][string]$CompletionFile
    )
    if (-not (Test-Path -LiteralPath $Base)) {
        return $Base
    }
    if (Test-Path -LiteralPath (Join-Path $Base $CompletionFile)) {
        return $Base
    }
    for ($version = 1; $version -le 99; $version++) {
        $candidate = "$Base-retry-v$version"
        if (-not (Test-Path -LiteralPath $candidate)) {
            return $candidate
        }
        if (Test-Path -LiteralPath (Join-Path $candidate $CompletionFile)) {
            return $candidate
        }
    }
    throw "No immutable retry slot remains for $Base"
}

function Resolve-TrainingLaunch {
    param([Parameter(Mandatory = $true)][string]$Base)
    $chain = @($Base)
    for ($version = 1; $version -le 99; $version++) {
        $chain += "$Base-continuation-v$version"
    }
    $latestExisting = $null
    foreach ($candidate in $chain) {
        if (-not (Test-Path -LiteralPath $candidate)) {
            if ($null -eq $latestExisting) {
                return [pscustomobject]@{ Complete = $false; Output = $candidate; Resume = $null }
            }
            $checkpoints = Get-ChildItem -LiteralPath $latestExisting `
                -Filter 'training-step-*.pt' | Sort-Object Name
            return [pscustomobject]@{
                Complete = $false
                Output = $candidate
                Resume = if ($checkpoints) { $checkpoints[-1].FullName } else { $null }
            }
        }
        if (Test-Path -LiteralPath (Join-Path $candidate 'training-report.json')) {
            return [pscustomobject]@{ Complete = $true; Output = $candidate; Resume = $null }
        }
        $latestExisting = $candidate
    }
    throw "No immutable continuation slot remains for $Base"
}

function Resolve-LogPair {
    param([Parameter(Mandatory = $true)][string]$BaseStem)
    for ($version = 0; $version -le 99; $version++) {
        $stem = if ($version -eq 0) { $BaseStem } else { "$BaseStem-retry-v$version" }
        $out = Join-Path $reports "$stem.out.log"
        $err = Join-Path $reports "$stem.err.log"
        if (-not (Test-Path -LiteralPath $out) -and -not (Test-Path -LiteralPath $err)) {
            return [pscustomobject]@{ Out = $out; Err = $err }
        }
    }
    throw "No immutable retry log slot remains for $BaseStem"
}

function Wait-GpuIdle {
    while ($true) {
        $usedMemory = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
            Select-Object -First 1).Trim()
        $memoryMiB = 0
        if (-not [int]::TryParse($usedMemory, [ref]$memoryMiB)) {
            throw "Could not parse local GPU memory use: $usedMemory"
        }
        $compute = (& nvidia-smi --query-compute-apps=pid,process_name,used_memory `
            --format=csv,noheader) -join "`n"
        if ($memoryMiB -lt 4096 -and $compute -notmatch '(?i)python|torch|cuda') {
            return
        }
        Write-Output "Waiting for local GPU idleness (memory MiB=$memoryMiB)."
        Start-Sleep -Seconds 30
    }
}

Wait-GpuIdle

$trainingLaunch = Resolve-TrainingLaunch -Base $runBase
$run = $trainingLaunch.Output
$checkpoint = Join-Path $run 'training-step-0020000.pt'
$audit = Resolve-ImmutableArtifactDirectory -Base $auditBase -CompletionFile 'audit-report.json'
$latents = $latentsBase
$trainStem = Split-Path $run -Leaf
$auditStem = Split-Path $audit -Leaf
$latentStem = Split-Path $latents -Leaf
$trainOut = Join-Path $reports "$trainStem.out.log"
$trainErr = Join-Path $reports "$trainStem.err.log"
$auditOut = Join-Path $reports "$auditStem.out.log"
$auditErr = Join-Path $reports "$auditStem.err.log"

$trainingArguments = @(
    (Join-Path $root 'scripts\run_mugen_sprite_autoencoder_2x_v1.py'),
    '--manifest', $manifest,
    '--profile', 'corpus20000',
    '--output', $run
)
if ($null -ne $trainingLaunch.Resume) {
    $resumeSha256 = (
        Get-FileHash -LiteralPath $trainingLaunch.Resume -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    $trainingArguments += @(
        '--resume-checkpoint', $trainingLaunch.Resume,
        '--expected-resume-sha256', $resumeSha256
    )
}
if (-not $trainingLaunch.Complete) {
    foreach ($log in @($trainOut, $trainErr)) {
        if (Test-Path -LiteralPath $log) {
            throw "Refusing to replace codec-training log: $log"
        }
    }
    $training = Start-Process -FilePath $python -ArgumentList $trainingArguments `
        -WorkingDirectory $root -WindowStyle Hidden -Wait -PassThru `
        -RedirectStandardOutput $trainOut -RedirectStandardError $trainErr
    if ($training.ExitCode -ne 0) {
        throw "Corpus codec training exited with code $($training.ExitCode)"
    }
}
if (-not (Test-Path -LiteralPath $checkpoint)) {
    throw 'Corpus codec training did not publish its final checkpoint'
}

$auditReportPath = Join-Path $audit 'audit-report.json'
if (-not (Test-Path -LiteralPath $auditReportPath)) {
    foreach ($log in @($auditOut, $auditErr)) {
        if (Test-Path -LiteralPath $log) {
            throw "Refusing to replace codec-audit log: $log"
        }
    }
    $auditing = Start-Process -FilePath $python -ArgumentList @(
        (Join-Path $root 'scripts\audit_mugen_sprite_autoencoder_2x_v1.py'),
        '--manifest', $manifest,
        '--run', $run,
        '--output', $audit,
        '--step', '20000',
        '--maximum-frames', '512'
    ) -WorkingDirectory $root -WindowStyle Hidden -Wait -PassThru `
        -RedirectStandardOutput $auditOut -RedirectStandardError $auditErr
    if ($auditing.ExitCode -ne 0) {
        throw "Corpus codec audit exited with code $($auditing.ExitCode)"
    }
}
$auditReport = Get-Content -LiteralPath $auditReportPath -Raw |
    ConvertFrom-Json
$metrics = $auditReport.aggregate_metrics
if (
    [double]$metrics.premultiplied_rgba_mae -gt 0.005 -or
    [double]$metrics.visible_rgb_mae -gt 0.04 -or
    [double]$metrics.alpha_iou_127 -lt 0.995
) {
    throw (
        'Corpus codec failed held-out quality gates: ' +
        "PM-RGBA=$($metrics.premultiplied_rgba_mae), " +
        "visible-RGB=$($metrics.visible_rgb_mae), alpha-IoU=$($metrics.alpha_iou_127)"
    )
}

$checkpointSha256 = (Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
$latentManifest = Join-Path $latents 'manifest.json'
if (-not (Test-Path -LiteralPath $latentManifest)) {
    $latentLogs = Resolve-LogPair -BaseStem $latentStem
    $encoding = Start-Process -FilePath $python -ArgumentList @(
        (Join-Path $root 'scripts\export_mugen_rgba_latents_2x_v1.py'),
        '--materialization', $manifest,
        '--checkpoint', $checkpoint,
        '--expected-checkpoint-sha256', $checkpointSha256,
        '--output', $latents,
        '--batch-sequences', '8',
        '--device', 'cuda'
    ) -WorkingDirectory $root -WindowStyle Hidden -Wait -PassThru `
        -RedirectStandardOutput $latentLogs.Out -RedirectStandardError $latentLogs.Err
    if ($encoding.ExitCode -ne 0) {
        throw "Broad latent export exited with code $($encoding.ExitCode)"
    }
}
if (-not (Test-Path -LiteralPath $latentManifest)) {
    throw 'Broad latent export did not publish a complete manifest'
}
