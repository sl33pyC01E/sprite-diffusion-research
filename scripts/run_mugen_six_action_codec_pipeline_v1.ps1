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
$run = Join-Path $experiments 'mugen-six-action-rgba-autoencoder-2x-v1-20000'
$checkpoint = Join-Path $run 'training-step-0020000.pt'
$audit = Join-Path $reports 'mugen-six-action-rgba-autoencoder-2x-step20000-audit-v1'
$latents = Join-Path $processed 'mugen-six-action-broad-rgba-latents-2x-v1'

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
foreach ($output in @($run, $audit, $latents)) {
    if (Test-Path -LiteralPath $output) {
        throw "Refusing to replace codec-pipeline artifact: $output"
    }
}

$usedMemory = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
    Select-Object -First 1).Trim()
$memoryMiB = 0
if (-not [int]::TryParse($usedMemory, [ref]$memoryMiB)) {
    throw "Could not parse local GPU memory use: $usedMemory"
}
$compute = (& nvidia-smi --query-compute-apps=pid,process_name,used_memory `
    --format=csv,noheader) -join "`n"
if ($memoryMiB -ge 4096 -or $compute -match '(?i)python|torch|cuda') {
    throw "Local GPU is not idle (memory MiB=$memoryMiB):`n$compute"
}

$trainOut = Join-Path $reports 'mugen-six-action-rgba-autoencoder-2x-v1-20000.out.log'
$trainErr = Join-Path $reports 'mugen-six-action-rgba-autoencoder-2x-v1-20000.err.log'
$auditOut = Join-Path $reports 'mugen-six-action-rgba-autoencoder-2x-step20000-audit-v1.out.log'
$auditErr = Join-Path $reports 'mugen-six-action-rgba-autoencoder-2x-step20000-audit-v1.err.log'
$latentOut = Join-Path $reports 'mugen-six-action-broad-rgba-latents-2x-v1.out.log'
$latentErr = Join-Path $reports 'mugen-six-action-broad-rgba-latents-2x-v1.err.log'
foreach ($log in @($trainOut, $trainErr, $auditOut, $auditErr, $latentOut, $latentErr)) {
    if (Test-Path -LiteralPath $log) {
        throw "Refusing to replace codec-pipeline log: $log"
    }
}

$training = Start-Process -FilePath $python -ArgumentList @(
    (Join-Path $root 'scripts\run_mugen_sprite_autoencoder_2x_v1.py'),
    '--manifest',
    $manifest,
    '--profile',
    'corpus20000',
    '--output',
    $run
) -WorkingDirectory $root -WindowStyle Hidden -Wait -PassThru `
    -RedirectStandardOutput $trainOut -RedirectStandardError $trainErr
if ($training.ExitCode -ne 0) {
    throw "Corpus codec training exited with code $($training.ExitCode)"
}
if (-not (Test-Path -LiteralPath $checkpoint)) {
    throw 'Corpus codec training did not publish its final checkpoint'
}

$auditing = Start-Process -FilePath $python -ArgumentList @(
    (Join-Path $root 'scripts\audit_mugen_sprite_autoencoder_2x_v1.py'),
    '--manifest',
    $manifest,
    '--run',
    $run,
    '--output',
    $audit,
    '--step',
    '20000',
    '--maximum-frames',
    '64'
) -WorkingDirectory $root -WindowStyle Hidden -Wait -PassThru `
    -RedirectStandardOutput $auditOut -RedirectStandardError $auditErr
if ($auditing.ExitCode -ne 0) {
    throw "Corpus codec audit exited with code $($auditing.ExitCode)"
}
$auditReport = Get-Content -LiteralPath (Join-Path $audit 'audit-report.json') -Raw |
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
$encoding = Start-Process -FilePath $python -ArgumentList @(
    (Join-Path $root 'scripts\export_mugen_rgba_latents_2x_v1.py'),
    '--materialization',
    $manifest,
    '--checkpoint',
    $checkpoint,
    '--expected-checkpoint-sha256',
    $checkpointSha256,
    '--output',
    $latents,
    '--batch-sequences',
    '8',
    '--device',
    'cuda'
) -WorkingDirectory $root -WindowStyle Hidden -Wait -PassThru `
    -RedirectStandardOutput $latentOut -RedirectStandardError $latentErr
if ($encoding.ExitCode -ne 0) {
    throw "Broad latent export exited with code $($encoding.ExitCode)"
}
if (-not (Test-Path -LiteralPath (Join-Path $latents 'manifest.json'))) {
    throw 'Broad latent export did not publish a complete manifest'
}
