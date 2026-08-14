param(
    [Parameter(Mandatory = $true)]
    [int]$WaitForCaptionProcessId,
    [Parameter(Mandatory = $true)]
    [int]$WaitForCoverageCaptionJoinId,
    [Parameter(Mandatory = $true)]
    [int]$WaitForCodecProcessId
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
$venvPython = Join-Path $root '.venv\Scripts\python.exe'
$processed = Join-Path $root 'data\processed'
$reports = Join-Path $root 'data\index\reports'
$experiments = Join-Path $root 'data\experiments'
$stillPlan = Join-Path $processed 'mugen-six-action-broad-still-plan-v1.json'
$latents = Join-Path $processed 'mugen-six-action-broad-rgba-latents-2x-v1\manifest.json'
$coverageCaptioned = Join-Path $processed 'mugen-six-action-dense-coverage-all-scales-captioned-v1.json'
$textBase = Join-Path $processed 'mugen-six-action-broad-sd14-clip-token-states-v1'
$motionArtifactsBase = Join-Path $processed 'mugen-six-action-dense-coverage-all-scales-motion-v1'
$stillBase = Join-Path $experiments 'mugen-six-action-still-dit-scratch-v1-step50000'
$motionBase = Join-Path $experiments 'mugen-six-action-dense-latent-motion-scratch-v1-step50000'
$clipModel = Join-Path $root 'data\models\stable-diffusion-v1-4-eb7ecef2ce03-training-components'
$clipIndexSha256 = '6c02b65f1d657f8db316c4976248b0ca6d2406b3396025e801b45c3ef6a91b47'

foreach ($processId in @(
    $WaitForCaptionProcessId,
    $WaitForCoverageCaptionJoinId,
    $WaitForCodecProcessId
)) {
    if (Get-Process -Id $processId -ErrorAction SilentlyContinue) {
        Wait-Process -Id $processId
    }
}
foreach ($input in @(
    $stillPlan,
    $latents,
    $coverageCaptioned,
    (Join-Path $clipModel 'source-index.json')
)) {
    if (-not (Test-Path -LiteralPath $input)) {
        throw "Model-pipeline input is incomplete: $input"
    }
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
            $checkpoints = Get-ChildItem -LiteralPath $latestExisting -Filter 'training-step-*.pt' |
                Sort-Object Name
            if (-not $checkpoints) {
                throw "Interrupted training has no resumable checkpoint: $latestExisting"
            }
            return [pscustomobject]@{
                Complete = $false
                Output = $candidate
                Resume = $checkpoints[-1].FullName
            }
        }
        if (Test-Path -LiteralPath (Join-Path $candidate 'training-report.json')) {
            return [pscustomobject]@{ Complete = $true; Output = $candidate; Resume = $null }
        }
        $latestExisting = $candidate
    }
    throw "No immutable continuation slot remains for $Base"
}

function Assert-GpuIdle {
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
}

function Invoke-LoggedProcess {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][object[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$LogStem
    )
    $out = Join-Path $reports "$LogStem.out.log"
    $err = Join-Path $reports "$LogStem.err.log"
    foreach ($log in @($out, $err)) {
        if (Test-Path -LiteralPath $log) {
            throw "Refusing to replace model-pipeline log: $log"
        }
    }
    $process = Start-Process -FilePath $Executable -ArgumentList $Arguments `
        -WorkingDirectory $root -WindowStyle Hidden -Wait -PassThru `
        -RedirectStandardOutput $out -RedirectStandardError $err
    if ($process.ExitCode -ne 0) {
        throw "$LogStem exited with code $($process.ExitCode)"
    }
}

Assert-GpuIdle

$textOutput = Resolve-ImmutableArtifactDirectory -Base $textBase -CompletionFile 'manifest.json'
$textManifest = Join-Path $textOutput 'manifest.json'
if (-not (Test-Path -LiteralPath $textManifest)) {
    Invoke-LoggedProcess -Executable $python -LogStem (Split-Path $textOutput -Leaf) -Arguments @(
        (Join-Path $root 'scripts\export_mugen_canonical_still_clip_states_v4.py'),
        '--plan', $stillPlan,
        '--model', $clipModel,
        '--expected-model-index-sha256', $clipIndexSha256,
        '--output', $textOutput,
        '--batch-size', '64',
        '--device', 'cuda'
    )
}

$motionArtifacts = Resolve-ImmutableArtifactDirectory `
    -Base $motionArtifactsBase -CompletionFile 'training-manifest.json'
$motionManifest = Join-Path $motionArtifacts 'training-manifest.json'
if (-not (Test-Path -LiteralPath $motionManifest)) {
    Invoke-LoggedProcess -Executable $venvPython `
        -LogStem (Split-Path $motionArtifacts -Leaf) -Arguments @(
            (Join-Path $root 'scripts\build_mugen_dense_motion_plan_v1.py'),
            $coverageCaptioned,
            $latents,
            $motionArtifacts
        )
}

$still = Resolve-TrainingLaunch -Base $stillBase
if (-not $still.Complete) {
    Assert-GpuIdle
    $arguments = @(
        (Join-Path $root 'scripts\train_mugen_latent_still_dit_v1.py'),
        '--plan', $stillPlan,
        '--latents', $latents,
        '--text', $textManifest,
        '--profile', 'corpus50000',
        '--output', $still.Output
    )
    if ($null -ne $still.Resume) {
        $resumeSha256 = (Get-FileHash -LiteralPath $still.Resume -Algorithm SHA256).Hash.ToLowerInvariant()
        $arguments += @('--resume-checkpoint', $still.Resume, '--expected-resume-sha256', $resumeSha256)
    }
    Invoke-LoggedProcess -Executable $python -Arguments $arguments `
        -LogStem (Split-Path $still.Output -Leaf)
}
if (-not (Test-Path -LiteralPath (Join-Path $still.Output 'training-report.json'))) {
    throw 'Still DiT did not publish its final report'
}

$motion = Resolve-TrainingLaunch -Base $motionBase
if (-not $motion.Complete) {
    Assert-GpuIdle
    $arguments = @(
        (Join-Path $root 'scripts\run_mugen_latent_motion_train_v1.py'),
        '--profile', 'corpus50000',
        '--manifest', $motionManifest,
        '--output', $motion.Output
    )
    if ($null -ne $motion.Resume) {
        $resumeSha256 = (Get-FileHash -LiteralPath $motion.Resume -Algorithm SHA256).Hash.ToLowerInvariant()
        $arguments += @('--resume-checkpoint', $motion.Resume, '--expected-resume-sha256', $resumeSha256)
    }
    Invoke-LoggedProcess -Executable $python -Arguments $arguments `
        -LogStem (Split-Path $motion.Output -Leaf)
}
if (-not (Test-Path -LiteralPath (Join-Path $motion.Output 'training-report.json'))) {
    throw 'Motion DiT did not publish its final report'
}
