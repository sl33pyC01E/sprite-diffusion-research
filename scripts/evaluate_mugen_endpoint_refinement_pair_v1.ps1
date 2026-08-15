param(
    [Parameter(Mandatory = $true)]
    [int]$WaitForProcessId
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
$manifest = Join-Path $root `
    'data\processed\mugen-six-action-dense-coverage-all-scales-motion-v1\training-manifest.json'
$reports = Join-Path $root 'data\index\reports'
$experiments = Join-Path $root 'data\experiments'
$inference = Join-Path $root 'data\inference'

if (Get-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue) {
    Wait-Process -Id $WaitForProcessId
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
        Start-Sleep -Seconds 15
    }
}

foreach ($variant in @('control', 'action')) {
    $runName = "mugen-six-action-dense-latent-motion-endpoint-$variant-from-step10000-v1-step3000"
    $run = Join-Path $experiments $runName
    if (-not (Test-Path -LiteralPath (Join-Path $run 'training-report.json') -PathType Leaf)) {
        throw "$variant refinement did not publish a complete training report"
    }
    $checkpoint = Join-Path $run 'checkpoint-ema.pt'
    $checkpointSha256 = (
        Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    $outputName = "$runName-gate-eval-v1"
    $output = Join-Path $inference $outputName
    $stdout = Join-Path $reports "$outputName.out.log"
    $stderr = Join-Path $reports "$outputName.err.log"
    foreach ($path in @($output, $stdout, $stderr)) {
        if (Test-Path -LiteralPath $path) {
            throw "Refusing to replace refinement evaluation artifact: $path"
        }
    }
    Wait-GpuIdle
    $arguments = @(
        (Join-Path $root 'scripts\evaluate_mugen_latent_motion_v1.py'),
        '--checkpoint', $checkpoint,
        '--expected-sha256', $checkpointSha256,
        '--output-name', $outputName,
        '--manifest', $manifest
    )
    $process = Start-Process -FilePath $python -ArgumentList $arguments `
        -WorkingDirectory $root -WindowStyle Hidden -Wait -PassThru `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    if ($process.ExitCode -ne 0) {
        throw "$variant refinement evaluation exited with code $($process.ExitCode)"
    }
    if (-not (Test-Path -LiteralPath (Join-Path $output 'evaluation-report.json') -PathType Leaf)) {
        throw "$variant refinement evaluation did not publish its report"
    }
}
