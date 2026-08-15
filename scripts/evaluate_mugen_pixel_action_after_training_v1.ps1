param(
    [Parameter(Mandatory = $true)]
    [int]$WaitForProcessId
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
$manifest = Join-Path $root `
    'data\processed\mugen-six-action-dense-coverage-all-scales-motion-v1\training-manifest.json'
$runName = 'mugen-six-action-dense-latent-motion-endpoint-pixel-action-from-step10000-v1-step3000'
$run = Join-Path (Join-Path $root 'data\experiments') $runName
$checkpoint = Join-Path $run 'checkpoint-ema.pt'
$trainingReport = Join-Path $run 'training-report.json'
$outputName = "$runName-gate-eval-v1"
$output = Join-Path (Join-Path $root 'data\inference') $outputName
$reports = Join-Path $root 'data\index\reports'
$stdout = Join-Path $reports "$outputName.out.log"
$stderr = Join-Path $reports "$outputName.err.log"

if (Get-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue) {
    Wait-Process -Id $WaitForProcessId
}
if (-not (Test-Path -LiteralPath $trainingReport -PathType Leaf)) {
    throw 'Pixel-action refinement did not publish a complete training report'
}
if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
    throw 'Pixel-action refinement did not publish its EMA checkpoint'
}
foreach ($path in @($output, $stdout, $stderr)) {
    if (Test-Path -LiteralPath $path) {
        throw "Refusing to replace pixel-action evaluation artifact: $path"
    }
}
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
        break
    }
    Start-Sleep -Seconds 15
}
$checkpointSha256 = (
    Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256
).Hash.ToLowerInvariant()
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
    throw "Pixel-action evaluation exited with code $($process.ExitCode)"
}
if (-not (Test-Path -LiteralPath (Join-Path $output 'evaluation-report.json') -PathType Leaf)) {
    throw 'Pixel-action evaluation did not publish its report'
}
