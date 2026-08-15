param(
    [Parameter(Mandatory = $true)]
    [int]$WaitForProcessId
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
$manifest = Join-Path $root `
    'data\processed\mugen-six-action-dense-coverage-all-scales-motion-v1\training-manifest.json'
$parent = Join-Path $root `
    'data\experiments\mugen-six-action-dense-latent-motion-scratch-v1-step50000\training-step-0010000.pt'
$parentSha256 = '8dd03f0757314c7977bec8546e255923ffc40a017c327ea1f013fff8e942de46'
$outputName = 'mugen-six-action-dense-latent-motion-endpoint-pixel-action-from-step10000-v1-step3000'
$output = Join-Path (Join-Path $root 'data\experiments') $outputName
$reports = Join-Path $root 'data\index\reports'
$stdout = Join-Path $reports "$outputName.out.log"
$stderr = Join-Path $reports "$outputName.err.log"

if (Get-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue) {
    Wait-Process -Id $WaitForProcessId
}
foreach ($variant in @('control', 'action')) {
    $evaluation = Join-Path (Join-Path $root 'data\inference') `
        "mugen-six-action-dense-latent-motion-endpoint-$variant-from-step10000-v1-step3000-gate-eval-v1\evaluation-report.json"
    if (-not (Test-Path -LiteralPath $evaluation -PathType Leaf)) {
        throw "$variant refinement evaluation is incomplete"
    }
}
foreach ($path in @($output, $stdout, $stderr)) {
    if (Test-Path -LiteralPath $path) {
        throw "Refusing to replace pixel-action artifact: $path"
    }
}
$actualParentSha256 = (
    Get-FileHash -LiteralPath $parent -Algorithm SHA256
).Hash.ToLowerInvariant()
if ($actualParentSha256 -ne $parentSha256) {
    throw 'Pixel-action parent checkpoint SHA-256 differs'
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
$arguments = @(
    (Join-Path $root 'scripts\run_mugen_latent_motion_refinement_v1.py'),
    '--profile', 'endpoint-pixel-action3000',
    '--manifest', $manifest,
    '--parent-checkpoint', $parent,
    '--expected-parent-sha256', $parentSha256,
    '--output', $output
)
$process = Start-Process -FilePath $python -ArgumentList $arguments `
    -WorkingDirectory $root -WindowStyle Hidden -Wait -PassThru `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr
if ($process.ExitCode -ne 0) {
    throw "Pixel-action refinement exited with code $($process.ExitCode)"
}
if (-not (Test-Path -LiteralPath (Join-Path $output 'training-report.json') -PathType Leaf)) {
    throw 'Pixel-action refinement did not publish its training report'
}
