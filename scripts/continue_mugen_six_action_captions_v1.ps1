param(
    [Parameter(Mandatory = $true)]
    [int]$WaitForProcessId
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
$processed = Join-Path $root 'data\processed'
$reports = Join-Path $root 'data\index\reports'
$captionInputs = Join-Path $processed 'mugen-six-action-broad-caption-inputs-v1\manifest.json'
$captions = Join-Path $processed 'mugen-six-action-broad-captions-v1'
$captionManifest = Join-Path $captions 'manifest.json'
$broad = Join-Path $processed 'mugen-six-action-broad-v1.json'
$dense = Join-Path $processed 'mugen-six-action-dense-v3.json'
$broadCaptioned = Join-Path $processed 'mugen-six-action-broad-captioned-v1.json'
$denseCaptioned = Join-Path $processed 'mugen-six-action-dense-captioned-v3.json'
$stillPlan = Join-Path $processed 'mugen-six-action-broad-still-plan-v1.json'

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

if (Get-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue) {
    Wait-Process -Id $WaitForProcessId
}
if (-not (Test-Path -LiteralPath $captionInputs)) {
    throw "Caption inputs are incomplete: $captionInputs"
}

$startedService = $false
try {
    if (-not (Test-Path -LiteralPath $captionManifest)) {
        $captionLogs = Resolve-LogPair -BaseStem 'mugen-six-action-broad-captions-v1'
        $serviceState = (& ssh -o BatchMode=yes -o ConnectTimeout=8 spark `
            'systemctl --user is-active qwen-122b.service || true').Trim()
        if ($serviceState -ne 'active') {
            $compute = (& ssh -o BatchMode=yes -o ConnectTimeout=8 spark `
                'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true') -join "`n"
            $blockingCompute = @()
            foreach ($line in ($compute -split "`r?`n")) {
                if (-not $line.Trim()) { continue }
                if ($line -notmatch '^\s*\d+\s*,\s*(.+)\s*,\s*(\d+)\s+MiB\s*$') {
                    $blockingCompute += $line
                    continue
                }
                $processName = $Matches[1].Trim()
                $memoryMiB = [int]$Matches[2]
                $idleStudioShell = (
                    $processName -match '/studio/unsloth_studio/' -and
                    $memoryMiB -le 1024
                )
                if (-not $idleStudioShell) {
                    $blockingCompute += $line
                }
            }
            if ($blockingCompute.Count -ne 0) {
                throw "Spark GPU has a competing model process:`n$($blockingCompute -join "`n")"
            }
            & ssh -o BatchMode=yes -o ConnectTimeout=8 spark `
                'systemctl --user start qwen-122b.service'
            if ($LASTEXITCODE -ne 0) {
                throw 'Could not start the existing Spark caption service'
            }
            $startedService = $true
        }
        $ready = $false
        for ($attempt = 0; $attempt -lt 60; $attempt++) {
            try {
                $response = Invoke-RestMethod -Uri 'http://spark:8080/v1/models' `
                    -Method Get -TimeoutSec 5
                if ($null -ne $response.data) {
                    $ready = $true
                    break
                }
            } catch {
                Start-Sleep -Seconds 3
            }
        }
        if (-not $ready) {
            throw 'Spark caption service did not become ready within three minutes'
        }
        $caption = Start-Process -FilePath $python -ArgumentList @(
            (Join-Path $root 'scripts\run_mugen_dense_spark_captions_v1.py'),
            $captionInputs,
            $captions,
            '--workers',
            '1'
        ) -WorkingDirectory $root -WindowStyle Hidden -Wait -PassThru `
            -RedirectStandardOutput $captionLogs.Out -RedirectStandardError $captionLogs.Err
        if ($caption.ExitCode -ne 0) {
            throw "Spark captioning exited with code $($caption.ExitCode)"
        }
    }
} finally {
    if ($startedService) {
        & ssh -o BatchMode=yes -o ConnectTimeout=8 spark `
            'systemctl --user stop qwen-122b.service'
    }
}

if (-not (Test-Path -LiteralPath $captionManifest)) {
    throw 'Spark captioning did not publish a complete manifest'
}
if (-not (Test-Path -LiteralPath $broadCaptioned)) {
    & $python (Join-Path $root 'scripts\build_mugen_dense_autoencoder_bridge_v1.py') `
        $broad $broadCaptioned --caption-manifest $captionManifest
    if ($LASTEXITCODE -ne 0) { throw 'Broad caption join failed' }
}
if (-not (Test-Path -LiteralPath $denseCaptioned)) {
    & $python (Join-Path $root 'scripts\build_mugen_dense_autoencoder_bridge_v1.py') `
        $dense $denseCaptioned --caption-manifest $captionManifest
    if ($LASTEXITCODE -ne 0) { throw 'Dense caption join failed' }
}
if (-not (Test-Path -LiteralPath $stillPlan)) {
    & $python (Join-Path $root 'scripts\build_mugen_dense_still_plan_v1.py') `
        $broadCaptioned $stillPlan
    if ($LASTEXITCODE -ne 0) { throw 'Broad still plan failed' }
}
