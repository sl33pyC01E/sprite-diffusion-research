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
$captionOut = Join-Path $reports 'mugen-six-action-broad-captions-v1.out.log'
$captionErr = Join-Path $reports 'mugen-six-action-broad-captions-v1.err.log'

if (Get-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue) {
    Wait-Process -Id $WaitForProcessId
}
if (-not (Test-Path -LiteralPath $captionInputs)) {
    throw "Caption inputs are incomplete: $captionInputs"
}

$startedService = $false
try {
    if (-not (Test-Path -LiteralPath $captionManifest)) {
        if ((Test-Path -LiteralPath $captionOut) -or (Test-Path -LiteralPath $captionErr)) {
            throw 'Refusing to replace caption logs; inspect and resume with new log paths'
        }
        $serviceState = (& ssh -o BatchMode=yes -o ConnectTimeout=8 spark `
            'systemctl --user is-active qwen-122b.service || true').Trim()
        if ($serviceState -ne 'active') {
            $compute = (& ssh -o BatchMode=yes -o ConnectTimeout=8 spark `
                'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true') -join "`n"
            if ($compute.Trim().Length -ne 0) {
                throw "Spark GPU is not idle:`n$compute"
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
            -RedirectStandardOutput $captionOut -RedirectStandardError $captionErr
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
foreach ($output in @($broadCaptioned, $denseCaptioned, $stillPlan)) {
    if (Test-Path -LiteralPath $output) {
        throw "Refusing to replace captioned corpus artifact: $output"
    }
}
& $python (Join-Path $root 'scripts\build_mugen_dense_autoencoder_bridge_v1.py') `
    $broad $broadCaptioned --caption-manifest $captionManifest
if ($LASTEXITCODE -ne 0) { throw 'Broad caption join failed' }
& $python (Join-Path $root 'scripts\build_mugen_dense_autoencoder_bridge_v1.py') `
    $dense $denseCaptioned --caption-manifest $captionManifest
if ($LASTEXITCODE -ne 0) { throw 'Dense caption join failed' }
& $python (Join-Path $root 'scripts\build_mugen_dense_still_plan_v1.py') `
    $broadCaptioned $stillPlan
if ($LASTEXITCODE -ne 0) { throw 'Broad still plan failed' }
