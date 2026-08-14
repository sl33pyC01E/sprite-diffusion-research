param(
    [Parameter(Mandatory = $true)]
    [int]$WaitForProcessId
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
$processed = Join-Path $root 'data\processed'
$reports = Join-Path $root 'data\index\reports'
$materializations = @(
    (Join-Path $processed 'mugen-iidx-jus-chibi-schema-core-b128-f8-v2'),
    (Join-Path $processed 'mugen-anime-ascension-schema-core-b128-f8-v2'),
    (Join-Path $processed 'mugen-mffa-schema-core-b128-f8-v2'),
    (Join-Path $processed 'mugen-anime-all-stars3-schema-core-b128-f8-v2')
)
$quality = Join-Path $reports 'mugen-six-action-coverage-all-scales-quality-audit-v1.json'
$dense = Join-Path $processed 'mugen-six-action-dense-coverage-all-scales-v1.json'
$bridge = Join-Path $processed 'mugen-six-action-dense-coverage-all-scales-autoencoder-v1.json'

if (Get-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue) {
    Wait-Process -Id $WaitForProcessId
}
foreach ($materialization in $materializations) {
    if (-not (Test-Path -LiteralPath (Join-Path $materialization 'materialization.json'))) {
        throw "Materialization is incomplete: $materialization"
    }
}
foreach ($output in @($quality, $dense, $bridge)) {
    if (Test-Path -LiteralPath $output) {
        throw "Refusing to replace coverage-tier artifact: $output"
    }
}

$auditArguments = @((Join-Path $root 'scripts\audit_mugen_stream_quality_v1.py'))
foreach ($materialization in $materializations) {
    $auditArguments += @('--materialization', $materialization)
}
$auditArguments += @(
    '--minimum-view-scale',
    '0.000001',
    '--output',
    $quality
)
& $python @auditArguments
if ($LASTEXITCODE -ne 0) { throw 'Coverage-tier quality audit failed' }

$manifestArguments = @((Join-Path $root 'scripts\build_mugen_dense_manifest_v1.py'))
foreach ($materialization in $materializations) {
    $manifestArguments += @('--materialization', $materialization)
}
$manifestArguments += @(
    '--quality-audit',
    $quality,
    '--output',
    $dense,
    '--tier',
    'dense'
)
& $python @manifestArguments
if ($LASTEXITCODE -ne 0) { throw 'Coverage-tier dense manifest failed' }

& $python (Join-Path $root 'scripts\build_mugen_dense_autoencoder_bridge_v1.py') `
    $dense $bridge
if ($LASTEXITCODE -ne 0) { throw 'Coverage-tier autoencoder bridge failed' }
