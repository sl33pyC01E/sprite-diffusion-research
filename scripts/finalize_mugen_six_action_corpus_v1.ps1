param(
    [Parameter(Mandatory = $true)]
    [int]$WaitForProcessId
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
$processed = Join-Path $root 'data\processed'
$reports = Join-Path $root 'data\index\reports'
$jus = Join-Path $processed 'mugen-iidx-jus-chibi-schema-core-b128-f8-v2'
$ascension = Join-Path $processed 'mugen-anime-ascension-schema-core-b128-f8-v2'
$mffa = Join-Path $processed 'mugen-mffa-schema-core-b128-f8-v2'
$allStars = Join-Path $processed 'mugen-anime-all-stars3-schema-core-b128-f8-v2'
$quality = Join-Path $reports 'mugen-six-action-full-quality-audit-v1.json'
$broad = Join-Path $processed 'mugen-six-action-broad-v1.json'
$dense = Join-Path $processed 'mugen-six-action-dense-v3.json'
$broadBridge = Join-Path $processed 'mugen-six-action-broad-autoencoder-v1.json'
$denseBridge = Join-Path $processed 'mugen-six-action-dense-autoencoder-v3.json'
$captionInputs = Join-Path $processed 'mugen-six-action-broad-caption-inputs-v1'

if (Get-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue) {
    Wait-Process -Id $WaitForProcessId
}
foreach ($materialization in @($jus, $ascension, $mffa, $allStars)) {
    if (-not (Test-Path -LiteralPath (Join-Path $materialization 'materialization.json'))) {
        throw "Materialization is incomplete: $materialization"
    }
}
if (-not (Test-Path -LiteralPath $quality)) {
    & $python (Join-Path $root 'scripts\audit_mugen_stream_quality_v1.py') `
        --materialization $jus `
        --materialization $ascension `
        --materialization $mffa `
        --materialization $allStars `
        --output $quality
    if ($LASTEXITCODE -ne 0) { throw 'Full MUGEN quality audit failed' }
}

foreach ($tier in @('broad', 'dense')) {
    $manifest = if ($tier -eq 'broad') { $broad } else { $dense }
    if (-not (Test-Path -LiteralPath $manifest)) {
        & $python (Join-Path $root 'scripts\build_mugen_dense_manifest_v1.py') `
            --materialization $jus `
            --materialization $ascension `
            --materialization $mffa `
            --materialization $allStars `
            --quality-audit $quality `
            --output $manifest `
            --tier $tier
        if ($LASTEXITCODE -ne 0) { throw "MUGEN $tier manifest failed" }
    }
}

if (-not (Test-Path -LiteralPath $broadBridge)) {
    & $python (Join-Path $root 'scripts\build_mugen_dense_autoencoder_bridge_v1.py') `
        $broad $broadBridge
    if ($LASTEXITCODE -ne 0) { throw 'Broad MUGEN autoencoder bridge failed' }
}
if (-not (Test-Path -LiteralPath $denseBridge)) {
    & $python (Join-Path $root 'scripts\build_mugen_dense_autoencoder_bridge_v1.py') `
        $dense $denseBridge
    if ($LASTEXITCODE -ne 0) { throw 'Dense MUGEN autoencoder bridge failed' }
}
if (-not (Test-Path -LiteralPath $captionInputs)) {
    & $python (Join-Path $root 'scripts\build_mugen_dense_caption_inputs_v1.py') `
        $broad $captionInputs
    if ($LASTEXITCODE -ne 0) { throw 'Broad MUGEN caption inputs failed' }
}
