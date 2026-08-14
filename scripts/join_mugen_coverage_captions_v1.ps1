param(
    [Parameter(Mandatory = $true)]
    [int]$WaitForCaptionProcessId,
    [Parameter(Mandatory = $true)]
    [int]$WaitForCoverageProcessId
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
$processed = Join-Path $root 'data\processed'
$dense = Join-Path $processed 'mugen-six-action-dense-coverage-all-scales-v1.json'
$captions = Join-Path $processed 'mugen-six-action-broad-captions-v1\manifest.json'
$output = Join-Path $processed 'mugen-six-action-dense-coverage-all-scales-captioned-v1.json'

foreach ($processId in @($WaitForCaptionProcessId, $WaitForCoverageProcessId)) {
    if (Get-Process -Id $processId -ErrorAction SilentlyContinue) {
        Wait-Process -Id $processId
    }
}
foreach ($input in @($dense, $captions)) {
    if (-not (Test-Path -LiteralPath $input)) {
        throw "Coverage caption-join input is incomplete: $input"
    }
}
if (Test-Path -LiteralPath $output) {
    throw "Refusing to replace coverage caption join: $output"
}
& $python (Join-Path $root 'scripts\build_mugen_dense_autoencoder_bridge_v1.py') `
    $dense $output --caption-manifest $captions
if ($LASTEXITCODE -ne 0) { throw 'Coverage caption join failed' }
