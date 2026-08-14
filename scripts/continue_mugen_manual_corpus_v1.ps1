param(
    [Parameter(Mandatory = $true)]
    [int]$WaitForProcessId
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
$runner = Join-Path $root 'scripts\materialize_mugen_manual_rar_core_v1.py'
$reports = Join-Path $root 'data\index\reports'
$jusOutput = Join-Path $root 'data\processed\mugen-iidx-jus-chibi-schema-core-b128-f8-v2'
$ascensionOutput = Join-Path $root 'data\processed\mugen-anime-ascension-schema-core-b128-f8-v2'

if (Get-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue) {
    Wait-Process -Id $WaitForProcessId
}

if (-not (Test-Path -LiteralPath $jusOutput)) {
    $retryOut = Join-Path $reports 'mugen-iidx-jus-full-materialization-v2-retry.out.log'
    $retryErr = Join-Path $reports 'mugen-iidx-jus-full-materialization-v2-retry.err.log'
    if ((Test-Path -LiteralPath $retryOut) -or (Test-Path -LiteralPath $retryErr)) {
        throw 'Refusing to replace JUS retry logs'
    }
    $retry = Start-Process -FilePath $python -ArgumentList @(
        $runner,
        '--profile',
        'iidx-jus-chibi-2000',
        '--retry-failed'
    ) -WorkingDirectory $root -WindowStyle Hidden -Wait -PassThru `
        -RedirectStandardOutput $retryOut -RedirectStandardError $retryErr
    if ($retry.ExitCode -ne 0) {
        throw "JUS retry exited with code $($retry.ExitCode)"
    }
}
if (-not (Test-Path -LiteralPath (Join-Path $jusOutput 'materialization.json'))) {
    throw 'JUS retry did not publish a complete materialization'
}

if (-not (Test-Path -LiteralPath $ascensionOutput)) {
    $ascensionOut = Join-Path $reports 'mugen-anime-ascension-full-materialization-v2.out.log'
    $ascensionErr = Join-Path $reports 'mugen-anime-ascension-full-materialization-v2.err.log'
    if ((Test-Path -LiteralPath $ascensionOut) -or (Test-Path -LiteralPath $ascensionErr)) {
        throw 'Refusing to replace Anime Ascension materialization logs'
    }
    $ascension = Start-Process -FilePath $python -ArgumentList @(
        $runner,
        '--profile',
        'anime-ascension-4000'
    ) -WorkingDirectory $root -WindowStyle Hidden -Wait -PassThru `
        -RedirectStandardOutput $ascensionOut -RedirectStandardError $ascensionErr
    if ($ascension.ExitCode -ne 0) {
        throw "Anime Ascension materialization exited with code $($ascension.ExitCode)"
    }
}
if (-not (Test-Path -LiteralPath (Join-Path $ascensionOutput 'materialization.json'))) {
    throw 'Anime Ascension did not publish a complete materialization'
}
