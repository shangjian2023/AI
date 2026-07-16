param(
    [string]$Participant = $env:COMPUTERNAME,
    [switch]$IncludeAdapters
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$Requirements = Join-Path $Root "requirements-team-opt125.txt"

function Invoke-Checked {
    param(
        [string]$Executable,
        [string[]]$Arguments
    )
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Executable $Arguments"
    }
}

if (-not (Test-Path -LiteralPath $VenvPython)) {
    $PyLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($PyLauncher) {
        Invoke-Checked $PyLauncher.Source @("-3.11", "-m", "venv", ".venv")
    } else {
        $Python = Get-Command python.exe -ErrorAction Stop
        $Version = & $Python.Source -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        if ($Version -ne "3.11") {
            throw "Python 3.11 is required. Found Python $Version."
        }
        Invoke-Checked $Python.Source @("-m", "venv", ".venv")
    }
}

$env:HF_ENDPOINT = if ($env:HF_ENDPOINT) { $env:HF_ENDPOINT } else { "https://hf-mirror.com" }
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Invoke-Checked $VenvPython @("-m", "pip", "install", "--upgrade", "pip")
Invoke-Checked $VenvPython @("-m", "pip", "install", "-r", $Requirements)
Invoke-Checked $VenvPython @("-m", "scripts.run_opt125_team_validation", "prepare")

$RunArguments = @(
    "-m", "scripts.run_opt125_team_validation", "run",
    "--participant", $Participant
)
if ($IncludeAdapters) {
    $RunArguments += "--include-adapters"
}
Invoke-Checked $VenvPython $RunArguments
