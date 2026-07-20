# Creates a project-local .venv and installs runtime dependencies.
# Usage (from repo root or this folder):
#   powershell -ExecutionPolicy Bypass -File .\env\setup_env.ps1

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $RepoRoot "env\requirements.txt"))) {
    Write-Error "Could not find env\requirements.txt. Run this script from the Capstone Project repo."
}

$VenvPath = Join-Path $RepoRoot ".venv"
$ReqPath = Join-Path $RepoRoot "env\requirements.txt"

Write-Host "==> Repo root: $RepoRoot"

$PythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $PythonCmd) {
    Write-Error "Python was not found on PATH. Install Python 3.11+ and retry."
}

$VersionText = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
Write-Host "==> Detected Python $VersionText"
$Major, $Minor = $VersionText.Split(".")
if ([int]$Major -lt 3 -or ([int]$Major -eq 3 -and [int]$Minor -lt 11)) {
    Write-Error "Python 3.11+ is required (found $VersionText)."
}

if (Test-Path $VenvPath) {
    Write-Host "==> Reusing existing .venv (delete .venv to recreate from scratch)"
} else {
    Write-Host "==> Creating virtual environment at .venv"
    & python -m venv $VenvPath
}

$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Error "Virtual environment Python not found at $VenvPython"
}

Write-Host "==> Upgrading pip"
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { Write-Error "pip upgrade failed" }

Write-Host "==> Installing requirements from env\requirements.txt"
& $VenvPython -m pip install -r $ReqPath
if ($LASTEXITCODE -ne 0) { Write-Error "pip install failed" }

Write-Host "==> Installing project package (editable)"
Push-Location $RepoRoot
& $VenvPython -m pip install -e .
$InstallCode = $LASTEXITCODE
Pop-Location
if ($InstallCode -ne 0) { Write-Error "editable install failed" }

Write-Host "==> Verifying key imports"
& $VenvPython -c "import pandas, openpyxl, requests, sklearn, rapidfuzz; print('OK: pandas, openpyxl, requests, sklearn, rapidfuzz')"
if ($LASTEXITCODE -ne 0) { Write-Error "import verification failed" }

Write-Host ""
Write-Host "Environment ready."
Write-Host "Next steps:"
Write-Host "  1. Activate:  .\.venv\Scripts\Activate.ps1"
Write-Host "  2. (Optional) Add On3 recruiting CSVs under data\raw\recruiting\"
Write-Host "  3. Compile:    python -m rookie_ppr.compile"
Write-Host "  4. Outputs:    data\output\rookie_ppr_master.xlsx and data\output\csv\"
Write-Host "  5. Scorer UI:  python -m rookie_ppr.ui"
Write-Host "     (or:        rookie-ppr-ui)"
