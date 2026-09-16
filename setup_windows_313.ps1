$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

if (Test-Path "venv") {
    Remove-Item -Recurse -Force "venv"
}

py -3.13 -m venv venv
& ".\venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
& ".\venv\Scripts\python.exe" -m pip install --only-binary=:all: -r requirements-py313.txt

# Remove internet-zone markers from official packages downloaded by pip.
Get-ChildItem ".\venv" -Recurse -File | Unblock-File

& ".\venv\Scripts\python.exe" -c "import numpy, pandas, xgboost; print('Python 3.13 environment OK')"

Write-Host ""
Write-Host "Setup complete. Run:"
Write-Host ".\venv\Scripts\python.exe live_predict.py 12919"