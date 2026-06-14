$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (Test-Path ".venv\Scripts\python.exe") {
    $python = ".venv\Scripts\python.exe"
} elseif (Test-Path "venv\Scripts\python.exe") {
    $python = "venv\Scripts\python.exe"
} else {
    $python = "python"
}

Write-Host "Starting local API on http://127.0.0.1:8000" -ForegroundColor Cyan
Start-Process "http://127.0.0.1:8000"

& $python run_api.py
