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

Write-Host "Starting Windows camera bridge on http://127.0.0.1:8123" -ForegroundColor Cyan
Start-Process "http://127.0.0.1:8123/health"

& $python -m uvicorn services.windows_bridge_app:app --host 127.0.0.1 --port 8123
