$ErrorActionPreference = "Stop"

param(
    [Parameter(Mandatory = $true)][string]$ServerUrl,
    [Parameter(Mandatory = $true)][string]$SessionId,
    [int]$CameraIndex = 0,
    [int]$IntervalMs = 300
)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (Test-Path ".venv\Scripts\python.exe") {
    $python = ".venv\Scripts\python.exe"
} elseif (Test-Path "venv\Scripts\python.exe") {
    $python = "venv\Scripts\python.exe"
} else {
    $python = "python"
}

Write-Host "Starting Windows remote companion for $ServerUrl with session $SessionId" -ForegroundColor Cyan
& $python -m services.windows_companion_agent --server-url $ServerUrl --session-id $SessionId --camera-index $CameraIndex --interval-ms $IntervalMs
