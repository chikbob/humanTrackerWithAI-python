@echo off
setlocal
cd /d "%~dp0\.."

if "%~1"=="" (
  echo Usage: scripts\start_windows_remote_companion.cmd ^<server_url^> ^<session_id^> [camera_index] [interval_ms]
  exit /b 1
)

if "%~2"=="" (
  echo Usage: scripts\start_windows_remote_companion.cmd ^<server_url^> ^<session_id^> [camera_index] [interval_ms]
  exit /b 1
)

if exist ".venv\Scripts\python.exe" (
  set "PYTHON=.venv\Scripts\python.exe"
) else if exist "venv\Scripts\python.exe" (
  set "PYTHON=venv\Scripts\python.exe"
) else (
  set "PYTHON=python"
)

set "CAMERA_INDEX=%~3"
if "%CAMERA_INDEX%"=="" set "CAMERA_INDEX=0"
set "INTERVAL_MS=%~4"
if "%INTERVAL_MS%"=="" set "INTERVAL_MS=300"

"%PYTHON%" -m services.windows_companion_agent --server-url "%~1" --session-id "%~2" --camera-index "%CAMERA_INDEX%" --interval-ms "%INTERVAL_MS%"
