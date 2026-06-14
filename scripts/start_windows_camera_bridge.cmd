@echo off
setlocal
cd /d "%~dp0\.."

if exist ".venv\Scripts\python.exe" (
  set "PYTHON=.venv\Scripts\python.exe"
) else if exist "venv\Scripts\python.exe" (
  set "PYTHON=venv\Scripts\python.exe"
) else (
  set "PYTHON=python"
)

start "" "http://127.0.0.1:8123/health"
"%PYTHON%" -m uvicorn services.windows_bridge_app:app --host 127.0.0.1 --port 8123
