@echo off
:: scripts/start_host_helper.bat
:: ──────────────────────────────
:: Launches the host restart helper via WSL Python in a minimized background window.
:: Run this once before starting the Docker stack.
::
:: The helper listens on port 7799 and lets Docker containers send
:: restart commands to the Windows host (e.g. restart LM Studio).
::
:: To have this run at login automatically:
::   Win+R → shell:startup → drop a shortcut to this file there.

setlocal

:: Resolve the repo root (one level above the scripts\ folder)
set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%.."

:: Verify WSL is available
where wsl >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] WSL not found. Ensure WSL2 is installed and a distro is set as default.
    pause
    exit /b 1
)

:: Convert the Windows script path to a WSL path using wslpath.
:: We pass the Windows path as a wsl argument — wslpath handles spaces correctly.
for /f "delims=" %%P in ('wsl wslpath -u "%REPO_ROOT%\scripts\host_restart_helper.py"') do (
    set "WSL_SCRIPT=%%P"
)

if not defined WSL_SCRIPT (
    echo [ERROR] Could not resolve WSL path for the helper script.
    pause
    exit /b 1
)

echo Starting host restart helper via WSL on port 7799...
start "Agent Stack - Host Helper" /min wsl python3 "%WSL_SCRIPT%"

echo Host helper launched in background WSL window.
