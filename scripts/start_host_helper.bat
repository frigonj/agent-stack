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

:: Read RESTART_HELPER_TOKEN out of .env so it reaches the helper's
:: environment — this process isn't launched via docker-compose, which is
:: what normally injects .env values into the agent containers.
set "RESTART_HELPER_TOKEN="
if exist "%REPO_ROOT%\.env" (
    for /f "usebackq tokens=1,* delims==" %%A in (`findstr /b "RESTART_HELPER_TOKEN=" "%REPO_ROOT%\.env"`) do (
        set "RESTART_HELPER_TOKEN=%%B"
    )
)

if not defined RESTART_HELPER_TOKEN (
    echo [ERROR] RESTART_HELPER_TOKEN is not set in .env.
    echo This server can restart containers and destroy Docker volumes, and it
    echo listens on 0.0.0.0, so it refuses to start without a shared secret.
    echo Generate one with: python -c "import secrets; print(secrets.token_hex(32))"
    echo and set RESTART_HELPER_TOKEN in %REPO_ROOT%\.env
    pause
    exit /b 1
)

echo Starting host restart helper via WSL on port 7799...
start "Agent Stack - Host Helper" /min wsl RESTART_HELPER_TOKEN="%RESTART_HELPER_TOKEN%" python3 "%WSL_SCRIPT%"

echo Host helper launched in background WSL window.
