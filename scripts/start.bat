@echo off
:: scripts/start.bat
:: ─────────────────
:: One-click startup for the agent stack on Windows.
::
:: What this does:
::   1. Starts the host restart helper (port 7799) in a background window
::   2. Starts: Redis, Postgres, Discord bridge, Orchestrator
::
:: Ephemeral agents (executor, code_search, document_qa, claude_code_agent)
:: are started automatically by the orchestrator when a task needs them,
:: and self-exit after 10 minutes of inactivity (AGENT_IDLE_TIMEOUT in .env).
::
:: To stop everything:
::   docker compose down

setlocal

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%.."

:: ── 1. Check for WSL ─────────────────────────────────────────────────────────
set "HAS_WSL=0"
where wsl >nul 2>&1
if %ERRORLEVEL%==0 set "HAS_WSL=1"

:: ── 2. Start host restart helper via WSL ─────────────────────────────────────
if "%HAS_WSL%"=="1" (
    echo [1/2] Starting host restart helper via WSL on port 7799...
    for /f "delims=" %%P in ('wsl wslpath -u "%REPO_ROOT%\scripts\host_restart_helper.py"') do (
        set "WSL_SCRIPT=%%P"
    )
    if defined WSL_SCRIPT (
        start "Agent Stack - Host Helper" /min wsl python3 "%WSL_SCRIPT%"
        echo       Listening on port 7799
    ) else (
        echo [WARN] Could not resolve WSL path — host helper skipped.
    )
) else (
    echo [1/2] WSL not found — skipping host restart helper.
    echo        Install WSL2 with a Linux distro to enable LM Studio restarts.
)

:: ── 3. Start Docker infra + Discord bridge ────────────────────────────────────
echo [2/2] Starting Docker infra + Discord bridge...
cd /d "%REPO_ROOT%"
docker compose up -d

echo.
echo ════════════════════════════════════════════════════════
echo   Agent stack is up
echo ════════════════════════════════════════════════════════
echo.
echo   Always-on:  Redis, Postgres, Discord bridge, Orchestrator
echo   On-demand:  executor, code_search, document_qa, claude_code_agent
echo               (started by orchestrator as needed, stop after 10 min idle)
echo.
echo   Logs:
echo     docker compose logs -f orchestrator
echo     docker compose logs -f discord_bridge
echo.
