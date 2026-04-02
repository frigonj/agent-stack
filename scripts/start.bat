@echo off
:: scripts/start.bat
:: ─────────────────
:: Usage:
::   start.bat          — start the full agent stack
::   start.bat status   — show running containers + agent task/queue status
::   start.bat stop     — stop all containers (docker compose down)
::
:: Ephemeral agents (executor, code_search, document_qa, claude_code_agent,
:: research) are started automatically by the orchestrator when a task needs
:: them, and self-exit after AGENT_IDLE_TIMEOUT seconds of inactivity.

setlocal

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%.."

:: ── Subcommand dispatch ───────────────────────────────────────────────────────
if /i "%~1"=="status" goto :status
if /i "%~1"=="stop"   goto :stop

:: ── START ─────────────────────────────────────────────────────────────────────

:: 1. Check for WSL
set "HAS_WSL=0"
where wsl >nul 2>&1
if %ERRORLEVEL%==0 set "HAS_WSL=1"

:: 2. Start host restart helper via WSL
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

:: 3. Start Docker infra + all always-on services
echo [2/2] Starting Docker infra...
cd /d "%REPO_ROOT%"
docker compose up -d

echo.
echo ════════════════════════════════════════════════════════
echo   Agent stack is up
echo ════════════════════════════════════════════════════════
echo.
echo   Always-on:  Redis, Postgres, Discord bridge, Orchestrator
echo   On-demand:  executor, code_search, document_qa, claude_code_agent, research
echo               (started by orchestrator as needed, stop after idle timeout)
echo.
echo   Commands:
echo     start.bat status              — live agent status ^& queue depths
echo     docker compose logs -f orchestrator
echo     docker compose logs -f discord_bridge
echo.
goto :eof

:: ── STATUS ────────────────────────────────────────────────────────────────────
:status
cd /d "%REPO_ROOT%"
echo.
echo ════════════════════════════════════════════════════════
echo   Container Status
echo ════════════════════════════════════════════════════════
docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Service}}"
echo.
echo ════════════════════════════════════════════════════════
echo   Agent Task Status  (via Redis agent:status:*)
echo ════════════════════════════════════════════════════════
docker compose exec -T redis redis-cli --no-auth-warning KEYS "agent:status:*" 2>nul | findstr /v "^$" > "%TEMP%\agent_keys.txt" 2>nul
if not exist "%TEMP%\agent_keys.txt" goto :status_unavail
for /f "delims=" %%K in (%TEMP%\agent_keys.txt) do (
    for /f "delims=" %%V in ('docker compose exec -T redis redis-cli --no-auth-warning GET "%%K" 2^>nul') do (
        echo   %%K  =  %%V
    )
)
del "%TEMP%\agent_keys.txt" >nul 2>&1
echo.
echo ════════════════════════════════════════════════════════
echo   Redis Stream Pending Counts  (agents:*)
echo ════════════════════════════════════════════════════════
docker compose exec -T redis redis-cli --no-auth-warning KEYS "agents:*" 2>nul | findstr /v "^$" | findstr /v "broadcast" > "%TEMP%\stream_keys.txt" 2>nul
if exist "%TEMP%\stream_keys.txt" (
    for /f "delims=" %%S in (%TEMP%\stream_keys.txt) do (
        for /f "delims=" %%L in ('docker compose exec -T redis redis-cli --no-auth-warning XLEN "%%S" 2^>nul') do (
            echo   %%S  length=%%L
        )
    )
    del "%TEMP%\stream_keys.txt" >nul 2>&1
)
echo.
goto :eof

:status_unavail
echo   (Redis not reachable — is the stack running? Run: start.bat)
echo.
goto :eof

:: ── STOP ──────────────────────────────────────────────────────────────────────
:stop
cd /d "%REPO_ROOT%"
echo Stopping all agent stack containers...
docker compose down
echo Done.
goto :eof
