#!/usr/bin/env python3
"""
scripts/host_restart_helper.py
────────────────────────────────
Lightweight HTTP server that runs on the Windows host and accepts restart
commands from Docker containers via host.docker.internal.

Usage:
  python scripts/host_restart_helper.py

Endpoints:
  GET  /health              — liveness check
  GET  /status              — docker ps output
  POST /restart/<service>   — restart a service

Services:
  lm-studio        — kill LM Studio.exe and relaunch
  orchestrator     — docker restart agent_orchestrator
  executor         — docker restart agent_executor
  code-search      — docker restart agent_code_search
  document-qa      — docker restart agent_document_qa
  discord          — docker restart agent_discord_bridge
  claude           — docker restart agent_claude_code
  redis            — docker restart agent_redis
  postgres         — docker restart agent_postgres
  all              — restart all agent containers (not infra)

To start on Windows login: add to Task Scheduler or drop a .bat in
  shell:startup  →  python "C:\\path\\to\\scripts\\host_restart_helper.py"
"""

from __future__ import annotations

import http.server
import json
import logging
import os
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

PORT = int(os.environ.get("RESTART_HELPER_PORT", "7799"))

# Maps friendly service names → Docker container names
CONTAINER_MAP: dict[str, str] = {
    "orchestrator": "agent_orchestrator",
    "executor": "agent_executor",
    "code-search": "agent_code_search",
    "code_search": "agent_code_search",
    "document-qa": "agent_document_qa",
    "document_qa": "agent_document_qa",
    "discord": "agent_discord_bridge",
    "claude": "agent_claude_code",
    "redis": "agent_redis",
    "postgres": "agent_postgres",
}

# Containers included in "restart all"
AGENT_CONTAINERS = [
    "agent_orchestrator",
    "agent_executor",
    "agent_code_search",
    "agent_document_qa",
    "agent_discord_bridge",
    "agent_claude_code",
]

# Common LM Studio install locations; first match wins
_LOCAL = os.environ.get("LOCALAPPDATA", "")
LM_STUDIO_PATHS = [
    Path(_LOCAL, "Programs", "lm-studio", "LM Studio.exe"),
    Path(_LOCAL, "LM-Studio", "LM Studio.exe"),
    Path(r"C:\Program Files\LM Studio", "LM Studio.exe"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _container_is_running(name: str) -> bool:
    """Return True if the container exists and is currently running."""
    try:
        r = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", name],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return r.returncode == 0 and r.stdout.strip() == "true"
    except Exception:
        return False


def _restart_container(name: str) -> dict:
    """
    Restart or start a container.

    If the container is running: docker restart (graceful stop + start).
    If the container is stopped:  docker start (bring it up without recreating).
    This lets the control channel revive agent containers that were never
    started (profile-gated) or that exited.
    """
    try:
        running = _container_is_running(name)
        cmd = ["docker", "restart" if running else "start", name]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode == 0:
            verb = "Restarted" if running else "Started"
            log.info("%s %s", verb.lower(), name)
            return {"ok": True, "message": f"{verb} {name}"}
        err = (r.stderr or r.stdout).strip()
        log.warning("failed to start/restart %s: %s", name, err)
        return {"ok": False, "message": err}
    except subprocess.TimeoutExpired:
        return {"ok": False, "message": f"Timeout starting {name}"}
    except FileNotFoundError:
        return {
            "ok": False,
            "message": "docker command not found — is Docker Desktop running?",
        }
    except Exception as e:
        return {"ok": False, "message": str(e)}


def _find_lm_studio() -> Path | None:
    # Check known install paths
    for p in LM_STUDIO_PATHS:
        if p.exists():
            return p

    # Fall back to querying running processes
    try:
        r = subprocess.run(
            [
                "wmic",
                "process",
                "where",
                "name='LM Studio.exe'",
                "get",
                "ExecutablePath",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in r.stdout.splitlines():
            line = line.strip()
            if line and "ExecutablePath" not in line:
                p = Path(line)
                if p.exists():
                    return p
    except Exception:
        pass

    return None


def _restart_lm_studio() -> dict:
    # 1. Kill existing process (ignore errors — it may not be running)
    kill = subprocess.run(
        ["taskkill", "/IM", "LM Studio.exe", "/F"],
        capture_output=True,
        text=True,
    )
    log.info("taskkill: %s", (kill.stdout or kill.stderr).strip())
    time.sleep(2)

    # 2. Find the executable
    lm_path = _find_lm_studio()
    if not lm_path:
        msg = (
            "LM Studio.exe not found in common locations. "
            "Set the correct path in LM_STUDIO_PATH env var or edit LM_STUDIO_PATHS "
            "in scripts/host_restart_helper.py."
        )
        log.error(msg)
        return {"ok": False, "message": msg}

    # 3. Relaunch detached
    try:
        subprocess.Popen(
            [str(lm_path)],
            creationflags=subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            if sys.platform == "win32"
            else 0,
        )
        log.info("launched LM Studio from %s", lm_path)
        return {"ok": True, "message": f"LM Studio restarted ({lm_path})"}
    except Exception as e:
        return {"ok": False, "message": f"Failed to launch {lm_path}: {e}"}


def _get_status() -> dict:
    try:
        r = subprocess.run(
            ["docker", "ps", "--format", "table {{.Names}}\t{{.Status}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return {"ok": True, "output": r.stdout}
    except Exception as e:
        return {"ok": False, "message": str(e)}


def _do_restart(service: str) -> dict:
    if service == "lm-studio":
        return _restart_lm_studio()
    if service == "all":
        results = [_restart_container(c) for c in AGENT_CONTAINERS]
        ok_count = sum(1 for r in results if r["ok"])
        return {
            "ok": ok_count > 0,
            "message": f"Restarted {ok_count}/{len(AGENT_CONTAINERS)} containers",
            "details": results,
        }
    if service in CONTAINER_MAP:
        return _restart_container(CONTAINER_MAP[service])
    # Accept raw container names (e.g. "agent_executor")
    return _restart_container(service)


# ── HTTP handler ──────────────────────────────────────────────────────────────


class RestartHandler(http.server.BaseHTTPRequestHandler):
    def log_message(
        self, fmt, *args
    ):  # silence default access log; structlog handles it
        pass

    def _send_json(self, status: int, data: dict) -> None:
        body = json.dumps(data, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path.rstrip("/")
        if path in ("", "/health"):
            self._send_json(
                200, {"ok": True, "service": "host_restart_helper", "port": PORT}
            )
        elif path == "/status":
            self._send_json(200, _get_status())
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        parts = urllib.parse.urlparse(self.path).path.strip("/").split("/")
        if len(parts) == 2 and parts[0] == "restart":
            service = parts[1].lower()
            log.info("restart request: %s", service)
            result = _do_restart(service)
            self._send_json(200, result)
        else:
            self._send_json(404, {"error": "not found"})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Allow overriding LM Studio path via env var
    lm_override = os.environ.get("LM_STUDIO_PATH")
    if lm_override:
        LM_STUDIO_PATHS.insert(0, Path(lm_override))

    server = http.server.HTTPServer(("0.0.0.0", PORT), RestartHandler)
    log.info("host_restart_helper listening on 0.0.0.0:%d", PORT)
    log.info("reachable from Docker at http://host.docker.internal:%d", PORT)
    log.info(
        "services: lm-studio, all, %s",
        ", ".join(k for k in CONTAINER_MAP if "_" not in k),
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
        server.server_close()
