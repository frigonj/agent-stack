"""
core/config.py
──────────────
Pydantic settings — loaded from environment variables.
All agents share this config class.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
    )

    # Redis
    redis_url: str = Field("redis://localhost:6379", validation_alias="REDIS_URL")

    # Long-term memory (PostgreSQL + pgvector)
    database_url: str = Field(
        "postgresql://agent:agent@postgres:5432/agentmem",
        validation_alias="DATABASE_URL",
    )

    # LM Studio
    lm_studio_url: str = Field(
        "http://host.docker.internal:1234", validation_alias="LM_STUDIO_URL"
    )
    lm_studio_model: str = Field("qwen3-vl-8b", validation_alias="LM_STUDIO_MODEL")

    # Claude fallback (used when ANTHROPIC_API_KEY is set and LM Studio circuit-breaks)
    claude_fallback_model: str = Field(
        "claude-haiku-4-5-20251001", validation_alias="CLAUDE_FALLBACK_MODEL"
    )

    # Agent identity
    agent_role: str = Field("agent", validation_alias="AGENT_ROLE")

    # Version stamp — set in docker-compose.yml; triggers intent flush on change
    agent_version: str = Field("", validation_alias="AGENT_VERSION")

    # Working directory containing docker-compose.yml (used for compose up fallback)
    compose_project_dir: str = Field(
        "/workspace/src", validation_alias="COMPOSE_PROJECT_DIR"
    )

    # Logging
    log_level: str = Field("INFO", validation_alias="LOG_LEVEL")
