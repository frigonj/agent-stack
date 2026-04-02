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
    database_url: str = Field("postgresql://agent:agent@postgres:5432/agentmem", validation_alias="DATABASE_URL")

    # LM Studio
    lm_studio_url: str = Field("http://host.docker.internal:1234", validation_alias="LM_STUDIO_URL")
    lm_studio_model: str = Field("qwen2.5-14b", validation_alias="LM_STUDIO_MODEL")

    # Agent identity
    agent_role: str = Field("agent", validation_alias="AGENT_ROLE")

    # Version stamp — set in docker-compose.yml; triggers intent flush on change
    agent_version: str = Field("", validation_alias="AGENT_VERSION")

    # Logging
    log_level: str = Field("INFO", validation_alias="LOG_LEVEL")
