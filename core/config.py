"""
core/config.py
──────────────
Pydantic settings — loaded from environment variables.
All agents share this config class.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Redis
    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")

    # Long-term memory (PostgreSQL + pgvector)
    database_url: str = Field("postgresql://agent:agent@postgres:5432/agentmem", env="DATABASE_URL")

    # LM Studio
    lm_studio_url: str = Field("http://host.docker.internal:1234", env="LM_STUDIO_URL")
    lm_studio_model: str = Field("qwen2.5-14b", env="LM_STUDIO_MODEL")

    # Agent identity
    agent_role: str = Field("agent", env="AGENT_ROLE")

    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        case_sensitive = False
