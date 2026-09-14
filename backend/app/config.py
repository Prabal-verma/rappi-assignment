"""Application configuration.

Everything the app can be tuned with lives here so that behaviour is
inspectable in one place — including the autonomy policy, which is a
business decision rather than a code detail.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["gemini", "anthropic", "openai", "deterministic"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- llm ----
    llm_provider: LLMProvider = "deterministic"
    llm_model: str = "gemini-2.5-flash"
    gemini_api_key: str = ""
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    llm_temperature: float = 0.0
    llm_timeout_seconds: int = 60
    llm_max_retries: int = 2

    # ---- database ----
    database_url: str = "sqlite+pysqlite:///./data/agent.db"

    # ---- agent ----
    agent_max_replan_attempts: int = 3
    agent_max_investigation_steps: int = 14

    # ---- autonomy policy ----
    # Landed order value at or below which the agent may execute without a
    # human, provided post-action validation is clean.
    autonomy_auto_approve_max_value_usd: float = 5000.0
    # Fractional deviation from the system recommendation beyond which a
    # human approves regardless of order value.
    autonomy_max_deviation_without_approval: float = 0.35

    # ---- supplier simulator ----
    supplier_sim_seed: int = 42

    # ---- app ----
    api_port: int = 8000
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def active_api_key(self) -> str:
        return {
            "gemini": self.gemini_api_key,
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
            "deterministic": "",
        }[self.llm_provider]

    @property
    def llm_is_live(self) -> bool:
        """True when a real model will be called."""
        return self.llm_provider != "deterministic" and bool(self.active_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
