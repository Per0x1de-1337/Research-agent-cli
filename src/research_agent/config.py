from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str | None = Field(
        default=None,
        validation_alias="OPENAI_API_KEY",
    )
    output_root: Path = Field(
        default=Path(".research-agent"),
        validation_alias="RESEARCH_AGENT_OUTPUT_ROOT",
    )
    planner_model: str = Field(
        default="gpt-5-mini",
        validation_alias="RESEARCH_AGENT_PLANNER_MODEL",
    )
    search_model: str = Field(
        default="gpt-4.1-mini",
        validation_alias="RESEARCH_AGENT_SEARCH_MODEL",
    )
    analyst_model: str = Field(
        default="gpt-5-mini",
        validation_alias="RESEARCH_AGENT_ANALYST_MODEL",
    )
    writer_model: str = Field(
        default="gpt-5.2",
        validation_alias="RESEARCH_AGENT_WRITER_MODEL",
    )
    critic_model: str = Field(
        default="gpt-5-mini",
        validation_alias="RESEARCH_AGENT_CRITIC_MODEL",
    )
    reasoning_effort: str = Field(
        default="medium",
        validation_alias="RESEARCH_AGENT_REASONING_EFFORT",
    )
    max_web_queries: int = Field(
        default=6,
        validation_alias="RESEARCH_AGENT_MAX_WEB_QUERIES",
    )
    max_research_passes: int = Field(
        default=2,
        validation_alias="RESEARCH_AGENT_MAX_RESEARCH_PASSES",
    )
    max_revisions: int = Field(
        default=1,
        validation_alias="RESEARCH_AGENT_MAX_REVISIONS",
    )
    max_source_chars: int = Field(
        default=16000,
        validation_alias="RESEARCH_AGENT_MAX_SOURCE_CHARS",
    )
    request_timeout_seconds: int = Field(
        default=180,
        validation_alias="RESEARCH_AGENT_REQUEST_TIMEOUT_SECONDS",
    )

    def ensure_output_root(self) -> Path:
        self.output_root.mkdir(parents=True, exist_ok=True)
        return self.output_root
