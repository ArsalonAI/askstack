"""Typed settings, mirroring `.env.example` field for field.

This is the only module that reads the environment. Everything else takes
`settings` (or an explicit argument), so an eval cell can construct its own
`Settings` without mutating global state.
"""

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ToolRetrievalMode = Literal["semantic", "native", "full"]
Effort = Literal["low", "medium", "high"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Corpus -----------------------------------------------------------
    corpus_repo: str = "fastapi/fastapi"
    # Symbolic ref. Ingest resolves it to a commit SHA and records that SHA on
    # the `ingest_runs` row — TRD §5.1 and §14.2 both require a *pinned*
    # revision, and `master` is not one.
    corpus_ref: str = "master"
    # Two floors: PRs and commits are windowed for cost, issues are not,
    # because the interpretive corpus lives in the archive. See `.env.example`.
    ingest_since: date | None = None
    ingest_issues_since: date | None = None
    github_token: str = ""
    areas_file: Path = Path("./areas.yaml")

    # --- Storage ----------------------------------------------------------
    database_url: str = "postgresql://askstack:askstack@localhost:5432/askstack"

    # --- Models -----------------------------------------------------------
    anthropic_api_key: str = ""
    agent_model: str = "claude-opus-5"
    agent_effort: Effort = "high"
    # extraction / consolidation / judge run at low effort
    batch_effort: Effort = "low"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # --- Observability ----------------------------------------------------
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    # --- Ablation flags (PRD §7.3) ---------------------------------------
    hybrid_enabled: bool = True
    memory_enabled: bool = True
    tool_retrieval_mode: ToolRetrievalMode = "semantic"
    tool_retrieval_k: int = 5
    # 0 = real tools only; >0 pads the catalog with synthetic defs (TRD §7.4)
    tool_catalog_size: int = 0
    memory_token_budget: int = 2000
    retrieval_top_k: int = 10
    tool_similarity_floor: float = 0.25

    # Kill switch for the structured facts path, NOT an ablation axis --
    # with it off, question classes 1-4 are unanswerable (TRD §6.4)
    structured_enabled: bool = True

    # --- Security ---------------------------------------------------------
    enable_run_snippet: bool = False  # TRD §15; not exposed until M2

    @field_validator("ingest_since", "ingest_issues_since", mode="before")
    @classmethod
    def _blank_is_unbounded(cls, value: object) -> object:
        """`INGEST_ISSUES_SINCE=` means full history, not a parse error."""
        return None if isinstance(value, str) and not value.strip() else value

    @property
    def sync_database_url(self) -> str:
        """`database_url` with the psycopg driver, for Alembic's sync engine."""
        _, _, rest = self.database_url.partition("://")
        return f"postgresql+psycopg://{rest}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
