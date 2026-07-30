"""`Settings` must stay in lockstep with `.env.example`.

`.env.example` is the documented surface — it is what a new checkout copies.
A field that exists in one and not the other is a setting that either can't be
configured or is silently ignored, and both fail quietly.
"""

import re
from pathlib import Path

from app.config import Settings

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_EXAMPLE = REPO_ROOT / ".env.example"

# Set by docker-compose for the Langfuse container, not read by the app.
COMPOSE_ONLY = {"langfuse_nextauth_secret", "langfuse_salt"}


def _env_example_keys() -> set[str]:
    text = ENV_EXAMPLE.read_text()
    return {
        m.group(1).lower()
        for m in re.finditer(r"^([A-Z][A-Z0-9_]*)=", text, flags=re.MULTILINE)
    }


def test_settings_covers_every_documented_key():
    documented = _env_example_keys() - COMPOSE_ONLY
    assert documented - set(Settings.model_fields) == set()


def test_no_undocumented_settings():
    # enable_run_snippet is deliberately absent from .env.example -- TRD §15
    # keeps it off and unexposed until M2.
    extra = set(Settings.model_fields) - _env_example_keys() - {"enable_run_snippet"}
    assert extra == set()


def test_sync_url_swaps_in_the_psycopg_driver():
    s = Settings(database_url="postgresql://u:p@host:5432/db")
    assert s.sync_database_url == "postgresql+psycopg://u:p@host:5432/db"
