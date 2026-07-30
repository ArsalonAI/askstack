"""Identifier decomposition — TRD §6.2, ADR 12."""

import pytest

from app.ingest.identifiers import decompose, split_identifier, tsv_input


@pytest.mark.parametrize(
    ("identifier", "parts"),
    [
        ("get_current_user", ["get", "current", "user"]),
        ("HTTPException", ["http", "exception"]),
        ("getURL", ["get", "url"]),
        ("APIRouter", ["api", "router"]),
        ("parse_obj_as", ["parse", "obj", "as"]),
        ("_private_helper", ["private", "helper"]),
        # No internal structure: decomposing costs tokens and buys nothing.
        ("router", []),
        ("HTTP", []),
        ("x", []),
    ],
)
def test_split_identifier(identifier, parts):
    assert split_identifier(identifier) == parts


def test_the_motivating_cases_from_the_trd():
    """§6.2 names these two directly: `HTTPException` must match a query
    saying "http exception", and `get_current_user` must match its parts."""
    assert {"http", "exception"} <= set(decompose("raise HTTPException(404)").split())
    assert {"get", "current", "user", "get_current_user"} <= set(
        decompose("def get_current_user():").split()
    )


def test_the_original_identifier_survives_decomposition():
    """A query that says `get_current_user` verbatim must still match."""
    assert "get_current_user" in decompose("get_current_user").split()


def test_output_is_deterministic():
    """Delta detection (§5.3) compares hashes of generated content. A set that
    serialised in iteration order would rewrite every unchanged row on every
    ingest, and the skip path is what keeps a re-run under a minute."""
    text = "class APIRouter:\n    def add_api_route(self, path): ...\n"
    assert decompose(text) == decompose(text)
    assert decompose(text) == " ".join(sorted(decompose(text).split()))


def test_prose_without_identifiers_is_left_alone():
    assert decompose("the quick brown fox") == ""
    assert tsv_input("the quick brown fox") == "the quick brown fox"


def test_tsv_input_appends_rather_than_replaces():
    out = tsv_input("raise HTTPException here")
    assert out.startswith("raise HTTPException here")
    assert "exception" in out
