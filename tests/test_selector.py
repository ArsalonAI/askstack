"""Tool selection — TRD §7.1, §7.2, §7.3, ablation axis C.

The semantic arm's ranking quality is a *measured* property, reported by the
eval, not asserted here. What these tests protect are the invariants that would
silently corrupt a measurement: a mode that quietly becomes a different mode, a
tool set that reorders between runs, and an embedding built from the wrong text.
"""

import pytest

from app.interfaces import ToolDef
from app.tools.registry import ALWAYS_INJECTED, BY_NAME, CATALOG
from app.tools.selector import (
    NATIVE_SEARCH_TOOL,
    FullToolSelector,
    NativeToolSelector,
    build_selector,
    embedding_text,
    to_api_tools,
)


class TestEmbeddingText:
    def test_uses_the_rendered_template_not_raw_json(self):
        """§7.1: raw JSON Schema embeds `"type": "object"` and `"required"`,
        which are identical across every tool and dilute the signal."""
        text = embedding_text(BY_NAME["stale_prs"])
        assert "stale_prs" in text
        assert '"type"' not in text
        assert "required" not in text

    def test_includes_parameter_descriptions(self):
        """Parameter descriptions carry most of what discriminates one tool
        from another."""
        text = embedding_text(BY_NAME["merged_prs"])
        assert "Parameters:" in text
        assert "area" in text

    def test_is_deterministic(self):
        """A template that reorders between runs re-embeds unchanged tools and
        makes `seed_tools.py --check` permanently dirty."""
        tool = BY_NAME["merged_prs"]
        assert embedding_text(tool) == embedding_text(tool)

    def test_every_tool_renders_non_empty(self):
        for tool in CATALOG:
            assert embedding_text(tool).strip(), tool.name


class TestApiTools:
    def test_output_is_sorted_by_name(self):
        """§10. The semantic arm builds its list from a vector query, which can
        reorder between runs; a reordered prefix invalidates the prompt cache
        with no symptom other than the bill."""
        shuffled = sorted(CATALOG, key=lambda t: t.name, reverse=True)
        names = [t["name"] for t in to_api_tools(shuffled)]
        assert names == sorted(names)

    def test_carries_only_the_wire_fields(self):
        entry = to_api_tools([BY_NAME["pr_state"]])[0]
        assert set(entry) == {"name", "description", "input_schema"}

    def test_deferred_tools_are_flagged(self):
        tools = to_api_tools(CATALOG, deferred=["merged_prs"])
        flagged = {t["name"] for t in tools if t.get("defer_loading")}
        assert flagged == {"merged_prs"}


class TestFullMode:
    async def test_returns_the_whole_catalog_regardless_of_query(self):
        selector = FullToolSelector(CATALOG)
        for query in ("what shipped last month", "hi", ""):
            assert len(await selector.select(query, k=2)) == len(CATALOG)

    async def test_ignores_k(self):
        """The control arm's cost is the point of the arm — an honest
        comparison needs it to actually broadcast everything."""
        assert len(await FullToolSelector(CATALOG).select("x", k=1)) == len(CATALOG)


class TestNativeMode:
    def test_defers_everything_except_the_loaded_set(self):
        """The API rejects a request in which every tool is deferred, so the
        anchor stays loaded — and so do the always-injected memory tools."""
        extra = NativeToolSelector(CATALOG).extra_request_params()
        loaded = {NativeToolSelector.ALWAYS_LOADED, *ALWAYS_INJECTED}
        assert not loaded & set(extra["defer"])
        assert len(extra["defer"]) == len(CATALOG) - len(loaded)

    def test_memory_tools_are_never_deferred(self):
        """ADR 11's failure mode, reintroduced by the provider's mechanism
        instead of ours: nobody phrases a question so that `memory_write` is
        the BM25 match, so a deferred memory tool never loads."""
        extra = NativeToolSelector(CATALOG).extra_request_params()
        assert not set(ALWAYS_INJECTED) & set(extra["defer"])

    def test_a_catalog_without_memory_defers_all_but_the_anchor(self):
        """Memory off: nothing to keep loaded but the anchor."""
        catalog = tuple(t for t in CATALOG if t.name not in ALWAYS_INJECTED)
        extra = NativeToolSelector(catalog).extra_request_params()
        assert len(extra["defer"]) == len(catalog) - 1

    def test_adds_the_provider_search_tool(self):
        extra = NativeToolSelector(CATALOG).extra_request_params()
        assert extra["extra_tools"] == [NATIVE_SEARCH_TOOL]

    def test_the_always_loaded_tool_is_real(self):
        assert NativeToolSelector.ALWAYS_LOADED in BY_NAME


class TestBuildSelector:
    @pytest.mark.parametrize(("mode", "cls"), [("full", "full"), ("native", "native")])
    def test_modes_resolve_to_their_own_arm(self, mode, cls):
        assert build_selector(mode, CATALOG).mode == cls

    def test_unknown_mode_raises_rather_than_falling_back(self):
        """A run that reports `semantic` in its config hash while actually
        broadcasting every tool poisons the comparison the thesis rests on."""
        with pytest.raises(NotImplementedError):
            build_selector("magic", CATALOG)

    def test_semantic_without_a_backend_raises(self):
        with pytest.raises(ValueError, match="pool"):
            build_selector("semantic", CATALOG)


def test_catalog_entries_are_tool_defs():
    assert all(isinstance(tool, ToolDef) for tool in CATALOG)
