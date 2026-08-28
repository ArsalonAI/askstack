"""The tool-scaling curve and its synthetic padding — TRD §14.5, §7.4, ADR 10.

The curve is the plot that carries PRD §5.4, and every way it can lie is
quiet: padding that repeats itself measures deduplication rather than scale,
padding that is obvious noise flatters the selector, and a catalog whose
*composition* changes with its size confounds the independent variable.

The measured curve itself lives in `evals/baselines/sweep.json` and is not
asserted here — these are the properties that have to hold for that file to
mean what it says.
"""

import json

import pytest

from app.tools.registry import CATALOG
from evals.sweep import (
    _full_tokens,
    _native_tokens,
    jaccard,
    prompt_tokens,
    recall,
)
from scripts.gen_synthetic_tools import DOMAINS, VERBS, generate


class TestSyntheticPadding:
    def test_capacity_reaches_the_largest_swept_size(self):
        """The 500-tool point is the one PRD §10 asks about; templates that
        cannot reach it would silently cap the curve."""
        capacity = sum(len(d["objects"]) for d in DOMAINS.values()) * len(VERBS)
        assert capacity >= 500 - len(CATALOG)

    def test_names_are_unique(self):
        """A catalog with repeated tools measures deduplication, not scale."""
        tools = generate(400)
        assert len({t.name for t in tools}) == len(tools)

    def test_sizes_are_prefix_stable(self):
        """The 50-tool point must be the first 50 of the 500.

        Without this the curve confounds catalog *size* with catalog
        *composition*, and a change between two points could be either.
        """
        assert [t.name for t in generate(50)] == [t.name for t in generate(500)][:50]

    def test_generation_is_deterministic(self):
        assert [t.name for t in generate(120)] == [t.name for t in generate(120)]

    def test_no_synthetic_name_collides_with_a_real_tool(self):
        """A synthetic shadowing a real name would make the real tool
        undispatchable — the padding would be sabotaging the system it is
        supposed to be a neutral background for."""
        real = {t.name for t in CATALOG}
        assert not real & {t.name for t in generate(500)}

    def test_everything_is_labelled_synthetic(self):
        """ADR 10 — unlabelled padding is dishonest measurement."""
        assert all(t.is_synthetic for t in generate(200))

    def test_distractors_share_vocabulary_with_the_real_catalog(self):
        """Padding that is obvious noise makes the curve flattering: a selector
        only has to beat gibberish to look good at scale.

        Measured over `embedding_text`, not over descriptions alone — §7.1
        embeds the name and parameter descriptions too, so that is the surface
        the distractors actually compete on.
        """
        from app.tools.selector import embedding_text

        synthetic = " ".join(embedding_text(t) for t in generate(500)).lower()
        real = " ".join(embedding_text(t) for t in CATALOG).lower()
        shared = {"list", "search", "status", "recent", "count", "window", "owner"}
        assert all(word in synthetic for word in shared)
        # The overlap has to be with the *real* catalog's vocabulary, not just
        # internally consistent across the padding.
        assert all(word in real for word in ("list", "search", "count", "window"))

    def test_small_catalogs_use_fewer_verb_shapes(self):
        """A consequence of prefix-stability worth knowing when reading the
        curve: padding is generated verb-major, so a 50-tool catalog carries
        only the first verb templates while a 500-tool one carries all ten.
        Small points on the curve therefore face *less shape-diverse*
        distractors, which if anything understates crowding at the low end —
        the direction that does not flatter the result.
        """
        shapes = lambda n: {t.name.split("_", 1)[0] for t in generate(n)}  # noqa: E731
        assert len(shapes(50)) < len(shapes(500))

    def test_asking_for_more_than_the_templates_yield_raises(self):
        """Silently returning fewer tools would mislabel a 300-tool point as
        500 and flatten the curve exactly where it matters."""
        with pytest.raises(ValueError, match="distinct tools"):
            generate(10_000)

    def test_zero_padding_is_empty_not_an_error(self):
        assert generate(0) == []


class TestPromptTokenModel:
    """The cost half of the curve. `semantic` being flat is the whole claim,
    so the arithmetic behind it has to be right."""

    def test_full_grows_with_the_catalog(self):
        assert _full_tokens(500) > _full_tokens(50) > _full_tokens(13)

    def test_full_is_roughly_linear_in_catalog_size(self):
        """Every definition, every request — so 10x the tools is ~10x the
        payload. A sublinear result would mean the model is wrong."""
        assert _full_tokens(500) / _full_tokens(50) == pytest.approx(10, rel=0.05)

    def test_native_is_cheaper_than_full_at_every_size(self):
        """Deferred tools send names, not definitions. If this ever inverts,
        `native` has stopped being a distinct arm."""
        for size in (13, 50, 200, 500):
            assert _native_tokens(size) < _full_tokens(size)

    def test_native_still_grows(self):
        """Deferred is not free — the API is still told the tools exist. A flat
        `native` column would overstate ADR 9's arm."""
        assert _native_tokens(500) > _native_tokens(50)

    def test_a_larger_payload_costs_more_tokens(self):
        ordered = sorted(CATALOG, key=lambda t: t.name)
        assert prompt_tokens(ordered) > prompt_tokens(ordered[:3])


class TestSelectionMetrics:
    def test_recall_is_independent_of_how_many_tools_were_offered(self):
        """Why the curve reports recall rather than jaccard: jaccard falls as
        `k` grows relative to `gold_tools` whether or not selection improved,
        which would confound the measurement with the arm's `k`."""
        gold = {"merged_prs"}
        assert recall({"merged_prs"}, gold) == 1.0
        assert recall({"merged_prs", "a", "b", "c", "d"}, gold) == 1.0
        assert jaccard({"merged_prs", "a", "b", "c", "d"}, gold) == pytest.approx(0.2)

    def test_recall_falls_when_the_gold_tool_is_displaced(self):
        assert recall({"a", "b"}, {"merged_prs"}) == 0.0

    def test_partial_recall_on_multi_tool_gold(self):
        assert recall({"search_docs"}, {"search_docs", "search_issues"}) == 0.5

    def test_empty_gold_does_not_divide_by_zero(self):
        assert recall(set(), set()) == 1.0


class TestCommittedCurve:
    """The recorded result. These assert its *shape*, not its values — the
    values move when the embedder or `k` moves, and should."""

    @pytest.fixture
    def curve(self):
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "evals" / "baselines" / "sweep.json"
        if not path.is_file():
            pytest.skip("sweep.json not recorded yet")
        return json.loads(path.read_text())

    def test_it_records_what_produced_it(self, curve):
        """A curve without its embedding model and `k` is uninterpretable —
        both are the reason the numbers are what they are."""
        assert curve["embedding_model"]
        assert curve["tool_retrieval_k"]

    def test_semantic_prompt_cost_is_flat_across_the_curve(self, curve):
        """PRD §5.4's cost claim. If this ever varies with catalog size, the
        thesis has broken."""
        semantic = {p["prompt_tokens"]["semantic"] for p in curve["points"]}
        assert len(semantic) == 1

    def test_full_prompt_cost_is_not_flat(self, curve):
        """The control arm has to actually grow, or there is nothing to beat."""
        full = [p["prompt_tokens"]["full"] for p in curve["points"]]
        assert full == sorted(full) and full[-1] > full[0] * 10

    def test_real_tool_count_is_constant(self, curve):
        """Only the padding varies. If the real catalog changed between points
        the curve would be measuring two things at once."""
        assert len({p["real_tools"] for p in curve["points"]}) == 1

    def test_crowd_out_explains_the_recall_column(self, curve):
        """The mechanism, asserted rather than asserted-in-prose: recall falls
        as distractors take top-k slots. Measuring crowd-out *after* selection
        always yields zero, which is how it was missed the first time."""
        first, last = curve["points"][0], curve["points"][-1]
        assert last["crowd_out_rate"] > first["crowd_out_rate"]
        assert last["tools_offered"] < first["tools_offered"]
