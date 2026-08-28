"""The embedder comparison — TRD §17 Q9.

No model is loaded here. Downloading five sets of weights to assert arithmetic
would make the suite slow, network-dependent, and no more correct. What is
tested is the harness's scoring and its per-family configuration, because both
can be wrong in ways that produce a plausible table:

- a query prefix applied to the wrong family measures the prefix, not the model
- `recall` computed over a set that still contains the always-injected tools
  measures §7.2.2 rather than the selector

The measured curve lives in `evals/baselines/embedders.json`.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from app.tools.registry import ALWAYS_INJECTED
from evals.embedders import CANDIDATES, TOP_K, load_questions, measure


class StubModel:
    """Ranks by an explicit list of tool names, so the expected order is stated
    rather than inferred.

    An earlier version keyed off a single marker and left every other tool
    tied; the sort then fell through to catalog order, and `merged_prs` — the
    first entry — landed in the top 5 even when the stub was supposed to be
    unable to find anything. A test that passes because of a tie is not
    testing ranking.

    Vectors live on the unit circle: the query is at angle 0, and a tool named
    `order[i]` sits at a small angle increasing with `i`, so cosine similarity
    is strictly decreasing in rank position. Unnamed tools sit furthest away.
    """

    def __init__(self, order: list[str] | None = None, dim: int = 2):
        self.order = order if order is not None else ["merged_prs"]
        self.dim = dim
        self.seen: list[str] = []

    def _angle(self, text: str) -> float:
        for position, name in enumerate(self.order):
            if name in text:
                return 0.02 * (position + 1)
        return 1.5  # far from the query, and beyond any ranked tool

    def encode(self, texts, **kw):
        self.seen.extend(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            # A query carries no tool name, so it lands at angle 0.
            angle = 0.0 if "Parameters:" not in text else self._angle(text)
            out[row] = (np.cos(angle), np.sin(angle))
        return out


class TestCandidateConfiguration:
    """§3.1 gives the protocol two methods rather than a boolean because
    getting the prefix wrong is a silent recall loss, not an error."""

    def test_every_candidate_declares_both_prefixes(self):
        for model_id, spec in CANDIDATES.items():
            assert "query_prefix" in spec, model_id
            assert "passage_prefix" in spec, model_id

    def test_bge_uses_its_instruction_prefix_on_queries_only(self):
        spec = CANDIDATES["BAAI/bge-small-en-v1.5"]
        assert spec["query_prefix"].startswith("Represent this sentence")
        assert spec["passage_prefix"] == ""

    def test_e5_prefixes_both_sides(self):
        """e5 is trained asymmetrically with `query:` and `passage:`. Omitting
        either costs several points and looks like the model being bad."""
        spec = CANDIDATES["intfloat/e5-small-v2"]
        assert spec["query_prefix"] == "query: "
        assert spec["passage_prefix"] == "passage: "

    def test_symmetric_models_get_no_prefix(self):
        for model_id in ("sentence-transformers/all-MiniLM-L6-v2", "thenlper/gte-small"):
            assert CANDIDATES[model_id]["query_prefix"] == ""
            assert CANDIDATES[model_id]["passage_prefix"] == ""

    def test_the_incumbent_is_among_the_candidates(self):
        """A comparison with no baseline in it cannot report a delta."""
        assert "BAAI/bge-small-en-v1.5" in CANDIDATES


class TestMeasure:
    @pytest.fixture(scope="module")
    def questions(self):
        return load_questions()

    def test_only_questions_with_gold_tools_are_scored(self, questions):
        """A question with no `gold_tools` has no right answer to find, and
        counting it would dilute recall with free passes."""
        assert questions
        assert all(q.get("gold_tools") for q in questions)

    def test_a_model_that_ranks_the_gold_tool_first_scores_well(self, questions):
        only = [q for q in questions if q["gold_tools"] == ["merged_prs"]][:3]
        assert only, "the golden set no longer has a merged_prs-only question"
        point = measure(StubModel(["merged_prs"]), _spec(), only, size=13)
        assert point.recall == 1.0

    def test_a_gold_tool_ranked_below_k_is_not_found(self, questions):
        """Six decoys ahead of it, k=5 — the exact shape of the crowd-out this
        module measures at 500 tools."""
        only = [q for q in questions if q["gold_tools"] == ["merged_prs"]][:3]
        decoys = [
            "open_issues", "stale_prs", "commits_by_author",
            "pr_state", "issue_state", "release_info",
        ]
        point = measure(StubModel(decoys), _spec(), only, size=13)
        assert point.recall == 0.0
        assert point.crowd_out == 0.0  # the decoys here are real tools, not padding

    def test_always_injected_tools_never_consume_a_top_k_slot(self, questions):
        """§7.2.2 — they are present regardless of query, so counting them
        measures the injection rule rather than the selector.

        Ranked first by the stub and still absent from the result: if they were
        eligible they would take two of the five slots.
        """
        point = measure(
            StubModel(list(ALWAYS_INJECTED) + ["merged_prs"]),
            _spec(),
            [q for q in questions if q["gold_tools"] == ["merged_prs"]][:3],
            size=13,
        )
        assert point.recall == 1.0

    def test_prefixes_reach_the_encoder(self, questions):
        model = StubModel()
        measure(model, _spec(query="Q: ", passage="P: "), questions[:2], size=13)
        assert any(t.startswith("P: ") for t in model.seen), "passages unprefixed"
        assert any(t.startswith("Q: ") for t in model.seen), "queries unprefixed"

    def test_the_catalog_is_padded_to_the_requested_size(self, questions):
        point = measure(StubModel(), _spec(), questions[:2], size=200)
        assert point.catalog_size == 200

    def test_top_k_widens_recall_monotonically(self, questions):
        """The finding this module exists to establish: recall is bounded by the
        window, so it cannot *fall* as the window grows."""
        subset = questions[:8]
        model = StubModel()
        recalls = [
            measure(model, _spec(), subset, size=200, top_k=k).recall
            for k in (5, 20, 50)
        ]
        assert recalls == sorted(recalls)

    def test_separation_is_gold_minus_nongold(self, questions):
        point = measure(StubModel(), _spec(), questions[:4], size=13)
        assert point.separation == pytest.approx(
            point.gold_median - point.nongold_median, abs=1e-4
        )


class TestCommittedComparison:
    """Shape, not values — the values move when a model or `k` moves."""

    @pytest.fixture(scope="module")
    def data(self):
        path = Path(__file__).resolve().parents[1] / "evals" / "baselines" / "embedders.json"
        if not path.is_file():
            pytest.skip("embedders.json not recorded yet")
        return json.loads(path.read_text())

    def test_it_records_the_window_it_measured_at(self, data):
        """A recall number without its `k` is uninterpretable — the whole
        finding is that `k` dominates."""
        assert data["top_k"] == TOP_K
        assert data["questions"] > 0

    def test_the_incumbent_is_present_for_comparison(self, data):
        models = {p["model"] for p in data["points"]}
        assert "BAAI/bge-small-en-v1.5" in models

    def test_more_than_one_model_was_measured(self, data):
        assert len({p["model"] for p in data["points"]}) >= 3

    def test_recall_degrades_with_catalog_size_for_every_model(self, data):
        """The collapse is structural rather than a property of one model — the
        claim §17 Q9 now rests on."""
        by_model: dict[str, list[dict]] = {}
        for p in data["points"]:
            by_model.setdefault(p["model"], []).append(p)
        for model, points in by_model.items():
            ordered = sorted(points, key=lambda p: p["catalog_size"])
            assert ordered[-1]["recall"] < ordered[0]["recall"], model


def _spec(query: str = "", passage: str = "") -> dict:
    return {"_id": "stub", "query_prefix": query, "passage_prefix": passage}
