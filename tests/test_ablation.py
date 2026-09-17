"""The memory ablation's analysis — pure functions, no database or model.

The paid run is only as trustworthy as what reads it: a bootstrap that
resamples the arms independently, or a loader that accepts the wrong arm,
would print a confident interval around the wrong number.
"""

import json

import pytest

from app.tools.registry import ALWAYS_INJECTED
from evals import ablation
from evals.runner import CACHE_FORMAT, Result


def record(qid, klass=1, *, success=True, latency_ms=1000, cost=0.1, f1=1.0):
    return {
        "qid": qid,
        "class": klass,
        "set_f1": f1,
        "success": success,
        "latency_ms": latency_ms,
        "cost_usd": cost,
    }


def result(klass, *, set_f1=None, gold=(), citations=()):
    metrics = {} if set_f1 is None else {"set_f1": set_f1}
    return Result(
        qid="q001",
        klass=klass,
        metrics=metrics,
        gold=list(gold),
        agent={"citations": [{"citation": c} for c in citations]},
    )


class TestSucceeded:
    def test_exact_needs_perfect_set_f1(self):
        assert ablation.succeeded(result(1, set_f1=1.0))
        assert not ablation.succeeded(result(2, set_f1=0.8))

    def test_interpretive_needs_a_cited_gold_passage(self):
        assert ablation.succeeded(result(5, gold=["issue:1#body"], citations=["issue:1#body"]))

    def test_interpretive_citing_only_other_passages_fails(self):
        assert not ablation.succeeded(
            result(6, gold=["issue:1#body"], citations=["issue:2#body"])
        )


class TestPercentile:
    def test_matches_numpy_linear_interpolation(self):
        values = [1.0, 2.0, 3.0, 4.0]
        assert ablation.percentile(values, 0.5) == 2.5
        assert ablation.percentile(values, 0.0) == 1.0
        assert ablation.percentile(values, 1.0) == 4.0
        assert ablation.percentile(values, 0.95) == pytest.approx(3.85)


class TestPairedBootstrap:
    def test_identical_arms_give_a_zero_width_difference(self):
        """Pairing is the point: the same resample hits both arms, so arms that
        agree on every question cannot disagree on any replicate."""
        rows = [record(f"q{i:03}", latency_ms=1000 * i) for i in range(20)]
        out = ablation.paired_bootstrap(
            rows, rows, lambda rs: ablation.mean([r["latency_ms"] for r in rs]), n_boot=500
        )
        assert out["diff_on_minus_off"]["estimate"] == 0
        assert out["diff_on_minus_off"]["ci95"] == [0, 0]
        lo, hi = out["on"]["ci95"]
        assert lo < out["on"]["estimate"] < hi

    def test_constant_shift_is_recovered_exactly(self):
        off = [record(f"q{i:03}", cost=0.01 * i) for i in range(30)]
        on = [{**r, "cost_usd": r["cost_usd"] + 0.05} for r in off]
        out = ablation.paired_bootstrap(
            on, off, lambda rs: ablation.mean([r["cost_usd"] for r in rs]), n_boot=500
        )
        lo, hi = out["diff_on_minus_off"]["ci95"]
        assert lo == pytest.approx(0.05)
        assert hi == pytest.approx(0.05)

    def test_is_deterministic_under_the_seed(self):
        rows = [record(f"q{i:03}", success=i % 3 == 0) for i in range(15)]
        metric = lambda rs: ablation.mean([float(r["success"]) for r in rs])  # noqa: E731
        assert ablation.paired_bootstrap(rows, rows, metric, n_boot=200) == (
            ablation.paired_bootstrap(rows, rows, metric, n_boot=200)
        )

    def test_stratification_preserves_the_group_sizes(self):
        """With strata, every replicate keeps the exact/interpretive split — so a
        metric that counts one group is constant across replicates."""
        rows = [record(f"q{i:03}", klass=1 if i < 6 else 5) for i in range(10)]
        count_exact = lambda rs: sum(r["class"] == 1 for r in rs)  # noqa: E731
        out = ablation.paired_bootstrap(
            rows, rows, count_exact, strata=lambda r: r["class"] == 1, n_boot=300
        )
        assert out["on"]["ci95"] == [6, 6]

    def test_refuses_unpaired_arms(self):
        with pytest.raises(AssertionError):
            ablation.paired_bootstrap(
                [record("q001")], [record("q002")], lambda rs: 0.0, n_boot=1
            )


class TestLoadArm:
    def _write(self, path, *, offered):
        selected = ["merged_prs", *(ALWAYS_INJECTED if offered else [])]
        row = {
            "cache_format": CACHE_FORMAT,
            "qid": "q001",
            "klass": 1,
            "metrics": {"set_f1": 1.0},
            "predicted": [],
            "gold": [],
            "agent": {
                "selected": selected,
                "called": ["merged_prs"],
                "citations": [],
                "usage": {"latency_ms": 900, "cost_usd": 0.02, "usage": {"output_tokens": 50}},
            },
        }
        path.write_text(json.dumps(row) + "\n")

    def test_accepts_a_correctly_labelled_arm(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ablation, "CACHE_DIR", tmp_path)
        self._write(tmp_path / "memory_on.jsonl", offered=True)
        [row] = ablation.load_arm("on")
        assert row["memory_tools_offered"] and row["success"]

    def test_rejects_an_off_arm_that_offered_memory_tools(self, tmp_path, monkeypatch):
        """The failure a shared cache would produce: the off arm silently
        replaying the on arm's answers."""
        monkeypatch.setattr(ablation, "CACHE_DIR", tmp_path)
        self._write(tmp_path / "memory_off.jsonl", offered=True)
        with pytest.raises(SystemExit, match="not run with memory off"):
            ablation.load_arm("off")


class TestIncomplete:
    def test_arms_that_stopped_together_are_still_incomplete(self):
        """The failure that actually happened: a credit balance ran out with both
        arms at q011, so they matched each other and nothing else."""
        partial = [record(f"q{i:03}") for i in range(1, 12)]
        expected = [f"q{i:03}" for i in range(1, 51)]
        problem = ablation.incomplete(partial, partial, expected)
        assert problem and "missing 39 of 50" in problem

    def test_complete_arms_pass(self):
        full = [record(f"q{i:03}") for i in range(1, 51)]
        assert ablation.incomplete(full, full, [r["qid"] for r in full]) is None


class TestLeaking:
    def test_memory_for_an_answered_question_is_harmless(self):
        """What the first run left behind: q001's on-arm turn wrote a memory. q001
        is cached and never re-asked, so nothing can read it."""
        assert ablation.leaking({"eval_q001"}, ["q001", "q012"], cached={"q001"}) == []

    def test_memory_for_a_question_still_to_run_blocks(self):
        assert ablation.leaking({"eval_q012"}, ["q001", "q012"], cached={"q001"}) == ["q012"]

    def test_fresh_run_treats_every_question_as_pending(self):
        assert ablation.leaking({"eval_q001"}, ["q001"], cached=set()) == ["q001"]
