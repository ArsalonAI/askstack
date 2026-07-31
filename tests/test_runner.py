"""Scorers — TRD §14.1.

Pure functions over lists, so none of this needs a database or a model. That
is deliberate: these are the numbers CI gates on, and a scorer bug would be
invisible behind a plausible-looking metric.
"""

from evals.runner import mrr_at_k, recall_at_k, set_f1


class TestSetF1:
    def test_exact_match(self):
        assert set_f1(["pr:1", "pr:2"], ["pr:2", "pr:1"]) == 1.0

    def test_empty_vs_empty_is_perfect(self):
        """"Which auth PRs merged in the last three months" can correctly answer
        *none*. Scoring that 0.0 would punish the one behaviour §5.2 wants —
        the held-out set's `expect_empty` questions depend on this."""
        assert set_f1([], []) == 1.0

    def test_empty_prediction_against_nonempty_gold(self):
        assert set_f1([], ["pr:1"]) == 0.0

    def test_nonempty_prediction_against_empty_gold(self):
        """Inventing entities where the answer is none is the §5.2 failure."""
        assert set_f1(["pr:1"], []) == 0.0

    def test_disjoint(self):
        assert set_f1(["pr:1"], ["pr:2"]) == 0.0

    def test_partial(self):
        # 1 hit, precision 1/2, recall 1/3 -> F1 = 2·(1/2·1/3)/(1/2+1/3) = 0.4
        assert set_f1(["pr:1", "pr:9"], ["pr:1", "pr:2", "pr:3"]) == 0.4

    def test_duplicates_do_not_inflate(self):
        assert set_f1(["pr:1", "pr:1"], ["pr:1"]) == 1.0


class TestRecallAtK:
    def test_counts_only_the_first_k(self):
        retrieved = ["a", "b", "c", "d", "e", "gold"]
        assert recall_at_k(retrieved, ["gold"], k=5) == 0.0
        assert recall_at_k(retrieved, ["gold"], k=6) == 1.0

    def test_fraction_of_gold_found(self):
        assert recall_at_k(["a", "b"], ["a", "z"], k=5) == 0.5

    def test_empty_gold_is_zero_not_a_crash(self):
        # Distinct from set-F1: an interpretive question with no gold chunks is
        # a malformed question, not a legitimate empty answer. build_gold's
        # validate rejects it; this only guarantees no ZeroDivisionError.
        assert recall_at_k(["a"], [], k=5) == 0.0


class TestMRRAtK:
    def test_reciprocal_of_first_hit(self):
        assert mrr_at_k(["a", "b", "gold"], ["gold"], k=10) == 1 / 3

    def test_first_gold_wins_not_the_best_one(self):
        assert mrr_at_k(["g2", "g1"], ["g1", "g2"], k=10) == 1.0

    def test_beyond_k_scores_zero(self):
        retrieved = [f"c{i}" for i in range(10)] + ["gold"]
        assert mrr_at_k(retrieved, ["gold"], k=10) == 0.0

    def test_no_hit(self):
        assert mrr_at_k(["a", "b"], ["gold"], k=10) == 0.0
