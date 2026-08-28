"""Episodic → semantic — TRD §8.2.

The clustering half is pure and tested here directly; it is also the half that
can corrupt memory silently, because a bad cluster does not raise — it produces
a confident consolidated statement about facts that had nothing to do with each
other.

The specific failure this module is built around: **"prefers async" and
"prefers sync" are near-identical vectors and opposite facts.** Cosine distance
cannot tell them apart, so the entity constraint and the explicit
`contradicts` field are what stand between that pair and a consolidated memory
asserting a confident average of the two.
"""

import numpy as np
import pytest

from app.interfaces import Memory
from app.memory.consolidation import (
    CONSOLIDATION_SCHEMA,
    DISTANCE_THRESHOLD,
    MIN_CLUSTER_SIZE,
    Cluster,
    cluster_memories,
    render_cluster,
)
from tests.test_memory_manager import NOW
from tests.test_memory_manager import memory as _memory


def mem(content, entities=(), memory_id="mem_x", age_days=0.0) -> Memory:
    return Memory(
        id=memory_id,
        user_id="u1",
        mem_type="episodic",
        content=content,
        entities=tuple(entities),
        confidence=0.8,
        revision=1,
        valid_from=_memory(age_days=age_days).valid_from,
        valid_to=None,
        created_by="extraction",
        source_session_id="sess_1",
        source_ids=(),
        trace_id=None,
    )


def unit(*components) -> np.ndarray:
    v = np.array(components, dtype=np.float32)
    return v / np.linalg.norm(v)


def tight(n: int, dim: int = 8) -> np.ndarray:
    """`n` vectors close enough to fall inside the 0.35 cosine threshold."""
    base = np.zeros(dim, dtype=np.float32)
    base[0] = 1.0
    out = []
    for i in range(n):
        v = base.copy()
        v[1] = 0.05 * i  # a small perturbation keeps cosine distance well under 0.35
        out.append(v / np.linalg.norm(v))
    return np.stack(out)


class TestClustering:
    def test_similar_memories_sharing_an_entity_cluster(self):
        memories = [
            mem("asked about auth in June", ["pr:1"], "m1"),
            mem("asked about auth in July", ["pr:1"], "m2"),
            mem("asked about auth in August", ["pr:1"], "m3"),
        ]
        clusters = cluster_memories(memories, tight(3))
        assert len(clusters) == 1
        assert set(clusters[0].ids) == {"m1", "m2", "m3"}

    def test_similar_memories_with_no_shared_entity_do_not_cluster(self):
        """The safeguard, stated as a test. These three are near-identical
        vectors; without the entity rule they would be consolidated into one
        statement despite being about unrelated objects."""
        memories = [
            mem("a", ["pr:1"], "m1"),
            mem("b", ["pr:2"], "m2"),
            mem("c", ["pr:3"], "m3"),
        ]
        assert cluster_memories(memories, tight(3)) == []

    def test_memories_with_no_entities_never_cluster(self):
        """A fact tied to no repository object is exactly the vague observation
        that should not become standing knowledge on the strength of sounding
        like two others."""
        memories = [mem("vague", (), f"m{i}") for i in range(3)]
        assert cluster_memories(memories, tight(3)) == []

    def test_a_pair_is_not_a_cluster(self):
        """Two similar facts are a coincidence; three can be a pattern."""
        memories = [mem("a", ["pr:1"], "m1"), mem("b", ["pr:1"], "m2")]
        assert cluster_memories(memories, tight(2)) == []

    def test_dissimilar_memories_do_not_cluster_even_sharing_an_entity(self):
        """The entity rule is a constraint on top of distance, not a substitute
        for it — otherwise everything about one PR would merge."""
        memories = [
            mem("auth is slow", ["pr:1"], "m1"),
            mem("docs are stale", ["pr:1"], "m2"),
            mem("CI is flaky", ["pr:1"], "m3"),
        ]
        orthogonal = np.stack([unit(1, 0, 0), unit(0, 1, 0), unit(0, 0, 1)])
        assert cluster_memories(memories, orthogonal) == []

    def test_below_the_minimum_returns_nothing_rather_than_raising(self):
        assert cluster_memories([mem("a", ["pr:1"])], np.zeros((1, 8))) == []
        assert cluster_memories([], np.empty((0, 0))) == []

    def test_partial_entity_overlap_is_not_enough(self):
        """`shared_entities` is an intersection across *all* members. Two of
        three sharing a PR would let a third unrelated fact ride along."""
        memories = [
            mem("a", ["pr:1", "pr:9"], "m1"),
            mem("b", ["pr:1"], "m2"),
            mem("c", ["pr:2"], "m3"),
        ]
        assert cluster_memories(memories, tight(3)) == []


class TestCluster:
    def test_shared_entities_is_an_intersection(self):
        cluster = Cluster(
            (mem("a", ["pr:1", "pr:2"], "m1"), mem("b", ["pr:2", "pr:3"], "m2"))
        )
        assert cluster.shared_entities == {"pr:2"}

    def test_render_includes_ids_so_contradictions_can_name_them(self):
        """`contradicts` refers to memory ids; a cluster rendered without them
        cannot express a contradiction at all."""
        rendered = render_cluster(Cluster((mem("prefers async", ["pr:1"], "mem_a"),)))
        assert "[mem_a]" in rendered
        assert "prefers async" in rendered

    def test_render_is_chronological(self):
        """"The newer wins" is the contradiction rule; the model cannot apply it
        from an unordered list."""
        rendered = render_cluster(
            Cluster(
                (
                    mem("newer", ["pr:1"], "m2", age_days=1),
                    mem("older", ["pr:1"], "m1", age_days=100),
                )
            )
        )
        assert rendered.index("older") < rendered.index("newer")

    def test_render_shows_confidence(self):
        assert "0.80" in render_cluster(Cluster((mem("a", ["pr:1"], "m1"),)))


class TestSchema:
    def test_contradicts_is_required_not_optional(self):
        """An optional field is one the model can quietly omit, and a silent
        omission here reads identically to "nothing conflicts"."""
        assert "contradicts" in CONSOLIDATION_SCHEMA["required"]

    def test_extra_properties_are_forbidden(self):
        assert CONSOLIDATION_SCHEMA["additionalProperties"] is False

    def test_every_property_is_required(self):
        assert set(CONSOLIDATION_SCHEMA["required"]) == set(
            CONSOLIDATION_SCHEMA["properties"]
        )


class TestSpecConstants:
    def test_threshold_matches_the_spec(self):
        """§8.2 fixes 0.35. It is deliberately not tuned — tuning it against one
        corpus fits the threshold to one user's memories."""
        assert DISTANCE_THRESHOLD == 0.35

    def test_minimum_cluster_size_matches_the_spec(self):
        assert MIN_CLUSTER_SIZE == 3


class TestPromptContract:
    def test_it_forbids_blending_contradictions(self):
        from app.memory.consolidation import CONSOLIDATION_PROMPT

        assert "Do not blend them" in CONSOLIDATION_PROMPT

    def test_it_names_the_async_sync_failure_directly(self):
        """The prompt teaches by the specific case rather than the abstraction,
        because the abstraction is what a model already believes it is doing."""
        from app.memory.consolidation import CONSOLIDATION_PROMPT

        assert "prefers async" in CONSOLIDATION_PROMPT

    def test_it_warns_against_consolidating_status(self):
        """The §5.2 failure again: status recorded as standing truth."""
        from app.memory.consolidation import CONSOLIDATION_PROMPT

        assert "goes stale" in CONSOLIDATION_PROMPT


class TestClampConfidence:
    def test_out_of_range_is_clamped_not_rejected(self):
        """`memories.confidence` has a CHECK constraint, so a model returning
        1.2 would fail the write at the end of a paid call."""
        from app.memory.consolidation import _clamp

        assert _clamp(1.5) == 1.0
        assert _clamp(-0.2) == 0.0
        assert _clamp(0.7) == pytest.approx(0.7)

    def test_garbage_falls_back_rather_than_raising(self):
        from app.memory.consolidation import _clamp

        assert 0.0 <= _clamp("very sure") <= 1.0
        assert 0.0 <= _clamp(None) <= 1.0


def test_now_is_stable():
    """Guards the shared fixture import from test_memory_manager."""
    assert NOW.year == 2026
