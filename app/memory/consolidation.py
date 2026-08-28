"""Episodic → semantic — TRD §8.2.

Extraction (§8.1) produces one episodic fact per session. Left alone, a user
who checks in weekly accumulates fifty near-duplicate observations of the same
standing preference, each competing for the same slice of a 2000-token budget.
Consolidation is what turns "asked about auth in June, asked about auth in
July, asked about auth in August" into one semantic memory that says what is
actually true about this user — and, critically, what *stopped* being true.

Two safeguards do the real work here, and both exist because embedding
similarity alone is dangerous in this specific way: **"prefers async" and
"prefers sync" are near-identical vectors.** Clustering on cosine distance
would happily merge them and hand the model a cluster whose members
contradict, and the consolidated statement would be a confident average of
two opposite facts. So clusters additionally require a shared entity, and the
model is asked to name contradictions explicitly rather than silently
reconciling them.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.interfaces import ConsolidationReport, Memory

log = logging.getLogger(__name__)

# §8.2. Cosine distance, average linkage. 0.35 is the spec's value and is not
# tuned here — tuning it against this corpus would fit the threshold to one
# user's memories, which is the same mistake as fitting a retrieval parameter
# to the golden set.
DISTANCE_THRESHOLD = 0.35

# §8.2 step 3. Two facts that happen to be similar are not a pattern; three is
# the smallest number that can be one. Consolidating pairs would also make the
# entity constraint nearly free to satisfy by accident.
MIN_CLUSTER_SIZE = 3

CONSOLIDATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "statement": {"type": "string"},
        "entities": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
        "contradicts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["statement", "entities", "confidence", "contradicts"],
    "additionalProperties": False,
}

CONSOLIDATION_PROMPT = """\
Below are episodic memories about one engineering manager, recorded across \
several sessions. They were grouped because they are semantically similar and \
share at least one referenced entity.

Write the single durable fact they add up to.

- **State what is true now**, not what was observed. "Tracks the auth \
workstream and the people working on it" — not "asked about auth three times".
- **If some of these contradict others, the newer wins**, and you must list \
the memory ids that are now wrong in `contradicts`. This is the whole reason \
you are being asked rather than an averaging function: "prefers async" and \
"prefers sync" are nearly identical as text and opposite as facts. Do not \
blend them. Name the loser.
- **Leave `contradicts` empty if nothing conflicts.** Most clusters are \
repetition, not disagreement, and inventing a contradiction destroys a memory \
that was fine.
- **Confidence is how durably true the consolidated fact is.** Repetition \
across sessions is evidence; a single loud statement is not.
- **Do not consolidate status into standing truth.** If these are observations \
about what had merged or was open at particular moments, the durable fact is \
about the manager's *interest*, not about the repository's state — that goes \
stale and the facts layer answers it better.

<memories>
{memories}
</memories>
"""


@dataclass(frozen=True)
class Cluster:
    """A group the model will be asked to consolidate."""

    memories: Sequence[Memory]

    @property
    def ids(self) -> list[str]:
        return [m.id for m in self.memories]

    @property
    def shared_entities(self) -> set[str]:
        sets = [set(m.entities) for m in self.memories]
        return set.intersection(*sets) if sets else set()


def cluster_memories(
    memories: Sequence[Memory], vectors: np.ndarray
) -> list[Cluster]:
    """§8.2 step 2 — agglomerative, cosine, average linkage, plus the entity rule.

    The entity constraint is applied *after* clustering rather than as a
    distance-matrix penalty. Both work; this way the two rules stay legible and
    a cluster rejected for having no shared entity is visibly a rejection
    rather than a distance that mysteriously exceeded a threshold.

    Memories with no entities at all can never satisfy the constraint and are
    skipped. That is intended: a fact tied to no repository object is exactly
    the kind of vague observation that should not be promoted to standing
    knowledge on the strength of sounding like two others.
    """
    if len(memories) < MIN_CLUSTER_SIZE:
        return []

    from sklearn.cluster import AgglomerativeClustering

    labels = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=DISTANCE_THRESHOLD,
        metric="cosine",
        linkage="average",
    ).fit_predict(vectors)

    grouped: dict[int, list[Memory]] = {}
    for label, memory in zip(labels, memories, strict=True):
        grouped.setdefault(int(label), []).append(memory)

    clusters = []
    for members in grouped.values():
        if len(members) < MIN_CLUSTER_SIZE:
            continue
        candidate = Cluster(tuple(members))
        # The safeguard. Without it, "prefers async" and "prefers sync" cluster
        # together on cosine distance and consolidate into a confident average
        # of two opposite facts.
        if not candidate.shared_entities:
            continue
        clusters.append(candidate)
    return clusters


def render_cluster(cluster: Cluster) -> str:
    """What the model reads. Ids are included because `contradicts` refers to
    them — a cluster rendered without ids cannot express a contradiction."""
    return "\n".join(
        f"[{m.id}] ({m.valid_from.date().isoformat()}, confidence {m.confidence:.2f}) "
        f"{m.content}"
        for m in sorted(cluster.memories, key=lambda m: m.valid_from)
    )


class Consolidator:
    """§3.7. Nightly, and via `POST /admin/consolidate`."""

    def __init__(self, store, embedder, client, settings) -> None:
        self.store = store
        self.embedder = embedder
        self.client = client
        self.settings = settings

    async def consolidate(self, user_id: str) -> ConsolidationReport:
        # One query, not two. Clustering treats `vectors[i]` as `episodic[i]`,
        # and fetching them separately would make that alignment depend on two
        # queries happening to return rows in the same order — which an
        # unordered embedding query does not guarantee. The failure would be
        # silent: clusters of facts that are not similar, and a plausible
        # consolidated memory written from them.
        episodic, vectors = await self.store.live_with_vectors(user_id, "episodic")
        if len(episodic) < MIN_CLUSTER_SIZE or vectors.size == 0:
            return ConsolidationReport(0, 0, 0, len(episodic))

        clusters = cluster_memories(episodic, vectors)
        clustered_ids = {mid for c in clusters for mid in c.ids}

        written = superseded = 0
        for cluster in clusters:
            result = await self._call(cluster)
            if result is None:
                continue

            statement = str(result.get("statement") or "").strip()
            if not statement:
                continue

            memory = await self.store.write(
                user_id,
                "semantic",
                statement,
                embedding=self.embedder.embed([statement])[0],
                entities=list(result.get("entities") or []),
                confidence=_clamp(result.get("confidence")),
                created_by="consolidation",
                source_session_id=None,
                # §8.2: the episodic rows are the *evidence*, and are never
                # deleted — `source_ids` is only meaningful while they exist.
                source_ids=cluster.ids,
                trace_id=None,
            )
            written += 1

            for contradicted in result.get("contradicts") or []:
                # Only ever supersede a memory that was in this cluster. A model
                # naming an id from outside it — or inventing one — would
                # otherwise close a memory nothing here had evidence about.
                if contradicted not in set(cluster.ids):
                    log.warning(
                        "consolidation named %s as contradicted, but it was not "
                        "in the cluster; ignoring",
                        contradicted,
                    )
                    continue
                try:
                    await self.store.supersede(contradicted, by=memory.id)
                    superseded += 1
                except Exception:  # noqa: BLE001
                    log.warning("could not supersede %s", contradicted, exc_info=True)

        return ConsolidationReport(
            clusters_formed=len(clusters),
            memories_written=written,
            memories_superseded=superseded,
            facts_skipped=len(episodic) - len(clustered_ids),
        )

    async def _call(self, cluster: Cluster) -> dict | None:
        """One structured-output call per cluster, at `batch_effort` (§10)."""
        try:
            response = await self.client.messages.create(
                model=self.settings.agent_model,
                max_tokens=CONSOLIDATION_MAX_TOKENS,
                output_config={
                    "effort": self.settings.batch_effort,
                    "format": {"type": "json_schema", "schema": CONSOLIDATION_SCHEMA},
                },
                messages=[
                    {
                        "role": "user",
                        "content": CONSOLIDATION_PROMPT.format(
                            memories=render_cluster(cluster)
                        ),
                    }
                ],
            )
        except Exception:  # noqa: BLE001 — a batch job, never a request path
            log.warning("consolidation call failed", exc_info=True)
            return None

        if response.stop_reason == "refusal":
            log.warning("consolidation refused: %s", response.stop_details)
            return None
        text = next((b.text for b in response.content if b.type == "text"), "")
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            log.warning("consolidation returned unparseable output")
            return None


def _clamp(value) -> float:
    """`memories.confidence` has a CHECK constraint; a model returning 1.2
    would fail the write at the end of a paid call."""
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


CONSOLIDATION_MAX_TOKENS = 16000  # §10, non-streaming batch route
