#!/usr/bin/env python3
"""Does the embedding model explain the tool-recall collapse? — TRD §17 Q9.

    python evals/embedders.py                  # every candidate, every size
    python evals/embedders.py --sizes 13 500
    python evals/embedders.py --models BAAI/bge-small-en-v1.5

`evals/sweep.py` measured tool recall falling **0.820 → 0.530** as the catalog
grows from 13 to 500, with 57.6% of the top-5 taken by distractors. §17 Q9
named a suspect before that curve existed: `bge-small-en-v1.5` places gold
tools at median cosine 0.556 and non-gold at 0.510, a band far too narrow to
separate 500 things. ADR 2 put the `Embedder` protocol in place as the swap
point for exactly this moment.

That is an accusation, not an experiment. This module runs the experiment.

**Entirely in memory — no Postgres, no migration, no API key.** Selection is
`argsort(catalog_vectors @ query_vector)[:k]`, which is what the pgvector query
computes; doing it here rather than through the database buys three things.
A 768-dimension candidate needs no migration off the `VECTOR(384)` column, so
models of different widths are comparable at all. Exact cosine removes HNSW's
approximation as a confound, so a difference between models is a difference
between *models*. And a run costs nothing and repeats exactly.

The number that matters is **separation**, not raw similarity. A model whose
gold and non-gold bands overlap cannot rank tools however high its scores are,
and a model with lower absolute similarity but a clean gap will retrieve
correctly at 500 tools where `bge-small` does not.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.tools.registry import ALWAYS_INJECTED, CATALOG  # noqa: E402
from app.tools.selector import embedding_text  # noqa: E402
from scripts.gen_synthetic_tools import generate  # noqa: E402

QUESTIONS = Path(__file__).parent / "golden" / "questions.yaml"
DEFAULT_SIZES = (13, 50, 200, 500)
TOP_K = 5

# Each family embeds queries differently, and getting it wrong is a silent
# several-point recall loss rather than an error — which is the whole reason
# §3.1 gives the protocol two methods instead of a boolean flag. bge wants an
# instruction prefix on queries only; e5 wants `query:`/`passage:` on both
# sides; MiniLM and gte are symmetric and want neither. A comparison that
# applied bge's prefix to all four would be measuring the prefix.
CANDIDATES: dict[str, dict[str, str]] = {
    "BAAI/bge-small-en-v1.5": {
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
        "note": "incumbent, 384d",
    },
    "sentence-transformers/all-MiniLM-L6-v2": {
        "query_prefix": "",
        "passage_prefix": "",
        "note": "symmetric, 384d",
    },
    "thenlper/gte-small": {
        "query_prefix": "",
        "passage_prefix": "",
        "note": "symmetric, 384d",
    },
    "intfloat/e5-small-v2": {
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "note": "asymmetric, 384d",
    },
    "BAAI/bge-base-en-v1.5": {
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
        "note": "larger, 768d — needs a migration to adopt",
    },
}


@dataclass(frozen=True)
class Point:
    model: str
    top_k: int
    catalog_size: int
    recall: float
    crowd_out: float
    gold_median: float
    nongold_median: float
    separation: float
    overlap: float


def load_questions() -> list[dict]:
    return [
        q
        for q in yaml.safe_load(QUESTIONS.read_text())
        if q.get("gold_tools")
    ]


def encode(model, texts: list[str], prefix: str) -> np.ndarray:
    vectors = model.encode(
        [prefix + t for t in texts],
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vectors.astype(np.float32, copy=False)


def measure(
    model, spec: dict, questions: list[dict], size: int, top_k: int = TOP_K
) -> Point:
    """One (model, catalog size) cell. No database, no network."""
    tools = list(CATALOG) + generate(max(size - len(CATALOG), 0))
    names = [t.name for t in tools]
    synthetic = np.array([t.is_synthetic for t in tools])
    # §7.1's rendered template, not the raw schema — the same text the real
    # selector embeds, or this measures a different system.
    catalog = encode(model, [embedding_text(t) for t in tools], spec["passage_prefix"])
    queries = encode(model, [q["question"] for q in questions], spec["query_prefix"])

    similarity = queries @ catalog.T  # both L2-normalised, so this is cosine

    recalls, crowd = [], []
    gold_scores, nongold_scores = [], []
    for row, question in zip(similarity, questions, strict=True):
        gold = set(question["gold_tools"])
        gold_idx = {i for i, n in enumerate(names) if n in gold}
        # §7.2.2 — the always-injected pair is present regardless of query and
        # measures nothing about the selector.
        eligible = [i for i, n in enumerate(names) if n not in ALWAYS_INJECTED]

        ranked = sorted(eligible, key=lambda i: -row[i])[:top_k]
        found = len(gold_idx & set(ranked))
        recalls.append(found / len(gold_idx) if gold_idx else 1.0)
        crowd.append(sum(1 for i in ranked if synthetic[i]) / len(ranked))

        gold_scores.extend(float(row[i]) for i in gold_idx)
        nongold_scores.extend(
            float(row[i]) for i in eligible if i not in gold_idx
        )

    gold_median = statistics.median(gold_scores) if gold_scores else 0.0
    nongold_median = statistics.median(nongold_scores) if nongold_scores else 0.0
    # The §17 Q9 measurement, repeated per model: what fraction of non-gold
    # tools score at or above the median gold tool? A model that cannot rank
    # has this near 0.5 however high its absolute similarities are.
    overlap = (
        sum(1 for s in nongold_scores if s >= gold_median) / len(nongold_scores)
        if nongold_scores
        else 0.0
    )
    return Point(
        model=spec["_id"],
        top_k=top_k,
        catalog_size=len(tools),
        recall=round(sum(recalls) / len(recalls), 4),
        crowd_out=round(sum(crowd) / len(crowd), 4),
        gold_median=round(gold_median, 4),
        nongold_median=round(nongold_median, 4),
        separation=round(gold_median - nongold_median, 4),
        overlap=round(overlap, 4),
    )


def print_table(points: list[Point]) -> None:
    by_model: dict[str, list[Point]] = {}
    for p in points:
        by_model.setdefault(p.model, []).append(p)

    print(
        f"\n{'model':<38}{'size':>6}{'k':>4}{'recall':>9}{'crowd':>8}"
        f"{'gold':>8}{'non-gold':>10}{'sep':>8}{'overlap':>9}"
    )
    for model, rows in by_model.items():
        for i, p in enumerate(rows):
            label = model.split("/")[-1] if i == 0 else ""
            print(
                f"{label:<38}{p.catalog_size:>6}{p.top_k:>4}{p.recall:>9.3f}"
                f"{p.crowd_out:>8.3f}"
                f"{p.gold_median:>8.3f}{p.nongold_median:>10.3f}"
                f"{p.separation:>8.3f}{p.overlap:>9.3f}"
            )
        print()

    print("  recall at the largest catalog, best first:")
    largest = max(p.catalog_size for p in points)
    ranked = sorted(
        (p for p in points if p.catalog_size == largest),
        key=lambda p: -p.recall,
    )
    baseline = next((p for p in ranked if "bge-small-en" in p.model), None)
    for p in ranked:
        delta = ""
        if baseline and p.model != baseline.model:
            delta = f"  ({p.recall - baseline.recall:+.3f} vs incumbent)"
        print(f"    {p.recall:.3f}  sep={p.separation:+.3f}  {p.model}{delta}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--models", nargs="+", default=list(CANDIDATES))
    parser.add_argument(
        "--top-k", type=int, nargs="+", default=[TOP_K],
        help="sweep k. The separation hypothesis predicts k barely matters; the "
             "window hypothesis predicts recall recovers as k grows.",
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)

    questions = load_questions()
    print(f"{len(questions)} golden questions carry gold_tools", file=sys.stderr)

    from sentence_transformers import SentenceTransformer

    points: list[Point] = []
    for model_id in args.models:
        spec = dict(CANDIDATES.get(model_id, {"query_prefix": "", "passage_prefix": ""}))
        spec["_id"] = model_id
        print(f"  loading {model_id}…", file=sys.stderr, flush=True)
        try:
            model = SentenceTransformer(model_id)
        except Exception as exc:  # noqa: BLE001
            # One unavailable model must not lose the whole comparison.
            print(f"  skipped {model_id}: {exc}", file=sys.stderr)
            continue
        for size in args.sizes:
            for k in args.top_k:
                point = measure(model, spec, questions, size, k)
                points.append(point)
                print(
                    f"    size={point.catalog_size:>4} k={k:<3} "
                    f"recall={point.recall:.3f} sep={point.separation:+.3f}",
                    file=sys.stderr,
                    flush=True,
                )

    if not points:
        print("no models could be loaded", file=sys.stderr)
        return 1

    print_table(points)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "top_k": TOP_K,
                    "questions": len(questions),
                    "points": [p.__dict__ for p in points],
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
