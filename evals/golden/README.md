# The golden set

50 questions in [`questions.yaml`](./questions.yaml), plus 15 sealed until M5 in
[`heldout.yaml`](./heldout.yaml). Authored at M0, **before any agent code existed** —
PRD §7.1.

## Pinned revision

| | |
|---|---|
| Repository | `fastapi/fastapi` |
| Commit | `95f8322ee1dc98b1a4b0dd2ed52a5e2c1a0f2c48` |
| Commit date | **2026-07-29** |
| Pull request / commit ingest floor | 2025-01-01 |
| Issue ingest floor | none — full history |

Every question's `as_of` is at or before the commit date, and at or after the floor
that applies to its class. `build_gold.py --check` aborts on either violation: a
question anchored after the pin cannot be answered correctly by any system, and one
anchored before the floor would score against a truncated corpus. Both would read as
retrieval regressions rather than as authoring mistakes.

## The freeze rule

**No question may be edited once an agent run has been scored against it.** Changes
require a new `id`; the old one stays in the file.

This is what makes "authored before the agent existed" a real claim. Without it the
set drifts, one reasonable-looking edit at a time, toward whatever the agent already
does well — and the number it produces stops meaning anything.

The rule is also duplicated at the top of `questions.yaml`, because a rule that lives
only in a README is a rule people edit files without reading.

## Layout

| File | Authored by | Frozen |
|---|---|---|
| `questions.yaml` | hand | yes, once scored |
| `heldout.yaml` | hand | yes; opened once, at M5 |
| `gold_entities.yaml` | `build_gold.py` | no — regenerate freely |
| `heldout_entities.yaml` | `build_gold.py` | no |

TRD §14.1 describes `build_gold.py` as writing generated sets "into the golden set".
They are written *beside* it instead. Entity sets are a pure function of
(query spec, pinned corpus), so regenerating them is routine — and routine
regeneration must never rewrite the artifact the freeze rule protects. Splitting the
files turns that from a convention into a file boundary.

## Composition

| Class | Count | Scored by | Ground truth |
|---|--:|---|---|
| 1 — delivery status | 8 | aggregate set-F1 | generated |
| 2 — change over time | 9 | aggregate set-F1 | generated |
| 3 — ownership and activity | 7 | aggregate set-F1 | generated |
| 4 — blockers and risk | 6 | aggregate set-F1 | generated |
| 5 — decision archaeology | 12 | recall@5, MRR@10 | hand-authored |
| 6 — scope and existence | 8 | recall@5, MRR@10 | hand-authored |

### How the interpretive questions were authored

`gold_chunks` for classes 5–6 were selected by **reading issue threads**, never by
querying the retriever. Candidates were surfaced from the facts layer on structural
signal alone — comment volume, and the `feature` / `bug` / `investigate` labels — and
then read. High-volume support requests were skipped: issues 2266 and 2269 have 200
chunks each but are titled *"Not able to create exe file"*, which is not decision
archaeology.

Citing whatever our own retrieval returns would make recall@5 measure itself. This is
the slow part of authoring, and it is slow on purpose.

## Question shapes excluded by construction

PRD §4 already sets this precedent: repo-native sources contain no sprints, so
*"what shipped last sprint"* is excluded rather than left to fail silently. Two more
shapes fail the same way against this corpus:

| Shape | Why | Measured |
|---|---|---|
| Milestone filters | FastAPI sets no milestones | 0 of 2,325 PRs, 0 of 3,542 issues |
| Open-issue questions | support moved to GitHub Discussions years ago | exactly 1 open issue |

Class 4 therefore leans on stale pull requests and labels. `FactsStore.open_issues`
and its `milestone=` parameter remain correct code with no data behind them *here*;
they are not dead, they are unexercised by this corpus.

## One question is deliberately unanswerable

`h011` asks which authentication pull requests merged in the last three months. The
answer is **none** — every auth-area change merged in February or March 2026. It
carries `expect_empty: true`.

This is PRD §5.2 in its purest form. A system that invents plausible work, or quietly
widens the window until it finds something, fails outright. **The scorer must treat an
empty prediction against an empty gold set as a perfect score**, not as a division by
zero.

## Verification log — TRD §14.1

§14.1 is candid about the weakness of generated ground truth: the scorer and the
answers share an implementation, so a bug in the SQL would be invisible. The required
mitigation is an independent check.

Each answer below was cross-checked against GitHub's **search API** — a different
endpoint from the one `scripts/ingest.py` walks, so a bug in our pagination or
filtering cannot hide in both. Run on **2026-07-30**, against the pinned revision.

### Window queries

| Question | Window | Ours | GitHub search | |
|---|---|--:|--:|:--:|
| q009 | 2026-07-22 → 07-29 | 42 | 42 | ✅ |
| q010 | 2026-06-29 → 07-29 | 116 | 116 | ✅ |
| q011 | June 2026 | 93 | 93 | ✅ |
| q012 | Q1 2026 | 234 | 234 | ✅ |
| q017 | all of 2025 | 545 | 545 | ✅ |

### Entity states

| Question | Entity | Ours | GitHub | |
|---|---|---|---|:--:|
| q001 | `pr:16105` | merged | merged | ✅ |
| q002 | `pr:15937` | merged | merged | ✅ |
| q003 | `pr:16102` | merged | merged | ✅ |
| q006 | `pr:15806` | **closed** | **closed** | ✅ |
| q007 | `pr:5718` | open | open | ✅ |
| q008 | `pr:15093` | merged | merged | ✅ |

**q006 is the trap.** Pull request 15806 is titled *"🔖 Release version 0.138.0"* and
was closed without merging. A system that reads the title, or that treats "closed" as
"done", reports shipped work that never shipped — the defining failure of PRD §5.2.

### Corpus-level counts, checked at ingest (step 2)

| | Ours | GitHub search | |
|---|--:|--:|:--:|
| Merged PRs since 2025-01-01 | 1,092 | 1,092 | ✅ |
| Closed issues, all time | 3,541 | 3,541 | ✅ |

## Regenerating

```bash
python evals/build_gold.py --check              # validate, write nothing
python evals/build_gold.py                      # regenerate gold_entities.yaml
python evals/build_gold.py --heldout            # regenerate heldout_entities.yaml
pytest tests/test_golden_set.py
```

Regenerating twice must produce a byte-identical file. If it does not, something in
the query path is non-deterministic and every baseline built on it is unreliable.
`tests/test_golden_set.py` asserts this, along with the schema, the date anchoring,
and that no two questions share an identical answer set — duplicated ground truth
silently double-counts whatever it tests.
