# PRD — askstack

A memory-first agent that answers an engineering manager's questions about delivery state, grounded in the repository.

**Status:** Draft · **Last updated:** 2026-07-28 · **Technical design:** [`TRD.md`](./TRD.md)

---

## 1. Problem

An engineering manager spends a meaningful share of every week reconstructing state that already exists in the repository: what shipped, what slipped, what's blocked on review, who's carrying which workstream, and why a decision was made six weeks ago. The information is all there. Assembling it means reading PR lists, cross-referencing issues, and asking people.

An agent could do this, except that stateless agents make it worse in two specific ways:

- **No persistence.** Every session starts cold. The manager re-establishes which workstreams they care about, who's on the team, and what they asked last week. The assistant that needs re-briefing every Monday is not saving anyone time.
- **Flat tool exposure.** Every tool definition is pushed into the prompt on every request. Prompt size and tool-selection error both grow with catalog size, so adding capability degrades the agent.

askstack treats **memory as the primary architecture**, not an add-on. A Memory Manager loads the manager's standing context at session start — their workstreams, their team, what they last asked and what the answer was. Tool definitions live in memory and are retrieved per query rather than broadcast.

**Thesis, to be proven with numbers rather than asserted:** persisted memory reduces turns-to-success across sessions, and semantic tool retrieval keeps tool-selection accuracy and prompt size flat as the tool catalog grows.

## 2. Goals / Non-goals

**Goals**

1. A Memory Manager orchestrating episodic, semantic, and procedural memory, loading a bounded context block at session start.
2. A procedural tool-retrieval layer: tool definitions stored as memory, only the relevant ones reaching the model.
3. An autonomous memory lifecycle — extraction, consolidation, write-back — where every write is attributable and reversible.
4. A 50-question golden set authored **before the agent exists**, plus an ablation harness gated in CI.
5. End-to-end tracing with per-query cost and latency attributable to each pipeline stage.

**Non-goals**

- Multi-tenant auth, RBAC, production hardening. Single user per `user_id`, no login.
- Real-time freshness. Repository ingest is a batch job, re-run on demand.
- Model fine-tuning or training. Everything is retrieval plus prompting.
- Beating a SOTA RAG benchmark. The deliverable is an honest measured comparison of *our own* configurations.
- Writing to the repository. askstack reads and reports; it never opens a PR, comments, or changes state.

## 3. User

One user: **an engineering manager responsible for a codebase they do not write in daily.** They need current, verifiable delivery state and the reasoning behind past decisions.

They are not the person implementing the code. Answers must not assume they will read a diff to check the claim.

### Scenarios

1. **The Monday check-in.** *"What's changed since we last spoke?"* — the agent knows when the last session was, and reports what merged, what closed, and what went stale in that window. **This scenario is unanswerable without memory**, which makes it the sharpest test of the whole thesis.
2. **Delivery status.** *"Did the auth migration ship?"* — answered from merge state, not from a document claiming it was planned.
3. **Blocked work.** *"Which PRs have been sitting in review over two weeks?"*
4. **Ownership.** *"Who's been working on the routing layer this quarter?"*
5. **Decision archaeology.** *"Why did we drop the sync client?"* — answered from issue and PR discussion, cited to the thread where it was argued.
6. **Correction.** The agent remembered a workstream wrong. The manager finds it in the memory panel, sees which session produced it, and reverts it.

## 4. Scope

One repository, ingested at a pinned revision. Default target is **FastAPI** (`fastapi/fastapi`, formerly `tiangolo/fastapi`) — active development, real milestones and releases, and a public PR and issue history deep enough to ask real status questions of. Configurable; any active repository works.

**Primary sources** are the delivery record: pull requests, commits, issues, labels, milestones, and releases.
**Supporting sources** are documentation and source code — used to confirm that something claimed to exist actually does.

This is an inversion of how a code-search tool is usually built, and it is deliberate. The manager's question is rarely "how does this work"; it is "is it done, and who did it."

**In scope:** one repository at a pinned revision, ingested as a batch job.
**Out of scope:** multiple repositories, external trackers (Linear, Jira), calendars, chat, and CI systems.

**A limit worth stating plainly:** repo-native sources contain no sprints. *"What shipped last sprint"* is not answerable and is excluded from evaluation by construction rather than failed silently. *"Since v0.110"* and *"in the last 30 days"* are answerable, because releases and dates are in the repository.

## 5. Product requirements

### 5.1 Question classes

The product must answer six classes. The distinction in the right-hand column drives how each is evaluated.

| # | Class | Example | Exactly checkable? |
|---|---|---|:--:|
| 1 | Delivery status | "Did the auth migration ship?" | ✅ |
| 2 | Change over time | "What shipped in the last month?" | ✅ |
| 3 | Ownership and activity | "Who's been working on routing?" | ✅ |
| 4 | Blockers and risk | "Which PRs are stale?" | ✅ |
| 5 | Decision archaeology | "Why did we drop the sync client?" | ✗ |
| 6 | Scope and existence | "Do we support websockets?" | partly |

Classes 1–4 have answers that are facts about the repository at the pinned revision — the exact set of pull requests merged in a window is not a matter of judgment. Classes 5–6 are interpretive. §6 gates the first group and reports on the second.

### 5.2 Verified status

**An answer about delivery state must reflect what actually merged, never what was merely proposed.** Reporting an open pull request as shipped work is the defining failure of this product — it is worse than returning nothing, because the manager acts on it.

Class 6 is where this bites hardest. "Do we support websockets?" can be answered affirmatively from a design document or an unmerged branch. The requirement is that any existence claim is confirmed against merge state before it is asserted.

### 5.3 Continuity across sessions

A session opens with the manager's standing context already loaded: workstreams they track, people they ask about, and what they asked previously along with the answer they got. The agent must not re-ask what it has been told. Continuity is bounded — a memory context that grows without limit crowds out the actual question.

### 5.4 Capability scaling

Adding tools must not degrade the agent. As the catalog grows from tens to hundreds of tools, tool-selection accuracy and prompt size must both stay approximately flat, measured against two baselines: exposing every tool, and the model provider's own tool-search feature.

### 5.5 Memory governance

Memory is written autonomously, which is only defensible if it is fully accountable. Every memory must be:

- **Inspectable** — visible to the user, with its content and confidence.
- **Attributable** — traceable to the session, and the specific request, that produced it.
- **Reversible** — restorable to any prior state in one action, with the reversal itself recorded.

Memory poisoning is the known failure mode of self-modifying agents. These three properties make autonomy recoverable rather than a one-way door.

### 5.6 Transparency

The interface surfaces the machinery, not just the answer. In one view the manager can see which memories were loaded and why, which tools were selected out of the full catalog, which sources were consulted, and which of those the answer actually cited.

This is a product requirement, not a debug affordance. A manager acting on a status report needs to see the evidence behind it — an unsourced claim about delivery state is a rumour with better formatting.

## 6. Success metrics

| Metric | What it measures | Applies to | Gated in CI? |
|---|---|---|:--:|
| Aggregate set-F1 | Did it return exactly the right pull requests, issues, or commits | classes 1–4 | ✅ |
| recall@5 | Did retrieval surface the right discussion | classes 5–6 | ✅ |
| MRR@10 | Did it rank it first | classes 5–6 | ✅ |
| Tool-selection accuracy | Was the right tool chosen from the catalog | all | ✅ |
| Citation resolution rate | Does every citation resolve to something actually consulted | all | ✅ |
| Citation grounding rate | Do the citations genuinely support the claims | all | report-only |
| Answer coverage | Are the expected points present | all | report-only |
| **Turns to success** | **Does memory make the manager faster across sessions** | cross-session suite | report-only |
| p95 latency, cost per query | Is it usable and affordable | all | report-only |

**Why some metrics gate and others do not.** The report-only metrics are LLM-judged and therefore non-deterministic; a hard gate on them produces flaky builds, and a red build nobody trusts is a gate that gets disabled. The gated metrics are mechanically computable and reproducible.

**Aggregate set-F1 is the strongest of these.** Because classes 1–4 have exactly checkable answers, ground truth is computed from the pinned revision rather than judged — precision and recall over the returned set, no model in the loop. That directly measures §5.2: an answer that includes an unmerged pull request loses precision, deterministically and every time.

**Turns to success is the headline.** It is report-only because the scenario suite is small enough that a hard threshold would be noise, but it is the number that decides whether the memory architecture was worth building.

## 7. Evaluation approach

### 7.1 Golden set

50 questions authored in M0, **before any agent code exists** — roughly 30 from classes 1–4 with exact ground truth, and 20 from classes 5–6 scored against source spans.

**Freeze rule:** no question may be edited after an agent run has been scored against it. Edits require a new ID. This is what makes "authored before the agent existed" a real claim rather than a slogan — without it, the set drifts toward whatever the agent already does well.

**Every question is anchored to a date.** "What shipped last month" resolves differently every week, so each question records the date it is asked *as of*, and the whole set is evaluated against one pinned repository revision. A question anchored after the pinned revision is a hard error in the eval runner, not a silent wrong answer.

A 15-question held-out set is authored at the same time and scored once, at M5, as an overfitting check.

### 7.2 Cross-session scenarios

10 scripted scenarios of 3 sessions each, modelled on recurring status check-ins: the manager asks about a workstream, comes back later, and asks what moved. The third session is only answerable efficiently if the first two were remembered. Run with memory on and off, measuring task success, turns to success, and cost per completed task.

### 7.3 Ablation axes

Three independent axes, each isolating one architectural claim:

| Axis | Arms | Claim under test |
|---|---|---|
| Hybrid retrieval | on / off | Combining dense and sparse beats dense alone |
| Memory | on / off | Persistence improves outcomes |
| Tool retrieval | semantic / provider-native / full exposure | Retrieved tools beat broadcasting them |

Plus a scaling sweep across catalog sizes, which is the plot that carries §5.4.

### 7.4 CI policy

Per pull request, the default configuration runs against the full golden set and is compared to a committed baseline. Any gated metric falling outside tolerance fails the build. The full ablation matrix and scaling sweep run nightly and never block.

Baselines change only through an explicit pull request that states why. A regression must never be absorbable by regenerating the baseline in the commit that caused it.

## 8. Milestones

| # | Milestone | Exit criteria |
|---|---|---|
| M0 | Corpus + golden set | Both the searchable index and the delivery record are ingested; 50 questions committed, date-anchored, and frozen |
| M1 | Retrieval + eval harness | Metrics reported for both question groups; baseline committed |
| M2 | Agent + tools + tracing | Cited status answers stream end to end; traces complete |
| M3 | Memory Manager | All three memory types load at session start, within budget |
| M4 | Lifecycle | Extraction, consolidation, write-back, provenance, revert |
| M5 | Ablations + CI | Matrix reproducible; PR gate live; scaling curve plotted; held-out set scored |
| M6 | UI | Transparency view per §5.6 |

## 9. Risks

| Risk | Mitigation |
|---|---|
| A confidently wrong status report | §5.2 — existence and delivery claims are confirmed against merge state; aggregate set-F1 penalises it deterministically |
| Memory poisoning degrades later sessions | §5.5 governance — provenance on every write, contradiction-aware consolidation, one-action revert |
| The mapping from area names to code paths is curated, so it can be wrong | Kept small and human-reviewed; ownership answers name the paths they resolved, so a bad mapping is visible rather than silent |
| Judge nondeterminism makes CI flaky | Judged metrics are report-only; gates use mechanical metrics only |
| Golden-set overfitting | Freeze rule plus a held-out set scored once, at M5 |
| Date-anchored answers rot | One pinned revision for the whole set; the runner fails on any question anchored past it |
| Synthetic tools inflate the tool-retrieval result | Real/synthetic split published with every number; accuracy also reported over real tools alone |
| Evaluation cost grows unbounded | Full matrix runs nightly on a budget, not per PR |

## 10. Open questions

1. Are the 10 cross-session scenarios hand-authored or generated from real repository history? Hand-authored is more defensible; generation is faster.
2. Does the tool-scaling sweep need to reach 1000 tools, or does 500 already make the point?

Technical open questions are tracked in [`TRD.md`](./TRD.md) §17.
