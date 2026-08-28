# askstack

[![tests](https://github.com/ArsalonAI/askstack/actions/workflows/tests.yml/badge.svg)](https://github.com/ArsalonAI/askstack/actions/workflows/tests.yml)
[![eval-sweep](https://github.com/ArsalonAI/askstack/actions/workflows/eval-sweep.yml/badge.svg)](https://github.com/ArsalonAI/askstack/actions/workflows/eval-sweep.yml)

A memory-first agent that answers an engineering manager's questions about delivery state — what shipped, what's blocked, who's working on what, and why something was decided — grounded in a repository's pull requests, commits, issues, docs, and code.

Stateless agents restart cold and push every tool definition into every prompt. askstack inverts both: a **Memory Manager** loads the manager's standing context at session start across episodic, semantic, and procedural memory, and tool definitions live in procedural memory so only the relevant ones per query reach the prompt. An extraction → consolidation → write-back loop lets the agent refine its own memory over time, with provenance on every write and one-call rollback.

Status questions get a **structured facts layer** alongside the semantic index, because "what shipped last month" has an exact answer and similarity search cannot produce it. That also makes the eval unusually sharp: roughly 30 of the 50 golden-set questions have provably correct answers, scored by set-F1 with no judge in the loop. Three ablation axes isolate the architectural claims — hybrid retrieval, memory, and tool retrieval — and two of the three are measured **without a model in the loop at all**, deterministically and at no cost ([ADR 18](./TRD.md#16-adr-log)).

The headline finding so far is a split verdict on the tool-retrieval thesis: across a 38× catalog the prompt stays **flat at 1,089 tokens** while full exposure reaches 82,615, and tool recall **falls from 0.820 to 0.530** as distractors crowd the top-`k`. Cheap prompts, half the right answers. [§14.5](./TRD.md) has the curve and the mechanism.

- **[`PRD.md`](./PRD.md)** — what we're building and why: problem, requirements, success metrics, evaluation approach, milestones.
- **[`TRD.md`](./TRD.md)** — how it's built: architecture, component interfaces, schema, retrieval and memory algorithms, API contract, performance budget, decision log.

## Quickstart

```bash
cp .env.example .env          # fill in ANTHROPIC_API_KEY, GITHUB_TOKEN
docker compose up -d          # Postgres + pgvector on :5432, Langfuse on :3000
uv sync --extra dev
uv run alembic -c scripts/migrations/alembic.ini upgrade head
```

Then open Langfuse at <http://localhost:3000>, create a project, and paste its keys into `.env`.

<details>
<summary>Without Docker</summary>

Against an existing Postgres 16+ with [pgvector](https://github.com/pgvector/pgvector) installed (`brew install pgvector`):

```bash
createdb askstack
psql -d askstack -f scripts/init_db.sql   # extensions; needs superuser
```

Point `DATABASE_URL` at it and run the migration as above. `pytest` uses a separate `askstack_test` database and creates it on first run — the schema tests downgrade to base on teardown, so they must never share a database with a real corpus.

</details>

## Running it

Ingest the corpus, embed the tool catalog, then start the service:

```bash
python scripts/ingest.py                  # ~30 min cold; re-runs skip unchanged chunks
python scripts/seed_tools.py              # tool embeddings for semantic selection
uv run uvicorn app.main:app               # http://localhost:8000
```

Ask it something. The response is an SSE stream, not JSON — every step of the
turn is an event, and the transparency view (PRD §5.6) is built on them:

```bash
curl -N localhost:8000/chat -H 'content-type: application/json' \
  -d '{"user_id":"you","session_id":null,"message":"Did pull request 15806 ship?"}'
```

`GET /healthz` reports what is actually reachable — corpus size, the pinned
revision every answer is dated against, whether a model key and Langfuse are
configured.

## Evaluating

Most of the eval surface needs no API key, and that is a design decision rather
than a convenience — a measurement runs through the agent only when the agent
is what is under test (ADR 18). Hybrid retrieval is a retriever configuration;
tool selection is a pgvector query against a local embedding model; prompt cost
is a property of the catalog. None of them get more true by being observed
through a paid agent turn.

**Free — no key, no network, deterministic:**

```bash
python evals/build_gold.py --check        # ground truth still reproduces from the pin
python evals/runner.py                    # retrieval: recall@5, MRR@10, set-F1
python evals/runner.py --question q033 --explain
python evals/sweep.py                     # the tool-scaling curve (§14.5)
python scripts/gen_synthetic_tools.py --size 200   # pad the catalog for the curve
```

**Paid — needs `ANTHROPIC_API_KEY`:**

```bash
python evals/runner.py --agent            # the full agent turn, 50 questions
python evals/scenarios.py                 # cross-session, memory on vs off
python evals/scenarios.py --check         # validate the suite, spends nothing
```

The two runner paths are deliberate. `--agent` measures the system a user meets
and is what the committed baseline gates on; the default path calls the
retriever and the facts layer directly, so a regression there is visible
without an agent run in the way. See TRD §14.1.

Both paid harnesses **refuse to write a baseline from a run in which anything
errored**. This is not defensive coding — it is the same failure twice: an
agent run that exhausted its credit balance partway through would otherwise
record the questions it never reached as zeros, and a cross-session run that
did the same would record "memory made the system worse" when what actually
happened was billing.

### The tool-scaling curve

`evals/sweep.py` measures both halves of the §5.4 claim at 13 / 50 / 200 / 500
tools, in about ninety seconds, for nothing:

| catalog | tool recall | crowd-out | real tools offered (k=5) | semantic prompt | full prompt |
|---:|---:|---:|---:|---:|---:|
| 13 | 0.820 | 0.000 | 4.48 | 1,089 | 2,148 |
| 500 | 0.530 | 0.576 | 1.94 | 1,089 | 82,615 |

The cost claim holds and the accuracy claim does not. The prompt stays flat
across a 38× catalog — 81,526 input tokens saved per request at 500 tools — but
recall falls from 0.820 to 0.530, because 57.6% of the top-5 becomes
distractors and the model is offered 1.94 real tools where `k` promised 5.
TRD §14.5 has the mechanism and §17 Q9 the diagnosis.

## Layout

```
app/            FastAPI service
  memory/       Memory Manager, versioned store, transcript extraction
  retrieval/    hybrid dense + sparse retrieval, RRF fusion
  tools/        MCP tool registry and semantic tool retrieval
scripts/        corpus ingest, migrations, synthetic tool padding
evals/
  golden/       50-question golden set (frozen once scored)
  scenarios/    10 cross-session check-ins (PRD §7.2)
  baselines/    committed metrics: main, retrieval, sweep, scenarios
ui/             three-pane web client
```
