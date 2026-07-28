# askstack

A memory-first agent that answers an engineering manager's questions about delivery state — what shipped, what's blocked, who's working on what, and why something was decided — grounded in a repository's pull requests, commits, issues, docs, and code.

Stateless agents restart cold and push every tool definition into every prompt. askstack inverts both: a **Memory Manager** loads the manager's standing context at session start across episodic, semantic, and procedural memory, and tool definitions live in procedural memory so only the relevant ones per query reach the prompt. An extraction → consolidation → write-back loop lets the agent refine its own memory over time, with provenance on every write and one-call rollback.

Status questions get a **structured facts layer** alongside the semantic index, because "what shipped last month" has an exact answer and similarity search cannot produce it. That also makes the eval unusually sharp: roughly 30 of the 50 golden-set questions have provably correct answers, scored by set-F1 with no judge in the loop. The rest is measured the usual way — an ablation matrix over hybrid retrieval × memory × tool retrieval, gated in CI against a committed baseline.

- **[`PRD.md`](./PRD.md)** — what we're building and why: problem, requirements, success metrics, evaluation approach, milestones.
- **[`TRD.md`](./TRD.md)** — how it's built: architecture, component interfaces, schema, retrieval and memory algorithms, API contract, performance budget, decision log.

## Quickstart

```bash
cp .env.example .env          # fill in ANTHROPIC_API_KEY, GITHUB_TOKEN
docker compose up -d          # Postgres + pgvector on :5432, Langfuse on :3000
```

Then open Langfuse at <http://localhost:3000>, create a project, and paste its keys into `.env`.

Ingest and the service arrive with M0/M2 — see the milestone table in the PRD.

## Layout

```
app/            FastAPI service
  memory/       Memory Manager: episodic, semantic, procedural
  retrieval/    hybrid dense + sparse retrieval, RRF fusion
  tools/        MCP tool registry and semantic tool retrieval
scripts/        corpus ingest, migrations, batch consolidation
evals/
  golden/       50-question golden set (frozen once scored)
  baselines/    committed metrics the CI gate compares against
ui/             three-pane web client
```
