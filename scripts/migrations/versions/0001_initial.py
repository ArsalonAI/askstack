"""initial schema — TRD §4

Transcribed as raw SQL rather than SQLAlchemy DDL on purpose: the schema is
already specified as SQL in the TRD, and re-expressing it in a second dialect
produces a diff nobody can eyeball against the doc.

Revision ID: 0001
Revises:
Create Date: 2026-07-29
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# The `vector` and `pg_trgm` extensions are owned by scripts/init_db.sql,
# which the Postgres image runs once on first boot. Creating them here would
# need superuser on every environment, so we check instead — and fail with an
# instruction rather than a bare "type vector does not exist" 40 lines later.
REQUIRED_EXTENSIONS = ("vector", "pg_trgm")


CORPUS = """
CREATE TABLE chunks (
    id            TEXT PRIMARY KEY,              -- = citation, stable across re-ingest
    source        TEXT NOT NULL CHECK (source IN ('docs','code','issue')),
    path          TEXT NOT NULL,
    anchor        TEXT NOT NULL,
    content       TEXT NOT NULL,
    content_sha   TEXT NOT NULL,
    token_count   INT  NOT NULL,
    embedding     VECTOR(384) NOT NULL,
    tsv           TSVECTOR NOT NULL,
    ingested_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX chunks_embedding_idx ON chunks
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX chunks_tsv_idx    ON chunks USING gin (tsv);
CREATE INDEX chunks_source_idx ON chunks (source);          -- source filter pushdown
CREATE INDEX chunks_sha_idx    ON chunks (content_sha);     -- ingest delta detection
"""

FACTS = """
-- Entities, not text. Every column here exists because some EM question
-- filters or aggregates on it.

CREATE TABLE pull_requests (
    number      INT PRIMARY KEY,
    title       TEXT NOT NULL,
    body        TEXT,
    state       TEXT NOT NULL CHECK (state IN ('open','merged','closed')),
    is_draft    BOOLEAN NOT NULL DEFAULT FALSE,
    author      TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL,
    merged_at   TIMESTAMPTZ,               -- NULL unless state = 'merged'
    closed_at   TIMESTAMPTZ,
    milestone   TEXT,
    additions   INT NOT NULL DEFAULT 0,
    deletions   INT NOT NULL DEFAULT 0,
    url         TEXT NOT NULL
);
-- Partial index: "what merged in a window" is the single hottest query, and
-- merged PRs are a minority of the table.
CREATE INDEX pr_merged_at_idx ON pull_requests (merged_at DESC)
    WHERE state = 'merged';
CREATE INDEX pr_state_idx     ON pull_requests (state, created_at DESC);
CREATE INDEX pr_author_idx    ON pull_requests (author, created_at DESC);
CREATE INDEX pr_milestone_idx ON pull_requests (milestone) WHERE milestone IS NOT NULL;

CREATE TABLE pr_files (
    pr_number  INT NOT NULL REFERENCES pull_requests(number) ON DELETE CASCADE,
    path       TEXT NOT NULL,
    additions  INT NOT NULL DEFAULT 0,
    deletions  INT NOT NULL DEFAULT 0,
    PRIMARY KEY (pr_number, path)
);
CREATE INDEX pr_files_path_idx ON pr_files (path text_pattern_ops);  -- area glob prefix match

CREATE TABLE pr_reviews (
    pr_number    INT NOT NULL REFERENCES pull_requests(number) ON DELETE CASCADE,
    reviewer     TEXT NOT NULL,
    state        TEXT NOT NULL,            -- approved | changes_requested | commented
    submitted_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (pr_number, reviewer, submitted_at)
);
CREATE INDEX pr_reviews_recent_idx ON pr_reviews (pr_number, submitted_at DESC);

CREATE TABLE commits (
    sha         TEXT PRIMARY KEY,
    author      TEXT NOT NULL,
    authored_at TIMESTAMPTZ NOT NULL,
    message     TEXT NOT NULL,
    pr_number   INT REFERENCES pull_requests(number) ON DELETE SET NULL
);
CREATE INDEX commits_author_idx ON commits (author, authored_at DESC);
CREATE INDEX commits_date_idx   ON commits (authored_at DESC);

CREATE TABLE commit_files (
    sha   TEXT NOT NULL REFERENCES commits(sha) ON DELETE CASCADE,
    path  TEXT NOT NULL,
    PRIMARY KEY (sha, path)
);
CREATE INDEX commit_files_path_idx ON commit_files (path text_pattern_ops);

CREATE TABLE issues (
    number        INT PRIMARY KEY,
    title         TEXT NOT NULL,
    body          TEXT,
    state         TEXT NOT NULL CHECK (state IN ('open','closed')),
    author        TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL,
    closed_at     TIMESTAMPTZ,
    closed_by_pr  INT REFERENCES pull_requests(number) ON DELETE SET NULL,
    milestone     TEXT,
    url           TEXT NOT NULL
);
CREATE INDEX issues_state_idx     ON issues (state, created_at);
CREATE INDEX issues_milestone_idx ON issues (milestone) WHERE milestone IS NOT NULL;

CREATE TABLE issue_labels (
    issue_number INT NOT NULL REFERENCES issues(number) ON DELETE CASCADE,
    label        TEXT NOT NULL,
    PRIMARY KEY (issue_number, label)
);
CREATE INDEX issue_labels_label_idx ON issue_labels (label);

CREATE TABLE releases (
    tag          TEXT PRIMARY KEY,
    name         TEXT,
    published_at TIMESTAMPTZ NOT NULL,
    body         TEXT,
    url          TEXT NOT NULL
);
CREATE INDEX releases_date_idx ON releases (published_at DESC);

-- Curated, not derived. See TRD §5.5.
CREATE TABLE areas (
    name       TEXT PRIMARY KEY,
    path_globs TEXT[] NOT NULL,
    description TEXT
);
"""

SESSIONS = """
CREATE TABLE sessions (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at    TIMESTAMPTZ,
    metadata    JSONB NOT NULL DEFAULT '{}'
);
CREATE INDEX sessions_user_idx ON sessions (user_id, started_at DESC);

CREATE TABLE messages (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    turn        INT  NOT NULL,
    role        TEXT NOT NULL CHECK (role IN ('user','assistant','tool')),
    content     JSONB NOT NULL,                  -- Anthropic content blocks, verbatim
    trace_id    TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (session_id, turn, role)
);
CREATE INDEX messages_session_idx ON messages (session_id, turn);
"""

MEMORY = """
CREATE TABLE memories (
    id                TEXT NOT NULL,             -- stable across revisions
    revision          INT  NOT NULL,
    user_id           TEXT NOT NULL,
    mem_type          TEXT NOT NULL CHECK (mem_type IN ('episodic','semantic','procedural')),
    content           TEXT NOT NULL,
    embedding         VECTOR(384) NOT NULL,
    entities          TEXT[] NOT NULL DEFAULT '{}',
    confidence        REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    valid_from        TIMESTAMPTZ NOT NULL DEFAULT now(),
    valid_to          TIMESTAMPTZ,               -- NULL = live
    superseded_by     TEXT,                      -- memories.id that replaced this
    created_by        TEXT NOT NULL
        CHECK (created_by IN ('extraction','consolidation','agent','human')),
    source_session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL,
    source_ids        TEXT[] NOT NULL DEFAULT '{}',  -- episodic ids, message ids
    trace_id          TEXT,
    PRIMARY KEY (id, revision)
);

-- The hot path: live memories for one user. Partial index keeps it small
-- even as revision history grows without bound.
CREATE INDEX memories_live_idx ON memories (user_id, mem_type)
    WHERE valid_to IS NULL;
CREATE INDEX memories_embedding_idx ON memories
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX memories_entities_idx ON memories USING gin (entities);
CREATE INDEX memories_session_idx  ON memories (source_session_id);

CREATE TABLE memory_audit (
    id          BIGSERIAL PRIMARY KEY,
    memory_id   TEXT NOT NULL,
    revision    INT  NOT NULL,
    op          TEXT NOT NULL CHECK (op IN ('create','supersede','revert')),
    before      JSONB,
    after       JSONB NOT NULL,
    actor       TEXT NOT NULL,
    trace_id    TEXT,
    at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX memory_audit_memory_idx ON memory_audit (memory_id, at DESC);
CREATE INDEX memory_audit_trace_idx  ON memory_audit (trace_id);
"""

TOOLS = """
CREATE TABLE tool_defs (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,
    description   TEXT NOT NULL,
    input_schema  JSONB NOT NULL,
    server        TEXT NOT NULL,
    is_synthetic  BOOLEAN NOT NULL DEFAULT FALSE,
    embedding     VECTOR(384) NOT NULL
);
CREATE INDEX tool_defs_embedding_idx ON tool_defs
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX tool_defs_synthetic_idx ON tool_defs (is_synthetic);
"""

EVAL = """
CREATE TABLE eval_runs (
    id           TEXT PRIMARY KEY,
    config_hash  TEXT NOT NULL,
    git_sha      TEXT NOT NULL,
    metrics      JSONB NOT NULL,
    started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at  TIMESTAMPTZ
);
CREATE INDEX eval_runs_config_idx ON eval_runs (config_hash, started_at DESC);
"""

INGEST = """
-- The completion marker TRD §4.2 requires: "a partial ingest leaves the
-- marker unset and the service refuses to start." `completed_at` is set only
-- once BOTH the facts layer and the semantic index have been written, which
-- is what keeps the two substrates from disagreeing about what exists.
CREATE TABLE ingest_runs (
    id              TEXT PRIMARY KEY,
    corpus_repo     TEXT NOT NULL,
    corpus_ref      TEXT NOT NULL,        -- symbolic ref as requested
    resolved_sha    TEXT NOT NULL,        -- the pinned revision actually ingested
    since           TIMESTAMPTZ,          -- ingest window floor; NULL = full history
    embedding_model TEXT NOT NULL,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at    TIMESTAMPTZ,          -- NULL = partial
    stats           JSONB NOT NULL DEFAULT '{}'
);
CREATE INDEX ingest_runs_complete_idx ON ingest_runs (completed_at DESC)
    WHERE completed_at IS NOT NULL;
"""

# Reverse dependency order.
DROP_ORDER = (
    "ingest_runs",
    "eval_runs",
    "tool_defs",
    "memory_audit",
    "memories",
    "messages",
    "sessions",
    "areas",
    "releases",
    "issue_labels",
    "issues",
    "commit_files",
    "commits",
    "pr_reviews",
    "pr_files",
    "pull_requests",
    "chunks",
)


def _require_extensions() -> None:
    conn = op.get_bind()
    installed = {
        row[0]
        for row in conn.exec_driver_sql(
            "SELECT extname FROM pg_extension WHERE extname = ANY(%s)",
            (list(REQUIRED_EXTENSIONS),),
        )
    }
    missing = [e for e in REQUIRED_EXTENSIONS if e not in installed]
    if missing:
        raise RuntimeError(
            f"missing Postgres extension(s): {', '.join(missing)}. "
            "They are created by scripts/init_db.sql, which the Postgres image runs "
            "on first boot. On a hand-rolled server, run it as superuser first."
        )


def upgrade() -> None:
    _require_extensions()
    for block in (CORPUS, FACTS, SESSIONS, MEMORY, TOOLS, EVAL, INGEST):
        op.execute(block)


def downgrade() -> None:
    for table in DROP_ORDER:
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
