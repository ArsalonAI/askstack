-- Runs once on first `docker compose up`. Schema itself lands with migration
-- 0001 (M0) -- M0's exit criteria is an ingested corpus, which needs tables.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
