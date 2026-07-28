-- Runs once on first `docker compose up`. Schema itself lands with migrations (M1).
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
