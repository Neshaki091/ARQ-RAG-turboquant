-- ============================================================
-- ARQ-RAG TurboQuant — Supabase Database Schema
-- Run this in: Supabase Dashboard > SQL Editor
-- ============================================================

-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Documents table
--    Uses 768 dimensions for Google text-embedding-004
CREATE TABLE IF NOT EXISTS documents (
  id          BIGSERIAL PRIMARY KEY,
  title       TEXT NOT NULL,
  content     TEXT NOT NULL,
  category    TEXT NOT NULL DEFAULT 'general',
  emoji       TEXT NOT NULL DEFAULT '📄',
  embedding   VECTOR(768),
  metadata    JSONB NOT NULL DEFAULT '{}',
  created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 3. HNSW index for fast cosine similarity search
--    ef_construction=128, m=16 are good defaults for RAG
CREATE INDEX IF NOT EXISTS documents_embedding_hnsw_idx
  ON documents
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 128);

-- 4. Query log table (optional, for analytics)
CREATE TABLE IF NOT EXISTS query_logs (
  id            BIGSERIAL PRIMARY KEY,
  query_text    TEXT NOT NULL,
  retrieved_ids BIGINT[] DEFAULT '{}',
  latency_ms    INTEGER,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 5. Match documents function (used by RAG retrieval)
CREATE OR REPLACE FUNCTION match_documents (
  query_embedding  VECTOR(768),
  match_threshold  FLOAT   DEFAULT 0.3,
  match_count      INT     DEFAULT 5
)
RETURNS TABLE (
  id          BIGINT,
  title       TEXT,
  content     TEXT,
  category    TEXT,
  emoji       TEXT,
  metadata    JSONB,
  similarity  FLOAT
)
LANGUAGE SQL STABLE
AS $$
  SELECT
    d.id,
    d.title,
    d.content,
    d.category,
    d.emoji,
    d.metadata,
    1 - (d.embedding <=> query_embedding) AS similarity
  FROM documents d
  WHERE 1 - (d.embedding <=> query_embedding) > match_threshold
  ORDER BY similarity DESC
  LIMIT match_count;
$$;

-- 6. Updated_at trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS set_updated_at ON documents;
CREATE TRIGGER set_updated_at
  BEFORE UPDATE ON documents
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();
