-- ============================================================================
-- Migration: Tạo mới bảng documents với VECTOR(1536)
-- Sử dụng text-embedding-004 (1536 dimensions)
-- Chạy script này trong Supabase SQL Editor
-- ============================================================================

-- 0. Bật extension pgvector (bỏ qua nếu đã có)
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Xóa bảng cũ nếu tồn tại (kèm function và index phụ thuộc)
DROP TABLE IF EXISTS documents CASCADE;

-- 2. Xóa function cũ nếu tồn tại
DROP FUNCTION IF EXISTS match_documents(VECTOR, FLOAT, INT);

-- 3. Tạo mới bảng documents với embedding 1536 chiều
CREATE TABLE documents (
    id          BIGSERIAL PRIMARY KEY,
    chunk_id    TEXT            UNIQUE NOT NULL,  -- MD5 hash để dedup khi upsert
    content     TEXT            NOT NULL,
    heading     TEXT,
    source      TEXT,
    chunk_index INT             DEFAULT 0,
    embedding   VECTOR(1536),
    created_at  TIMESTAMPTZ     DEFAULT NOW()
);

-- 4. Tạo index IVFFlat cho tìm kiếm vector cosine
CREATE INDEX idx_documents_embedding
ON documents
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 10);

-- 5. Tạo function match_documents để tìm kiếm ngữ nghĩa
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding VECTOR(1536),
    match_threshold FLOAT DEFAULT 0.5,
    match_count INT DEFAULT 5
)
RETURNS TABLE (
    id              BIGINT,
    content         TEXT,
    heading         TEXT,
    source          TEXT,
    chunk_index     INT,
    similarity      FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        d.id,
        d.content,
        d.heading,
        d.source,
        d.chunk_index,
        1 - (d.embedding <=> query_embedding) AS similarity
    FROM documents d
    WHERE 1 - (d.embedding <=> query_embedding) > match_threshold
    ORDER BY d.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- ============================================================================
-- ✅ Tạo mới hoàn tất!
-- Tiếp theo: restart server → truy cập /api/embed để chạy lại embedding
-- ============================================================================
