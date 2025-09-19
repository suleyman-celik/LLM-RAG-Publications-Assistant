-- ============================================
-- Initialize pgvector extension
-- ============================================

-- Install the pgvector extension if not already installed
CREATE EXTENSION IF NOT EXISTS vector;

-- Create schema if needed
CREATE SCHEMA IF NOT EXISTS public;

-- ============================================
-- Embedding model metadata table
-- ============================================

-- Drop tables if they exist
DROP TABLE IF EXISTS public.model_embed;
CREATE TABLE IF NOT EXISTS public.model_embed (
    model_name TEXT PRIMARY KEY,       -- Unique name of the embedding model
    dimension INT NOT NULL,            -- Dimension of the vectors produced by this model
    table_name TEXT NOT NULL           -- Corresponding table where vectors are stored
);

-- ============================================
-- Create the items table to store content and embeddings
-- ============================================

-- Notes:
-- - 'content' stores the textual data to vectorize.
-- - 'embedding' stores the 768-dimensional vector produced by the embedding model.
-- - Using 768 dimensions is critical to match the model output.
-- - Optionally, you could normalize vectors before storing for faster similarity search.
-- - Consider using a separate table for content and referencing it via foreign key if needed.

/*
For simplicity, we are directly adding the content into this table as
a column containing text data. It could easily be a foreign key pointing to
another table instead that has the content you want to vectorize for
semantic search, just storing here the vectorized content in our "items" table.

"768" dimensions for our vector embedding is critical - that is the
number of dimensions our open source embeddings model output, for later in the
blog post.
*/

-- ============================================
-- Default items table (can be used for most common model)
-- https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
-- ============================================

-- - Drop tables if they exist
DROP TABLE IF EXISTS public.text_chunks;
CREATE TABLE IF NOT EXISTS public.text_chunks (
    id SERIAL PRIMARY KEY,
    section_content TEXT NOT NULL,
    type TEXT,
    vector VECTOR(768),           -- (1536) pgvector column adjust this based on your OpenAI model
    hash TEXT UNIQUE,             -- content hash for deduplication
    group_id INT,
    source_link TEXT,
    source_name TEXT,
    source_note TEXT,
    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- - Drop tables if they exist
-- DROP TABLE IF EXISTS public.items;
-- CREATE TABLE IF NOT EXISTS public.items (
--     id BIGSERIAL PRIMARY KEY,
--     content TEXT NOT NULL,             -- Text data to be vectorized
--     embedding VECTOR(768)              -- Embedding vector (update dimension per model)
--     -- created_at TIMESTAMPTZ DEFAULT NOW(),
--     -- updated_at TIMESTAMPTZ DEFAULT NOW()
-- );

-- - Drop tables if they exist
-- DROP TABLE IF EXISTS public.items_128;
-- CREATE TABLE IF NOT EXISTS public.items_128 (
--     id BIGSERIAL PRIMARY KEY,
--     content TEXT NOT NULL,
--     embedding VECTOR(128)
-- );

-- - Drop tables if they exist
-- DROP TABLE IF EXISTS public.items_384;
-- CREATE TABLE IF NOT EXISTS public.items_384 (
--     id BIGSERIAL PRIMARY KEY,
--     content TEXT NOT NULL,
--     embedding VECTOR(384)
-- );

-- - Drop tables if they exist
-- DROP TABLE IF EXISTS public.items_512;
-- CREATE TABLE IF NOT EXISTS public.items_512 (
--     id BIGSERIAL PRIMARY KEY,
--     content TEXT NOT NULL,
--     embedding VECTOR(512)
-- );

-- - Drop tables if they exist
-- DROP TABLE IF EXISTS public.items_768;
-- CREATE TABLE IF NOT EXISTS public.items_768 (
--     id BIGSERIAL PRIMARY KEY,
--     content TEXT NOT NULL,
--     embedding VECTOR(768)
-- );

-- - Drop tables if they exist
-- DROP TABLE IF EXISTS public.items_1024;
-- CREATE TABLE IF NOT EXISTS public.items_1024 (
--     id BIGSERIAL PRIMARY KEY,
--     content TEXT NOT NULL,
--     embedding VECTOR(1024)
-- );

-- ============================================
-- Optional: Dynamic index creation template
-- NOTICE:  ivfflat index created with little data
-- DETAIL:  This will cause low recall.
-- HINT:  Drop the index until the table has more data.
-- If you are going to insert many embeddings, leave it as is. After bulk inserting, you can rebuild the index for better recall:
-- REINDEX INDEX idx_items_768_embedding;
-- ============================================

-- - Using ivfflat index for approximate nearest neighbor search
-- - Adjust 'lists' depending on dataset size (higher for larger datasets)
-- - Drop indexes if they exist
-- DROP INDEX IF EXISTS idx_items_embedding;
-- CREATE INDEX IF NOT EXISTS idx_items_embedding
-- ON items USING ivfflat (embedding vector_l2_ops)
-- WITH (lists = 100);

-- - Drop indexes if they exist
-- DROP INDEX IF EXISTS idx_items_384_embedding;
-- CREATE INDEX IF NOT EXISTS idx_items_384_embedding
-- ON items_384 USING ivfflat (embedding vector_l2_ops)
-- WITH (lists = 100);

-- - Drop indexes if they exist
-- DROP INDEX IF EXISTS idx_items_768_embedding;
-- CREATE INDEX IF NOT EXISTS idx_items_768_embedding
-- ON items_768 USING ivfflat (embedding vector_l2_ops)
-- WITH (lists = 100);

-- ============================================
-- Optional: Trigger to auto-update 'updated_at'
-- ============================================

-- CREATE OR REPLACE FUNCTION update_updated_at_column()
-- RETURNS TRIGGER AS $$
-- BEGIN
--     NEW.updated_at = NOW();
--     RETURN NEW;
-- END;
-- $$ LANGUAGE plpgsql;

-- DROP TRIGGER IF EXISTS trg_update_items_updated_at ON items;

-- CREATE TRIGGER trg_update_items_updated_at
-- BEFORE UPDATE ON items
-- FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();


-- CREATE OR REPLACE FUNCTION update_updated_at_column()
-- RETURNS TRIGGER AS $$
-- BEGIN
--     NEW.updated_at = NOW();
--     RETURN NEW;
-- END;
-- $$ LANGUAGE plpgsql;

-- DROP TRIGGER IF EXISTS trg_update_items_768_updated_at ON items_768;

-- CREATE TRIGGER trg_update_items_768_updated_at
-- BEFORE UPDATE ON items_768
-- FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();