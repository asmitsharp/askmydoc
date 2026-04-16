CREATE TABLE IF NOT EXISTS chunks (
    id         UUID PRIMARY KEY,
    content    TEXT        NOT NULL,
    source     TEXT        NOT NULL,
    chunk_index INTEGER    NOT NULL,
    page_start INTEGER    DEFAULT 0,
    page_end   INTEGER    DEFAULT 0,
    metadata   JSONB      DEFAULT '{}',
    tsv        TSVECTOR   GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

-- This GIN index makes tsquery lookups fast (without it, every search = full table scan)
CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING GIN (tsv);
