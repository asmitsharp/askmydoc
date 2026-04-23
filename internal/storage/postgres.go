package storage

import (
	"context"
	"fmt"

	"github.com/ashmitsharp/askmydocs/internal/ingestion"
	"github.com/jackc/pgx/v5/pgxpool"
)

type BM25Store struct {
	pool *pgxpool.Pool
}

func NewBM25Store(ctx context.Context, connStr string) (*BM25Store, error) {
	pool, err := pgxpool.New(ctx, connStr)
	if err != nil {
		return nil, fmt.Errorf("unable to connect to database: %w", err)
	}

	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("unable to ping database: %w", err)
	}
	store := &BM25Store{pool: pool}
	if err := store.ensureSchema(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("ensure schema failed: %w", err)
	}

	return store, nil

}

func (b *BM25Store) ensureSchema(ctx context.Context) error {
	sql := `
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

        CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING GIN (tsv);
    `

	_, err := b.pool.Exec(ctx, sql)
	if err != nil {
		return fmt.Errorf("create schema: %w", err)
	}

	return nil
}

func (b *BM25Store) Upsert(ctx context.Context, chunks []ingestion.Chunk) error {
	tx, err := b.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin transaction: %w", err)
	}
	defer tx.Rollback(ctx)

	stmt := `INSERT INTO chunks (id, content, source, chunk_index, page_start, page_end, metadata)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (id) DO UPDATE SET
            content = EXCLUDED.content,
            source = EXCLUDED.source,
            chunk_index = EXCLUDED.chunk_index,
            page_start = EXCLUDED.page_start,
            page_end = EXCLUDED.page_end,
            metadata = EXCLUDED.metadata
    `

	for _, chunk := range chunks {
		metadata := "{}"
		if _, err := tx.Exec(ctx, stmt, chunk.ID, chunk.Content,
			chunk.Source,
			chunk.Index,
			chunk.PageStart,
			chunk.PageEnd,
			metadata); err != nil {
			return fmt.Errorf("upsert chunk %s: %w", chunk.ID, err)
		}
	}

	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit transaction: %w", err)
	}
	return nil
}

func (b *BM25Store) Search(ctx context.Context, query string, topK int, filter *MetadataFilter) ([]SearchResult, error) {
	// Use OR-based matching: plainto_tsquery uses AND between all terms, which
	// fails for natural-language questions (e.g. "What was TCS's total revenue")
	// because all 9+ words must appear in a single chunk. OR matching retrieves
	// chunks containing ANY query terms, ranked by ts_rank (which scores higher
	// when more terms match).
	stmt := `
        SELECT id, content, source, chunk_index, ts_rank(tsv, q) AS score
        FROM chunks, to_tsquery('english', 
            array_to_string(
                array(SELECT lexeme FROM unnest(to_tsvector('english', $1)) ORDER BY lexeme),
                ' | '
            )
        ) q
        WHERE tsv @@ q
    `
	args := []any{query}

	if filter != nil && filter.Filename != "" {
		stmt += ` AND source = $2`
		args = append(args, filter.Filename)

		stmt += ` ORDER BY score DESC LIMIT $3`
		args = append(args, topK)
	} else {
		stmt += ` ORDER BY score DESC LIMIT $2`
		args = append(args, topK)
	}
	rows, err := b.pool.Query(ctx, stmt, args...)
	if err != nil {
		return nil, fmt.Errorf("search query: %w", err)
	}
	defer rows.Close()

	var results []SearchResult
	for rows.Next() {
		var id, content, source string
		var chunkIndex int
		var score float32

		if err := rows.Scan(&id, &content, &source, &chunkIndex, &score); err != nil {
			return nil, fmt.Errorf("scan row: %w", err)
		}
		result := SearchResult{
			ID:    id,
			Score: score,
			Payload: map[string]any{
				"content":     content,
				"source":      source,
				"chunk_index": chunkIndex,
			},
		}

		results = append(results, result)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("rows error: %w", err)
	}

	return results, nil
}

func (b *BM25Store) Close() {
	b.pool.Close()
}
