package storage

import (
	"context"

	"github.com/ashmitsharp/askmydocs/internal/ingestion"
)

// point is a chunk vector + its metadata payload.
type Point struct {
	ID      string
	Vector  []float32
	Payload map[string]any
}

type SearchResult struct {
	ID      string
	Score   float32
	Payload map[string]any
}

type MetadataFilter struct {
	Filename string `json:"filename,omitempty"`
}

type VectorStore interface {
	Upsert(ctx context.Context, points []Point) error
	Search(ctx context.Context, vector []float32, topK int, filter *MetadataFilter) ([]SearchResult, error)
	Delete(ctx context.Context, ids []string) error
}

type ChunkStore interface {
	Upsert(ctx context.Context, chunks []ingestion.Chunk) error
}
