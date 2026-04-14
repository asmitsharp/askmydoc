package storage

import "context"

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

type VectorStore interface {
	Upsert(ctx context.Context, points []Point) error
	Search(ctx context.Context, vector []float32, topK int) ([]SearchResult, error)
	Delete(ctx context.Context, ids []string) error
}
