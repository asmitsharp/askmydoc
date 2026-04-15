package pipeline

import (
	"context"
	"fmt"

	"github.com/ashmitsharp/askmydocs/internal/embedding"
	"github.com/ashmitsharp/askmydocs/internal/ingestion"
	"github.com/ashmitsharp/askmydocs/internal/storage"
)

type PipeLine struct {
	loader   *ingestion.Router
	chunker  *ingestion.Chunker
	embedder embedding.Embedding
	store    storage.VectorStore
}

func NewPipeLine(loader *ingestion.Router, chunker *ingestion.Chunker, embedder embedding.Embedding, store storage.VectorStore) *PipeLine {
	return &PipeLine{
		loader:   loader,
		chunker:  chunker,
		embedder: embedder,
		store:    store,
	}
}

func (p *PipeLine) Ingest(ctx context.Context, filepath string) (int, error) {
	document, err := p.loader.Load(filepath)
	if err != nil {
		return 0, fmt.Errorf("failed to load document: %w", err)
	}

	chunks := []ingestion.Chunk{}
	if len(document.Pages) > 0 {
		chunks = p.chunker.ChunkPages(document)
	} else {
		chunks = p.chunker.Chunk(document)
	}

	if len(chunks) == 0 {
		return 0, fmt.Errorf("there are 0 chunks generated for the file : %s", document.Source)
	}

	extractedChunks := make([]string, 0, len(chunks))
	for _, chunk := range chunks {
		extractedChunks = append(extractedChunks, chunk.Content)
	}

	vectors, err := p.embedder.Embed(ctx, extractedChunks)
	if err != nil {
		return 0, fmt.Errorf("error embedding the chunks : %w", err)
	}

	points := make([]storage.Point, len(chunks))
	for i, chunk := range chunks {
		points[i].ID = chunk.ID
		points[i].Vector = vectors[i]
		points[i].Payload = map[string]any{
			"source":      chunk.Source,
			"pageStart":   chunk.PageStart,
			"pageEnd":     chunk.PageEnd,
			"chunk_index": chunk.Index,
			"content":     chunk.Content,
		}
	}

	if err := p.store.Upsert(ctx, points); err != nil {
		return 0, fmt.Errorf("failed to upsert points: %w", err)
	}

	fmt.Printf("Successfully ingested %d chunks from %s\n", len(chunks), document.Source)
	return len(chunks), nil

}
