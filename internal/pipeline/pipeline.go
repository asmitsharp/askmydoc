package pipeline

import (
	"context"
	"fmt"

	"github.com/ashmitsharp/askmydocs/internal/embedding"
	"github.com/ashmitsharp/askmydocs/internal/ingestion"
	"github.com/ashmitsharp/askmydocs/internal/storage"
)

type PipeLine struct {
	loader     *ingestion.Router
	chunker    *ingestion.Chunker
	embedder   embedding.Embedding
	store      storage.VectorStore
	chunkStore storage.ChunkStore
}

func NewPipeLine(loader *ingestion.Router, chunker *ingestion.Chunker, embedder embedding.Embedding, store storage.VectorStore, chunkStore storage.ChunkStore) *PipeLine {
	return &PipeLine{
		loader:     loader,
		chunker:    chunker,
		embedder:   embedder,
		store:      store,
		chunkStore: chunkStore,
	}
}

func (p *PipeLine) Ingest(ctx context.Context, filepath string, originalFilename string) (int, error) {
	document, err := p.loader.Load(filepath)
	if err != nil {
		return 0, fmt.Errorf("failed to load document: %w", err)
	}

	if originalFilename != "" {
		document.Source = originalFilename
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
		windowContent := chunk.Content

		if i > 0 && chunks[i-1].Source == chunk.Source {
			windowContent = chunks[i-1].Content + "\n\n" + windowContent
		}

		if i < len(chunks)-1 && chunks[i+1].Source == chunk.Source {
			windowContent = windowContent + "\n\n" + chunks[i+1].Content
		}

		points[i].ID = chunk.ID
		points[i].Vector = vectors[i]
		points[i].Payload = map[string]any{
			"source":      chunk.Source,
			"pageStart":   chunk.PageStart,
			"pageEnd":     chunk.PageEnd,
			"chunk_index": chunk.Index,
			"content":     chunk.Content,
			"window":      windowContent,
		}
	}

	// concurrent write to two differet data stores
	errChan := make(chan error, 2)

	// write to qdrant
	go func() {
		err := p.store.Upsert(ctx, points)
		errChan <- err
	}()

	// write to postgres
	go func() {
		err := p.chunkStore.Upsert(ctx, chunks)
		errChan <- err
	}()

	var qdrantErr, postgresErr error
	for i := 0; i < 2; i++ {
		err := <-errChan
		if i == 0 {
			qdrantErr = err
		} else {
			postgresErr = err
		}
	}

	if qdrantErr != nil || postgresErr != nil {
		return 0, fmt.Errorf("ingest failed: qdrant=%v, postgres=%v", qdrantErr, postgresErr)
	}

	fmt.Printf("Successfully ingested %d chunks from %s\n", len(chunks), document.Source)
	return len(chunks), nil

}
