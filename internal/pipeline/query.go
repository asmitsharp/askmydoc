package pipeline

import (
	"context"
	"fmt"
	"strings"

	"github.com/ashmitsharp/askmydocs/internal/embedding"
	"github.com/ashmitsharp/askmydocs/internal/ingestion"
	"github.com/ashmitsharp/askmydocs/internal/llm"
	"github.com/ashmitsharp/askmydocs/internal/retrieval"
	"github.com/ashmitsharp/askmydocs/internal/storage"
)

type Answer struct {
	Text string
}

type Citation struct {
	Source     string
	ChunkIndex int
	Score      float32
	Content    string
}

type QueryPipeLine struct {
	embedder   embedding.Embedding
	store      storage.VectorStore
	bm25Store  *storage.BM25Store
	llm        llm.LLM
	topK       int
	maxTokens  int
	fusionTopN int
	rerankTopN int
	reranker   retrieval.Reranker
}

func NewQueryPipeLine(embedder embedding.Embedding, store storage.VectorStore, bm25Store *storage.BM25Store, llm llm.LLM, reranker retrieval.Reranker) *QueryPipeLine {
	return &QueryPipeLine{
		embedder:   embedder,
		store:      store,
		bm25Store:  bm25Store,
		llm:        llm,
		topK:       5,
		maxTokens:  3000,
		fusionTopN: 20,
		rerankTopN: 5,
		reranker:   reranker,
	}
}

func (q *QueryPipeLine) Execute(ctx context.Context, question string) (*Answer, []Citation, error) {
	questionVector, err := q.embedder.Embed(ctx, []string{question})
	if err != nil {
		return nil, nil, fmt.Errorf("question vector cannot be created : %w", err)
	}

	// Concurrent searches
	vectorCh := make(chan []storage.SearchResult, 1)
	bm25Ch := make(chan []storage.SearchResult, 1)
	errCh := make(chan error, 2)

	go func() {
		results, err := q.store.Search(ctx, questionVector[0], q.fusionTopN)
		if err != nil {
			errCh <- err
			return
		}
		vectorCh <- results
	}()

	go func() {
		results, err := q.bm25Store.Search(ctx, question, q.fusionTopN)
		if err != nil {
			errCh <- err
			return
		}
		bm25Ch <- results
	}()

	var vectorResults, bm25Results []storage.SearchResult
	select {
	case err := <-errCh:
		return nil, nil, fmt.Errorf("search error: %w", err)
	case vectorResults = <-vectorCh:
		select {
		case err := <-errCh:
			return nil, nil, fmt.Errorf("search error: %w", err)
		case bm25Results = <-bm25Ch:
			// Both done
		}
	}

	// Fuse results
	fused := retrieval.ReciprocalRankFusion(vectorResults, bm25Results, q.fusionTopN)
	if len(fused) == 0 {
		return nil, nil, fmt.Errorf("no relevant results found after fusion")
	}

	// Optional reranking
	var finalResults []retrieval.FusedResult
	if q.reranker != nil {
		documents := make([]string, len(fused))
		for i, fr := range fused {
			documents[i] = fr.Payload["content"].(string)
		}

		rerankResults, err := q.reranker.Rerank(ctx, question, documents, q.rerankTopN)
		if err != nil {
			return nil, nil, fmt.Errorf("reranking failed: %w", err)
		}

		// Reorder based on rerank indices
		finalResults = make([]retrieval.FusedResult, len(rerankResults))
		for i, rr := range rerankResults {
			finalResults[i] = fused[rr.Index]
			finalResults[i].RRFScore = rr.RelevanceScore
		}
	} else {
		// Take top rerankTopN from fused
		if len(fused) > q.rerankTopN {
			finalResults = fused[:q.rerankTopN]
		} else {
			finalResults = fused
		}
	}

	selected := make([]Citation, 0, len(finalResults))
	tokenUsed := 0
	for _, fr := range finalResults {
		content, ok := fr.Payload["content"].(string)
		if !ok || strings.TrimSpace(content) == "" {
			continue
		}

		tokenChunk := ingestion.EstimateTokens(content)
		if tokenUsed+tokenChunk > q.maxTokens {
			break
		}

		tokenUsed += tokenChunk
		selected = append(selected, Citation{
			Source:     toString(fr.Payload["source"]),
			ChunkIndex: toInt(fr.Payload["chunk_index"]),
			Score:      float32(fr.RRFScore),
			Content:    content,
		})
	}

	if len(selected) == 0 {
		return nil, nil, fmt.Errorf("no chunks fit within the %d token budget", q.maxTokens)
	}

	prompt := q.buildPrompt(question, selected)
	answerText, err := q.llm.Complete(ctx, prompt)
	if err != nil {
		return nil, nil, fmt.Errorf("error generating answer: %w", err)
	}

	return &Answer{Text: answerText}, selected, nil
}

func (q *QueryPipeLine) buildPrompt(question string, citations []Citation) string {
	var sb strings.Builder
	sb.WriteString("You are a helpful assistant. Answer the user's question using ONLY the context provided below. If the answer is not in the context, say 'I don't know based on the provided documents. For every claim you make, add a citation in the format [Source: filename, chunk N]'\n\n")
	sb.WriteString("Context:\n")

	for _, citation := range citations {
		fmt.Fprintf(&sb, "[Source: %s, Chunk %d]\n%s\n", citation.Source, citation.ChunkIndex+1, citation.Content)
	}

	sb.WriteString("\nQuestion: ")
	sb.WriteString(question)
	sb.WriteString("\nAnswer:")
	return sb.String()
}

func toString(value any) string {
	switch v := value.(type) {
	case string:
		return v
	default:
		return ""
	}
}

func toInt(value any) int {
	switch v := value.(type) {
	case int:
		return v
	case int64:
		return int(v)
	case float64:
		return int(v)
	default:
		return -1
	}
}
