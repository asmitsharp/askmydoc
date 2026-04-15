package pipeline

import (
	"context"
	"fmt"
	"strings"

	"github.com/ashmitsharp/askmydocs/internal/embedding"
	"github.com/ashmitsharp/askmydocs/internal/ingestion"
	"github.com/ashmitsharp/askmydocs/internal/llm"
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
	embedder  embedding.Embedding
	store     storage.VectorStore
	llm       llm.LLM
	topK      int
	maxTokens int
}

func NewQueryPipeLine(embedder embedding.Embedding, store storage.VectorStore, llm llm.LLM) *QueryPipeLine {
	return &QueryPipeLine{
		embedder:  embedder,
		store:     store,
		llm:       llm,
		topK:      5,
		maxTokens: 3000,
	}
}

func (q *QueryPipeLine) Execute(ctx context.Context, question string) (*Answer, []Citation, error) {
	questionVector, err := q.embedder.Embed(ctx, []string{question})
	if err != nil {
		return nil, nil, fmt.Errorf("question vector cannot be created : %w", err)
	}

	results, err := q.store.Search(ctx, questionVector[0], q.topK)
	if err != nil {
		return nil, nil, fmt.Errorf("error while searching the database : %w", err)
	}
	if len(results) == 0 {
		return nil, nil, fmt.Errorf("for the given question could not find relevant vectors")
	}

	selected := make([]Citation, 0, len(results))
	tokenUsed := 0
	for _, result := range results {
		content, ok := result.Payload["content"].(string)
		if !ok || strings.TrimSpace(content) == "" {
			continue
		}

		tokenChunk := ingestion.EstimateTokens(content)
		if tokenUsed+tokenChunk > q.maxTokens {
			break
		}

		tokenUsed += tokenChunk
		selected = append(selected, Citation{
			Source:     toString(result.Payload["source"]),
			ChunkIndex: toInt(result.Payload["chunk_index"]),
			Score:      result.Score,
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
