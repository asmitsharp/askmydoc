package embedding

import (
	"context"
	"fmt"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
)

type GeminiClient struct {
	model  string
	client *genai.Client
}

func NewGeminiClient(ctx context.Context, apiKey string) (*GeminiClient, error) {
	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		return nil, fmt.Errorf("failed to create gemini client: %w", err)
	}

	return &GeminiClient{
		model:  "text-embedding-004",
		client: client,
	}, nil
}

func (g *GeminiClient) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	model := g.client.EmbeddingModel(g.model)
	
	batch := model.NewBatch()
	for _, text := range texts {
		batch.AddContent(genai.Text(text))
	}

	resp, err := model.BatchEmbedContents(ctx, batch)
	if err != nil {
		return nil, fmt.Errorf("error embedding texts with gemini: %w", err)
	}

	var embeddings [][]float32
	for _, e := range resp.Embeddings {
		embeddings = append(embeddings, e.Values)
	}

	return embeddings, nil
}

func (g *GeminiClient) GetModelName() string {
	return g.model
}
