package embedding

import (
	"context"
	"fmt"
	"strings"

	"github.com/openai/openai-go/v3" // imported as openai
	"github.com/openai/openai-go/v3/option"
)

type OpenAIClient struct {
	model  string
	client openai.Client
}

func NewOpenAIClient(apiKey string) *OpenAIClient {
	return &OpenAIClient{
		model: openai.EmbeddingModelTextEmbedding3Small,
		client: openai.NewClient(
			option.WithAPIKey(apiKey),
		),
	}
}

func (o *OpenAIClient) Embed(ctx context.Context, text []string) ([][]float32, error) {
	if len(text) == 0 {
		return nil, fmt.Errorf("openai embed: input cannot be empty")
	}

	for i, value := range text {
		if strings.TrimSpace(value) == "" {
			return nil, fmt.Errorf("openai embed: input %d cannot be empty", i)
		}
	}

	resp, err := o.client.Embeddings.New(ctx, openai.EmbeddingNewParams{
		Input: openai.EmbeddingNewParamsInputUnion{
			OfArrayOfStrings: text,
		},
		Model:          o.model,
		EncodingFormat: openai.EmbeddingNewParamsEncodingFormatFloat,
	})
	if err != nil {
		return nil, fmt.Errorf("openai embed: %w", err)
	}
	if len(resp.Data) != len(text) {
		return nil, fmt.Errorf("openai embed: got %d embeddings for %d inputs", len(resp.Data), len(text))
	}

	vectors := make([][]float32, len(resp.Data))
	for _, item := range resp.Data {
		if item.Index < 0 || int(item.Index) >= len(vectors) {
			return nil, fmt.Errorf("openai embed: response index %d out of range", item.Index)
		}

		vector := make([]float32, len(item.Embedding))
		for i, value := range item.Embedding {
			vector[i] = float32(value)
		}
		vectors[item.Index] = vector
	}

	return vectors, nil
}

func (o *OpenAIClient) GetModelName() string {
	return o.model
}
