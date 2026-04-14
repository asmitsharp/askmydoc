package embedding

import (
	"context"
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"golang.org/x/time/rate"
)

type OpenAIClient struct {
	model      string
	client     openai.Client
	batchSize  int
	maxRetries int
	baseDelay  time.Duration
	limiter    *rate.Limiter
}

func NewOpenAIClient(apiKey string) *OpenAIClient {
	return &OpenAIClient{
		model: openai.EmbeddingModelTextEmbedding3Small,
		client: openai.NewClient(
			option.WithAPIKey(apiKey),
		),
		batchSize:  512,
		maxRetries: 3,
		baseDelay:  100 * time.Millisecond,
		limiter:    rate.NewLimiter(5, 10),
	}
}

func (o *OpenAIClient) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, fmt.Errorf("openai embed: input cannot be empty")
	}

	for i, value := range texts {
		if strings.TrimSpace(value) == "" {
			return nil, fmt.Errorf("openai embed: input %d cannot be empty", i)
		}
	}

	batches := batchTexts(texts, o.batchSize)
	allVectors := make([][]float32, 0, len(texts))
	for _, batch := range batches {
		start := time.Now()
		vectors, err := o.batchEmbedRetry(ctx, batch)
		if err != nil {
			return nil, err
		}
		allVectors = append(allVectors, vectors...)
		duration := time.Since(start)
		log.Printf("openai batch size=%d took=%v", len(batch), duration)
	}

	return allVectors, nil
}

func (o *OpenAIClient) GetModelName() string {
	return o.model
}

func (o *OpenAIClient) batchEmbedRetry(ctx context.Context, batch []string) ([][]float32, error) {
	var lastErr error
	for attempt := 0; attempt <= o.maxRetries; attempt++ {
		if err := o.limiter.Wait(ctx); err != nil {
			return nil, err
		}
		resp, err := o.client.Embeddings.New(ctx, openai.EmbeddingNewParams{
			Input: openai.EmbeddingNewParamsInputUnion{
				OfArrayOfStrings: batch,
			},
			Model:          o.model,
			EncodingFormat: openai.EmbeddingNewParamsEncodingFormatFloat,
		})
		if err == nil {
			if len(resp.Data) != len(batch) {
				return nil, fmt.Errorf("openai embed: got %d embeddings for %d inputs", len(resp.Data), len(batch))
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

		lastErr = err
		if attempt == o.maxRetries {
			break
		}

		delay := o.baseDelay * time.Duration(1<<attempt)
		log.Printf("openai retry attempt=%d delay=%v error=%v", attempt+1, delay, err)
		select {
		case <-time.After(delay):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	return nil, fmt.Errorf("openai embed failed after %d retries: %w", o.maxRetries, lastErr)
}
