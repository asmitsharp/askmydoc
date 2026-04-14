package embedding

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"golang.org/x/time/rate"
)

type HFClient struct {
	apiKey     string
	model      string
	endpoint   string
	httpClient *http.Client
	batchSize  int
	maxRetries int
	baseDelay  time.Duration
	limiter    *rate.Limiter
}

func NewHFClient(apiKey string) *HFClient {
	return &HFClient{
		apiKey:     apiKey,
		model:      "sentence-transformers/all-MiniLM-L6-v2",
		endpoint:   "https://router.huggingface.co/hf-inference/models",
		httpClient: &http.Client{Timeout: 30 * time.Second},
		batchSize:  32,
		maxRetries: 3,
		baseDelay:  100 * time.Millisecond,
		limiter:    rate.NewLimiter(5, 10),
	}
}

func (h *HFClient) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, fmt.Errorf("huggingface embed: input cannot be empty")
	}
	for i, t := range texts {
		if strings.TrimSpace(t) == "" {
			return nil, fmt.Errorf("huggingface embed: input %d cannot be empty", i)
		}
	}
	batches := batchTexts(texts, h.batchSize)
	allVectors := make([][]float32, 0, len(texts))
	for _, batch := range batches {
		start := time.Now()
		vectors, err := h.batchEmbedRetry(ctx, batch)
		if err != nil {
			return nil, err
		}
		allVectors = append(allVectors, vectors...)
		duration := time.Since(start)
		log.Printf("huggingface batch size=%d took=%v", len(batch), duration)
	}

	return allVectors, nil
}

func (h *HFClient) GetModelName() string {
	return h.model
}

func (h *HFClient) batchEmbedRetry(ctx context.Context, batch []string) ([][]float32, error) {
	var lastErr error

	for attempt := 0; attempt <= h.maxRetries; attempt++ {

		if err := h.limiter.Wait(ctx); err != nil {
			return nil, err
		}

		payload := map[string]interface{}{
			"inputs": batch,
		}

		jsonData, err := json.Marshal(payload)
		if err != nil {
			return nil, fmt.Errorf("huggingface embed: marshal: %w", err)
		}

		req, err := http.NewRequestWithContext(ctx, "POST", fmt.Sprintf("%s/%s/pipeline/feature-extraction", h.endpoint, h.model), bytes.NewBuffer(jsonData))
		if err != nil {
			return nil, fmt.Errorf("huggingface embed: create request: %w", err)
		}

		req.Header.Set("Authorization", "Bearer "+h.apiKey)
		req.Header.Set("Content-Type", "application/json")

		resp, err := h.httpClient.Do(req)
		if err != nil {
			lastErr = err
		} else {
			defer resp.Body.Close()

			if resp.StatusCode >= 200 && resp.StatusCode < 300 {
				var vectors [][]float32
				if err := json.NewDecoder(resp.Body).Decode(&vectors); err != nil {
					return nil, fmt.Errorf("huggingface embed: decode response: %w", err)
				}

				if len(vectors) != len(batch) {
					return nil, fmt.Errorf("huggingface embed: got %d embeddings for %d inputs", len(vectors), len(batch))
				}

				return vectors, nil
			}

			body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))

			if resp.StatusCode == 429 || resp.StatusCode >= 500 {
				lastErr = fmt.Errorf("huggingface retryable status %d: %s", resp.StatusCode, string(body))
			} else {
				return nil, fmt.Errorf("huggingface embed: status %d: %s", resp.StatusCode, string(body))
			}
		}

		if attempt == h.maxRetries {
			break
		}

		delay := h.baseDelay * time.Duration(1<<attempt)
		log.Printf("huggingface retry attempt=%d delay=%v error=%v", attempt+1, delay, err)
		select {
		case <-time.After(delay):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	return nil, fmt.Errorf("huggingface embed failed after %d retries: %w", h.maxRetries, lastErr)
}
