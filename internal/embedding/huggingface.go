package embedding

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

type HFClient struct {
	apiKey     string
	model      string
	endpoint   string
	httpClient *http.Client
}

func NewHFClient(apiKey string) *HFClient {
	return &HFClient{
		apiKey:     apiKey,
		model:      "sentence-transformers/all-MiniLM-L6-v2",
		endpoint:   "https://router.huggingface.co/hf-inference/models",
		httpClient: &http.Client{Timeout: 30 * time.Second},
	}
}

func (h *HFClient) Embed(ctx context.Context, text []string) ([][]float32, error) {
	payload := map[string]interface{}{
		"inputs": text,
	}

	jsonData, _ := json.Marshal(payload)

	req, err := http.NewRequestWithContext(ctx, "POST", fmt.Sprintf("%s/%s/pipeline/feature-extraction", h.endpoint, h.model), bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("huggingface embed: create request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+h.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := h.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("huggingface embed: request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == 503 {
		return nil, fmt.Errorf("model is loading (HF free tier cold start), retry in 20s")
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return nil, fmt.Errorf("huggingface embed: status %d: %s", resp.StatusCode, string(body))
	}

	var embeddings [][]float32
	if err := json.NewDecoder(resp.Body).Decode(&embeddings); err != nil {
		return nil, fmt.Errorf("huggingface embed: decode response: %w", err)
	}

	return embeddings, nil
}

func (h *HFClient) GetModelName() string {
	return h.model
}
