package retrieval

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

type Reranker interface {
	Rerank(ctx context.Context, query string, documents []string, topN int) ([]CohereResult, error)
}

type CohereReranker struct {
	apiKey     string
	model      string
	baseURL    string
	httpClient *http.Client
}

func NewCohereReranker(apiKey, model string) *CohereReranker {
	return &CohereReranker{
		apiKey:     apiKey,
		model:      model,
		baseURL:    "https://api.cohere.com",
		httpClient: &http.Client{Timeout: 30 * time.Second},
	}
}

type CohereRequest struct {
	Model     string   `json:"model"`
	Query     string   `json:"query"`
	Documents []string `json:"documents"`
	TopN      int      `json:"top_n"`
}

type CohereResult struct {
	Index          int     `json:"index"`
	RelevanceScore float64 `json:"relevance_score"`
}

type CohereResponse struct {
	Results []CohereResult `json:"results"`
}

func (c *CohereReranker) Rerank(ctx context.Context, query string, documents []string, topN int) ([]CohereResult, error) {
	reqBody := CohereRequest{
		Model:     c.model,
		Query:     query,
		Documents: documents,
		TopN:      topN,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("cohere rerank: marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/v2/rerank", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("cohere rerank: create request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("cohere rerank: do request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("cohere rerank: status %d: %s", resp.StatusCode, string(body))
	}

	var response CohereResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, fmt.Errorf("cohere rerank: decode response: %w", err)
	}

	return response.Results, nil
}
