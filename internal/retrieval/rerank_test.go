package retrieval

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"
)

func createMockCohereServer(responseBody string, statusCode int) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify request method and path
		if r.Method != "POST" || r.URL.Path != "/v2/rerank" {
			w.WriteHeader(http.StatusNotFound)
			return
		}

		// Verify headers
		if r.Header.Get("Authorization") != "Bearer test-key" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}

		w.WriteHeader(statusCode)
		w.Write([]byte(responseBody))
	}))
}

func TestCohereReranker_Rerank_Success(t *testing.T) {
	// Arrange
	mockResponse := `{
		"results": [
			{"index": 1, "relevance_score": 0.95},
			{"index": 0, "relevance_score": 0.85},
			{"index": 2, "relevance_score": 0.75}
		]
	}`

	server := createMockCohereServer(mockResponse, http.StatusOK)
	defer server.Close()

	reranker := &CohereReranker{
		apiKey:     "test-key",
		model:      "rerank-v3.5",
		baseURL:    server.URL,
		httpClient: &http.Client{Timeout: 5 * time.Second},
	}

	documents := []string{"doc1", "doc2", "doc3"}

	// Act
	results, err := reranker.Rerank(context.Background(), "test query", documents, 3)

	// Assert
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	if len(results) != 3 {
		t.Errorf("Expected 3 results, got %d", len(results))
	}

	// Verify order matches API response
	if results[0].Index != 1 || results[0].RelevanceScore != 0.95 {
		t.Errorf("First result incorrect: index=%d, score=%.2f", results[0].Index, results[0].RelevanceScore)
	}

	if results[1].Index != 0 || results[1].RelevanceScore != 0.85 {
		t.Errorf("Second result incorrect: index=%d, score=%.2f", results[1].Index, results[1].RelevanceScore)
	}

	if results[2].Index != 2 || results[2].RelevanceScore != 0.75 {
		t.Errorf("Third result incorrect: index=%d, score=%.2f", results[2].Index, results[2].RelevanceScore)
	}
}

func TestCohereReranker_RequestFormat(t *testing.T) {
	var receivedBody map[string]interface{}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&receivedBody)
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"results": []}`))
	}))
	defer server.Close()

	reranker := &CohereReranker{
		apiKey:     "test-key",
		model:      "rerank-v3.5",
		baseURL:    server.URL,
		httpClient: &http.Client{Timeout: 5 * time.Second},
	}

	documents := []string{"doc1", "doc2"}

	_, err := reranker.Rerank(context.Background(), "test query", documents, 2)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Assert request body structure
	expected := map[string]interface{}{
		"model":     "rerank-v3.5",
		"query":     "test query",
		"documents": []interface{}{"doc1", "doc2"},
		"top_n":     float64(2), // JSON numbers are float64
	}

	if !reflect.DeepEqual(receivedBody, expected) {
		t.Errorf("Request body mismatch.\nExpected: %+v\nActual: %+v", expected, receivedBody)
	}
}

func TestCohereReranker_Rerank_APIError(t *testing.T) {
	server := createMockCohereServer(`{"error": "invalid api key"}`, http.StatusUnauthorized)
	defer server.Close()

	reranker := &CohereReranker{
		apiKey:     "invalid-key",
		model:      "rerank-v3.5",
		baseURL:    server.URL,
		httpClient: &http.Client{Timeout: 5 * time.Second},
	}

	_, err := reranker.Rerank(context.Background(), "query", []string{"doc"}, 1)

	if err == nil {
		t.Error("Expected error for 401 status")
	}

	if !strings.Contains(err.Error(), "401") {
		t.Errorf("Error should mention status code: %v", err)
	}
}

func TestCohereReranker_Rerank_Timeout(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(100 * time.Millisecond) // Longer than client timeout
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	reranker := &CohereReranker{
		apiKey:     "test",
		model:      "test",
		baseURL:    server.URL,
		httpClient: &http.Client{Timeout: 50 * time.Millisecond},
	}

	_, err := reranker.Rerank(context.Background(), "query", []string{"doc"}, 1)

	if err == nil {
		t.Error("Expected timeout error")
	}
}

func TestCohereReranker_Rerank_MalformedResponse(t *testing.T) {
	server := createMockCohereServer(`invalid json`, http.StatusOK)
	defer server.Close()

	reranker := &CohereReranker{
		apiKey:     "test-key",
		model:      "rerank-v3.5",
		baseURL:    server.URL,
		httpClient: &http.Client{Timeout: 5 * time.Second},
	}

	_, err := reranker.Rerank(context.Background(), "query", []string{"doc"}, 1)

	if err == nil {
		t.Error("Expected error for malformed JSON response")
	}

	if !strings.Contains(err.Error(), "decode response") {
		t.Errorf("Error should mention decode failure: %v", err)
	}
}