package storage

import (
	"context"
	"os"
	"testing"

	"github.com/ashmitsharp/askmydocs/internal/ingestion"
)

func TestBM25Search(t *testing.T) {
	dsn := os.Getenv("POSTGRES_DSN")
	if dsn == "" {
		dsn = "postgres://askmydocs:askmydocs@localhost:5432/askmydocs?sslmode=disable"
	}

	ctx := context.Background()
	store, err := NewBM25Store(ctx, dsn)
	if err != nil {
		t.Skipf("skipping: cannot connect to PostgreSQL: %v", err)
	}
	defer store.Close()
	testChunks := []ingestion.Chunk{
		{
			ID: "00000000-0000-0000-0000-000000000001",
			Content: "The system uses OAuth2 for secure access management. " +
				"Users authenticate through the identity provider.",
			Source: "auth-guide.md", Index: 0,
		},
		{
			ID: "00000000-0000-0000-0000-000000000002",
			Content: "Database connections are pooled using pgxpool. " +
				"Each request gets a connection from the pool.",
			Source: "infra-guide.md", Index: 0,
		},
		{
			ID: "00000000-0000-0000-0000-000000000003",
			Content: "The authentication token has a default expiry of 3600 seconds. " +
				"After token expiry, the client must re-authenticate using a refresh token.",
			Source: "auth-guide.md", Index: 1,
		},
	}
	// Upsert test chunks
	if err := store.Upsert(ctx, testChunks); err != nil {
		t.Fatalf("failed to upsert test chunks: %v", err)
	}

	// Test 1: Search for "authentication token expiry" — should find chunk 3
	results, err := store.Search(ctx, "authentication token expiry", 10)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if len(results) == 0 {
		t.Fatalf("expected results, got 0")
	}

	// Verify chunk 3 is in results
	chunk3Found := false
	var chunk3Score float32
	var chunk1Score float32

	for _, result := range results {
		if result.ID == "00000000-0000-0000-0000-000000000003" {
			chunk3Found = true
			chunk3Score = result.Score
		}
		if result.ID == "00000000-0000-0000-0000-000000000001" {
			chunk1Score = result.Score
		}
	}

	if !chunk3Found {
		t.Fatalf("chunk 3 not found in results")
	}

	// Verify chunk 2 is NOT in results (it's about databases, not auth)
	for _, result := range results {
		if result.ID == "00000000-0000-0000-0000-000000000002" {
			t.Fatalf("chunk 2 should not be in results for 'authentication token expiry'")
		}
	}

	// Verify chunk 3 scores higher than chunk 1
	if chunk3Score <= chunk1Score {
		t.Fatalf("chunk 3 score (%f) should be higher than chunk 1 score (%f)", chunk3Score, chunk1Score)
	}

	t.Logf("✓ Chunk 3 ranked first with score %f", chunk3Score)
	t.Logf("✓ Chunk 1 ranked second with score %f", chunk1Score)

	// Test 2: Search for something that exists in NO chunks
	emptyResults, err := store.Search(ctx, "kubernetes deployment", 10)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if len(emptyResults) != 0 {
		t.Fatalf("expected 0 results for 'kubernetes deployment', got %d", len(emptyResults))
	}

	t.Logf("✓ Empty search returns 0 results as expected")
}
