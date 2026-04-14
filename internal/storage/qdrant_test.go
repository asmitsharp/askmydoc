package storage

import (
	"bufio"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestQdrantStoreIntegration(t *testing.T) {
	loadDotEnvForTest(t)

	addr := os.Getenv("QDRANT_ADDR")
	if strings.TrimSpace(addr) == "" {
		t.Skip("set QDRANT_ADDR=localhost to run Qdrant integration tests")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	store, err := NewQdrantStore(ctx, addr, 6334, "test_askmydocs", 4)
	if err != nil {
		t.Fatalf("NewQdrantStore: %v", err)
	}
	defer store.Close()

	points := []Point{
		{ID: "550e8400-e29b-41d4-a716-446655440000", Vector: []float32{0.1, 0.2, 0.3, 0.4}, Payload: map[string]any{"content": "hello world", "source": "test.txt", "index": 0}},
		{ID: "550e8400-e29b-41d4-a716-446655440001", Vector: []float32{0.9, 0.8, 0.7, 0.6}, Payload: map[string]any{"content": "goodbye world", "source": "test.txt", "index": 1}},
	}

	if err := store.Upsert(ctx, points); err != nil {
		t.Fatalf("Upsert: %v", err)
	}

	results, err := store.Search(ctx, []float32{0.1, 0.2, 0.3, 0.4}, 2)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected at least one search result")
	}
	if results[0].ID != "550e8400-e29b-41d4-a716-446655440000" {
		t.Errorf("expected closest point to be index 0, got %q", results[0].ID)
	}

	if err := store.Delete(ctx, []string{"550e8400-e29b-41d4-a716-446655440000", "550e8400-e29b-41d4-a716-446655440001"}); err != nil {
		t.Fatalf("Delete: %v", err)
	}
}

func loadDotEnvForTest(t *testing.T) {
	t.Helper()
	path, err := findDotEnv()
	if err != nil {
		return
	}
	f, err := os.Open(path)
	if err != nil {
		return
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		value = strings.Trim(strings.TrimSpace(value), `"'`)
		if key == "" || os.Getenv(key) != "" {
			continue
		}
		t.Setenv(key, value)
	}
}

func findDotEnv() (string, error) {
	dir, _ := os.Getwd()
	for {
		p := filepath.Join(dir, ".env")
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", os.ErrNotExist
		}
		dir = parent
	}
}
