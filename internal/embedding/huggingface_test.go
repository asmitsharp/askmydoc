package embedding

import (
	"bufio"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestHFClientEmbedIntegration(t *testing.T) {
	loadDotEnvForTest(t)

	provider := strings.ToLower(strings.TrimSpace(os.Getenv("EMBEDDING_PROVIDER")))
	if provider != ProviderHuggingFace && provider != "hf" {
		t.Skip("set EMBEDDING_PROVIDER=huggingface to run Hugging Face integration test")
	}

	apiKey := os.Getenv("HF_API_KEY")
	if strings.TrimSpace(apiKey) == "" {
		t.Skip("set HF_API_KEY to run Hugging Face integration test")
	}

	client := NewHFClient(apiKey)
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()

	embeddings, err := client.Embed(ctx, []string{"hello", "world"})
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if len(embeddings) != 2 {
		t.Fatalf("len(embeddings) = %d, want 2", len(embeddings))
	}
	dimensions := len(embeddings[0])
	if dimensions == 0 {
		t.Fatal("first embedding vector is empty")
	}
	for i, embedding := range embeddings {
		if len(embedding) != dimensions {
			t.Fatalf("embedding %d dimensions = %d, want %d", i, len(embedding), dimensions)
		}
	}
}

func loadDotEnvForTest(t *testing.T) {
	t.Helper()

	path, err := findDotEnv()
	if err != nil {
		return
	}

	file, err := os.Open(path)
	if err != nil {
		return
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
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
	if err := scanner.Err(); err != nil {
		t.Fatalf("read .env: %v", err)
	}
}

func findDotEnv() (string, error) {
	dir, err := os.Getwd()
	if err != nil {
		return "", err
	}

	for {
		path := filepath.Join(dir, ".env")
		if _, err := os.Stat(path); err == nil {
			return path, nil
		}

		parent := filepath.Dir(dir)
		if parent == dir {
			return "", os.ErrNotExist
		}
		dir = parent
	}
}
