package embedding

import "testing"

func TestNewClientOpenAI(t *testing.T) {
	client, err := NewClient("openai", "test-key")
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	if client.GetModelName() != "text-embedding-3-small" {
		t.Fatalf("model = %q, want text-embedding-3-small", client.GetModelName())
	}
}

func TestNewClientHuggingFaceAlias(t *testing.T) {
	client, err := NewClient("hf", "test-key")
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	if client.GetModelName() != "sentence-transformers/all-MiniLM-L6-v2" {
		t.Fatalf("model = %q, want sentence-transformers/all-MiniLM-L6-v2", client.GetModelName())
	}
}

func TestNewClientUnsupportedProvider(t *testing.T) {
	_, err := NewClient("unknown", "test-key")
	if err == nil {
		t.Fatal("expected unsupported provider error")
	}
}
