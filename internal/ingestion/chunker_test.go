package ingestion

import (
	"strings"
	"testing"
)

func TestChunkReturnsSingleChunkForShortDocument(t *testing.T) {
	chunker := &Chunker{
		MaxChunkSize: 20,
		Separators:   []string{" "},
	}

	chunks := chunker.Chunk(&Document{Content: "short text"})
	if len(chunks) != 1 {
		t.Fatalf("len(chunks) = %d, want 1", len(chunks))
	}
	if chunks[0].Content != "short text" {
		t.Fatalf("Content = %q, want %q", chunks[0].Content, "short text")
	}
}

func TestChunkRecursivelySplitsLargeDocument(t *testing.T) {
	chunker := &Chunker{
		MaxChunkSize: 10,
		Separators:   []string{"\n\n", " ", ""},
	}

	chunks := chunker.Chunk(&Document{Content: "alpha beta gamma\n\ndelta epsilon zeta"})

	if len(chunks) == 0 {
		t.Fatal("expected chunks")
	}
	for _, chunk := range chunks {
		if chunk.Content == "" {
			t.Fatal("expected no empty chunks")
		}
		if len(chunk.Content) > chunker.MaxChunkSize {
			t.Fatalf("chunk %q has len %d, want <= %d", chunk.Content, len(chunk.Content), chunker.MaxChunkSize)
		}
	}
}

func TestChunkHardSplitsWhenNoSeparatorCanHelp(t *testing.T) {
	chunker := &Chunker{
		MaxChunkSize: 5,
		Separators:   []string{" "},
	}

	chunks := chunker.Chunk(&Document{Content: "abcdefghijkl"})
	got := make([]string, 0, len(chunks))
	for _, chunk := range chunks {
		got = append(got, chunk.Content)
	}

	want := []string{"abcde", "fghij", "kl"}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("chunks = %#v, want %#v", got, want)
	}
}
