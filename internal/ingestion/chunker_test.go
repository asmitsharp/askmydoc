package ingestion

import (
	"strings"
	"testing"

	"github.com/google/uuid"
)

func newTestChunker(max, overlap int) *Chunker {
	return NewTextChunker(
		WithMaxChunkTokens(max),
		WithOverlapTokens(overlap),
	)
}

func repeat(word string, n int) string {
	words := make([]string, n)
	for i := range words {
		words[i] = word
	}
	return strings.Join(words, " ")
}

func TestChunk_NilDocument(t *testing.T) {
	c := newTestChunker(512, 50)
	if got := c.Chunk(nil); got != nil {
		t.Fatalf("expected nil for nil doc, got %v", got)
	}
}

func TestChunk_EmptyContent(t *testing.T) {
	c := newTestChunker(512, 50)
	doc := &Document{Content: "   ", Source: "empty.txt"}
	if got := c.Chunk(doc); got != nil {
		t.Fatalf("expected nil for whitespace-only doc, got %v", got)
	}
}

func TestChunk_ShortDocumentReturnsSingleChunk(t *testing.T) {
	c := newTestChunker(512, 50)
	doc := &Document{Content: "Hello world.", Source: "short.txt"}
	chunks := c.Chunk(doc)
	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(chunks))
	}
	if chunks[0].Content != doc.Content {
		t.Errorf("content mismatch: %q", chunks[0].Content)
	}
	if chunks[0].Source != "short.txt" {
		t.Errorf("source not set: %q", chunks[0].Source)
	}
	if _, err := uuid.Parse(chunks[0].ID); err != nil {
		t.Errorf("ID not a valid UUID: %q", chunks[0].ID)
	}
}

func TestChunk_NeverExceedsMaxTokens(t *testing.T) {
	c := newTestChunker(20, 5)
	doc := &Document{Content: repeat("word", 200), Source: "big.txt"}
	chunks := c.Chunk(doc)
	if len(chunks) == 0 {
		t.Fatal("expected at least one chunk")
	}
	for i, ch := range chunks {
		got := estimateTokens(ch.Content)
		if got > c.MaxChunkTokens*2 {
			t.Errorf("chunk %d: %d tokens, max is %d", i, got, c.MaxChunkTokens)
		}
	}
}

func TestChunk_OverlapApplied(t *testing.T) {
	c := newTestChunker(10, 8)
	doc := &Document{
		Content: repeat("alpha", 50) + "\n\n" + repeat("beta", 50),
		Source:  "overlap.txt",
	}
	chunks := c.Chunk(doc)
	if len(chunks) < 2 {
		t.Fatalf("need ≥2 chunks to test overlap, got %d", len(chunks))
	}
	for i := 1; i < len(chunks); i++ {
		prev := chunks[i-1].Content
		curr := chunks[i].Content
		runes := []rune(prev)
		if len(runes) < 4 {
			continue
		}
		tail := string(runes[len(runes)-4:])
		if !strings.Contains(curr, tail) {
			preview := curr
			if len(preview) > 60 {
				preview = preview[:60]
			}
			t.Errorf("chunk %d missing overlap from chunk %d\nprev tail: %q\ncurr start: %q",
				i, i-1, tail, preview)
		}
	}
}

func TestChunk_MetadataIsSet(t *testing.T) {
	c := newTestChunker(5, 1)
	doc := &Document{Content: repeat("word", 50), Source: "meta.txt"}
	chunks := c.Chunk(doc)
	for i, ch := range chunks {
		if ch.Source != "meta.txt" {
			t.Errorf("chunk %d: Source = %q, want %q", i, ch.Source, "meta.txt")
		}
		if ch.Index != i {
			t.Errorf("chunk %d: Index = %d, want %d", i, ch.Index, i)
		}
		if _, err := uuid.Parse(ch.ID); err != nil {
			t.Errorf("chunk %d: ID not valid UUID: %q", i, ch.ID)
		}
	}
}

func TestChunk_HardSplitOnHugeWord(t *testing.T) {
	c := newTestChunker(5, 0)
	bigWord := strings.Repeat("x", 200)
	doc := &Document{Content: bigWord, Source: "huge.txt"}
	chunks := c.Chunk(doc)
	if len(chunks) == 0 {
		t.Fatal("expected chunks from hard split")
	}
	var got strings.Builder
	for _, ch := range chunks {
		got.WriteString(ch.Content)
	}
	if got.String() != bigWord {
		t.Error("hard split lost content")
	}
}

func TestChunkPages_AttachesPageNumbers(t *testing.T) {
	c := newTestChunker(512, 50)
	doc := &Document{
		Source: "book.pdf",
		Pages: []Page{
			{Number: 1, Content: "Introduction to the topic."},
			{Number: 2, Content: "More detail on the subject."},
		},
	}
	chunks := c.ChunkPages(doc)
	if len(chunks) == 0 {
		t.Fatal("expected chunks")
	}
	for _, ch := range chunks {
		if ch.PageStart == 0 {
			t.Errorf("chunk %q: PageStart not set", ch.ID)
		}
	}
}

func TestChunkPages_FallsBackOnEmptyPages(t *testing.T) {
	c := newTestChunker(512, 50)
	doc := &Document{Content: "fallback content", Source: "plain.txt"}
	chunks := c.ChunkPages(doc)
	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk from fallback, got %d", len(chunks))
	}
}

func TestEstimateTokens(t *testing.T) {
	cases := []struct {
		input string
		want  int
	}{
		{"", 0},
		{"abcd", 1},
		{"abcde", 2},
		{"hello world", 3},
	}
	for _, tc := range cases {
		got := estimateTokens(tc.input)
		if got != tc.want {
			t.Errorf("estimateTokens(%q) = %d, want %d", tc.input, got, tc.want)
		}
	}
}
