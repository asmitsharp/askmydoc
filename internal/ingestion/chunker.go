package ingestion

import (
	"strings"
	"unicode/utf8"

	"github.com/google/uuid"
)

type Chunk struct {
	Content   string
	ID        string
	Index     int
	Source    string
	PageStart int
	PageEnd   int
}

type ChunkerOption func(*Chunker)

func WithMaxChunkTokens(n int) ChunkerOption { return func(c *Chunker) { c.MaxChunkTokens = n } }
func WithOverlapTokens(n int) ChunkerOption  { return func(c *Chunker) { c.OverlapTokens = n } }
func WithSeparators(seps []string) ChunkerOption {
	return func(c *Chunker) { c.Separators = seps }
}

type Chunker struct {
	MaxChunkTokens int
	OverlapTokens  int
	Separators     []string
}

func NewMarkdownChunker(opts ...ChunkerOption) *Chunker {
	c := &Chunker{
		MaxChunkTokens: 2048,
		OverlapTokens:  50,
		Separators: []string{
			"\n# ",
			"\n## ",
			"\n### ",
			"\n#### ",
			"\n\n",
			"\n",
			". ",
			" ",
			"",
		},
	}

	for _, opt := range opts {
		opt(c)
	}
	return c
}

func NewTextChunker(opts ...ChunkerOption) *Chunker {
	c := &Chunker{
		MaxChunkTokens: 512,
		OverlapTokens:  50,
		Separators:     []string{"\n\n", "\n", ". ", " ", ""},
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

func (c *Chunker) Chunk(doc *Document) []Chunk {
	if doc == nil || strings.TrimSpace(doc.Content) == "" {
		return nil
	}

	if EstimateTokens(doc.Content) <= c.MaxChunkTokens {
		return []Chunk{
			{
				ID:      chunkID(doc.Content),
				Content: doc.Content,
				Index:   0,
				Source:  doc.Source,
			},
		}
	}

	splits := c.splitText(doc.Content, c.Separators)
	return c.mergeSplits(splits, doc.Source)
}

func (c *Chunker) ChunkPages(doc *Document) []Chunk {
	if doc == nil || len(doc.Pages) == 0 {
		return c.Chunk(doc)
	}

	var all []Chunk
	globalIndex := 0

	for _, page := range doc.Pages {
		if strings.TrimSpace(page.Content) == "" {
			continue
		}

		splits := c.splitText(page.Content, c.Separators)
		pageChunks := c.mergeSplits(splits, doc.Source)

		for _, ch := range pageChunks {
			ch.Index = globalIndex
			ch.ID = chunkID(ch.Content)
			ch.PageStart = page.Number
			ch.PageEnd = page.Number
			all = append(all, ch)
			globalIndex++
		}
	}

	return all
}

func (c *Chunker) splitText(text string, separators []string) []string {
	if EstimateTokens(text) <= c.MaxChunkTokens {
		return []string{text}
	}
	if len(separators) == 0 {
		return c.hardSplit(text)
	}

	sep := separators[0]
	if sep == "" {
		return c.hardSplit(text)
	}

	var result []string
	for _, part := range strings.Split(text, sep) {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}

		if EstimateTokens(part) <= c.MaxChunkTokens {
			result = append(result, part)
		} else {
			result = append(result, c.splitText(part, separators[1:])...)
		}
	}

	return result
}

func (c *Chunker) mergeSplits(splits []string, source string) []Chunk {
	var chunks []Chunk
	current := ""
	overlap := ""

	flush := func() {
		if current == "" {
			return
		}

		idx := len(chunks)
		chunks = append(chunks, Chunk{
			ID:      chunkID(current),
			Content: current,
			Index:   idx,
			Source:  source,
		})

		runes := []rune(current)
		if c.OverlapTokens > 0 && len(runes) > c.OverlapTokens {
			overlap = string(runes[len(runes)-c.OverlapTokens:])
		} else {
			overlap = current
		}
		current = ""
	}

	for _, split := range splits {
		split = strings.TrimSpace(split)
		if split == "" {
			continue
		}

		if current == "" && overlap != "" {
			current = overlap
		}

		candidate := split
		if current != "" {
			candidate = current + "\n\n" + split
		}

		if EstimateTokens(candidate) <= c.MaxChunkTokens {
			current = candidate
		} else {
			flush()
			current = split
		}
	}

	flush()
	return chunks
}

func (c *Chunker) hardSplit(text string) []string {
	if text == "" {
		return nil
	}

	runes := []rune(text)
	runeLimit := c.MaxChunkTokens * 4

	var chunks []string
	for len(runes) > 0 {
		end := runeLimit
		if end > len(runes) {
			end = len(runes)
		}
		chunks = append(chunks, string(runes[:end]))
		runes = runes[end:]
	}
	return chunks
}

func estimateTokens(text string) int {
	count := utf8.RuneCountInString(text)
	if count == 0 {
		return 0
	}
	return (count + 3) / 4
}

func EstimateTokens(text string) int {
	return estimateTokens(text)
}

// docNamespace is a UUID namespace for document chunks.
// This ensures consistent UUIDs across runs for the same source+index.
var docNamespace = uuid.Must(uuid.Parse("6ba7b810-9dad-11d1-80b4-00c04fd430c8"))

// chunkID generates a deterministic UUID v5 for a chunk.
// The same source and index will always produce the same UUID,
// making re-indexing idempotent and compatible with Qdrant.
func chunkID(content string) string {
	// Create a namespace for this source to avoid collisions
	// across different sources
	finalUUID := uuid.NewSHA1(docNamespace, []byte(content))
	return finalUUID.String()
}
