package ingestion

import "strings"

type Chunk struct {
	Content string
}

type Chunker struct {
	MaxChunkSize int
	Overlap      int
	Separators   []string
}

func NewRecursiveChunker() *Chunker {
	return &Chunker{
		MaxChunkSize: 2048,
		Overlap:      200,
		Separators:   []string{"\n\n", "\n", ".", " ", ""},
	}
}

func (c *Chunker) Chunk(doc *Document) []Chunk {
	if doc == nil || strings.TrimSpace(doc.Content) == "" {
		return nil
	}

	if len(doc.Content) <= c.MaxChunkSize {
		return []Chunk{
			{Content: doc.Content},
		}
	}

	splits := c.splitText(doc.Content, c.Separators)
	return c.mergeSplits(splits)
}

func (c *Chunker) splitText(text string, separators []string) []string {
	if len(text) <= c.MaxChunkSize {
		return []string{text}
	}
	if len(separators) == 0 {
		return c.hardSplit(text)
	}

	sep := separators[0]
	if sep == "" {
		return c.hardSplit(text)
	}

	parts := strings.Split(text, sep)
	chunks := []string{}
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}

		if len(part) <= c.MaxChunkSize {
			chunks = append(chunks, part)
		} else {
			chunks = append(chunks, c.splitText(part, separators[1:])...)
		}
	}

	return chunks
}

func (c *Chunker) mergeSplits(splits []string) []Chunk {
	chunks := make([]Chunk, 0, len(splits))
	current := ""
	for _, split := range splits {
		split = strings.TrimSpace(split)
		if split == "" {
			continue
		}

		if current == "" {
			current = split
			continue
		}
		next := current + "\n\n" + split
		if len(next)+len(split) <= c.MaxChunkSize {
			current += "\n\n" + split
		}
		chunks = append(chunks, Chunk{Content: current})
		current = split
	}
	if current != "" {
		chunks = append(chunks, Chunk{Content: current})
	}
	return chunks
}

func (c *Chunker) hardSplit(text string) []string {
	if text == "" {
		return nil
	}

	chunks := []string{}
	for len(text) > c.MaxChunkSize {
		chunks = append(chunks, text[:c.MaxChunkSize])
		text = text[c.MaxChunkSize:]
	}
	if text != "" {
		chunks = append(chunks, text)
	}

	return chunks
}
