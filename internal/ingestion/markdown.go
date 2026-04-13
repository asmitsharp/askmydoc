package ingestion

import (
	"fmt"
	"os"
)

type MarkdownLoader struct{}

func (m MarkdownLoader) Load(path string) (*Document, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to load the file : %w", err)
	}

	document := &Document{
		Content: string(content),
	}

	return document, nil
}

func (m MarkdownLoader) Supports(ext string) bool {
	return ext == ".md" || ext == ".markdown"
}
