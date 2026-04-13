package ingestion

import (
	"fmt"
	"os"
)

type TextLoader struct{}

func (t *TextLoader) Load(path string) (*Document, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("text loader : %w", err)
	}
	return &Document{Content: string(data)}, nil
}

func (t *TextLoader) Supports(ext string) bool {
	return ext == ".txt"
}
