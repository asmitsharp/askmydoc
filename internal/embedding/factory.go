package embedding

import (
	"context"
	"fmt"
	"strings"
)

const (
	ProviderOpenAI      = "openai"
	ProviderHuggingFace = "huggingface"
	ProviderGemini      = "gemini"
)

func NewClient(provider, apiKey string) (Embedding, error) {
	switch strings.ToLower(strings.TrimSpace(provider)) {
	case ProviderOpenAI:
		return NewOpenAIClient(apiKey), nil
	case ProviderHuggingFace, "hf":
		return NewHFClient(apiKey), nil
	case ProviderGemini:
		importCtx := context.Background()
		return NewGeminiClient(importCtx, apiKey)
	default:
		return nil, fmt.Errorf("unsupported embedding provider %q", provider)
	}
}
