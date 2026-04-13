package embedding

import (
	"fmt"
	"strings"
)

const (
	ProviderOpenAI      = "openai"
	ProviderHuggingFace = "huggingface"
)

func NewClient(provider, apiKey string) (Embedding, error) {
	switch strings.ToLower(strings.TrimSpace(provider)) {
	case ProviderOpenAI:
		return NewOpenAIClient(apiKey), nil
	case ProviderHuggingFace, "hf":
		return NewHFClient(apiKey), nil
	default:
		return nil, fmt.Errorf("unsupported embedding provider %q", provider)
	}
}
