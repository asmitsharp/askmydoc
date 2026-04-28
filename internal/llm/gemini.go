package llm

import (
	"context"
	"fmt"
	"strings"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
)

type GeminiClient struct {
	model  string
	client *genai.Client
}

func NewGeminiClient(ctx context.Context, apiKey string) (*GeminiClient, error) {
	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		return nil, fmt.Errorf("failed to create gemini client: %w", err)
	}

	return &GeminiClient{
		model:  "gemini-2.5-flash",
		client: client,
	}, nil
}

func (g *GeminiClient) Generate(ctx context.Context, prompt string) (*GenerateResponse, error) {
	if prompt == "" {
		return nil, fmt.Errorf("prompt cannot be empty")
	}

	model := g.client.GenerativeModel(g.model)

	resp, err := model.GenerateContent(ctx, genai.Text(prompt))
	if err != nil {
		return nil, fmt.Errorf("error getting response from gemini : %w", err)
	}

	if len(resp.Candidates) == 0 || len(resp.Candidates[0].Content.Parts) == 0 {
		return nil, fmt.Errorf("no content returned from gemini")
	}

	part := resp.Candidates[0].Content.Parts[0]
	usageEstimated := false
	inputTokens := 0
	outputTokens := 0
	if resp.UsageMetadata != nil {
		inputTokens = int(resp.UsageMetadata.PromptTokenCount)
		outputTokens = int(resp.UsageMetadata.CandidatesTokenCount)
	} else {
		usageEstimated = true
		// fallback heuristic when SDK response has no usage metadata.
		inputTokens = estimateTokens(prompt)
	}

	if textPart, ok := part.(genai.Text); ok {
		text := string(textPart)
		if usageEstimated {
			outputTokens = estimateTokens(text)
		}
		return &GenerateResponse{
			Text:           text,
			Model:          g.model,
			Provider:       "gemini",
			InputTokens:    inputTokens,
			OutputTokens:   outputTokens,
			UsageEstimated: usageEstimated,
		}, nil
	}

	return nil, fmt.Errorf("unexpected content format returned from gemini")
}

func estimateTokens(text string) int {
	if strings.TrimSpace(text) == "" {
		return 0
	}
	// Approximation: ~4 characters per token for English-like text.
	return len(text)/4 + 1
}
