package llm

import "context"

type GenerateResponse struct {
	Text           string
	Model          string
	Provider       string
	InputTokens    int
	OutputTokens   int
	CostUSD        float64
	UsageEstimated bool
}

type LLM interface {
	Generate(ctx context.Context, prompt string) (*GenerateResponse, error)
}
