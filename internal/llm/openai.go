package llm

import (
	"context"
	"fmt"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"
)

type OpenAIClient struct {
	model  string
	client openai.Client
}

func NewOpenAIClient(apiKey string) *OpenAIClient {
	return &OpenAIClient{
		model: openai.ChatModelGPT4oMini,
		client: openai.NewClient(
			option.WithAPIKey(apiKey),
		),
	}
}

func (o *OpenAIClient) Generate(ctx context.Context, prompt string) (*GenerateResponse, error) {
	if prompt == "" {
		return nil, fmt.Errorf("prompt cannot be empty")
	}

	resp, err := o.client.Responses.New(ctx, responses.ResponseNewParams{
		Input: responses.ResponseNewParamsInputUnion{OfString: openai.String(prompt)},
		Model: o.model,
	})

	if err != nil {
		return nil, fmt.Errorf("error getting response from openai : %w", err)
	}

	return &GenerateResponse{
		Text:           resp.OutputText(),
		Model:          string(o.model),
		Provider:       "openai",
		InputTokens:    int(resp.Usage.InputTokens),
		OutputTokens:   int(resp.Usage.OutputTokens),
		UsageEstimated: false,
	}, nil
}
