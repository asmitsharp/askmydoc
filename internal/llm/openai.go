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

func (o *OpenAIClient) Complete(ctx context.Context, prompt string) (string, error) {
	if prompt == "" {
		return "", fmt.Errorf("prompt cannot be empty")
	}

	resp, err := o.client.Responses.New(ctx, responses.ResponseNewParams{
		Input: responses.ResponseNewParamsInputUnion{OfString: openai.String(prompt)},
		Model: o.model,
	})

	if err != nil {
		return "", fmt.Errorf("error getting response from openai : %w", err)
	}

	return resp.OutputText(), nil
}
