// internal/llm/groq.go
package llm

import (
	"context"
	"fmt"
	"log"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/sashabaranov/go-openai" // Groq uses OpenAI-compatible API
)

const maxRetries = 4

type GroqLLM struct {
	client *openai.Client
	model  string
}

func NewGroqLLM(apiKey, model string) *GroqLLM {
	config := openai.DefaultConfig(apiKey)
	config.BaseURL = "https://api.groq.com/openai/v1"

	if model == "" {
		model = "meta-llama/llama-4-scout-17b-16e-instruct"
	}

	return &GroqLLM{
		client: openai.NewClientWithConfig(config),
		model:  model,
	}
}

func (g *GroqLLM) Complete(ctx context.Context, prompt string) (string, error) {
	var lastErr error

	for attempt := range maxRetries {
		resp, err := g.client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
			Model: g.model,
			Messages: []openai.ChatCompletionMessage{
				{Role: openai.ChatMessageRoleUser, Content: prompt},
			},
			MaxTokens: 1024,
		})

		if err == nil {
			return resp.Choices[0].Message.Content, nil
		}

		lastErr = err
		errStr := err.Error()

		// Only retry on 429 — fail fast on anything else (400, 401, 500...)
		if !strings.Contains(errStr, "429") {
			return "", fmt.Errorf("groq completion failed: %w", err)
		}

		wait := parseGroqRetryAfter(errStr)
		log.Printf("[Groq] TPM rate limit on attempt %d/%d, waiting %s...",
			attempt+1, maxRetries, wait.Round(time.Millisecond))

		select {
		case <-ctx.Done():
			return "", fmt.Errorf("context cancelled during groq retry: %w", ctx.Err())
		case <-time.After(wait):
		}
	}

	return "", fmt.Errorf("groq: exhausted %d retries: %w", maxRetries, lastErr)
}

func parseGroqRetryAfter(errMsg string) time.Duration {
	re := regexp.MustCompile(`try again in (\d+(?:\.\d+)?)s`)
	matches := re.FindStringSubmatch(errMsg)
	if len(matches) < 2 {
		return 10 * time.Second
	}

	seconds, err := strconv.ParseFloat(matches[1], 64)
	if err != nil {
		return 10 * time.Second
	}

	// Add 1s buffer on top of what Groq says
	return time.Duration(seconds*float64(time.Second)) + time.Second
}
