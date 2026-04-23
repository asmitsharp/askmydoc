package eval

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ashmitsharp/askmydocs/internal/llm"
)

// Judge implements the LLM-as-a-Judge pattern for evaluating answer faithfulness.
type Judge struct {
	llm llm.LLM
}

// JudgeResponse is the structured JSON response from the judge LLM.
type JudgeResponse struct {
	Score  float64 `json:"score"`
	Reason string  `json:"reason"`
}

func NewJudge(llmClient llm.LLM) *Judge {
	return &Judge{llm: llmClient}
}

// Evaluate scores the faithfulness of an answer against the retrieved context chunks.
func (j *Judge) Evaluate(ctx context.Context, answer string, chunks []string) (float64, error) {
	prompt := buildJudgePrompt(answer, chunks)

	response, err := j.llm.Complete(ctx, prompt)
	if err != nil {
		return 0, fmt.Errorf("judge LLM call failed: %w", err)
	}

	score, err := parseJudgeResponse(response)
	if err != nil {
		return 0, fmt.Errorf("failed to parse judge response: %w", err)
	}

	return score, nil
}

func buildJudgePrompt(answer string, chunks []string) string {
	var sb strings.Builder

	sb.WriteString(`You are an impartial evaluation judge for a Retrieval-Augmented Generation system.

Your task: Determine if the ANSWER is fully supported by the CONTEXT provided.

## CONTEXT (Retrieved Chunks):
`)

	for i, chunk := range chunks {
		truncated := chunk
		if len(chunk) > 800 {
			truncated = chunk[:800] + "..."
		}
		fmt.Fprintf(&sb, "--- Chunk %d ---\n%s\n\n", i+1, truncated)
	}

	sb.WriteString("## ANSWER TO EVALUATE:\n")
	sb.WriteString(answer)
	sb.WriteString(`

## INSTRUCTIONS:
1. Break the answer into individual claims/statements.
2. For each claim, check if it is directly supported by the context above.
3. A claim about something NOT being in the documents is valid only if
   the context genuinely does not contain that information.
4. Ignore citation formatting — focus only on factual content.

Reply with ONLY a JSON object in this exact format:
{"score": <float between 0.0 and 1.0>, "reason": "<brief explanation>"}

Where score = (number of supported claims) / (total claims).
If all claims are supported, score = 1.0.
If the answer says "I don't know" and the context truly lacks the info, score = 1.0.
`)

	return sb.String()
}

func parseJudgeResponse(response string) (float64, error) {
	response = strings.TrimSpace(response)

	var judgeResp JudgeResponse
	if err := json.Unmarshal([]byte(response), &judgeResp); err == nil {
		return clampScore(judgeResp.Score), nil
	}

	// If direct parse fails we try to extract JSON from markdown code blocks or surrounding text
	start := strings.Index(response, "{")
	end := strings.LastIndex(response, "}")
	if start != -1 && end != -1 && end > start {
		jsonStr := response[start : end+1]
		if err := json.Unmarshal([]byte(jsonStr), &judgeResp); err == nil {
			return clampScore(judgeResp.Score), nil
		}
	}

	return 0, fmt.Errorf("could not extract valid JSON score from judge response: %s", response)
}

// clampScore ensures the score is within the valid [0.0, 1.0] range.
func clampScore(score float64) float64 {
	if score < 0 {
		return 0
	}
	if score > 1 {
		return 1
	}
	return score
}
