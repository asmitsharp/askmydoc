package embedding

import "context"

type Embedding interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
	GetModelName() string
}

var (
	_ Embedding = (*HFClient)(nil)
	_ Embedding = (*OpenAIClient)(nil)
)
