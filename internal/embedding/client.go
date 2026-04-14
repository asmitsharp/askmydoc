package embedding

import "context"

// Embed takes a list of texts and returns embeddings in the same order.
type Embedding interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
	GetModelName() string
}

var (
	_ Embedding = (*HFClient)(nil)
	_ Embedding = (*OpenAIClient)(nil)
)
