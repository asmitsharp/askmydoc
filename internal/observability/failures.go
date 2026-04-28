package observability

import (
	"context"
	"errors"
	"strings"
)

type FailureReason string

const (
	FailureUnknown FailureReason = "unknown"

	FailureBadRequest          FailureReason = "bad_request"
	FailureRequestTimeout      FailureReason = "request_timeout"
	FailureRequestCancelled    FailureReason = "request_cancelled"
	FailureValidationFailed    FailureReason = "validation_failed"
	FailureNoRelevantResults   FailureReason = "no_relevant_results"
	FailureTokenBudgetExceeded FailureReason = "token_budget_exceeded"

	FailureEmbeddingTimeout     FailureReason = "embedding_timeout"
	FailureEmbeddingUnavailable FailureReason = "embedding_unavailable"
	FailureEmbeddingProvider    FailureReason = "embedding_provider_error"

	FailureQdrantUnavailable FailureReason = "qdrant_unavailable"
	FailureQdrantTimeout     FailureReason = "qdrant_timeout"
	FailureQdrantQuery       FailureReason = "qdrant_query_error"
	FailureQdrantUpsert      FailureReason = "qdrant_upsert_error"

	FailurePostgresUnavailable FailureReason = "postgres_unavailable"
	FailurePostgresTimeout     FailureReason = "postgres_timeout"
	FailureBM25Query           FailureReason = "bm25_query_error"

	FailureRerankerTimeout     FailureReason = "reranker_timeout"
	FailureRerankerUnavailable FailureReason = "reranker_unavailable"
	FailureRerankerProvider    FailureReason = "reranker_provider_error"

	FailureLLMTimeout     FailureReason = "llm_timeout"
	FailureLLMUnavailable FailureReason = "llm_unavailable"
	FailureLLMProvider    FailureReason = "llm_provider_error"
	FailureLLMParse       FailureReason = "llm_parse_error"

	FailureIngestionLoad    FailureReason = "ingestion_load_error"
	FailureIngestionChunk   FailureReason = "ingestion_chunk_error"
	FailureIngestionOCR     FailureReason = "ingestion_ocr_error"
	FailureIngestionPersist FailureReason = "ingestion_persist_error"
	FailureTaskQueue        FailureReason = "task_queue_error"
)

func (f FailureReason) String() string {
	if f == "" {
		return string(FailureUnknown)
	}
	return string(f)
}

type AppError struct {
	Reason  FailureReason
	Message string
	Err     error
}

func (e *AppError) Error() string {
	if e == nil {
		return ""
	}
	if e.Message == "" && e.Err != nil {
		return e.Err.Error()
	}
	if e.Err == nil {
		return e.Message
	}
	return e.Message + ": " + e.Err.Error()
}

func (e *AppError) Unwrap() error { return e.Err }

func NewFailure(reason FailureReason, message string, err error) error {
	return &AppError{Reason: reason, Message: message, Err: err}
}

func ReasonOf(err error) FailureReason {
	if err == nil {
		return FailureUnknown
	}
	var appErr *AppError
	if errors.As(err, &appErr) && appErr.Reason != "" {
		return appErr.Reason
	}
	if errors.Is(err, context.Canceled) {
		return FailureRequestCancelled
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return FailureRequestTimeout
	}
	l := strings.ToLower(err.Error())
	switch {
	case strings.Contains(l, "token budget"), strings.Contains(l, "budget exceeded"):
		return FailureTokenBudgetExceeded
	case strings.Contains(l, "qdrant"), strings.Contains(l, "vector search"):
		return FailureQdrantUnavailable
	case strings.Contains(l, "bm25"), strings.Contains(l, "postgres"):
		return FailureBM25Query
	case strings.Contains(l, "rerank"):
		return FailureRerankerProvider
	case strings.Contains(l, "embedding"):
		return FailureEmbeddingProvider
	case strings.Contains(l, "llm"), strings.Contains(l, "openai"), strings.Contains(l, "gemini"), strings.Contains(l, "groq"):
		return FailureLLMProvider
	case strings.Contains(l, "no relevant results"):
		return FailureNoRelevantResults
	default:
		return FailureUnknown
	}
}
