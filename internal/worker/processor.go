package worker

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/ashmitsharp/askmydocs/internal/pipeline"
	"github.com/ashmitsharp/askmydocs/internal/task"
	"github.com/hibiken/asynq"
)

// TaskProcessor implements asynq.Handler
type TaskProcessor struct {
	pipeline *pipeline.PipeLine
}

func NewTaskProcessor(pipeline *pipeline.PipeLine) *TaskProcessor {
	return &TaskProcessor{
		pipeline: pipeline,
	}
}

// ProcessTask is the function that gets called when a job is pulled from Redis
func (processor *TaskProcessor) ProcessTask(ctx context.Context, t *asynq.Task) error {
	switch t.Type() {
	case task.TypeDocumentIngestion:
		return processor.handleDocumentIngestion(ctx, t)
	default:
		return fmt.Errorf("unexpected task type: %s", t.Type())
	}
}

func (processor *TaskProcessor) handleDocumentIngestion(ctx context.Context, t *asynq.Task) error {
	var payload task.DocumentIngestionPayload
	if err := json.Unmarshal(t.Payload(), &payload); err != nil {
		return fmt.Errorf("json.Unmarshal failed: %v: %w", err, asynq.SkipRetry)
	}

	fmt.Printf("Worker processing file: %s\n", payload.OriginalFilename)

	_, err := processor.pipeline.Ingest(ctx, payload.FilePath, payload.OriginalFilename)
	if err != nil {
		return fmt.Errorf("pipeline.Ingest failed: %w", err)
	}

	if err := os.Remove(payload.FilePath); err != nil {
		fmt.Printf("warning: failed to delete file %s: %v\n", payload.FilePath, err)
	}

	fmt.Printf("Successfully finished processing %s\n", payload.OriginalFilename)
	return nil
}
