package task

import (
	"encoding/json"

	"github.com/hibiken/asynq"
)

const (
	TypeDocumentIngestion = "document:ingest"
)

// DocumentIngestionPayload holds the data needed for the worker to process the file
type DocumentIngestionPayload struct {
	FilePath         string `json:"file_path"`
	OriginalFilename string `json:"original_filename"`
}

// NewDocumentIngestionTask creates a new Asynq task for document ingestion
func NewDocumentIngestionTask(filePath, originalFilename string) (*asynq.Task, error) {
	payload, err := json.Marshal(DocumentIngestionPayload{
		FilePath:         filePath,
		OriginalFilename: originalFilename,
	})
	if err != nil {
		return nil, err
	}
	return asynq.NewTask(TypeDocumentIngestion, payload), nil
}
