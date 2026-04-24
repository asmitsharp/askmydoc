package api

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/ashmitsharp/askmydocs/internal/pipeline"
	"github.com/ashmitsharp/askmydocs/internal/storage"
	"github.com/ashmitsharp/askmydocs/internal/task"
	"github.com/hibiken/asynq"
)

type Handler struct {
	ingest      *pipeline.PipeLine
	query       *pipeline.QueryPipeLine
	asynqClient *asynq.Client
}

func NewHandler(ingest *pipeline.PipeLine, query *pipeline.QueryPipeLine, asynqClient *asynq.Client) *Handler {
	return &Handler{
		ingest:      ingest,
		query:       query,
		asynqClient: asynqClient,
	}
}

type QueryResponse struct {
	Answer    string         `json:"answer"`
	Citation  []citationJson `json:"citations"`
	LatencyMs int64          `json:"latency_ms"`
}

type citationJson struct {
	Source     string  `json:"source"`
	ChunkIndex int     `json:"chunk_index"`
	Score      float32 `json:"score"`
	Content    string  `json:"content"`
}

type QueryRequest struct {
	Question string                  `json:"question"`
	Filter   *storage.MetadataFilter `json:"filter,omitempty"`
}

type IngestResponse struct {
	Status   string `json:"status"`
	Filename string `json:"filename"`
	Message  string `json:"message"`
}

func (h *Handler) Routes() *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /ingest", h.HandleIngest)
	mux.HandleFunc("POST /query", h.HandleQuery)
	mux.HandleFunc("GET /health", h.HandleHealth)
	return mux
}

func (h *Handler) HandleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"status":    "ok",
		"vector_db": true,
		"llm":       true,
	})
}

func (h *Handler) HandleQuery(w http.ResponseWriter, r *http.Request) {
	req := QueryRequest{}
	err := json.NewDecoder(r.Body).Decode(&req)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if strings.TrimSpace(req.Question) == "" {
		writeError(w, http.StatusBadRequest, "question cannot be empty")
		return
	}
	start := time.Now()
	answer, citations, queryErr := h.query.Execute(r.Context(), req.Question, req.Filter)
	if queryErr != nil {
		writeError(w, http.StatusInternalServerError, queryErr.Error())
		return
	}
	latency := time.Since(start).Milliseconds()
	citationsJSON := make([]citationJson, len(citations))
	for i, citation := range citations {
		citationsJSON[i] = citationJson{
			Source:     citation.Source,
			ChunkIndex: citation.ChunkIndex,
			Score:      citation.Score,
			Content:    citation.Content,
		}
	}

	resp := QueryResponse{
		Answer:    answer.Text,
		Citation:  citationsJSON,
		LatencyMs: latency,
	}
	writeJSON(w, http.StatusOK, resp)
}

func (h *Handler) HandleIngest(w http.ResponseWriter, r *http.Request) {
	err := r.ParseMultipartForm(32 << 20)
	if err != nil {
		writeError(w, http.StatusBadRequest, "failed to parse form")
		return
	}

	file, header, fileErr := r.FormFile("file")
	if fileErr != nil {
		writeError(w, http.StatusBadRequest, "file field is required")
		return
	}
	defer file.Close()

	tmpFile, tmpFileErr := os.CreateTemp("", "askmydoc-upload-*"+filepath.Ext(header.Filename))
	if tmpFileErr != nil {
		writeError(w, http.StatusInternalServerError, "failed to create temp file")
		return
	}
	// NOTE: We do not defer os.Remove here because the worker needs the file.

	_, err = io.Copy(tmpFile, file)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to save uploaded file")
		return
	}
	tmpFile.Close()

	t, err := task.NewDocumentIngestionTask(tmpFile.Name(), header.Filename)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to create task")
		return
	}

	info, err := h.asynqClient.Enqueue(t)
	if err != nil {
		writeError(w, http.StatusInternalServerError, fmt.Sprintf("failed to enqueue task: %v", err))
		return
	}

	resp := map[string]string{
		"status":  "processing",
		"job_id":  info.ID,
		"message": "File uploaded and queued for processing",
	}
	writeJSON(w, http.StatusAccepted, resp)

}
func writeJSON(w http.ResponseWriter, status int, data any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}
