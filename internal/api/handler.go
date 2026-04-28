package api

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/ashmitsharp/askmydocs/internal/observability"
	"github.com/ashmitsharp/askmydocs/internal/pipeline"
	"github.com/ashmitsharp/askmydocs/internal/storage"
	"github.com/ashmitsharp/askmydocs/internal/task"
	"github.com/hibiken/asynq"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.opentelemetry.io/otel/trace"
)

type Handler struct {
	ingest      *pipeline.PipeLine
	query       *pipeline.QueryPipeLine
	asynqClient *asynq.Client
	metrics     *observability.Metrics
}

func NewHandler(ingest *pipeline.PipeLine, query *pipeline.QueryPipeLine, asynqClient *asynq.Client, metrics *observability.Metrics) *Handler {
	return &Handler{
		ingest:      ingest,
		query:       query,
		asynqClient: asynqClient,
		metrics:     metrics,
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
	mux.HandleFunc("GET /metrics", promhttp.Handler().ServeHTTP)
	return h.withTraceHeader(h.withRequestMetrics(mux))
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
		writeError(r.Context(), w, http.StatusBadRequest, err.Error(), observability.FailureBadRequest)
		return
	}
	if strings.TrimSpace(req.Question) == "" {
		writeError(r.Context(), w, http.StatusBadRequest, "question cannot be empty", observability.FailureValidationFailed)
		return
	}
	start := time.Now()
	answer, citations, queryErr := h.query.Execute(r.Context(), req.Question, req.Filter)
	if queryErr != nil {
		reason := observability.ReasonOf(queryErr)
		writeError(r.Context(), w, http.StatusInternalServerError, queryErr.Error(), reason)
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
		writeError(r.Context(), w, http.StatusBadRequest, "failed to parse form", observability.FailureBadRequest)
		return
	}

	file, header, fileErr := r.FormFile("file")
	if fileErr != nil {
		writeError(r.Context(), w, http.StatusBadRequest, "file field is required", observability.FailureBadRequest)
		return
	}
	defer file.Close()

	tmpFile, tmpFileErr := os.CreateTemp("", "askmydoc-upload-*"+filepath.Ext(header.Filename))
	if tmpFileErr != nil {
		writeError(r.Context(), w, http.StatusInternalServerError, "failed to create temp file", observability.FailureIngestionLoad)
		return
	}
	// NOTE: We do not defer os.Remove here because the worker needs the file.

	_, err = io.Copy(tmpFile, file)
	if err != nil {
		writeError(r.Context(), w, http.StatusInternalServerError, "failed to save uploaded file", observability.FailureIngestionLoad)
		return
	}
	tmpFile.Close()

	t, err := task.NewDocumentIngestionTask(tmpFile.Name(), header.Filename)
	if err != nil {
		writeError(r.Context(), w, http.StatusInternalServerError, "failed to create task", observability.FailureTaskQueue)
		return
	}

	info, err := h.asynqClient.Enqueue(t)
	if err != nil {
		writeError(r.Context(), w, http.StatusInternalServerError, fmt.Sprintf("failed to enqueue task: %v", err), observability.FailureTaskQueue)
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
	_ = json.NewEncoder(w).Encode(data)
}

func writeError(ctx context.Context, w http.ResponseWriter, status int, msg string, reason observability.FailureReason) {
	traceID := currentTraceID(ctx)
	writeJSON(w, status, map[string]string{
		"error":          msg,
		"failure_reason": reason.String(),
		"trace_id":       traceID,
	})
}

func (h *Handler) withTraceHeader(next http.Handler) *http.ServeMux {
	mux := http.NewServeMux()
	mux.Handle("/", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if traceID := currentTraceID(r.Context()); traceID != "" {
			w.Header().Set("X-Trace-Id", traceID)
		}
		next.ServeHTTP(w, r)
	}))
	return mux
}

func (h *Handler) withRequestMetrics(next http.Handler) *http.ServeMux {
	mux := http.NewServeMux()
	mux.Handle("/", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if h.metrics == nil || r.URL.Path != "/query" {
			next.ServeHTTP(w, r)
			return
		}
		start := time.Now()
		recorder := &statusRecorder{ResponseWriter: w, statusCode: http.StatusOK}
		next.ServeHTTP(recorder, r)
		statusLabel := "success"
		if recorder.statusCode >= 400 {
			statusLabel = "error"
		}
		h.metrics.RequestsTotal.WithLabelValues(statusLabel).Inc()
		h.metrics.RequestLatencySeconds.WithLabelValues(statusLabel).Observe(time.Since(start).Seconds())
	}))
	return mux
}

type statusRecorder struct {
	http.ResponseWriter
	statusCode int
}

func (r *statusRecorder) WriteHeader(statusCode int) {
	r.statusCode = statusCode
	r.ResponseWriter.WriteHeader(statusCode)
}

func currentTraceID(ctx context.Context) string {
	spanCtx := trace.SpanContextFromContext(ctx)
	if !spanCtx.HasTraceID() {
		return ""
	}
	return spanCtx.TraceID().String()
}
