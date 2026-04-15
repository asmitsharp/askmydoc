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
)

type Handler struct {
	ingest *pipeline.PipeLine
	query  *pipeline.QueryPipeLine
}

func NewHandler(ingest *pipeline.PipeLine, query *pipeline.QueryPipeLine) *Handler {
	return &Handler{
		ingest: ingest,
		query:  query,
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
	Question string `json:"question"`
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
	}
	start := time.Now()
	answer, citations, queryErr := h.query.Execute(r.Context(), req.Question)
	if queryErr != nil {
		writeError(w, http.StatusInternalServerError, queryErr.Error())
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
	defer os.Remove(tmpFile.Name())

	_, err = io.Copy(tmpFile, file)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to save uploaded file")
		return
	}
	tmpFile.Close()

	numChunks, err := h.ingest.Ingest(r.Context(), tmpFile.Name())
	if err != nil {
		writeError(w, http.StatusInternalServerError, fmt.Sprintf("ingestion failed: %v", err))
		return
	}

	resp := IngestResponse{
		Status:   "ingested",
		Filename: header.Filename,
		Message:  fmt.Sprintf("Successfully ingested %d chunks", numChunks),
	}
	writeJSON(w, http.StatusOK, resp)

}
func writeJSON(w http.ResponseWriter, status int, data any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}
