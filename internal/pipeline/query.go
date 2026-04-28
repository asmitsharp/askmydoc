package pipeline

import (
	"context"
	"fmt"
	"log"
	"regexp"
	"strings"
	"time"

	"github.com/ashmitsharp/askmydocs/internal/embedding"
	"github.com/ashmitsharp/askmydocs/internal/ingestion"
	"github.com/ashmitsharp/askmydocs/internal/llm"
	"github.com/ashmitsharp/askmydocs/internal/observability"
	"github.com/ashmitsharp/askmydocs/internal/retrieval"
	"github.com/ashmitsharp/askmydocs/internal/storage"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

type Answer struct {
	Text string
}

type Citation struct {
	Source     string
	ChunkIndex int
	Score      float32
	Content    string
}

type Timings struct {
	RetrievalMs  int64
	GenerationMs int64
	TotalMs      int64
}

type QueryPipeLine struct {
	embedder   embedding.Embedding
	store      storage.VectorStore
	bm25Store  *storage.BM25Store
	llm        llm.LLM
	reranker   retrieval.Reranker
	topK       int
	maxTokens  int
	fusionTopN int
	rerankTopN int
	rerankSem  chan struct{}
	llmSem     chan struct{}
	metrics    *observability.Metrics
	costCalc   *observability.CostCalculator
	budgetUSD  float64
	tracer     trace.Tracer
}

func NewQueryPipeLine(
	embedder embedding.Embedding,
	store storage.VectorStore,
	bm25Store *storage.BM25Store,
	llm llm.LLM,
	reranker retrieval.Reranker,
	metrics *observability.Metrics,
	costCalc *observability.CostCalculator,
) *QueryPipeLine {
	if costCalc == nil {
		costCalc = observability.NewCostCalculator()
	}
	return &QueryPipeLine{
		embedder:   embedder,
		store:      store,
		bm25Store:  bm25Store,
		llm:        llm,
		reranker:   reranker,
		topK:       5,
		maxTokens:  3000,
		fusionTopN: 20,
		rerankTopN: 5,
		rerankSem:  make(chan struct{}, 1),
		llmSem:     make(chan struct{}, 2),
		metrics:    metrics,
		costCalc:   costCalc,
		budgetUSD:  0.05,
		tracer:     otel.Tracer("askmydocs.pipeline"),
	}
}

// acquireSem acquires a semaphore slot, respecting context cancellation.
func acquireSem(ctx context.Context, sem chan struct{}) error {
	select {
	case sem <- struct{}{}:
		return nil
	case <-ctx.Done():
		return fmt.Errorf("context cancelled while waiting for API slot: %w", ctx.Err())
	}
}

func releaseSem(sem chan struct{}) {
	<-sem
}

// Execute runs the full RAG pipeline. Timings are always collected internally;
// use ExecuteWithTimings if you need them exposed.
func (q *QueryPipeLine) Execute(
	ctx context.Context,
	question string,
	filter *storage.MetadataFilter,
) (*Answer, []Citation, error) {
	answer, citations, _, err := q.execute(ctx, question, filter)
	return answer, citations, err
}

// ExecuteWithTimings is the same as Execute but also returns timing breakdowns.
func (q *QueryPipeLine) ExecuteWithTimings(
	ctx context.Context,
	question string,
	filter *storage.MetadataFilter,
) (*Answer, []Citation, *Timings, error) {
	return q.execute(ctx, question, filter)
}

// execute is the single internal implementation used by both public methods.
func (q *QueryPipeLine) execute(
	ctx context.Context,
	question string,
	filter *storage.MetadataFilter,
) (*Answer, []Citation, *Timings, error) {
	ctx, rootSpan := q.tracer.Start(ctx, "rag.query")
	rootSpan.SetAttributes(
		attribute.Int("question.length", len(question)),
		attribute.Int("fusion_top_n", q.fusionTopN),
		attribute.Int("rerank_top_n", q.rerankTopN),
	)
	defer rootSpan.End()
	totalStart := time.Now()

	// ── Retrieval phase ──────────────────────────────────────────────────────

	retrievalStart := time.Now()

	selected, err := q.retrieve(ctx, question, filter)
	if err != nil {
		reason := observability.ReasonOf(err)
		rootSpan.RecordError(err)
		rootSpan.SetStatus(codes.Error, reason.String())
		if q.metrics != nil {
			q.metrics.RequestFailuresTotal.WithLabelValues("retrieve", reason.String()).Inc()
		}
		return nil, nil, nil, err
	}

	retrievalDuration := time.Since(retrievalStart)

	// ── Generation phase ─────────────────────────────────────────────────────

	generationStart := time.Now()

	answer, citations, err := q.generate(ctx, question, selected)
	if err != nil {
		reason := observability.ReasonOf(err)
		rootSpan.RecordError(err)
		rootSpan.SetStatus(codes.Error, reason.String())
		if q.metrics != nil {
			q.metrics.RequestFailuresTotal.WithLabelValues("generate", reason.String()).Inc()
		}
		return nil, nil, nil, err
	}

	generationDuration := time.Since(generationStart)

	timings := &Timings{
		RetrievalMs:  retrievalDuration.Milliseconds(),
		GenerationMs: generationDuration.Milliseconds(),
		TotalMs:      time.Since(totalStart).Milliseconds(),
	}

	return answer, citations, timings, nil
}

// retrieve handles embedding, dual search, fusion, reranking, and token budgeting.
func (q *QueryPipeLine) retrieve(
	ctx context.Context,
	question string,
	filter *storage.MetadataFilter,
) ([]Citation, error) {
	embedStart := time.Now()
	embedCtx, embedSpan := q.tracer.Start(ctx, "embed_query")
	questionVector, err := q.embedder.Embed(embedCtx, []string{question})
	embedSpan.End()
	if q.metrics != nil {
		q.metrics.StageLatencySeconds.WithLabelValues("embed_query").Observe(time.Since(embedStart).Seconds())
	}
	if err != nil {
		return nil, observability.NewFailure(observability.FailureEmbeddingProvider, "question embedding failed", err)
	}

	// Run vector search and BM25 concurrently.
	// Use a struct to carry both result and error so we don't need separate channels.
	type searchResult struct {
		results []storage.SearchResult
		err     error
	}

	vectorCh := make(chan searchResult, 1)
	bm25Ch := make(chan searchResult, 1)

	go func() {
		start := time.Now()
		stageCtx, span := q.tracer.Start(ctx, "vector_search")
		log.Println("[DEBUG] Qdrant vector search starting...")
		results, err := q.store.Search(stageCtx, questionVector[0], q.fusionTopN, filter)
		span.End()
		if q.metrics != nil {
			q.metrics.StageLatencySeconds.WithLabelValues("vector_search").Observe(time.Since(start).Seconds())
		}
		log.Printf("[DEBUG] Qdrant vector search done (%d results, err=%v)", len(results), err)
		vectorCh <- searchResult{results, err}
	}()

	go func() {
		start := time.Now()
		stageCtx, span := q.tracer.Start(ctx, "bm25_search")
		log.Println("[DEBUG] BM25 search starting...")
		results, err := q.bm25Store.Search(stageCtx, question, q.fusionTopN, filter)
		span.End()
		if q.metrics != nil {
			q.metrics.StageLatencySeconds.WithLabelValues("bm25_search").Observe(time.Since(start).Seconds())
		}
		log.Printf("[DEBUG] BM25 search done (%d results, err=%v)", len(results), err)
		bm25Ch <- searchResult{results, err}
	}()

	// Collect both results. The select pattern you had before could miss
	// the second error if both goroutines failed simultaneously.
	var vectorRes, bm25Res searchResult
	for range 2 {
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("search cancelled: %w", ctx.Err())
		case r := <-vectorCh:
			vectorRes = r
		case r := <-bm25Ch:
			bm25Res = r
		}
	}

	if vectorRes.err != nil {
		return nil, observability.NewFailure(observability.FailureQdrantQuery, "vector search error", vectorRes.err)
	}
	if bm25Res.err != nil {
		return nil, observability.NewFailure(observability.FailureBM25Query, "bm25 search error", bm25Res.err)
	}

	// Fuse
	fusionStart := time.Now()
	_, fusionSpan := q.tracer.Start(ctx, "rrf_fusion")
	fused := retrieval.ReciprocalRankFusion(vectorRes.results, bm25Res.results, q.fusionTopN)
	fusionSpan.End()
	if q.metrics != nil {
		q.metrics.StageLatencySeconds.WithLabelValues("rrf_fusion").Observe(time.Since(fusionStart).Seconds())
	}
	log.Printf("[DEBUG] Fusion done (%d results)", len(fused))
	if len(fused) == 0 {
		return nil, observability.NewFailure(observability.FailureNoRelevantResults, "no relevant results found after fusion", nil)
	}

	// Rerank
	finalResults, err := q.rerank(ctx, question, fused)
	if err != nil {
		return nil, err
	}

	// Token budget selection
	selected := make([]Citation, 0, len(finalResults))
	tokenUsed := 0
	for _, fr := range finalResults {
		content, ok := fr.Payload["window"].(string)
		if !ok || strings.TrimSpace(content) == "" {
			content, _ = fr.Payload["content"].(string)
		}

		tokens := ingestion.EstimateTokens(content)
		if tokenUsed+tokens > q.maxTokens {
			break
		}
		tokenUsed += tokens

		selected = append(selected, Citation{
			Source:     toString(fr.Payload["source"]),
			ChunkIndex: toInt(fr.Payload["chunk_index"]),
			Score:      float32(fr.RRFScore),
			Content:    content,
		})
	}

	if len(selected) == 0 {
		return nil, fmt.Errorf("no chunks fit within the %d token budget", q.maxTokens)
	}

	return selected, nil
}

// rerank applies the reranker if configured, otherwise falls back to top-N slice.
func (q *QueryPipeLine) rerank(
	ctx context.Context,
	question string,
	fused []retrieval.FusedResult,
) ([]retrieval.FusedResult, error) {
	stageStart := time.Now()
	_, span := q.tracer.Start(ctx, "rerank")
	defer func() {
		span.End()
		if q.metrics != nil {
			q.metrics.StageLatencySeconds.WithLabelValues("rerank").Observe(time.Since(stageStart).Seconds())
		}
	}()
	if q.reranker == nil {
		log.Println("[DEBUG] Reranker disabled, using top fused results")
		if len(fused) > q.rerankTopN {
			return fused[:q.rerankTopN], nil
		}
		return fused, nil
	}

	log.Println("[DEBUG] Reranking starting...")

	documents := make([]string, len(fused))
	for i, fr := range fused {
		if window, ok := fr.Payload["window"].(string); ok && strings.TrimSpace(window) != "" {
			documents[i] = window
		} else {
			documents[i], _ = fr.Payload["content"].(string)
		}
	}

	// Protect the rerank API call
	if err := acquireSem(ctx, q.rerankSem); err != nil {
		return nil, observability.NewFailure(observability.FailureRerankerTimeout, "rerank semaphore", err)
	}
	rerankResults, err := q.reranker.Rerank(ctx, question, documents, q.rerankTopN)
	releaseSem(q.rerankSem)
	if err != nil {
		return nil, observability.NewFailure(observability.FailureRerankerProvider, "reranking failed", err)
	}

	log.Printf("[DEBUG] Reranking done (%d results)", len(rerankResults))

	finalResults := make([]retrieval.FusedResult, len(rerankResults))
	for i, rr := range rerankResults {
		finalResults[i] = fused[rr.Index]
		finalResults[i].RRFScore = rr.RelevanceScore
		if q.metrics != nil {
			q.metrics.RetrievalScores.WithLabelValues(fmt.Sprintf("%d", i+1)).Set(float64(rr.RelevanceScore))
		}
	}
	return finalResults, nil
}

// generate calls the LLM and parses citations from the response.
func (q *QueryPipeLine) generate(
	ctx context.Context,
	question string,
	selected []Citation,
) (*Answer, []Citation, error) {
	stageStart := time.Now()
	stageCtx, span := q.tracer.Start(ctx, "llm_generate")
	defer func() {
		span.End()
		if q.metrics != nil {
			q.metrics.StageLatencySeconds.WithLabelValues("llm_generate").Observe(time.Since(stageStart).Seconds())
		}
	}()
	log.Printf("[DEBUG] Generating answer for %d citations...", len(selected))

	prompt := q.buildPrompt(question, selected)
	estimatedInputTokens := ingestion.EstimateTokens(prompt)
	estimatedOutputTokens := 1024
	estimatedCostUSD := q.costCalc.Estimate("", estimatedInputTokens, estimatedOutputTokens)
	if estimatedCostUSD > q.budgetUSD {
		return nil, nil, observability.NewFailure(
			observability.FailureTokenBudgetExceeded,
			fmt.Sprintf("estimated request cost %.6f exceeds budget %.2f", estimatedCostUSD, q.budgetUSD),
			nil,
		)
	}

	// Protect the LLM call
	if err := acquireSem(stageCtx, q.llmSem); err != nil {
		return nil, nil, observability.NewFailure(observability.FailureLLMTimeout, "llm semaphore", err)
	}
	response, err := q.llm.Generate(stageCtx, prompt)
	releaseSem(q.llmSem)
	if err != nil {
		return nil, nil, observability.NewFailure(observability.FailureLLMProvider, "llm generation failed", err)
	}
	if response.CostUSD <= 0 {
		response.CostUSD = q.costCalc.Estimate(response.Model, response.InputTokens, response.OutputTokens)
	}
	span.SetAttributes(
		attribute.String("llm.model", response.Model),
		attribute.String("llm.provider", response.Provider),
		attribute.Int("llm.input_tokens", response.InputTokens),
		attribute.Int("llm.output_tokens", response.OutputTokens),
		attribute.Float64("llm.cost_usd", response.CostUSD),
		attribute.Bool("llm.usage_estimated", response.UsageEstimated),
	)
	if q.metrics != nil {
		q.metrics.TokensUsedTotal.WithLabelValues("input", response.Provider, response.Model).Add(float64(response.InputTokens))
		q.metrics.TokensUsedTotal.WithLabelValues("output", response.Provider, response.Model).Add(float64(response.OutputTokens))
		q.metrics.CostUSDTotal.WithLabelValues(response.Provider, response.Model, fmt.Sprintf("%t", response.UsageEstimated)).Add(response.CostUSD)
	}

	log.Println("[DEBUG] LLM generation done")

	validCitations, answerText := q.parseCitations(response.Text, selected)

	return &Answer{Text: answerText}, validCitations, nil
}

// parseCitations extracts [Source: X, Chunk N] markers from the answer text,
// validates them against the selected citations, and strips any hallucinated ones.
func (q *QueryPipeLine) parseCitations(answerText string, selected []Citation) ([]Citation, string) {
	re := regexp.MustCompile(`\[Source: (.*?), Chunk (\d+)\]`)
	matches := re.FindAllStringSubmatch(answerText, -1)

	validCitations := make([]Citation, 0)
	citationSet := make(map[string]bool)

	for _, match := range matches {
		if len(match) != 3 {
			continue
		}

		src := match[1]
		var chunkIdx int
		_, _ = fmt.Sscanf(match[2], "%d", &chunkIdx)
		chunkIdx-- // prompt uses 1-indexed chunks

		found := false
		for _, sel := range selected {
			if sel.Source == src && sel.ChunkIndex == chunkIdx {
				key := fmt.Sprintf("%s-%d", sel.Source, sel.ChunkIndex)
				if !citationSet[key] {
					validCitations = append(validCitations, sel)
					citationSet[key] = true
				}
				found = true
				break
			}
		}

		if !found {
			// Hallucinated citation — strip it from the answer
			answerText = strings.Replace(answerText, match[0], "", 1)
		}
	}

	return validCitations, answerText
}

func (q *QueryPipeLine) buildPrompt(question string, citations []Citation) string {
	var sb strings.Builder

	// Fixed: citation instruction is now separate from the "I don't know" clause
	sb.WriteString("You are a precise document assistant. Follow these rules:\n")
	sb.WriteString("1. Answer ONLY using the context provided below.\n")
	sb.WriteString("2. For every claim, cite its source using: [Source: filename, Chunk N]\n")
	sb.WriteString("3. If the answer is not in the context, respond: \"I don't know based on the provided documents.\"\n")
	sb.WriteString("4. Do not invent information not present in the context.\n\n")
	sb.WriteString("Context:\n")

	for _, c := range citations {
		fmt.Fprintf(&sb, "[Source: %s, Chunk %d]\n%s\n\n", c.Source, c.ChunkIndex+1, c.Content)
	}

	sb.WriteString("Question: ")
	sb.WriteString(question)
	sb.WriteString("\nAnswer:")

	return sb.String()
}

func toString(value any) string {
	if v, ok := value.(string); ok {
		return v
	}
	return ""
}

func toInt(value any) int {
	switch v := value.(type) {
	case int:
		return v
	case int64:
		return int(v)
	case float64:
		return int(v)
	default:
		return -1
	}
}
