package eval

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/ashmitsharp/askmydocs/internal/pipeline"
)

const perQueryTimeout = 2 * time.Minute
const minGapBetweenQueries = 20 * time.Second

type GoldenTestCase struct {
	ID                     string `json:"id"`
	Question               string `json:"question"`
	ExpectedAnswerContains string `json:"expected_answer_contains"`
	SourceDocument         string `json:"source_document"`
	Category               string `json:"category"`
}

type TestResult struct {
	ID                  string   `json:"id"`
	Question            string   `json:"question"`
	Category            string   `json:"category"`
	ContextRecall       float64  `json:"context_recall"`
	AnswerCorrectness   float64  `json:"answer_correctness"`
	Faithfulness        float64  `json:"faithfulness"`
	RetrievalLatencyMs  int64    `json:"retrieval_latency_ms"`
	GenerationLatencyMs int64    `json:"generation_latency_ms"`
	TotalLatencyMs      int64    `json:"total_latency_ms"`
	ActualAnswer        string   `json:"actual_answer"`
	RetrievedSources    []string `json:"retrieved_sources"`
	Pass                bool     `json:"pass"`
	FailureReason       string   `json:"failure_reason,omitempty"`
}

type EvalReport struct {
	Timestamp          string       `json:"timestamp"`
	TotalQueries       int          `json:"total_queries"`
	ContextRecallScore float64      `json:"context_recall_score"`
	CorrectnessScore   float64      `json:"correctness_score"`
	FaithfulnessScore  float64      `json:"faithfulness_score"`
	LatencyStats       LatencyStats `json:"latency_stats"`
	PassedThresholds   bool         `json:"passed_thresholds"`
	Results            []TestResult `json:"results"`
}

type LatencyStats struct {
	RetrievalAvgMs  int64 `json:"retrieval_avg_ms"`
	RetrievalP50Ms  int64 `json:"retrieval_p50_ms"`
	RetrievalP95Ms  int64 `json:"retrieval_p95_ms"`
	GenerationAvgMs int64 `json:"generation_avg_ms"`
	GenerationP50Ms int64 `json:"generation_p50_ms"`
	GenerationP95Ms int64 `json:"generation_p95_ms"`
	TotalAvgMs      int64 `json:"total_avg_ms"`
	TotalP50Ms      int64 `json:"total_p50_ms"`
	TotalP95Ms      int64 `json:"total_p95_ms"`
}

type Runner struct {
	queryPipeline *pipeline.QueryPipeLine
	judge         *Judge
	thresholds    Thresholds
}

type Thresholds struct {
	ContextRecall float64
	Faithfulness  float64
	Correctness   float64
}

func NewRunner(pipeline *pipeline.QueryPipeLine, judge *Judge, thresholds Thresholds) *Runner {
	return &Runner{
		queryPipeline: pipeline,
		judge:         judge,
		thresholds:    thresholds,
	}
}

// Run executes the full evaluation harness against the golden dataset.
func (r *Runner) Run(goldenDataPath string) (*EvalReport, error) {
	testdata, err := os.ReadFile(goldenDataPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read golden dataset at %s: %w", goldenDataPath, err)
	}

	var testCases []GoldenTestCase
	if err := json.Unmarshal(testdata, &testCases); err != nil {
		return nil, fmt.Errorf("failed to unmarshal golden dataset: %w", err)
	}

	log.Printf("Loaded %d test cases from %s", len(testCases), goldenDataPath)

	// 2. Loop through each test case and evaluate
	results := make([]TestResult, 0, len(testCases))
	ctx := context.Background()

	var lastQueryAt time.Time
	for i, tc := range testCases {
		if i > 0 {
			elapsed := time.Since(lastQueryAt)
			if remaining := minGapBetweenQueries - elapsed; remaining > 0 {
				log.Printf("Rate limiting: waiting %s before next query...", remaining.Round(time.Millisecond))
				time.Sleep(remaining)
			}
		}

		log.Printf("[%d/%d] Evaluating: %s — %s", i+1, len(testCases), tc.ID, tc.Question)
		lastQueryAt = time.Now()

		result := r.evaluateTestCase(ctx, tc)
		results = append(results, result)

		status := "✅ PASS"
		if !result.Pass {
			status = fmt.Sprintf("❌ FAIL (%s)", result.FailureReason)
		}
		log.Printf("[%d/%d] %s: %s", i+1, len(testCases), tc.ID, status)
	}

	// 3. Compute aggregate scores
	report := r.buildReport(results)

	// 4. Write report to eval_results.json
	reportJSON, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("failed to marshal eval report: %w", err)
	}

	if err := os.WriteFile("eval_results.json", reportJSON, 0644); err != nil {
		return nil, fmt.Errorf("failed to write eval_results.json: %w", err)
	}

	log.Printf("Full results written to: eval_results.json")

	return report, nil
}

// evaluateTestCase runs a single golden test case through the pipeline and scores it.
func (r *Runner) evaluateTestCase(ctx context.Context, tc GoldenTestCase) TestResult {
	result := TestResult{
		ID:       tc.ID,
		Question: tc.Question,
		Category: tc.Category,
	}

	// Add per-query timeout so a slow backend doesn't hang the entire eval
	queryCtx, cancel := context.WithTimeout(ctx, perQueryTimeout)
	defer cancel()

	// Execute the query pipeline with timing data
	answer, citations, timings, err := r.queryPipeline.ExecuteWithTimings(queryCtx, tc.Question, nil)
	if err != nil {
		result.Pass = false
		result.FailureReason = fmt.Sprintf("pipeline error: %v", err)
		return result
	}

	result.ActualAnswer = answer.Text
	result.RetrievalLatencyMs = timings.RetrievalMs
	result.GenerationLatencyMs = timings.GenerationMs
	result.TotalLatencyMs = timings.TotalMs

	// Collect retrieved sources for the report
	sources := make([]string, 0, len(citations))
	sourceSet := make(map[string]bool)
	for _, c := range citations {
		if !sourceSet[c.Source] {
			sources = append(sources, c.Source)
			sourceSet[c.Source] = true
		}
	}
	result.RetrievedSources = sources

	// --- Score: Context Recall ---
	// Check if any citation's source matches the expected source document
	for _, citation := range citations {
		if citation.Source == tc.SourceDocument {
			result.ContextRecall = 1.0
			break
		}
	}

	// --- Score: Answer Correctness ---
	// Case-insensitive substring match
	if strings.Contains(strings.ToLower(answer.Text), strings.ToLower(tc.ExpectedAnswerContains)) {
		result.AnswerCorrectness = 1.0
	}

	// --- Score: Faithfulness (LLM-as-Judge) ---
	chunks := make([]string, len(citations))
	for i, c := range citations {
		chunks[i] = c.Content
	}

	log.Printf("[DEBUG] Starting judge evaluation for %s...", tc.ID)
	faithScore, err := r.judge.Evaluate(queryCtx, answer.Text, chunks)
	if err != nil {
		log.Printf("WARNING: Judge evaluation failed for %s: %v (defaulting to 0.0)", tc.ID, err)
		result.Faithfulness = 0.0
	} else {
		log.Printf("[DEBUG] Judge evaluation DONE for %s (score=%.2f)", tc.ID, faithScore)
		result.Faithfulness = faithScore
	}

	// --- Determine Pass/Fail ---
	var failures []string
	if result.ContextRecall < 1.0 {
		failures = append(failures, "Context Recall FAIL")
	}
	if result.AnswerCorrectness < 1.0 {
		failures = append(failures, "Correctness FAIL")
	}
	if result.Faithfulness < 0.5 {
		failures = append(failures, fmt.Sprintf("Faithfulness FAIL (%.2f)", result.Faithfulness))
	}

	if len(failures) == 0 {
		result.Pass = true
	} else {
		result.Pass = false
		result.FailureReason = strings.Join(failures, "; ")
	}

	return result
}

// buildReport aggregates individual test results into the final EvalReport.
func (r *Runner) buildReport(results []TestResult) *EvalReport {
	n := float64(len(results))
	if n == 0 {
		return &EvalReport{
			Timestamp: time.Now().Format(time.RFC3339),
		}
	}

	var recallSum, correctnessSum, faithfulnessSum float64
	for _, res := range results {
		recallSum += res.ContextRecall
		correctnessSum += res.AnswerCorrectness
		faithfulnessSum += res.Faithfulness
	}

	avgRecall := recallSum / n
	avgCorrectness := correctnessSum / n
	avgFaithfulness := faithfulnessSum / n

	latencyStats := computeLatencyStats(results)

	passedThresholds := avgRecall >= r.thresholds.ContextRecall &&
		avgCorrectness >= r.thresholds.Correctness &&
		avgFaithfulness >= r.thresholds.Faithfulness

	return &EvalReport{
		Timestamp:          time.Now().Format(time.RFC3339),
		TotalQueries:       len(results),
		ContextRecallScore: avgRecall,
		CorrectnessScore:   avgCorrectness,
		FaithfulnessScore:  avgFaithfulness,
		LatencyStats:       latencyStats,
		PassedThresholds:   passedThresholds,
		Results:            results,
	}
}

// PrintReport prints a formatted evaluation summary to stdout.
func PrintReport(report *EvalReport) {
	fmt.Println("══════════════════════════════════════════════════════")
	fmt.Println("  AskMyDocs RAG Evaluation Report")
	fmt.Println("══════════════════════════════════════════════════════")
	fmt.Printf("  Queries:     %d\n", report.TotalQueries)
	fmt.Printf("  Timestamp:   %s\n", report.Timestamp)
	fmt.Println("──────────────────────────────────────────────────────")
	fmt.Println("  METRIC              SCORE    THRESHOLD    STATUS")
	fmt.Println("──────────────────────────────────────────────────────")

	printMetricRow("Context Recall", report.ContextRecallScore, 0.80)
	printMetricRow("Answer Correctness", report.CorrectnessScore, 0.75)
	printMetricRow("Faithfulness", report.FaithfulnessScore, 0.85)

	fmt.Println("──────────────────────────────────────────────────────")
	fmt.Println("  LATENCY             AVG      P50      P95")
	fmt.Println("──────────────────────────────────────────────────────")

	fmt.Printf("  Retrieval           %-8s %-8s %s\n",
		fmtMs(report.LatencyStats.RetrievalAvgMs),
		fmtMs(report.LatencyStats.RetrievalP50Ms),
		fmtMs(report.LatencyStats.RetrievalP95Ms))

	fmt.Printf("  Generation          %-8s %-8s %s\n",
		fmtMs(report.LatencyStats.GenerationAvgMs),
		fmtMs(report.LatencyStats.GenerationP50Ms),
		fmtMs(report.LatencyStats.GenerationP95Ms))

	fmt.Printf("  Total (E2E)         %-8s %-8s %s\n",
		fmtMs(report.LatencyStats.TotalAvgMs),
		fmtMs(report.LatencyStats.TotalP50Ms),
		fmtMs(report.LatencyStats.TotalP95Ms))

	fmt.Println("══════════════════════════════════════════════════════")

	if report.PassedThresholds {
		fmt.Println("  OVERALL: ✅ PASS (all thresholds met)")
	} else {
		fmt.Println("  OVERALL: ❌ FAIL (thresholds not met)")
	}
	fmt.Println("══════════════════════════════════════════════════════")

	// Print failed queries
	var failed []TestResult
	for _, r := range report.Results {
		if !r.Pass {
			failed = append(failed, r)
		}
	}

	if len(failed) > 0 {
		fmt.Printf("\n  Failed queries (%d):\n", len(failed))
		for _, f := range failed {
			fmt.Printf("    %s [%s] — %s\n", f.ID, f.Category, f.FailureReason)
		}
	}

	fmt.Println()
	fmt.Println("  Full results written to: eval_results.json")
}

func printMetricRow(name string, score, threshold float64) {
	status := "✅ PASS"
	if score < threshold {
		status = "❌ FAIL"
	}
	fmt.Printf("  %-20s %.2f     >= %.2f      %s\n", name, score, threshold, status)
}

func fmtMs(ms int64) string {
	return fmt.Sprintf("%dms", ms)
}
