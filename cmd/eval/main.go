package main

import (
	"context"
	"flag"
	"log"
	"os"
	"strconv"
	"strings"

	"github.com/joho/godotenv"

	"github.com/ashmitsharp/askmydocs/internal/embedding"
	"github.com/ashmitsharp/askmydocs/internal/eval"
	"github.com/ashmitsharp/askmydocs/internal/llm"
	"github.com/ashmitsharp/askmydocs/internal/pipeline"
	"github.com/ashmitsharp/askmydocs/internal/retrieval"
	"github.com/ashmitsharp/askmydocs/internal/storage"
)

type Config struct {
	EmbeddingProvider string
	HFApiKey          string
	OpenAIApiKey      string
	QdrantHost        string
	QdrantPort        int
	QdrantCollection  string
	PostgresDSN       string
	RerankEnabled     bool
	CohereApiKey      string
	LLMProvider       string
	GeminiApiKey      string
	GroqApiKey        string
}

func main() {
	// --- CLI flags ---
	goldenDataPath := flag.String("dataset", "testdata/golden_set.json", "Path to the golden dataset JSON file")
	flag.Parse()

	if err := godotenv.Load(); err != nil {
		log.Printf("no .env file loaded, continuing with environment: %v", err)
	}

	cfg := readConfig()

	log.Printf("═══ AskMyDocs Evaluation Harness ═══")
	log.Printf("Embedding provider: %s", cfg.EmbeddingProvider)
	log.Printf("LLM provider: %s", cfg.LLMProvider)
	log.Printf("Qdrant: %s:%d collection=%s", cfg.QdrantHost, cfg.QdrantPort, cfg.QdrantCollection)
	log.Printf("Dataset: %s", *goldenDataPath)

	ctx := context.Background()

	// Embedder
	embeddingKey := cfg.OpenAIApiKey
	if strings.EqualFold(cfg.EmbeddingProvider, embedding.ProviderHuggingFace) {
		embeddingKey = cfg.HFApiKey
	} else if strings.EqualFold(cfg.EmbeddingProvider, "gemini") {
		embeddingKey = cfg.GeminiApiKey
	}

	embedder, err := embedding.NewClient(cfg.EmbeddingProvider, embeddingKey)
	if err != nil {
		log.Fatalf("failed to create embedder: %v", err)
	}

	// Vector store
	var vectorSize uint64
	if strings.EqualFold(cfg.EmbeddingProvider, embedding.ProviderHuggingFace) {
		vectorSize = 384
	} else if strings.EqualFold(cfg.EmbeddingProvider, "gemini") {
		vectorSize = 768
	} else {
		vectorSize = 1536
	}

	store, err := storage.NewQdrantStore(ctx, cfg.QdrantHost, cfg.QdrantPort, cfg.QdrantCollection, vectorSize)
	if err != nil {
		log.Fatalf("failed to create qdrant store: %v", err)
	}
	defer func() {
		if err := store.Close(); err != nil {
			log.Printf("qdrant close error: %v", err)
		}
	}()

	// BM25 store
	bm25store, err := storage.NewBM25Store(ctx, cfg.PostgresDSN)
	if err != nil {
		log.Fatalf("failed to create bm25 store: %v", err)
	}
	defer bm25store.Close()

	// LLM client
	var llmClient llm.LLM
	var judgeLLM llm.LLM

	if strings.EqualFold(cfg.LLMProvider, "gemini") {
		client, err := llm.NewGeminiClient(ctx, cfg.GeminiApiKey)
		if err != nil {
			log.Fatalf("failed to create gemini client: %v", err)
		}
		llmClient = client
		judgeLLM = client
	} else if strings.EqualFold(cfg.LLMProvider, "groq") {
		llmClient = llm.NewGroqLLM(cfg.GroqApiKey, "meta-llama/llama-4-scout-17b-16e-instruct")
		judgeLLM = llm.NewGroqLLM(cfg.GroqApiKey, "llama-3.1-8b-instant")
	} else {
		client := llm.NewOpenAIClient(cfg.OpenAIApiKey)
		llmClient = client
		judgeLLM = client
	}

	// Reranker
	var reranker retrieval.Reranker
	if cfg.RerankEnabled {
		reranker = retrieval.NewCohereReranker(cfg.CohereApiKey, "rerank-english-v3.0")
	}

	// --- Build pipelines ---
	queryPipeLine := pipeline.NewQueryPipeLine(embedder, store, bm25store, llmClient, reranker)

	// --- Build evaluation components ---
	judge := eval.NewJudge(judgeLLM)

	thresholds := eval.Thresholds{
		ContextRecall: 0.80,
		Faithfulness:  0.85,
		Correctness:   0.75,
	}

	runner := eval.NewRunner(queryPipeLine, judge, thresholds)

	// --- Run evaluation ---
	report, err := runner.Run(*goldenDataPath)
	if err != nil {
		log.Fatalf("Evaluation failed: %v", err)
	}

	// --- Print terminal report ---
	eval.PrintReport(report)

	// --- Exit with appropriate code ---
	if !report.PassedThresholds {
		log.Println("Thresholds NOT met — exiting with code 1")
		os.Exit(1)
	}

	log.Println("All thresholds met — exiting with code 0")
}

func readConfig() Config {
	qdrantPort := 6334
	if p := os.Getenv("QDRANT_PORT"); p != "" {
		if v, err := strconv.Atoi(p); err == nil {
			qdrantPort = v
		}
	}

	return Config{
		EmbeddingProvider: os.Getenv("EMBEDDING_PROVIDER"),
		HFApiKey:          os.Getenv("HF_API_KEY"),
		OpenAIApiKey:      os.Getenv("OPENAI_API_KEY"),
		QdrantHost:        defaultString(os.Getenv("QDRANT_HOST"), "localhost"),
		QdrantPort:        qdrantPort,
		QdrantCollection:  defaultString(os.Getenv("QDRANT_COLLECTION"), "askmydocs"),
		PostgresDSN:       defaultString(os.Getenv("POSTGRES_DSN"), "postgres://askmydocs:askmydocs@localhost:5432/askmydocs?sslmode=disable"),
		RerankEnabled:     os.Getenv("RERANK_ENABLED") == "true",
		CohereApiKey:      os.Getenv("COHERE_API_KEY"),
		LLMProvider:       defaultString(os.Getenv("LLM_PROVIDER"), "openai"),
		GeminiApiKey:      os.Getenv("GEMINI_API_KEY"),
		GroqApiKey:        os.Getenv("GROQ_API_KEY"),
	}
}

func defaultString(value, fallback string) string {
	if value == "" {
		return fallback
	}
	return value
}
