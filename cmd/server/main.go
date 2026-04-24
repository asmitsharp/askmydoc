package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/ashmitsharp/askmydocs/internal/api"
	"github.com/ashmitsharp/askmydocs/internal/embedding"
	"github.com/ashmitsharp/askmydocs/internal/ingestion"
	"github.com/ashmitsharp/askmydocs/internal/llm"
	"github.com/ashmitsharp/askmydocs/internal/pipeline"
	"github.com/ashmitsharp/askmydocs/internal/retrieval"
	"github.com/ashmitsharp/askmydocs/internal/storage"
	"github.com/ashmitsharp/askmydocs/internal/task"
	"github.com/ashmitsharp/askmydocs/internal/worker"
	"github.com/hibiken/asynq"
	"github.com/joho/godotenv"
)

type Config struct {
	Port              string
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
	RedisAddr         string
}

func main() {
	if err := godotenv.Load(); err != nil {
		log.Printf("no .env file loaded, continuing with environment: %v", err)
	}
	cfg := readConfig()

	log.Printf("Embedding provider: %s", cfg.EmbeddingProvider)
	log.Printf("Qdrant: %s:%d collection=%s", cfg.QdrantHost, cfg.QdrantPort, cfg.QdrantCollection)

	ctx := context.Background()

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

	bm25store, err := storage.NewBM25Store(ctx, cfg.PostgresDSN)
	if err != nil {
		log.Fatalf("failed to create bm25 store: %v", err)
	}
	defer bm25store.Close()

	var llmClient llm.LLM
	if strings.EqualFold(cfg.LLMProvider, "gemini") {
		client, err := llm.NewGeminiClient(ctx, cfg.GeminiApiKey)
		if err != nil {
			log.Fatalf("failed to create gemini client: %v", err)
		}
		llmClient = client
	} else {
		llmClient = llm.NewOpenAIClient(cfg.OpenAIApiKey)
	}

	var reranker retrieval.Reranker
	if cfg.RerankEnabled {
		reranker = retrieval.NewCohereReranker(cfg.CohereApiKey, "rerank-english-v3.0")
	} else {
		reranker = nil
	}

	loader := ingestion.NewRouter()
	chunker := ingestion.NewTextChunker()
	ingestionPipeLine := pipeline.NewPipeLine(loader, chunker, embedder, store, bm25store)
	queryPipeLine := pipeline.NewQueryPipeLine(embedder, store, bm25store, llmClient, reranker)

	redisOpt := asynq.RedisClientOpt{Addr: cfg.RedisAddr}
	asynqClient := asynq.NewClient(redisOpt)
	defer asynqClient.Close()

	handler := api.NewHandler(ingestionPipeLine, queryPipeLine, asynqClient)

	asynqServer := asynq.NewServer(
		redisOpt,
		asynq.Config{
			Concurrency: 10,
		},
	)

	taskProcessor := worker.NewTaskProcessor(ingestionPipeLine)
	asynqMux := asynq.NewServeMux()
	asynqMux.HandleFunc(task.TypeDocumentIngestion, taskProcessor.ProcessTask)

	go func() {
		log.Printf("Asynq Worker Server starting on %s", cfg.RedisAddr)
		if err := asynqServer.Run(asynqMux); err != nil {
			log.Fatalf("could not run asynq server: %v", err)
		}
	}()
	server := &http.Server{
		Addr:         ":" + cfg.Port,
		Handler:      handler.Routes(),
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 5 * time.Minute,
		IdleTimeout:  120 * time.Second,
	}

	go func() {
		log.Printf("Server starting on :%s", cfg.Port)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("server error: %v", err)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("Shutting down...")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := server.Shutdown(shutdownCtx); err != nil {
		log.Printf("server shutdown error: %v", err)
	}
}

func readConfig() Config {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	qdrantPort := 6334
	if p := os.Getenv("QDRANT_PORT"); p != "" {
		if v, err := strconv.Atoi(p); err == nil {
			qdrantPort = v
		}
	}

	return Config{
		Port:              port,
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
		RedisAddr:         defaultString(os.Getenv("REDIS_ADDR"), "localhost:6379"),
	}
}

func defaultString(value, fallback string) string {
	if value == "" {
		return fallback
	}
	return value
}
