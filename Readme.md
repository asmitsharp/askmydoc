# AskMyDocs

[![CI](https://github.com/ashmitsharp/askmydocs/actions/workflows/ci.yml/badge.svg)](https://github.com/ashmitsharp/askmydocs/actions/workflows/ci.yml)

A production-grade RAG (Retrieval-Augmented Generation) system built in Go. Upload documents, ask questions, get answers grounded in your content.

This started as a portfolio project to go deep on how retrieval actually works — not just "embed and search" but the full pipeline: hybrid retrieval, reranking, sliding window context, token budgeting. The kind of stuff that separates a demo from something you'd trust in production.

---

## What it does

You upload a document (PDF, Markdown, plain text). It gets chunked, embedded, and indexed into both a vector store and a full-text search store. When you ask a question, it runs both a vector search and a BM25 search concurrently, fuses the results with Reciprocal Rank Fusion, optionally reranks with Cohere, and passes the best chunks to an LLM to generate a grounded answer with citations.

---

## Architecture

```
┌─────────────┐     POST /ingest      ┌──────────────────────────────────────────────┐
│   Client    │ ───────────────────►  │                Ingestion Pipeline             │
└─────────────┘                       │                                               │
                                      │  Router → Loader → Chunker → Embedder        │
                                      │                    │              │            │
                                      │                    ▼              ▼            │
                                      │              PostgreSQL        Qdrant          │
                                      │              (BM25/TSV)    (Vector Store)      │
                                      └──────────────────────────────────────────────┘

┌─────────────┐     POST /query       ┌──────────────────────────────────────────────┐
│   Client    │ ───────────────────►  │                Query Pipeline                 │
└─────────────┘                       │                                               │
                                      │  Embed Question                               │
                                      │       │                                       │
                                      │  ┌────┴────┐                                  │
                                      │  ▼         ▼                                  │
                                      │ Qdrant   Postgres   (concurrent)              │
                                      │  └────┬────┘                                  │
                                      │       ▼                                       │
                                      │      RRF Fusion                               │
                                      │       ▼                                       │
                                      │  Cohere Rerank  (optional)                   │
                                      │       ▼                                       │
                                      │  Token Budget → LLM → Answer + Citations     │
                                      └──────────────────────────────────────────────┘
```

### Tech stack

- **Go** — the whole backend, nothing fancy
- **Qdrant** — vector store, cosine similarity search
- **PostgreSQL** — BM25 full-text search via `tsvector` + GIN index
- **OpenAI** — `text-embedding-3-small` for embeddings, `gpt-4o-mini` for generation
- **Cohere** — `rerank-english-v3.0` as the cross-encoder reranker
- **Docker Compose** — Qdrant + Postgres local setup
- **Kubernetes** — full production-style deployment via `kind`
- **Observability** — OpenTelemetry + Langfuse (Tracing), Prometheus (Metrics), Grafana (Dashboards)

Optional alternatives (swappable via env vars):
- HuggingFace (`all-MiniLM-L6-v2`) or Gemini (`text-embedding-004`) for embeddings
- Gemini (`gemini-2.5-flash`) as the LLM

---

## Why these decisions

**Hybrid retrieval (vector + BM25)**

Vector search alone misses exact keyword matches. BM25 alone misses semantic similarity. The formula question test makes this obvious — "what is `v(T+1)`" is a keyword match, while "how do you derive 1-minute volume" is semantic. Running both and fusing gives you coverage neither approach has alone.

**Reciprocal Rank Fusion**

I considered weighted score combination but RRF is simpler and more robust — it only cares about rank position, not the raw scores, so you don't have to normalize across two completely different scoring systems (cosine similarity vs. ts_rank). k=60 is the standard starting point.

**Cohere as cross-encoder reranker**

The bi-encoder embeddings we use for retrieval are fast but they score query and document independently. A cross-encoder sees them together, which is significantly more accurate at relevance scoring. The tradeoff is speed — you can't run it over thousands of candidates. The pattern here is: retrieve 20 candidates cheaply, rerank the top 5 expensively. Cohere's `rerank-english-v3.0` is solid for this.

**Sentence window retrieval**

Chunks are stored small for precise retrieval, but each chunk also stores a `window` field containing the previous and next chunks. When we pass context to the LLM, we pass the window, not just the chunk. This way the LLM has the surrounding context without the retrieval step being confused by large, noisy chunks.

**Deterministic chunk IDs (UUID v5)**

Chunk IDs are SHA1 UUIDs derived from content. Same content always gets the same ID. This makes ingestion idempotent — re-uploading a document doesn't create duplicates, it just upserts. Qdrant and Postgres both use `ON CONFLICT DO UPDATE`.

**Concurrent search and ingestion**

Both the vector search and BM25 search run in separate goroutines and are joined via channels. Same for the dual-write during ingestion (Qdrant + Postgres). No reason to serialize these — they're independent operations hitting different stores.

**OCR fallback for PDFs**

PDFs with embedded text use `ledongthuc/pdf` for extraction. Scanned PDFs fall back to Tesseract OCR via `gen2brain/go-fitz` for rendering pages to images. The threshold is 40 non-whitespace runes — below that, we assume the embedded extraction failed and try OCR.

**Token budget**

Before building the prompt, we estimate token counts (rune count / 4, rough but good enough) and enforce a 3000-token ceiling across all selected chunks. This prevents context overflow and keeps latency predictable.

**Langfuse over Jaeger for Tracing**

While Jaeger is great for microservices, Langfuse is specifically built for LLM observability. It allows us to view not just latency, but the exact prompts, retrieved contexts, generation outputs, and token counts directly in the trace tree.

**Business-level Metrics (Prometheus)**

Metrics are emitted from pipeline boundaries (e.g., `rag_retrieval_stage_latency_seconds`) rather than relying purely on vendor SDK instrumentation. This aligns observability with our business logic so we know exactly how much time is spent in BM25 vs Vector search vs Reranking.

**Pre-flight Cost Guardrails**

Before sending a prompt to the LLM, we estimate the token cost based on model pricing. If it exceeds a soft budget (e.g., $0.05), we abort the query. This prevents runaway usage from exceptionally large contexts or malicious queries.

**Typed Failure Taxonomy**

Instead of passing opaque string errors up the stack, we classify errors into typed enums (e.g., `FailureLLMTimeout`, `FailureQdrantUnavailable`). This allows for highly reliable failure rate monitoring and Grafana alerting, categorizing errors by root cause rather than string matching.

**StatefulSets for databases (Kubernetes)**

Qdrant and PostgreSQL run as StatefulSets rather than Deployments. StatefulSets give stable network identities and ordered pod management, which matters for databases — a Deployment could reschedule a pod to a different node and detach its PVC. PersistentVolumeClaims are declared separately so storage survives pod restarts and redeployments.

**HPA over manual scaling**

The API layer runs behind a HorizontalPodAutoscaler (min: 2, max: 6) targeting 70% CPU utilization. Scale-up is intentionally aggressive (2 pods per 60s) while scale-down is conservative (1 pod per 300s, 5-minute stabilization window) to avoid thrashing under bursty RAG workloads.

**Init containers for dependency ordering**

Rather than using `depends_on` style hacks, the API pod uses init containers that poll `nc -z` against Postgres and Qdrant before the main container starts. This is the correct Kubernetes pattern — liveness/readiness probes protect the pod after startup, but init containers protect it during startup.

---

## Project structure

```
askmydocs/
├── cmd/server/main.go          # entrypoint, wires everything together
├── internal/
│   ├── api/                    # HTTP handlers, request/response types
│   ├── embedding/              # OpenAI, HuggingFace, Gemini clients
│   ├── ingestion/              # loaders (text, PDF, markdown), chunker
│   ├── llm/                    # OpenAI, Gemini generation clients
│   ├── observability/          # telemetry, metrics, cost calculation, structured errors
│   ├── pipeline/               # ingestion pipeline, query pipeline
│   ├── retrieval/              # RRF fusion, Cohere reranker
│   └── storage/                # Qdrant vector store, Postgres BM25 store
├── k8s/
│   ├── api/                    # Deployment, Service, HPA
│   ├── postgres/               # StatefulSet, Service, PVC
│   ├── qdrant/                 # StatefulSet, Service, PVC
│   ├── redis/                  # Deployment, Service
│   ├── namespace.yml
│   ├── configmap.yml
│   └── secrets/
├── migrations/                 # SQL schema
├── docker/
│   ├── docker-compose.yml
│   ├── prometheus/             # Prometheus config
│   └── grafana/                # Grafana dashboards and config
└── .env                        # not committed
```

---

## Running locally

**Prerequisites**

- Go 1.21+
- Docker + Docker Compose
- OpenAI API key (minimum)
- Cohere API key (optional, for reranking)

**1. Start the infrastructure**

```bash
cd docker
docker compose up -d
```

This starts the entire local stack:
- **Qdrant**: 6333/6334 (Vector Store)
- **Postgres**: 5432 (BM25)
- **Prometheus**: 9090 (Metrics)
- **Grafana**: 3001 (Dashboards)
- **Langfuse**: 3000 (Tracing)

**2. Configure environment**

Create a `.env` file in the project root:

```env
PORT=8080

# Embedding — pick one
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-...

# Or HuggingFace
# EMBEDDING_PROVIDER=huggingface
# HF_API_KEY=hf_...

# Or Gemini
# EMBEDDING_PROVIDER=gemini
# GEMINI_API_KEY=...

# LLM — pick one
LLM_PROVIDER=openai
# LLM_PROVIDER=gemini

# Reranking — optional but recommended
RERANK_ENABLED=true
COHERE_API_KEY=...

# Storage
QDRANT_HOST=localhost
QDRANT_PORT=6334
QDRANT_COLLECTION=askmydocs
POSTGRES_DSN=postgres://askmydocs:askmydocs@localhost:5432/askmydocs?sslmode=disable

# Observability (Tracing via Langfuse)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

**3. Run the server**

```bash
go run ./cmd/server/main.go
```

**4. Ingest a document**

```bash
curl -X POST http://localhost:8080/ingest \
  -F "file=@/path/to/your/document.pdf"
```

Supports `.pdf`, `.md`, `.markdown`, `.txt`.

**5. Query it**

```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "your question here"}'
```

**6. Health check**

```bash
curl http://localhost:8080/health
```

---

## Running on Kubernetes

A full Kubernetes deployment is available using `kind` for local clusters. The stack is fully tested end-to-end.

**What's deployed:**
- **Qdrant & PostgreSQL** — StatefulSets with PVC-backed storage (500Mi each)
- **Redis** — Deployment with liveness/readiness probes
- **API** — Deployment with HPA scaling 2–6 replicas at 70% CPU, rolling updates with zero `maxUnavailable`, init containers for startup dependency ordering, and startup/readiness/liveness probes

See [`docs/kubernetes_setup.md`](docs/kubernetes_setup.md) for the complete walkthrough — cluster creation, secret management, image loading into kind, and end-to-end validation via port-forward.

> **Remaining:** ServiceMonitor integration (Prometheus Operator) and Ingress are not yet configured. Currently tested via `kubectl port-forward`.

---

## Running tests

Unit tests (no external dependencies):

```bash
go test ./internal/ingestion/... ./internal/retrieval/... ./internal/embedding/...
```

Integration tests (require running Postgres):

```bash
go test ./internal/storage/...
```

Integration tests (require running Qdrant, set env var):

```bash
QDRANT_ADDR=localhost go test ./internal/storage/...
```

HuggingFace embedding integration test (requires `EMBEDDING_PROVIDER=huggingface` and `HF_API_KEY`):

```bash
go test ./internal/embedding/...
```

---

## API reference

### `POST /ingest`

Multipart form upload. Field name: `file`.

```json
{
  "status": "ingested",
  "filename": "document.pdf",
  "message": "Successfully ingested 12 chunks"
}
```

### `POST /query`

```json
// request
{ "question": "your question" }

// response (headers: X-Trace-Id: 00-xxxxx...)
{
  "answer": "...",
  "citations": [
    {
      "source": "document.pdf",
      "chunk_index": 2,
      "score": 0.9950,
      "content": "..."
    }
  ],
  "latency_ms": 1823
}
```

### `GET /health`

```json
{ "status": "ok", "vector_db": true, "llm": true }
```

### `GET /metrics`

Prometheus metrics endpoint exposing request latencies, failure counts, and token usage.

```text
# HELP rag_requests_total Total number of RAG requests
# TYPE rag_requests_total counter
rag_requests_total{status="success"} 42
...
```

---

## Evaluation Pipeline

To ensure the RAG system performs reliably, AskMyDocs includes a fully automated evaluation pipeline that runs against a "golden dataset" of Q/A pairs. This pipeline measures three critical metrics:
- **Context Recall:** Did we retrieve the correct source document?
- **Answer Correctness:** Does the generated answer contain the expected core facts?
- **Faithfulness (LLM-as-a-Judge):** Is the answer fully supported by the retrieved context chunks, without any hallucinations?

### Running the Evaluation

The evaluation harness reads the golden dataset, runs each query through the full pipeline, and generates a detailed report.

```bash
# Run the evaluation pipeline
go run cmd/eval/main.go -dataset testdata/golden_set.json
```

This will output a terminal summary and write the detailed query-by-query results to `eval_results.json`.

### The Dual-Model Architecture

The evaluation pipeline is built to use a dual-model architecture. For example, when using Groq as the LLM provider, we configure it as follows:
- **Generation Model:** `meta-llama/llama-4-scout-17b-16e-instruct` (or similar)
- **Judge Model:** `llama-3.1-8b-instant`

**Why use two different models, and why is the judge smaller?**
1. **Token Limits & Cost:** Evaluating an entire dataset of 50+ queries consumes a massive amount of tokens. Running both generation and evaluation on a massive 70B+ model would quickly exhaust daily free-tier API limits (e.g., 100k tokens/day).
2. **Task Complexity:** The generation model has the heavy lifting—it must synthesize an answer, perform multi-hop reasoning, and structure citations. The **Judge model**, however, only has a focused validation task: verify if the generated answer is factually supported by the provided text chunks. It does not need vast world knowledge or external context. Even on "hard-edge" inferential queries where the answer isn't explicitly stated, the judge only needs to confirm if the generated inference logically maps back to the provided chunks. A smaller, lightning-fast 8B parameter model is highly effective for this strict verification task.

---

## What's next

- Streaming responses
- ServiceMonitor integration for Prometheus Operator
- Ingress controller configuration