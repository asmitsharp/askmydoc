package observability

import "github.com/prometheus/client_golang/prometheus"

type Metrics struct {
	RequestsTotal         *prometheus.CounterVec
	RequestLatencySeconds *prometheus.HistogramVec
	StageLatencySeconds   *prometheus.HistogramVec
	TokensUsedTotal       *prometheus.CounterVec
	CostUSDTotal          *prometheus.CounterVec
	RequestFailuresTotal  *prometheus.CounterVec
	RetrievalScores       *prometheus.GaugeVec
}

func InitMetrics(registry *prometheus.Registry) *Metrics {
	m := &Metrics{
		RequestsTotal: prometheus.NewCounterVec(
			prometheus.CounterOpts{Name: "rag_requests_total", Help: "Total number of rag requests"},
			[]string{"status"},
		),
		RequestLatencySeconds: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "rag_request_latency_seconds",
				Help:    "End-to-end request latency in seconds",
				Buckets: prometheus.DefBuckets,
			},
			[]string{"status"},
		),
		StageLatencySeconds: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "rag_retrieval_stage_latency_seconds",
				Help:    "Latency per retrieval/generation stage in seconds",
				Buckets: []float64{0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10},
			},
			[]string{"stage"},
		),
		TokensUsedTotal: prometheus.NewCounterVec(
			prometheus.CounterOpts{Name: "rag_tokens_used_total", Help: "Total number of LLM tokens used"},
			[]string{"kind", "provider", "model"},
		),
		CostUSDTotal: prometheus.NewCounterVec(
			prometheus.CounterOpts{Name: "rag_cost_usd_total", Help: "Total LLM cost in USD"},
			[]string{"provider", "model", "estimated"},
		),
		RequestFailuresTotal: prometheus.NewCounterVec(
			prometheus.CounterOpts{Name: "rag_request_failures_total", Help: "Total number of request failures"},
			[]string{"stage", "reason"},
		),
		RetrievalScores: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{Name: "rag_retrieval_scores", Help: "Per-rank retrieval score"},
			[]string{"rank"},
		),
	}
	if registry == nil {
		registry = prometheus.DefaultRegisterer.(*prometheus.Registry)
	}
	registry.MustRegister(
		m.RequestsTotal,
		m.RequestLatencySeconds,
		m.StageLatencySeconds,
		m.TokensUsedTotal,
		m.CostUSDTotal,
		m.RequestFailuresTotal,
		m.RetrievalScores,
	)
	return m
}
