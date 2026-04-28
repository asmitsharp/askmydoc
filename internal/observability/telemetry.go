package observability

import (
	"context"
	"encoding/base64"
	"fmt"
	"strings"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	"go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.40.0"
)

type TelemetryConfig struct {
	OTLPEndpoint      string
	OTLPHeaders       map[string]string
	ServiceName       string
	ServiceVersion    string
	Environment       string
	InsecureTransport bool
}

func InitTelemetry(ctx context.Context, cfg TelemetryConfig) (func(context.Context) error, error) {
	exporterOpts := []otlptracehttp.Option{
		otlptracehttp.WithEndpointURL(strings.TrimSpace(cfg.OTLPEndpoint)),
		otlptracehttp.WithHeaders(cfg.OTLPHeaders),
	}
	if cfg.InsecureTransport {
		exporterOpts = append(exporterOpts, otlptracehttp.WithInsecure())
	}
	traceExporter, err := otlptracehttp.New(ctx, exporterOpts...)
	if err != nil {
		return nil, err
	}

	res, err := resource.New(ctx,
		resource.WithTelemetrySDK(),
		resource.WithAttributes(
			semconv.ServiceName(cfg.ServiceName),
			semconv.ServiceVersion(cfg.ServiceVersion),
			semconv.DeploymentEnvironmentName(cfg.Environment),
		),
	)
	if err != nil {
		return nil, err
	}

	tracerProvider := trace.NewTracerProvider(
		trace.WithResource(res),
		trace.WithBatcher(traceExporter, trace.WithBatchTimeout(time.Second)),
	)
	otel.SetTracerProvider(tracerProvider)
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
		propagation.TraceContext{},
		propagation.Baggage{},
	))
	return tracerProvider.Shutdown, nil
}

func BuildLangfuseOTLPHeaders(publicKey, secretKey string) map[string]string {
	if strings.TrimSpace(publicKey) == "" || strings.TrimSpace(secretKey) == "" {
		return map[string]string{}
	}
	raw := fmt.Sprintf("%s:%s", strings.TrimSpace(publicKey), strings.TrimSpace(secretKey))
	token := base64.StdEncoding.EncodeToString([]byte(raw))
	return map[string]string{
		"Authorization":                "Basic " + token,
		"x-langfuse-ingestion-version": "4",
	}
}
