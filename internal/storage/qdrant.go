package storage

import (
	"context"
	"fmt"

	"github.com/qdrant/go-client/qdrant"
)

const upsertBatchSize = 128

type QdrantStore struct {
	client     *qdrant.Client
	collection string
}

func NewQdrantStore(ctx context.Context, host string, port int, collection string, vectorSize uint64) (*QdrantStore, error) {
	client, err := qdrant.NewClient(&qdrant.Config{
		Host: host,
		Port: port,
	})
	if err != nil {
		return nil, fmt.Errorf("qdrant: connect %s:%d: %w", host, port, err)
	}

	if err := ensureCollection(ctx, client, collection, vectorSize); err != nil {
		return nil, err
	}

	return &QdrantStore{client: client, collection: collection}, nil
}

func (q *QdrantStore) Close() error {
	return q.client.Close()
}

func (q *QdrantStore) Upsert(ctx context.Context, points []Point) error {
	for start := 0; start < len(points); start += upsertBatchSize {
		end := start + upsertBatchSize
		if end > len(points) {
			end = len(points)
		}
		if err := q.upsertBatch(ctx, points[start:end]); err != nil {
			return err
		}
	}
	return nil
}

func (q *QdrantStore) upsertBatch(ctx context.Context, batch []Point) error {
	pts := make([]*qdrant.PointStruct, len(batch))
	for i, p := range batch {
		pts[i] = &qdrant.PointStruct{
			Id:      qdrant.NewIDUUID(p.ID),
			Vectors: qdrant.NewVectors(p.Vector...),
			Payload: toQdrantPayload(p.Payload),
		}
	}

	_, err := q.client.Upsert(ctx, &qdrant.UpsertPoints{
		CollectionName: q.collection,
		Points:         pts,
	})
	if err != nil {
		return fmt.Errorf("qdrant: upsert: %w", err)
	}
	return nil
}

// returns the topK most similar points to the query vector.
func (q *QdrantStore) Search(ctx context.Context, vector []float32, topK int) ([]SearchResult, error) {
	limit := uint64(topK)
	withPayload := qdrant.NewWithPayload(true)

	results, err := q.client.Query(ctx, &qdrant.QueryPoints{
		CollectionName: q.collection,
		Query:          qdrant.NewQuery(vector...),
		Limit:          &limit,
		WithPayload:    withPayload,
	})
	if err != nil {
		return nil, fmt.Errorf("qdrant: search: %w", err)
	}

	out := make([]SearchResult, len(results))
	for i, r := range results {
		out[i] = SearchResult{
			ID:      r.GetId().GetUuid(),
			Score:   r.GetScore(),
			Payload: fromQdrantPayload(r.GetPayload()),
		}
	}
	return out, nil
}

func (q *QdrantStore) Delete(ctx context.Context, ids []string) error {
	pointIDs := make([]*qdrant.PointId, len(ids))
	for i, id := range ids {
		pointIDs[i] = qdrant.NewIDUUID(id)
	}

	_, err := q.client.Delete(ctx, &qdrant.DeletePoints{
		CollectionName: q.collection,
		Points: &qdrant.PointsSelector{
			PointsSelectorOneOf: &qdrant.PointsSelector_Points{
				Points: &qdrant.PointsIdsList{Ids: pointIDs},
			},
		},
	})
	if err != nil {
		return fmt.Errorf("qdrant: delete: %w", err)
	}
	return nil
}

func ensureCollection(ctx context.Context, client *qdrant.Client, name string, vectorSize uint64) error {
	exists, err := client.CollectionExists(ctx, name)
	if err != nil {
		return fmt.Errorf("qdrant: check collection %q: %w", name, err)
	}
	if exists {
		return nil
	}

	err = client.CreateCollection(ctx, &qdrant.CreateCollection{
		CollectionName: name,
		VectorsConfig: qdrant.NewVectorsConfig(&qdrant.VectorParams{
			Size:     vectorSize,
			Distance: qdrant.Distance_Cosine,
		}),
	})
	if err != nil {
		return fmt.Errorf("qdrant: create collection %q: %w", name, err)
	}
	return nil
}

func toQdrantPayload(m map[string]any) map[string]*qdrant.Value {
	out := make(map[string]*qdrant.Value, len(m))
	for k, v := range m {
		out[k] = anyToValue(v)
	}
	return out
}

func anyToValue(v any) *qdrant.Value {
	switch val := v.(type) {
	case string:
		return qdrant.NewValueString(val)
	case int:
		return qdrant.NewValueInt(int64(val))
	case int64:
		return qdrant.NewValueInt(val)
	case float64:
		return qdrant.NewValueDouble(val)
	case bool:
		return qdrant.NewValueBool(val)
	default:
		return qdrant.NewValueString(fmt.Sprintf("%v", val))
	}
}

func fromQdrantPayload(m map[string]*qdrant.Value) map[string]any {
	out := make(map[string]any, len(m))
	for k, v := range m {
		switch kind := v.GetKind().(type) {
		case *qdrant.Value_StringValue:
			out[k] = kind.StringValue
		case *qdrant.Value_IntegerValue:
			out[k] = kind.IntegerValue
		case *qdrant.Value_DoubleValue:
			out[k] = kind.DoubleValue
		case *qdrant.Value_BoolValue:
			out[k] = kind.BoolValue
		default:
			out[k] = v.String()
		}
	}
	return out
}
