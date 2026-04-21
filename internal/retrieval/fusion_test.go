package retrieval

import (
	"testing"

	"github.com/ashmitsharp/askmydocs/internal/storage"
)

func TestReciprocalRankFusion(t *testing.T) {
	tests := []struct {
		name          string
		vectorResults []storage.SearchResult
		bm25Results   []storage.SearchResult
		topN          int
		wantLen       int
		wantTopID     string
	}{
		{
			name:          "both empty",
			vectorResults: []storage.SearchResult{},
			bm25Results:   []storage.SearchResult{},
			topN:          5,
			wantLen:       0,
			wantTopID:     "",
		},
		{
			name:          "only vector results",
			vectorResults: createSearchResults([]string{"v1", "v2", "v3"}),
			bm25Results:   nil,
			topN:          2,
			wantLen:       2,
			wantTopID:     "v1",
		},
		{
			name:          "only bm25 results",
			vectorResults: nil,
			bm25Results:   createSearchResults([]string{"b1", "b2"}),
			topN:          5,
			wantLen:       2,
			wantTopID:     "b1",
		},
		{
			name:          "both with intersection and different ranks",
			vectorResults: createSearchResults([]string{"v1", "v2", "v3"}),
			bm25Results:   createSearchResults([]string{"b2", "v1", "b1"}),
			topN:          5,
			wantLen:       5,
			wantTopID:     "v1", // v1 ranks 0 (vector) and 1 (bm25)
		},
		{
			name:          "topN limits output",
			vectorResults: createSearchResults([]string{"id1", "id2", "id3"}),
			bm25Results:   createSearchResults([]string{"id2", "id4", "id1"}),
			topN:          2,
			wantLen:       2,
			wantTopID:     "id2", // id2 score (1/62 + 1/61) > id1 score (1/61 + 1/63)
		},
		{
			name:          "disjoint results all appear",
			vectorResults: createSearchResults([]string{"D", "E", "F"}),
			bm25Results:   createSearchResults([]string{"A", "B", "C"}),
			topN:          10,
			wantLen:       6,
			wantTopID:     "A", // BM25 rank 0 has highest score
		},
		{
			name:          "identical rankings boost scores",
			vectorResults: createSearchResults([]string{"A", "B", "C"}),
			bm25Results:   createSearchResults([]string{"A", "B", "C"}),
			topN:          5,
			wantLen:       3,
			wantTopID:     "A", // A gets double the score
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ReciprocalRankFusion(tt.vectorResults, tt.bm25Results, tt.topN)
			if len(got) != tt.wantLen {
				t.Errorf("ReciprocalRankFusion() returned %d results, want %d", len(got), tt.wantLen)
			}
			if tt.wantLen > 0 && tt.wantTopID != "" {
				if got[0].ChunkId != tt.wantTopID {
					t.Errorf("ReciprocalRankFusion() top result ID = %v, want %v", got[0].ChunkId, tt.wantTopID)
				}
			}
		})
	}
}

func createSearchResults(ids []string) []storage.SearchResult {
	result := make([]storage.SearchResult, len(ids))
	for i, id := range ids {
		result[i] = storage.SearchResult{
			ID: id,
			Payload: map[string]interface{}{
				"content":     "test content for " + id,
				"source":      "test.txt",
				"chunk_index": i,
			},
		}
	}
	return result
}
