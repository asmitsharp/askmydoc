package retrieval

import (
	"testing"

	"github.com/ashmitsharp/askmydocs/internal/storage"
)

func TestReciprocalRankFusion_Disjoint(t *testing.T) {
	vectorResults := createSearchResults([]string{"D", "E", "F"})
	bm25Results := createSearchResults([]string{"A", "B", "C"})

	fused := ReciprocalRankFusion(vectorResults, bm25Results, 10)

	if len(fused) != 6 {
		t.Errorf("Expected 6 results, got %d", len(fused))
	}

	expectedIDs := map[string]bool{"A": true, "B": true, "C": true, "D": true, "E": true, "F": true}
	for _, fr := range fused {
		if !expectedIDs[fr.ChunkId] {
			t.Errorf("Unexpected ID: %s", fr.ChunkId)
		}
		delete(expectedIDs, fr.ChunkId)
	}

	scores := map[string]float32{
		"A": 1.0 / (0 + 60 + 1),
		"B": 1.0 / (1 + 60 + 2),
		"C": 1.0 / (2 + 60 + 3),
		"D": 1.0 / (0 + 60 + 4),
		"E": 1.0 / (1 + 60 + 5),
		"F": 1.0 / (2 + 60 + 6),
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
