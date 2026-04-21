package retrieval

import (
	"log"
	"sort"

	"github.com/ashmitsharp/askmydocs/internal/storage"
)

const (
	k int = 60
)

type FusedResult struct {
	RRFScore   float64
	ChunkId    string
	Payload    map[string]interface{}
	VectorRank int
	BM25Rank   int
}

func ReciprocalRankFusion(vectorResults, bm25Results []storage.SearchResult, topN int) []FusedResult {
	if topN == 0 {
		log.Println("topN is 0")
	}

	if len(vectorResults) == 0 && len(bm25Results) == 0 {
		log.Println("both vector and bm25 results are empty")
		return nil
	}

	scoreMap := make(map[string]FusedResult)

	for i, res := range bm25Results {
		score := 1.0 / (float64(i) + float64(k) + 1.0)

		scoreMap[res.ID] = FusedResult{
			RRFScore: score,
			ChunkId:  res.ID,
			Payload:  res.Payload,
			BM25Rank: i,
		}
	}

	for i, res := range vectorResults {
		if _, ok := scoreMap[res.ID]; ok {
			score := 1.0 / (float64(i) + float64(k) + 1.0)
			fr := scoreMap[res.ID]
			fr.RRFScore += score
			fr.VectorRank = i
			scoreMap[res.ID] = fr
		} else {
			score := 1.0 / (float64(i) + float64(k) + 1.0)
			scoreMap[res.ID] = FusedResult{
				RRFScore:   score,
				ChunkId:    res.ID,
				Payload:    res.Payload,
				VectorRank: i,
			}
		}
	}

	fusedResult := make([]FusedResult, 0, len(scoreMap))

	for _, fr := range scoreMap {
		fusedResult = append(fusedResult, fr)
	}
	sort.Slice(fusedResult, func(i, j int) bool {
		return fusedResult[i].RRFScore > fusedResult[j].RRFScore
	})

	if len(fusedResult) > topN {
		fusedResult = fusedResult[:topN]
	}

	return fusedResult
}
