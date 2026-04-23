package eval

import "sort"

// computePercentile returns the value at the given percentile from a sorted slice of int64 values.
func computePercentile(sorted []int64, percentile float64) int64 {
	if len(sorted) == 0 {
		return 0
	}

	index := int(float64(len(sorted)-1) * percentile)
	if index >= len(sorted) {
		index = len(sorted) - 1
	}

	return sorted[index]
}

// computeLatencyStats calculates aggregate latency statistics (average, P50, P95)
func computeLatencyStats(results []TestResult) LatencyStats {
	if len(results) == 0 {
		return LatencyStats{}
	}

	retrievals := make([]int64, len(results))
	generations := make([]int64, len(results))
	totals := make([]int64, len(results))

	var retrievalSum, generationSum, totalSum int64

	for i, r := range results {
		retrievals[i] = r.RetrievalLatencyMs
		generations[i] = r.GenerationLatencyMs
		totals[i] = r.TotalLatencyMs

		retrievalSum += r.RetrievalLatencyMs
		generationSum += r.GenerationLatencyMs
		totalSum += r.TotalLatencyMs
	}

	sort.Slice(retrievals, func(i, j int) bool { return retrievals[i] < retrievals[j] })
	sort.Slice(generations, func(i, j int) bool { return generations[i] < generations[j] })
	sort.Slice(totals, func(i, j int) bool { return totals[i] < totals[j] })

	n := int64(len(results))

	return LatencyStats{
		RetrievalAvgMs:  retrievalSum / n,
		RetrievalP50Ms:  computePercentile(retrievals, 0.50),
		RetrievalP95Ms:  computePercentile(retrievals, 0.95),
		GenerationAvgMs: generationSum / n,
		GenerationP50Ms: computePercentile(generations, 0.50),
		GenerationP95Ms: computePercentile(generations, 0.95),
		TotalAvgMs:      totalSum / n,
		TotalP50Ms:      computePercentile(totals, 0.50),
		TotalP95Ms:      computePercentile(totals, 0.95),
	}
}
