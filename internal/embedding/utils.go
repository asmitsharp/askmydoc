package embedding

func batchTexts(texts []string, batchSize int) [][]string {
	var batches [][]string
	if batchSize <= 0 {
		panic("batchSize must be > 0")
	}
	for start := 0; start < len(texts); start += batchSize {
		end := start + batchSize
		if end > len(texts) {
			end = len(texts)
		}
		batches = append(batches, texts[start:end])
	}
	return batches
}
