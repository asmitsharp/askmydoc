package observability

type Price struct {
	InputPer1K  float64
	OutputPer1K float64
}

type CostCalculator struct {
	defaultPrice Price
	byModel      map[string]Price
}

func NewCostCalculator() *CostCalculator {
	return &CostCalculator{
		// conservative fallback estimate (USD per 1k tokens)
		defaultPrice: Price{InputPer1K: 0.0005, OutputPer1K: 0.0015},
		byModel: map[string]Price{
			"gpt-4o-mini":                               {InputPer1K: 0.00015, OutputPer1K: 0.0006},
			"gemini-2.5-flash":                          {InputPer1K: 0.0003, OutputPer1K: 0.0025},
			"llama-3.1-8b-instant":                      {InputPer1K: 0.00005, OutputPer1K: 0.00008},
			"meta-llama/llama-4-scout-17b-16e-instruct": {InputPer1K: 0.00011, OutputPer1K: 0.00034},
		},
	}
}

func (c *CostCalculator) Estimate(model string, inputTokens, outputTokens int) float64 {
	price, ok := c.byModel[model]
	if !ok {
		price = c.defaultPrice
	}
	return (float64(inputTokens)/1000.0)*price.InputPer1K + (float64(outputTokens)/1000.0)*price.OutputPer1K
}
