package activationfunctions

func Step(x []float64) []float64 {
	y := make([]float64, len(x))
	for i, divX := range x {
		if divX > 0 {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
	return y
}

func Arange(min float64, max float64, step float64) []float64 {
	var result []float64
	for i := min; i < max; i += step {
		result = append(result, i)
	}
	return result
}
