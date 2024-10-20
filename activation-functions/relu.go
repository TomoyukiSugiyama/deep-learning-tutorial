package activationfunctions

func ReLU(x []float64) []float64 {
	y := make([]float64, len(x))
	for i, divX := range x {
		if divX > 0 {
			y[i] = divX
		} else {
			y[i] = 0
		}
	}
	return y
}
