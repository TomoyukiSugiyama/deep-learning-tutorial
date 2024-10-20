package activationfunctions

import "math"

func Sigmoid(x []float64) []float64 {
	y := make([]float64, len(x))
	for i, divX := range x {
		y[i] = 1 / (1 + math.Exp(-divX))
	}
	return y
}
