package activationfunctions

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func Sigmoid(x []float64) []float64 {
	y := make([]float64, len(x))
	for i, divX := range x {
		y[i] = 1 / (1 + math.Exp(-divX))
	}
	return y
}

func SigmoidVec(x mat.Matrix) mat.Matrix {
	sigmoid := func(i, j int, x float64) float64 {
		return 1 / (1 + math.Exp(-x))
	}
	r, c := x.Dims()
	y := mat.NewDense(r, c, nil)
	y.Apply(sigmoid, x)
	return y
}
