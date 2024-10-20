package activationfunctions

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func Sigmoid(x mat.Matrix) mat.Matrix {
	sigmoid := func(i, j int, x float64) float64 {
		return 1 / (1 + math.Exp(-x))
	}
	r, c := x.Dims()
	y := mat.NewDense(r, c, nil)
	y.Apply(sigmoid, x)
	return y
}
