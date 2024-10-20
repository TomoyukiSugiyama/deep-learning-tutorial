package activationfunctions

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func Softmax(x *mat.Dense) *mat.Dense {
	max := mat.Max(x)
	expf := func(i, j int, x float64) float64 {
		return math.Exp(x - max)
	}
	r, c := x.Dims()
	exp := mat.NewDense(r, c, nil)

	exp.Apply(expf, x)
	sum := mat.Sum(exp)

	divf := func(i, j int, x float64) float64 {
		return x / sum
	}

	y := mat.NewDense(r, c, nil)
	y.Apply(divf, exp)

	return y
}
