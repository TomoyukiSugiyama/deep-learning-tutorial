package activationfunctions

import (
	"math"

	"tutorial/display"

	"gonum.org/v1/gonum/mat"
)

func Softmax(x *mat.Dense) *mat.Dense {
	expf := func(i, j int, x float64) float64 {
		return math.Exp(x)
	}
	r, c := x.Dims()
	exp := mat.NewDense(r, c, nil)

	exp.Apply(expf, x)
	display.Print(exp)
	sum := mat.Sum(exp)

	divf := func(i, j int, x float64) float64 {
		return x / sum
	}

	y := mat.NewDense(r, c, nil)
	y.Apply(divf, exp)

	return y
}
