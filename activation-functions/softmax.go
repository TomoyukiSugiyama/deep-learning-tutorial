package activationfunctions

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func Softmax(x *mat.Dense) *mat.Dense {
	maxList := make([]float64, 0)
	r, c := x.Dims()
	for i := 0; i < r; i++ {
		max := -999.0
		for j := 0; j < c; j++ {
			if x.At(i, j) > max {
				max = x.At(i, j)
			}
		}
		maxList = append(maxList, max)
	}
	expf := func(i, j int, x float64) float64 {
		return math.Exp(x - maxList[i])
	}
	exp := mat.NewDense(r, c, nil)
	exp.Apply(expf, x)
	sumList := make([]float64, 0)
	for i := 0; i < r; i++ {
		sum := 0.0
		for j := 0; j < c; j++ {
			sum += exp.At(i, j)
		}
		sumList = append(sumList, sum)
	}

	y := mat.NewDense(r, c, nil)

	scale := func(i, j int, x float64) float64 {
		return x / sumList[i]
	}
	y.Apply(scale, exp)

	return y
}
