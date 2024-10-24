package calc

import (
	"gonum.org/v1/gonum/mat"
)

func Add(a *mat.Dense, b *mat.Dense) *mat.Dense {
	add := func(i, j int, v float64) float64 {
		return v + b.At(0, j)
	}

	out := mat.NewDense(a.RawMatrix().Rows, a.RawMatrix().Cols, nil)
	out.Apply(add, a)

	return out
}
