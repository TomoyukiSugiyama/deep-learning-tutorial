package calc

import (
	"gonum.org/v1/gonum/mat"
)

func Sum(a *mat.Dense) *mat.Dense {

	out := mat.NewDense(1, a.RawMatrix().Cols, nil)

	for r := 0; r < a.RawMatrix().Rows; r++ {
		for c := 0; c < a.RawMatrix().Cols; c++ {
			out.Set(0, c, out.At(0, c)+a.At(r, c))
		}
	}

	return out
}
