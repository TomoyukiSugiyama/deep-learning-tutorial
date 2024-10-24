package calc

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func CrossEntropyError(y *mat.Dense, t *mat.Dense) float64 {
	delta := 1e-7
	r, c := y.Caps()
	sum := 0.0

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			sum += t.At(i, j) * math.Log(y.At(i, j)+delta)
		}
	}
	return -sum / float64(r)
}
