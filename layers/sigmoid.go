package layers

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type Sigmoid struct {
	out *mat.Dense
}

func InitSigmoid() *Sigmoid {

	return &Sigmoid{}
}

func (s *Sigmoid) Forward(x *mat.Dense) *mat.Dense {
	sigmoid := func(i, j int, v float64) float64 {
		return 1 / (1 + math.Exp(-v))
	}
	out := mat.NewDense(x.RawMatrix().Rows, x.RawMatrix().Cols, nil)
	out.Apply(sigmoid, x)
	s.out = out
	return out
}

func (s *Sigmoid) Backward(dout *mat.Dense) *mat.Dense {
	sigmoidBackward := func(i, j int, v float64) float64 {
		return v * (1.0 - v)
	}
	dx := mat.NewDense(dout.RawMatrix().Rows, dout.RawMatrix().Cols, nil)
	dx.Apply(sigmoidBackward, s.out)
	dx.MulElem(dx, dout)
	return dx
}
