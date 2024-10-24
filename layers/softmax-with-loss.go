package layers

import (
	af "tutorial/activation-functions"
	"tutorial/calc"

	"gonum.org/v1/gonum/mat"
)

type SoftmaxWithLoss struct {
	loss float64
	y    *mat.Dense
	t    *mat.Dense
}

func InitSoftmaxWithLoss() *SoftmaxWithLoss {
	return &SoftmaxWithLoss{}
}

func (s *SoftmaxWithLoss) Forward(x, t *mat.Dense) float64 {
	s.t = t
	s.y = af.Softmax(x)
	s.loss = calc.CrossEntropyError(s.y, t)
	return s.loss
}

func (s *SoftmaxWithLoss) Backward() *mat.Dense {
	batchSize := s.t.RawMatrix().Rows
	dx := mat.NewDense(batchSize, s.y.RawMatrix().Cols, nil)
	dx.Sub(s.y, s.t)
	dx.Scale(1/float64(batchSize), dx)
	return dx
}
