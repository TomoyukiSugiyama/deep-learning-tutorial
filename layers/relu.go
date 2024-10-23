package layers

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

type ReLU struct {
	mask [][]bool
}

func InitReLU() *ReLU {
	return &ReLU{}
}

func (r *ReLU) Forward(x *mat.Dense) *mat.Dense {
	r.mask = make([][]bool, x.RawMatrix().Rows)
	for i := range r.mask {
		r.mask[i] = make([]bool, x.RawMatrix().Cols)
	}

	relu := func(i, j int, v float64) float64 {
		if v <= 0 {
			r.mask[i][j] = true
			return 0
		}
		r.mask[i][j] = false
		return v
	}

	out := mat.NewDense(x.RawMatrix().Rows, x.RawMatrix().Cols, nil)
	out.Apply(relu, x)
	fmt.Println(r.mask)
	return out
}

func (r *ReLU) Backward(dout *mat.Dense) *mat.Dense {
	dRelu := func(i, j int, v float64) float64 {
		if r.mask[i][j] {
			fmt.Println("mask", r.mask[i][j])
			return 0.0
		}
		return v
	}
	dx := mat.NewDense(dout.RawMatrix().Rows, dout.RawMatrix().Cols, nil)
	dx.Apply(dRelu, dout)
	return dx
}
