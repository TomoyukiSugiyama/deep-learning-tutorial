package layers

import (
	"tutorial/calc"

	"gonum.org/v1/gonum/mat"
)

type Affine struct {
	w  *mat.Dense
	b  *mat.Dense
	x  *mat.Dense
	dw *mat.Dense
	db *mat.Dense
}

func InitAffine(w *mat.Dense, b *mat.Dense) *Affine {
	return &Affine{w: w, b: b}
}

func (a *Affine) Forward(x *mat.Dense) *mat.Dense {
	a.x = x
	xw := mat.NewDense(x.RawMatrix().Rows, a.w.RawMatrix().Cols, nil)
	xw.Mul(x, a.w)
	out := calc.Add(xw, a.b)
	return out
}

func (a *Affine) Backward(dout *mat.Dense) *mat.Dense {
	dx := mat.NewDense(dout.RawMatrix().Rows, a.w.RawMatrix().Rows, nil)
	dx.Mul(dout, a.w.T())

	a.dw = mat.NewDense(a.x.RawMatrix().Cols, dout.RawMatrix().Cols, nil)
	a.dw.Mul(a.x.T(), dout)

	a.db = mat.NewDense(1, dout.RawMatrix().Cols, nil)
	a.db = calc.Sum(dout)

	return dx
}

func (a *Affine) GetGrads() (*mat.Dense, *mat.Dense) {
	return a.dw, a.db
}

func (a *Affine) UpdateParams(learningRate float64) {
	dwTmp := mat.NewDense(a.dw.RawMatrix().Rows, a.dw.RawMatrix().Cols, nil)
	dbTmp := mat.NewDense(a.db.RawMatrix().Rows, a.db.RawMatrix().Cols, nil)
	dwTmp.Scale(learningRate, a.dw)
	dbTmp.Scale(learningRate, a.db)
	a.w.Sub(a.w, dwTmp)
	a.b.Sub(a.b, dbTmp)
}
