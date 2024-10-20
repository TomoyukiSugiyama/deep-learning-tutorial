package network

import (
	af "tutorial/activation-functions"
	"tutorial/display"

	"gonum.org/v1/gonum/mat"
)

var w1, w2, w3, b1, b2, b3 *mat.Dense

func InitMetwork() {
	w1 = mat.NewDense(2, 3, []float64{0.1, 0.3, 0.5, 0.2, 0.4, 0.6})
	b1 = mat.NewDense(1, 3, []float64{0.1, 0.2, 0.3})
	w2 = mat.NewDense(3, 2, []float64{0.1, 0.4, 0.2, 0.5, 0.3, 0.6})
	b2 = mat.NewDense(1, 2, []float64{0.1, 0.2})
	w3 = mat.NewDense(2, 2, []float64{0.1, 0.3, 0.2, 0.4})
	b3 = mat.NewDense(1, 2, []float64{0.1, 0.2})
}

func Forward(x *mat.Dense) {
	InitMetwork()
	a1 := mat.NewDense(1, 3, nil)
	a1.Mul(x, w1)
	a1.Add(a1, b1)
	z1 := af.SigmoidVec(a1)

	a2 := mat.NewDense(1, 2, nil)
	a2.Mul(z1, w2)
	a2.Add(a2, b2)
	z2 := af.SigmoidVec(a2)

	a3 := mat.NewDense(1, 2, nil)
	a3.Mul(z2, w3)
	a3.Add(a3, b3)
	y := af.Identity(a3)

	display.Print(y)
}
