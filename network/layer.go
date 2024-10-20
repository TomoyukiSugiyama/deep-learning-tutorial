package network

import (
	af "tutorial/activation-functions"
	"tutorial/display"

	"github.com/po3rin/gomnist"
	"gonum.org/v1/gonum/mat"
)

type Network struct {
	w1, w2, w3, b1, b2, b3 *mat.Dense
}

func GetData() gomnist.MNIST {
	l := gomnist.NewLoader("./datasets", gomnist.Normalization(true), gomnist.OneHotLabel(false))

	dataset, err := l.Load()
	if err != nil {
		panic(err)
	}
	_ = dataset.TrainData.At(0, 135)

	_ = dataset.TrainLabels

	return dataset
}

func InitNetwork() Network {
	n := Network{}
	n.w1 = mat.NewDense(2, 3, []float64{0.1, 0.3, 0.5, 0.2, 0.4, 0.6})
	n.b1 = mat.NewDense(1, 3, []float64{0.1, 0.2, 0.3})
	n.w2 = mat.NewDense(3, 2, []float64{0.1, 0.4, 0.2, 0.5, 0.3, 0.6})
	n.b2 = mat.NewDense(1, 2, []float64{0.1, 0.2})
	n.w3 = mat.NewDense(2, 2, []float64{0.1, 0.3, 0.2, 0.4})
	n.b3 = mat.NewDense(1, 2, []float64{0.1, 0.2})
	return n
}

func (n Network) Forward(x *mat.Dense) {
	a1 := mat.NewDense(1, 3, nil)
	a1.Mul(x, n.w1)
	a1.Add(a1, n.b1)
	z1 := af.Sigmoid(a1)

	a2 := mat.NewDense(1, 2, nil)
	a2.Mul(z1, n.w2)
	a2.Add(a2, n.b2)
	z2 := af.Sigmoid(a2)

	a3 := mat.NewDense(1, 2, nil)
	a3.Mul(z2, n.w3)
	a3.Add(a3, n.b3)
	y := af.Identity(a3)

	display.Print(y)
}
