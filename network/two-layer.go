package network

import (
	"fmt"
	"math"
	"math/rand"

	af "tutorial/activation-functions"
	"tutorial/display"

	"gonum.org/v1/gonum/mat"
)

type TwoLayerNetwork struct {
	w1, w2, b1, b2 *mat.Dense
	inputSize      int
	hiddenSize     int
	outputSize     int
}

func InitTwoLayerNetwork(inputSize int, hiddenSize int, outputSize int) *TwoLayerNetwork {
	w1 := generateRandomMatrix(inputSize, hiddenSize)
	fmt.Print("w1 : ")
	fmt.Println(w1.Caps())
	w2 := generateRandomMatrix(hiddenSize, outputSize)
	fmt.Print("w2 : ")
	fmt.Println(w2.Caps())
	b1 := generateZeroMatrix(1, hiddenSize)
	fmt.Print("b1 : ")
	fmt.Println(b1.Caps())
	b2 := generateZeroMatrix(1, outputSize)
	fmt.Print("b2 : ")
	fmt.Println(b2.Caps())
	display.Print(b2)
	return &TwoLayerNetwork{w1, w2, b1, b2, inputSize, hiddenSize, outputSize}
}

func generateRandomMatrix(row int, column int) *mat.Dense {
	data := make([]float64, row*column)
	for i := range data {
		data[i] = rand.NormFloat64()
	}
	return mat.NewDense(row, column, data)
}

func generateZeroMatrix(row int, column int) *mat.Dense {
	data := make([]float64, row*column)
	return mat.NewDense(row, column, data)
}

func (n *TwoLayerNetwork) Predict(x *mat.Dense) *mat.Dense {
	a1 := mat.NewDense(n.inputSize, n.hiddenSize, nil)
	a1.Mul(x, n.w1)
	ab1 := Add(a1, n.b1)
	z1 := af.Sigmoid(ab1)

	a2 := mat.NewDense(n.hiddenSize, n.outputSize, nil)
	a2.Mul(z1, n.w2)
	ab2 := Add(a2, n.b2)
	y := af.Softmax(ab2)
	return y
}

func (n *TwoLayerNetwork) Loss(x *mat.Dense, t *mat.Dense) float64 {
	y := n.Predict(x)
	return crossEntropyError(y, t)
}

func crossEntropyError(y *mat.Dense, t *mat.Dense) float64 {
	delta := 1e-7
	r, c := y.Caps()
	sum := 0.0
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			sum += t.At(i, j) * math.Log(y.At(i, j)+delta)
		}
	}
	return -sum
}

func NumericalGradient(f func(*mat.Dense) float64, x *mat.Dense) *mat.Dense {
	h := 1e-4
	r, c := x.Caps()
	grad := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			tmpVal := x.At(i, j)
			x.Set(i, j, tmpVal+h)
			fxh1 := f(x)
			x.Set(i, j, tmpVal-h)
			fxh2 := f(x)
			grad.Set(i, j, (fxh1-fxh2)/(2*h))
			x.Set(i, j, tmpVal)
		}
	}
	return grad
}
