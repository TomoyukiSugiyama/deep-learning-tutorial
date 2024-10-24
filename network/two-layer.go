package network

import (
	"fmt"
	"math/rand"

	"tutorial/display"
	l "tutorial/layers"

	"gonum.org/v1/gonum/mat"
)

type TwoLayerNetwork struct {
	w1, w2, b1, b2                 *mat.Dense
	w1Grad, w2Grad, b1Grad, b2Grad *mat.Dense

	inputSize  int
	hiddenSize int
	outputSize int

	layers    *l.Layers
	lastLayer *l.SoftmaxWithLoss
}

func InitTwoLayerNetwork(inputSize int, hiddenSize int, outputSize int) *TwoLayerNetwork {
	generateRandomMatrix := func(row int, column int) *mat.Dense {
		const weightInitStd = 0.01
		data := make([]float64, row*column)
		for i := range data {
			data[i] = rand.NormFloat64() * weightInitStd
		}
		return mat.NewDense(row, column, data)
	}

	generateZeroMatrix := func(row int, column int) *mat.Dense {
		data := make([]float64, row*column)
		return mat.NewDense(row, column, data)
	}

	fmt.Println(">> InitTwoLayerNetwork")
	w1 := generateRandomMatrix(inputSize, hiddenSize)
	fmt.Print(">> w1 : ")
	fmt.Println(w1.Caps())
	w2 := generateRandomMatrix(hiddenSize, outputSize)
	fmt.Print(">> w2 : ")
	fmt.Println(w2.Caps())
	b1 := generateZeroMatrix(1, hiddenSize)
	fmt.Print(">> b1 : ")
	fmt.Println(b1.Caps())
	b2 := generateZeroMatrix(1, outputSize)
	fmt.Print(">> b2 : ")
	fmt.Println(b2.Caps())

	layers := l.InitLayers()
	layers.AddLayer(l.InitAffine(w1, b1))
	layers.AddLayer(l.InitReLU())
	layers.AddLayer(l.InitAffine(w2, b2))
	lastLayer := l.InitSoftmaxWithLoss()

	tn := TwoLayerNetwork{
		w1:         w1,
		w2:         w2,
		b1:         b1,
		b2:         b2,
		inputSize:  inputSize,
		hiddenSize: hiddenSize,
		outputSize: outputSize,
		layers:     layers,
		lastLayer:  lastLayer,
	}
	return &tn
}

func (n *TwoLayerNetwork) Predict(x *mat.Dense) *mat.Dense {
	for _, layer := range n.layers.Layers {
		x = layer.Forward(x)
		display.Print(x)
	}

	// batchSize := x.RawMatrix().Rows
	// // ab1 = x * w1 + b1
	// // z1 = sigmoid(ab1)
	// a1 := mat.NewDense(batchSize, n.hiddenSize, nil)
	// a1.Mul(x, n.w1)
	// ab1 := Add(a1, n.b1)
	// z1 := af.Sigmoid(ab1)
	// // ab2 = z1 * w2 + b2
	// // y = softmax(ab2)
	// a2 := mat.NewDense(batchSize, n.outputSize, nil)
	// a2.Mul(z1, n.w2)
	// ab2 := Add(a2, n.b2)
	// y := af.Softmax(ab2)
	return x
}

func (n *TwoLayerNetwork) Loss(x *mat.Dense, t *mat.Dense) float64 {
	y := n.Predict(x)
	return n.lastLayer.Forward(y, t)
}

func (n *TwoLayerNetwork) Gradient(x *mat.Dense, t *mat.Dense) {
	n.Loss(x, t)

	dout := n.lastLayer.Backward()
	for i := len(n.layers.Layers) - 1; i >= 0; i-- {
		dout = n.layers.Layers[i].Backward(dout)
	}

	// Affine1
	n.w1Grad, n.b1Grad = n.layers.Layers[0].GetGrads()
	// Affine2
	n.w2Grad, n.b2Grad = n.layers.Layers[2].GetGrads()
}

// func (n *TwoLayerNetwork) crossEntropyError(y *mat.Dense, t *mat.Dense) float64 {
// 	delta := 1e-7
// 	r, c := y.Caps()
// 	sum := 0.0
// 	for i := 0; i < r; i++ {
// 		for j := 0; j < c; j++ {
// 			sum += t.At(i, j) * math.Log(y.At(i, j)+delta)
// 		}
// 	}
// 	return -sum / float64(r)
// }

func numericalGradient(f func(*mat.Dense) float64, w *mat.Dense) *mat.Dense {
	h := 1e-4
	r, c := w.Caps()
	grad := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			tmp := w.At(i, j)
			w.Set(i, j, tmp+h)
			fxh1 := f(w)
			w.Set(i, j, tmp-h)
			fxh2 := f(w)
			grad.Set(i, j, (fxh1-fxh2)/(2*h))
			w.Set(i, j, tmp)
		}
	}
	return grad
}

func NumericalGradientTest() {
	f := func(x *mat.Dense) float64 {
		return x.At(0, 0)*x.At(0, 0) + x.At(1, 0)*x.At(1, 0)
	}
	x := mat.NewDense(2, 1, []float64{3, 0})
	grad := numericalGradient(f, x)
	display.Print(grad)
}

func (n *TwoLayerNetwork) NumericalGradient(x *mat.Dense, t *mat.Dense) {
	lossW := func(w *mat.Dense) float64 {
		return n.Loss(x, t)
	}
	n.w1Grad = numericalGradient(lossW, n.w1)
	n.w2Grad = numericalGradient(lossW, n.w2)
	n.b1Grad = numericalGradient(lossW, n.b1)
	n.b2Grad = numericalGradient(lossW, n.b2)
}

func updateParams(w *mat.Dense, grad *mat.Dense, lr float64) {
	r, c := w.Caps()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			w.Set(i, j, w.At(i, j)-lr*grad.At(i, j))
		}
	}
}

func (n *TwoLayerNetwork) UpdateParams(lr float64) {
	// updateParams(n.w1, n.w1Grad, lr)
	// updateParams(n.w2, n.w2Grad, lr)
	// updateParams(n.b1, n.b1Grad, lr)
	// updateParams(n.b2, n.b2Grad, lr)

	for _, layer := range n.layers.Layers {
		layer.UpdateParams(lr)
	}
}

func (n *TwoLayerNetwork) Accuracy(x *mat.Dense, t *mat.Dense) float64 {
	y := n.Predict(x)
	maxIndexList := maxIndexList(y)
	accuracyCount := 0
	for i, j := range maxIndexList {
		if int(t.At(i, j)) == 1 {
			accuracyCount++
		}
	}
	return float64(accuracyCount) / float64(t.RawMatrix().Rows)
}

func maxIndexList(d mat.Matrix) []int {
	r, c := d.Dims()

	maxList := make([]float64, r)
	maxIndexList := make([]int, r)
	for i := 0; i < r; i++ {
		maxList[i] = -99.0
		for j := 0; j < c; j++ {
			if d.At(i, j) > maxList[i] {
				maxList[i] = d.At(i, j)
				maxIndexList[i] = j
			}
		}
	}
	return maxIndexList
}

func (n *TwoLayerNetwork) GetGrads() (*mat.Dense, *mat.Dense, *mat.Dense, *mat.Dense) {
	w1Grad := mat.DenseCopyOf(n.w1Grad)
	b1Grad := mat.DenseCopyOf(n.b1Grad)
	w2Grad := mat.DenseCopyOf(n.w2Grad)
	b2Grad := mat.DenseCopyOf(n.b2Grad)

	return w1Grad, b1Grad, w2Grad, b2Grad
}
