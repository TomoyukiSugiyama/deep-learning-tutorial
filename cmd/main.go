package main

import (
	"fmt"
	af "tutorial/activation-functions"
	"tutorial/display"
	"tutorial/network"

	"gonum.org/v1/gonum/mat"
)

func main() {
	x := af.Arange(-5.0, 5.0, 0.1)
	yStep := af.Step(x)
	display.New(display.Settings{Title: "Step Function", X: "X", Y: "Y", Dataset: display.Dataset{X: x, Y: yStep}, Output: "step.png"}).Show()

	yReLU := af.ReLU(x)
	display.New(display.Settings{Title: "ReLU Function", X: "X", Y: "Y", Dataset: display.Dataset{X: x, Y: yReLU}, Output: "relu.png"}).Show()

	xVec := mat.NewDense(3, 3, []float64{-4, -3, -2, -1, 0, 1, 2, 3, 4})
	yVec := af.Sigmoid(xVec)
	display.Print(yVec)

	xMulVec := mat.NewDense(1, 2, []float64{3, 2})
	yMulVec := mat.NewDense(1, 2, []float64{2, 4})
	yMulTVec := yMulVec.T()
	mul := mat.NewDense(1, 1, nil)
	mul.Mul(xMulVec, yMulTVec)
	display.Print(mul)

	xSoftmax := mat.NewDense(1, 3, []float64{0.3, 2.9, 4.0})
	ySoftmax := af.Softmax(xSoftmax)
	display.Print(ySoftmax)

	dataset := network.GetData()
	network := network.InitNetwork()
	// display.Print(dataset.TrainData)
	r, c := dataset.TestData.Dims()
	testData := mat.DenseCopyOf(dataset.TestData)
	var xMat *mat.Dense
	accuracyCount := 0
	// display.Print(dataset.TestLabels)
	for i := 0; i < r; i++ {
		xMat = testData.Slice(i, i+1, 0, c).(*mat.Dense)
		xUnFT := mat.DenseCopyOf(mat.NewDense(28, 28, xMat.RawRowView(0)).T())
		xMatT := mat.NewDense(1, 784, xUnFT.RawMatrix().Data)
		// need to set Normalization = false
		// display.Save(strconv.Itoa(i), xMatT)
		y := network.Forward(xMatT)
		fmt.Println("--------------------")
		fmt.Println("Index: ", i)
		display.Print(y)
		maxIndex := maxIndex(y)
		fmt.Println("Predicted: ", maxIndex)
		fmt.Println("Actual: ", int(dataset.TestLabels.At(i, 0)))
		if int(dataset.TestLabels.At(i, 0)) == maxIndex {
			accuracyCount++
		}
	}

	fmt.Println("Accuracy Count: ", accuracyCount)
	fmt.Println("Total Count: ", r)
	fmt.Println("Accuracy: ", float64(accuracyCount)/float64(r))

}

func maxIndex(d mat.Matrix) int {
	r, c := d.Dims()
	max := -9999.0
	maxIndex := 0
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			if d.At(i, j) > max {
				max = d.At(i, j)
				maxIndex = i*c + j
			}
		}
	}
	return maxIndex
}
