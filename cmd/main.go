package main

import (
	"fmt"
	"strconv"
	af "tutorial/activation-functions"
	"tutorial/display"
	"tutorial/network"

	"gonum.org/v1/gonum/mat"
)

func main() {
	dataset := network.GetData()
	network := network.InitNetwork()

	const batchSize = 100
	fmt.Println("Batch Size: ", batchSize)
	r, _ := dataset.TestData.Dims()
	testData := mat.DenseCopyOf(dataset.TestData)
	batchList := createBatchList(testData, batchSize)

	accuracyCount := 0

	for i, batch := range batchList {
		y := network.Forward(batch)
		maxIndexList := maxIndexList(y)
		for j, maiIndex := range maxIndexList {
			if int(dataset.TestLabels.At(i*batchSize+j, 0)) == maiIndex {
				accuracyCount++
			}
		}
		fmt.Println("batch("+strconv.Itoa(i)+") Accuracy Count: ", accuracyCount)
	}

	fmt.Println("Accuracy Count: ", accuracyCount)
	fmt.Println("Total Count: ", r)
	fmt.Println("Accuracy: ", float64(accuracyCount)/float64(r))

}

func dataT(d *mat.Dense) *mat.Dense {
	const imageHeight = 28
	const imageWidth = 28
	r, c := d.Dims()
	dataT := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		xMat := d.Slice(i, i+1, 0, c).(*mat.Dense)
		xUnFT := mat.DenseCopyOf(mat.NewDense(imageHeight, imageWidth, xMat.RawMatrix().Data).T())
		xMatT := mat.NewDense(1, c, xUnFT.RawMatrix().Data)
		dataT.SetRow(i, xMatT.RawMatrix().Data)
	}

	return dataT
}

func createBatchList(d *mat.Dense, batchSize int) []*mat.Dense {
	dataT := dataT(d)
	r, c := dataT.Dims()
	var batchList []*mat.Dense
	for i := 0; i < r; i += batchSize {
		slice := mat.DenseCopyOf(dataT.Slice(i, i+batchSize, 0, c).(*mat.Dense))
		batchList = append(batchList, slice)
	}
	return batchList
}

func maxIndexList(d mat.Matrix) []int {
	r, c := d.Dims()
	maxList := make([]float64, r)
	maxIndexList := make([]int, r)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			if d.At(i, j) > maxList[i] {
				maxList[i] = d.At(i, j)
				maxIndexList[i] = j
			}
		}
	}
	return maxIndexList
}

func Test() {
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
}
