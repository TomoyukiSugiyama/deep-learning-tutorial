package main

import (
	"fmt"
	"math/rand"
	"strconv"
	"time"
	af "tutorial/activation-functions"
	"tutorial/display"
	"tutorial/layers"
	"tutorial/network"

	"gonum.org/v1/gonum/mat"
)

func main() {
	SoftmaxWithLoss()
	// Affine()
	// Sigmoid()
	// ReLU()
	// CalcPrice()
	// Training()
	// TestNetwork()
	// TrainTwoLayerNetwork()

}

func SoftmaxWithLoss() {
	x := []float64{1.0, 2.0, 3.0, 4.0}
	t := []float64{0.0, 0.0, 1.0, 0.0}
	xMat := mat.NewDense(1, 4, x)
	tMat := mat.NewDense(1, 4, t)
	fmt.Println("Input")
	fmt.Println("x")
	display.Print(xMat)
	fmt.Println("t")
	display.Print(tMat)
	softmaxWithLossLayer := layers.InitSoftmaxWithLoss()
	loss := softmaxWithLossLayer.Forward(xMat, tMat)
	fmt.Println("Loss: ", loss)
	dx := softmaxWithLossLayer.Backward()
	fmt.Println("dx")
	display.Print(dx)
}

func Affine() {
	x := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	w := []float64{1.0, 2.0, 3.0, 4.0}
	b := []float64{1.0, 2.0}
	xMat := mat.NewDense(3, 2, x)
	wMat := mat.NewDense(2, 2, w)
	bMat := mat.NewDense(1, 2, b)
	fmt.Println("Input")
	fmt.Println("x")
	display.Print(xMat)
	fmt.Println("w")
	display.Print(wMat)
	fmt.Println("b")
	display.Print(bMat)
	affineLayer := layers.InitAffine(wMat, bMat)
	fmt.Println("Forward")
	y := affineLayer.Forward(xMat)
	display.Print(y)
	fmt.Println("Backward")
	dout := mat.NewDense(3, 2, []float64{5.0, -2.0, 7.0, 4.0, 3.0, 1.0})
	fmt.Println("dout")
	display.Print(dout)
	dx := affineLayer.Backward(dout)
	fmt.Println("dx")
	display.Print(dx)
}

func Sigmoid() {
	x := []float64{-1.0, 1.0, 2.0, -2.0}
	xMat := mat.NewDense(2, 2, x)
	display.Print(xMat)
	sigmoidLayer := layers.InitSigmoid()
	y := sigmoidLayer.Forward(xMat)
	display.Print(y)
	dout := mat.NewDense(2, 2, []float64{5.0, -2.0, 7.0, 4.0})
	dx := sigmoidLayer.Backward(dout)
	display.Print(dx)
}

func ReLU() {
	x := []float64{-1.0, 1.0, 2.0, -2.0}
	xMat := mat.NewDense(2, 2, x)
	display.Print(xMat)
	reluLayer := layers.InitReLU()
	y := reluLayer.Forward(xMat)
	display.Print(y)
	dout := mat.NewDense(2, 2, []float64{5.0, -2.0, 7.0, 4.0})
	dx := reluLayer.Backward(dout)
	display.Print(dx)
}

func CalcPrice() {
	const apple = 100
	const appleNum = 2
	const orange = 150
	const orangeNum = 3
	const tax = 1.1

	mulAppleLayer := layers.InitMul()
	mulOrangeLayer := layers.InitMul()
	addAppleOrangeLayer := layers.InitAdd()
	mulTaxLayer := layers.InitMul()

	applePrice := mulAppleLayer.Forward(apple, appleNum)
	orangePrice := mulOrangeLayer.Forward(orange, orangeNum)
	allPrice := addAppleOrangeLayer.Forward(applePrice, orangePrice)
	price := mulTaxLayer.Forward(allPrice, tax)
	fmt.Println("Price: ", price)

	dPrice := 1.0
	dAllPrice, dTax := mulTaxLayer.Backward(dPrice)
	dApplePrice, dOrangePrice := addAppleOrangeLayer.Backward(dAllPrice)
	dOrange, dOrangeNum := mulOrangeLayer.Backward(dOrangePrice)
	dApple, dAppleNum := mulAppleLayer.Backward(dApplePrice)

	fmt.Println("dApple: ", dApple)
	fmt.Println("dAppleNum: ", dAppleNum)
	fmt.Println("dOrange: ", dOrange)
	fmt.Println("dOrangeNum: ", dOrangeNum)

	fmt.Println("dTax: ", dTax)
}

func Training() {
	fmt.Println("Training Start")
	defer fmt.Println("Training End")
	const batchSize = 100
	dataset := network.GetData()
	trainData := mat.DenseCopyOf(dataset.TrainData)
	trainLavels := mat.DenseCopyOf(dataset.TrainLabels)
	network := network.InitNetwork()

	const iteration = 1
	fmt.Println("Iteration: ", iteration)
	fmt.Println("Batch Size: ", batchSize)

	accuracyCount := 0
	for i := 0; i < iteration; i++ {
		batch, t := randomChoice(trainData, trainLavels, batchSize)
		y := network.Forward(batch)
		maxIndexList := maxIndexList(y)
		for j, maiIndex := range maxIndexList {
			if int(t.At(j, 0)) == maiIndex {
				accuracyCount++
			}
		}
		fmt.Println("batch("+strconv.Itoa(i)+") Accuracy Count: ", accuracyCount)
	}
	fmt.Println("Accuracy Count: ", accuracyCount)
	fmt.Println("Total Count: ", batchSize*iteration)
	fmt.Println("Accuracy: ", float64(accuracyCount)/float64(batchSize*iteration))
}

func randomChoice(d *mat.Dense, t *mat.Dense, size int) (*mat.Dense, *mat.Dense) {
	dataT := dataT(d)
	rnd := rand.New(rand.NewSource(time.Now().UnixNano()))
	randomData := mat.NewDense(size, dataT.RawMatrix().Cols, nil)
	randomLabel := mat.NewDense(size, t.RawMatrix().Cols, nil)
	for i := 0; i < size; i++ {
		r := rnd.Intn(size)
		randomData.SetRow(i, dataT.RawRowView(r))
		randomLabel.SetRow(i, t.RawRowView(r))
	}
	return randomData, randomLabel
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

func TestNetwork() {
	fmt.Println("Test Start")
	defer fmt.Println("Test End")

	dataset := network.GetData()
	network := network.InitNetwork()

	const batchSize = 100
	fmt.Println("Batch Size: ", batchSize)
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
	r, _ := dataset.TestData.Dims()
	fmt.Println("Total Count: ", r)
	fmt.Println("Accuracy: ", float64(accuracyCount)/float64(r))
}

func TrainTwoLayerNetwork() {
	fmt.Println("Train Start")
	defer fmt.Println("Train End")

	const inputSize = 784
	const hiddenSize = 50
	const outputSize = 10
	const batchSize = 100
	dataset := network.GetData()
	trainData := mat.DenseCopyOf(dataset.TrainData)
	trainLabels := mat.DenseCopyOf(dataset.TrainLabels)
	testData := mat.DenseCopyOf(dataset.TestData)
	testLabels := mat.DenseCopyOf(dataset.TestLabels)
	network := network.InitTwoLayerNetwork(inputSize, hiddenSize, outputSize)

	const iteration = 1000
	const leaningRate = 0.1
	trainLossList := make([]float64, iteration)
	iterationList := make([]float64, iteration)
	trainAccList := make([]float64, iteration)
	testAccList := make([]float64, iteration)
	fmt.Println(">> Iteration: ", iteration)
	fmt.Println(">> Leaning Rate: ", leaningRate)
	fmt.Println(">> Train Total Count: ", trainData.RawMatrix().Rows)
	fmt.Println(">> Test Total Count: ", testData.RawMatrix().Rows)

	for i := 0; i < iteration; i++ {
		batch, t := randomChoice(trainData, trainLabels, batchSize)
		network.NumericalGradient(batch, t)
		network.UpdateParams(leaningRate)
		// check result
		trainLossList[i] = network.Loss(batch, t)
		iterationList[i] = float64(i)
		fmt.Println("Loss (", strconv.Itoa(i), ") :", trainLossList[i])
		trainAccList[i] = network.Accuracy(trainData, trainLabels)
		fmt.Println("Train Accuracy: ", trainAccList[i])
		testAccList[i] = network.Accuracy(testData, testLabels)
		fmt.Println("Test Accuracy: ", testAccList[i])
	}

	loss := display.Settings{
		Title:   "Loss",
		X:       "Iteration",
		Y:       "Loss",
		Dataset: display.Dataset{X: iterationList, Y: trainLossList},
		Output:  "loss.png",
	}
	display.New(loss).Show()

	train := display.Settings{
		Title:   "Train Accuracy",
		X:       "Iteration",
		Y:       "Accuracy",
		Dataset: display.Dataset{X: iterationList, Y: trainAccList},
		Output:  "train-accuracy.png",
	}
	display.New(train).Show()

	test := display.Settings{
		Title:   "Test Accuracy",
		X:       "Iteration",
		Y:       "Accuracy",
		Dataset: display.Dataset{X: iterationList, Y: testAccList},
		Output:  "test-accuracy.png",
	}
	display.New(test).Show()

}
