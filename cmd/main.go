package main

import (
	"fmt"
	af "tutorial/activation-functions"

	"tutorial/display"

	"gonum.org/v1/gonum/mat"
)

func main() {
	x := af.Arange(-5.0, 5.0, 0.1)
	yStep := af.Step(x)
	display.New(display.Settings{Title: "Step Function", X: "X", Y: "Y", Dataset: display.Dataset{X: x, Y: yStep}, Output: "step.png"}).Show()

	ySig := af.Sigmoid(x)
	display.New(display.Settings{Title: "Sigmoid Function", X: "X", Y: "Y", Dataset: display.Dataset{X: x, Y: ySig}, Output: "sigmoid.png"}).Show()

	yReLU := af.ReLU(x)
	display.New(display.Settings{Title: "ReLU Function", X: "X", Y: "Y", Dataset: display.Dataset{X: x, Y: yReLU}, Output: "relu.png"}).Show()

	xVec := mat.NewDense(3, 3, []float64{-4, -3, -2, -1, 0, 1, 2, 3, 4})
	yVec := af.SigmoidVec(xVec)
	fmt.Println(yVec)
}
