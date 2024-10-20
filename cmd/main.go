package main

import (
	af "tutorial/activation-functions"
	"tutorial/display"
)

func main() {
	x := af.Arange(-5.0, 5.0, 0.1)
	yStep := af.Step(x)
	display.New(display.Settings{Title: "Step Function", X: "X", Y: "Y", Dataset: display.Dataset{X: x, Y: yStep}, Output: "step.png"}).Show()

	ySig := af.Sigmoid(x)
	display.New(display.Settings{Title: "Sigmoid Function", X: "X", Y: "Y", Dataset: display.Dataset{X: x, Y: ySig}, Output: "sigmoid.png"}).Show()
}
