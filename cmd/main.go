package main

import (
	af "tutorial/activation-functions"
	"tutorial/display"
)

func main() {
	x := af.Arange(-5.0, 5.0, 0.1)
	y := af.Step(x)

	display.New(display.Settings{Title: "Step Function", X: "X", Y: "Y", Dataset: display.Dataset{X: x, Y: y}}).Show()
}
