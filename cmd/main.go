package main

import (
	"tutorial/display"
)

func main() {
	display.New(display.Settings{Title: "Title", X: "X", Y: "Y"}).Show()
}
