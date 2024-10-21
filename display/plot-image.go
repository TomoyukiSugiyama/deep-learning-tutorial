package display

import (
	"image"
	"image/color"
	"image/png"
	"os"

	"gonum.org/v1/gonum/mat"
)

func Save(i string, d mat.Matrix) {
	x := 0
	y := 0
	width := 28
	height := 28

	img := image.NewGray(image.Rect(x, y, width, height))
	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			img.Set(j, i, color.Gray{uint8(d.At(0, i*height+j))})
		}
	}

	file, _ := os.Create("outputs/" + i + ".png")
	defer file.Close()

	if err := png.Encode(file, img); err != nil {
		panic(err)
	}
}
