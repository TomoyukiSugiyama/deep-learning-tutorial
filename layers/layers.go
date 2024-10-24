package layers

import (
	"gonum.org/v1/gonum/mat"
)

type Layers struct {
	Layers []Layer
}

type Layer interface {
	Forward(x *mat.Dense) *mat.Dense
	Backward(dout *mat.Dense) *mat.Dense
	GetGrads() (*mat.Dense, *mat.Dense)
}

func InitLayers() *Layers {
	return &Layers{}
}

func (l *Layers) AddLayer(layer Layer) {
	l.Layers = append(l.Layers, layer)
}
