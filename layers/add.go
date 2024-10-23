package layers

type Add struct {
}

func InitAdd() *Add {
	return &Add{}
}

func (a *Add) Forward(x float64, y float64) float64 {
	return x + y
}

func (a *Add) Backward(dout float64) (float64, float64) {
	dx := dout
	dy := dout
	return dx, dy
}
