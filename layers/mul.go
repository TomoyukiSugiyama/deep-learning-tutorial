package layers

type Mul struct {
	x float64
	y float64
}

func InitMul() *Mul {

	return &Mul{}
}

func (m *Mul) Forward(x float64, y float64) float64 {
	m.x = x
	m.y = y
	return x * y
}

func (m *Mul) Backward(dout float64) (float64, float64) {
	dx := dout * m.y
	dy := dout * m.x
	return dx, dy
}
