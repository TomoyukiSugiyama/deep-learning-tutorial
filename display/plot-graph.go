package display

import (
	"math/rand/v2"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

type Display interface {
	New() display
	Show()
}

type display struct {
	p *plot.Plot
}

func New() display {
	p := plot.New()
	p.Title.Text = "Plotutil example"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"
	d := display{
		p: plot.New(),
	}

	return d
}

func (d display) Show() {
	err := plotutil.AddLinePoints(d.p,
		"First", randomPoints(15),
		"Second", randomPoints(15),
		"Third", randomPoints(15))
	if err != nil {
		panic(err)
	}

	if err := d.p.Save(4*vg.Inch, 4*vg.Inch, "outputs/points.png"); err != nil {
		panic(err)
	}
}
func randomPoints(n int) plotter.XYs {
	pts := make(plotter.XYs, n)
	for i := range pts {
		if i == 0 {
			pts[i].X = rand.Float64()
		} else {
			pts[i].X = pts[i-1].X + rand.Float64()
		}
		pts[i].Y = pts[i].X + 10*rand.Float64()
	}
	return pts
}
