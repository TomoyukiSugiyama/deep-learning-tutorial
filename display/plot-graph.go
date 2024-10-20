package display

import (
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
	p   *plot.Plot
	xys plotter.XYs
}

type Settings struct {
	Title   string
	X       string
	Y       string
	Dataset Dataset
}

type Dataset struct {
	X []float64
	Y []float64
}

func New(s Settings) display {
	p := plot.New()
	p.Title.Text = s.Title
	p.X.Label.Text = s.X
	p.Y.Label.Text = s.Y
	xys := xysFrom(s.Dataset)

	d := display{
		p:   p,
		xys: xys,
	}

	return d
}

func (d display) Show() {
	err := plotutil.AddLinePoints(d.p,
		"data", d.xys)
	if err != nil {
		panic(err)
	}

	if err := d.p.Save(4*vg.Inch, 4*vg.Inch, "outputs/points.png"); err != nil {
		panic(err)
	}
}

func xysFrom(dataset Dataset) plotter.XYs {
	pts := make(plotter.XYs, len(dataset.X))
	for i := range pts {
		pts[i].X = dataset.X[i]
		pts[i].Y = dataset.Y[i]
	}

	return pts
}
