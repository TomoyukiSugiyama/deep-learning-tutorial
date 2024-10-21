package network

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
	af "tutorial/activation-functions"

	"github.com/po3rin/gomnist"
	"gonum.org/v1/gonum/mat"
)

type Network struct {
	w1, w2, w3, b1, b2, b3 *mat.Dense
}

func GetData() gomnist.MNIST {
	l := gomnist.NewLoader("./datasets", gomnist.Normalization(true), gomnist.OneHotLabel(false))

	dataset, err := l.Load()
	if err != nil {
		panic(err)
	}
	return dataset
}

func convertFromCSV(filename string) (int, int, []float64) {
	file, err := os.Open("./network/weights/" + filename)
	if err != nil {
		panic(err)
	}
	defer file.Close()
	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		panic(err)
	}
	r := len(records)
	c := len(records[0])
	data := make([]float64, r*c)
	for i, recordR := range records {
		for j, recordC := range recordR {
			data[i*c+j], err = strconv.ParseFloat(recordC, 64)
			if err != nil {
				panic(err)
			}
		}
	}
	return r, c, data
}

func InitNetwork() Network {
	n := Network{}
	n.w1 = mat.NewDense(convertFromCSV("w1.csv"))
	fmt.Println(n.w1.Caps())
	n.b1 = mat.NewDense(convertFromCSV("b1.csv"))
	fmt.Println(n.b1.Caps())
	n.w2 = mat.NewDense(convertFromCSV("w2.csv"))
	fmt.Println(n.w2.Caps())
	n.b2 = mat.NewDense(convertFromCSV("b2.csv"))
	fmt.Println(n.b2.Caps())
	n.w3 = mat.NewDense(convertFromCSV("w3.csv"))
	fmt.Println(n.w3.Caps())
	n.b3 = mat.NewDense(convertFromCSV("b3.csv"))
	fmt.Println(n.b3.Caps())
	return n
}

func (n Network) Forward(x *mat.Dense) *mat.Dense {
	a1R, a1C := n.b1.Caps()
	// a1 = x * w1 + b1
	// z1 = sigmoid(a1)
	a1 := mat.NewDense(a1R, a1C, nil)
	a1.Mul(x, n.w1)
	a1.Add(a1, n.b1)
	z1 := af.Sigmoid(a1)
	// a2 = z1 * w2 + b2
	// z2 = sigmoid(a2)
	a2R, a2C := n.b2.Caps()
	a2 := mat.NewDense(a2R, a2C, nil)
	a2.Mul(z1, n.w2)
	a2.Add(a2, n.b2)
	z2 := af.Sigmoid(a2)
	// a3 = z2 * w3 + b3
	// y = softmax(a3)
	a3R, a3C := n.b3.Caps()
	a3 := mat.NewDense(a3R, a3C, nil)
	a3.Mul(z2, n.w3)
	a3.Add(a3, n.b3)
	y := af.Softmax(a3)

	return y
}
