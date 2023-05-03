package main

import (
	"fmt"
	"log"
	"math"
	"os"

	"github.com/YohayAiTe/elefas"
)

func main() {
	// load weights
	weightsFile, err := os.Open("model.npz")
	if err != nil {
		log.Fatal(err)
	}
	weights, err := elefas.LoadNumpyDataFrames[float32](weightsFile)
	if err != nil {
		log.Fatal(err)
	}

	// build model
	model := elefas.NewModel[float32](1)
	l := model.AddLayer(&elefas.FlattenLayer[float32]{}, nil)
	l = l.AddLayer(elefas.NewDenseLayer(weights[0], weights[1]))
	l = l.AddLayer(elefas.NewReLUActivation[float32]())
	l = l.AddLayer(elefas.NewDenseLayer(weights[2], weights[3]))
	l = l.AddLayer(elefas.NewReLUActivation[float32]())
	l = l.AddLayer(elefas.NewDenseLayer(weights[4], weights[5]))
	l = l.AddLayer(elefas.NewReLUActivation[float32]())
	l = l.AddLayer(elefas.NewDenseLayer(weights[6], weights[7]))
	l = l.AddLayer(&elefas.SoftmaxActivation[float32]{Axis: -1})
	model.SetOutput(l, 0)

	// load the test data
	testDataFile, err := os.Open("test_data.npy")
	if err != nil {
		log.Fatal(err)
	}
	testData, err := elefas.LoadNumpyDataFrame[float32](testDataFile)
	if err != nil {
		log.Fatal(err)
	}
	testExpectedFile, err := os.Open("test_expected.npy")
	if err != nil {
		log.Fatal(err)
	}
	testExpected, err := elefas.LoadNumpyDataFrame[float32](testExpectedFile)
	if err != nil {
		log.Fatal(err)
	}

	// predict and compute loss
	prediction := model.Predict(testData)[0]
	var sum float32
	for i := 0; i < testExpected.Dim(0); i++ {
		for c := 0; c < testExpected.Dim(1); c++ {
			sum += testExpected.At(i, c) * float32(math.Log(float64(prediction.At(i, c))))
		}
	}
	fmt.Printf("test loss: %f\n", -sum/float32(testExpected.Dim(0)))
}
