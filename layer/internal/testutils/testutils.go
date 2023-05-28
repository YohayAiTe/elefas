package testutils

import (
	"math"
	"math/rand"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"

	"github.com/YohayAiTe/elefas"
)

type PythonLayerData[T elefas.SizedNumber] struct {
	Name    string
	Weights []elefas.DataFrame[T]
}

func TestLayerByPython[T elefas.SizedNumber](t *testing.T, pythonData PythonLayerData[T],
	layer elefas.Layer[T], input elefas.DataFrame[T], epsilon T) {

	weightsFile, err := os.CreateTemp("testdata", "weights_*.npz")
	if err != nil {
		t.Fatalf("error creating temp weights file: %v", err)
	}
	defer func() {
		name := weightsFile.Name()
		weightsFile.Close()
		os.Remove(name)
	}()
	err = elefas.SaveNumpyDataFrames(pythonData.Weights, weightsFile)
	if err != nil {
		t.Fatalf("error writing to temp weights file: %v", err)
	}

	inputFile, err := os.CreateTemp("testdata", "input_*.npy")
	if err != nil {
		t.Fatalf("error creating temp input file: %v", err)
	}
	defer func() {
		name := inputFile.Name()
		inputFile.Close()
		os.Remove(name)
	}()
	err = elefas.SaveNumpyDataFrame(input, inputFile)
	if err != nil {
		t.Fatalf("error writing to temp input file: %v", err)
	}

	outputFile, err := os.CreateTemp("testdata", "output_*.npy")
	if err != nil {
		t.Fatalf("error creating temp output file: %v", err)
	}
	outputFileName := outputFile.Name()
	defer os.Remove(outputFileName)
	outputFile.Close()

	pythonCommand := exec.Command("python3", "testdata/keras_script.py", pythonData.Name,
		weightsFile.Name(), inputFile.Name(), outputFileName)

	if out, err := pythonCommand.CombinedOutput(); err != nil {
		t.Fatalf("error running python script: %v; its output is:\n%s", err, out)
	}

	outputFile, err = os.Open(outputFileName)
	if err != nil {
		t.Fatalf("error reopening output file: %v", err)
	}
	defer outputFile.Close()

	pythonOutput, err := elefas.LoadNumpyDataFrame[T](outputFile)
	if err != nil {
		t.Fatalf("error loading output: %v", err)
	}

	computedOutput := layer.Apply(input)
	if computedOutput.DimCount() != pythonOutput.DimCount() {
		t.Fatalf("computed output and python output have different number of dimensions: %d-%d",
			computedOutput.DimCount(), pythonOutput.DimCount())
	}
	for i := 0; i < computedOutput.DimCount(); i++ {
		if computedOutput.Dim(i) != pythonOutput.Dim(i) {
			t.Fatalf("computed output and python output differ in dimension %d: %d-%d",
				i, computedOutput.Dim(i), pythonOutput.Dim(i))
		}
	}
	for i := 0; i < computedOutput.TotalSize(); i++ {
		diff := computedOutput.FlatAt(i) - pythonOutput.FlatAt(i)
		if diff < 0 {
			diff = -diff
		}
		if diff > epsilon {
			t.Fatalf("computed output and python output differ in flat index %d by more than epsilon(%v): (%v)-(%v)",
				i, epsilon, computedOutput.FlatAt(i), pythonOutput.FlatAt(i))
		}
	}
}

func RandomDataFrame[T elefas.SizedNumber](r *rand.Rand, dims []int) elefas.DataFrame[T] {
	df := elefas.MakeDataFrame[T](dims)
	switch any(T(0)).(type) {
	case int8:
		for i := 0; i < df.TotalSize(); i++ {
			df.SetFlatAt(T(r.Intn(math.MaxInt8)), i)
		}
	case int16:
		for i := 0; i < df.TotalSize(); i++ {
			df.SetFlatAt(T(r.Intn(math.MaxInt16)), i)
		}
	case int32:
		for i := 0; i < df.TotalSize(); i++ {
			df.SetFlatAt(T(r.Int31()), i)
		}
	case int64:
		for i := 0; i < df.TotalSize(); i++ {
			df.SetFlatAt(T(r.Int63()), i)
		}
	case uint8:
		for i := 0; i < df.TotalSize(); i++ {
			df.SetFlatAt(T(r.Intn(math.MaxUint8)), i)
		}
	case uint16:
		for i := 0; i < df.TotalSize(); i++ {
			df.SetFlatAt(T(r.Intn(math.MaxUint16)), i)
		}
	case uint32:
		for i := 0; i < df.TotalSize(); i++ {
			df.SetFlatAt(T(r.Int31()), i)
		}
	case uint64:
		for i := 0; i < df.TotalSize(); i++ {
			df.SetFlatAt(T(r.Int63()), i)
		}
	case float32:
		for i := 0; i < df.TotalSize(); i++ {
			df.SetFlatAt(T(r.Float32()), i)
		}
	case float64:
		for i := 0; i < df.TotalSize(); i++ {
			df.SetFlatAt(T(r.Float64()), i)
		}
	}
	return df
}

func DimString(dims []int) string {
	var sb strings.Builder
	for i, dim := range dims {
		if i > 0 {
			sb.WriteByte('x')
		}
		sb.WriteString(strconv.Itoa(dim))
	}
	return sb.String()
}
