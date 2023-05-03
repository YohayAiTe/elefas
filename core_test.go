package elefas_test

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"

	"github.com/YohayAiTe/elefas"
)

type pythonLayerData[T elefas.SizedNumber] struct {
	name    string
	weights []elefas.DataFrame[T]
}

func testLayerByPython[T elefas.SizedNumber](t *testing.T, pythonData pythonLayerData[T],
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
	err = elefas.SaveNumpyDataFrames(pythonData.weights, weightsFile)
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

	pythonCommand := exec.Command("python3", "testdata/keras_script.py", pythonData.name,
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

func randomDataFrame[T elefas.SizedNumber](r *rand.Rand, dims []int) elefas.DataFrame[T] {
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

func dimString(dims []int) string {
	var sb strings.Builder
	for i, dim := range dims {
		if i > 0 {
			sb.WriteByte('x')
		}
		sb.WriteString(strconv.Itoa(dim))
	}
	return sb.String()
}

func denseTestFunc[T elefas.SizedNumber](t *testing.T, r *rand.Rand, dims [][3]int, epsilon T) {
	t.Parallel()
	for i := 0; i < len(dims); i++ {
		t.Run(fmt.Sprintf("(%d)x%dx%d", dims[i][0], dims[i][1], dims[i][2]), func(t *testing.T) {
			batchSize := dims[i][0]
			inputSize := dims[i][1]
			outputSize := dims[i][2]

			kernel := randomDataFrame[T](r, []int{inputSize, outputSize})
			bias := randomDataFrame[T](r, []int{outputSize})
			input := randomDataFrame[T](r, []int{batchSize, inputSize})

			testLayerByPython[T](t, pythonLayerData[T]{
				name:    "dense",
				weights: []elefas.DataFrame[T]{kernel, bias},
			}, elefas.NewDenseLayer(kernel, bias), input, epsilon)
		})
	}
}

func TestDense(t *testing.T) {
	r := rand.New(rand.NewSource(0))
	dims := [][3]int{
		{1, 1, 1}, {1, 1, 2}, {1, 2, 1}, {1, 2, 3}, {1, 10, 3}, {1, 2, 10}, {1, 10, 10},
		{5, 1, 1}, {5, 1, 2}, {5, 2, 1}, {5, 2, 3}, {5, 10, 3}, {5, 2, 10}, {5, 10, 10},
		{1, 48, 48}, {1, 49, 49}, {16, 256, 256},
	}
	t.Run("float32", func(t *testing.T) {
		denseTestFunc[float32](t, r, dims, 1e-4)
	})
	t.Run("float64", func(t *testing.T) {
		denseTestFunc(t, r, dims, 1e-5)
	})
}

func denseBenchFunc[T elefas.SizedNumber](b *testing.B, r *rand.Rand, dims [][3]int) {
	for i := 0; i < len(dims); i++ {
		b.Run(fmt.Sprintf("(%d)x%dx%d", dims[i][0], dims[i][1], dims[i][2]), func(b *testing.B) {
			batchSize := dims[i][0]
			inputSize := dims[i][1]
			outputSize := dims[i][2]

			kernel := randomDataFrame[T](r, []int{inputSize, outputSize})
			bias := randomDataFrame[T](r, []int{outputSize})
			input := randomDataFrame[T](r, []int{batchSize, inputSize})

			layer := elefas.NewDenseLayer(kernel, bias)

			for i := 0; i < b.N; i++ {
				_ = layer.Apply(input)
			}
		})
	}
}

func BenchmarkDense(b *testing.B) {
	r := rand.New(rand.NewSource(0))
	dims := [][3]int{
		{1, 1, 1}, {1, 1, 2}, {1, 2, 1}, {1, 2, 3}, {1, 10, 3}, {1, 2, 10}, {1, 10, 10},
		{5, 1, 1}, {5, 1, 2}, {5, 2, 1}, {5, 2, 3}, {5, 10, 3}, {5, 2, 10}, {5, 10, 10},
		{16, 256, 256},
	}
	b.Run("float32", func(b *testing.B) {
		denseBenchFunc[float32](b, r, dims)
	})
	b.Run("float64", func(b *testing.B) {
		denseBenchFunc[float64](b, r, dims)
	})
}
