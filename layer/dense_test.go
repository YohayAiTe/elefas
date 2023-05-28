package layer_test

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/YohayAiTe/elefas"
	"github.com/YohayAiTe/elefas/layer"
	"github.com/YohayAiTe/elefas/layer/internal/testutils"
)

func denseTestFunc[T elefas.SizedNumber](t *testing.T, r *rand.Rand, dims [][3]int, epsilon T) {
	t.Parallel()
	for i := 0; i < len(dims); i++ {
		t.Run(fmt.Sprintf("(%d)x%dx%d", dims[i][0], dims[i][1], dims[i][2]), func(t *testing.T) {
			batchSize := dims[i][0]
			inputSize := dims[i][1]
			outputSize := dims[i][2]

			kernel := testutils.RandomDataFrame[T](r, []int{inputSize, outputSize})
			bias := testutils.RandomDataFrame[T](r, []int{outputSize})
			input := testutils.RandomDataFrame[T](r, []int{batchSize, inputSize})

			testutils.TestLayerByPython[T](t, testutils.PythonLayerData[T]{
				Name:    "dense",
				Weights: []elefas.DataFrame[T]{kernel, bias},
			}, layer.NewDense(kernel, bias), input, epsilon)
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

			kernel := testutils.RandomDataFrame[T](r, []int{inputSize, outputSize})
			bias := testutils.RandomDataFrame[T](r, []int{outputSize})
			input := testutils.RandomDataFrame[T](r, []int{batchSize, inputSize})

			layer := layer.NewDense(kernel, bias)

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
