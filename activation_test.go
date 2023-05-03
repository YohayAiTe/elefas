package elefas_test

import (
	"math"
	"math/rand"
	"strconv"
	"testing"

	"github.com/YohayAiTe/elefas"
)

type reluTestCase struct {
	dims                               []int
	MaxValue, NegativeSlope, Threshold float64
}

func reluTestFunc[T elefas.SizedNumber](t *testing.T, r *rand.Rand, testCases []reluTestCase, epsilon T) {
	for _, testCase := range testCases {
		t.Run(dimString(testCase.dims), func(t *testing.T) {
			input := randomDataFrame[T](r, testCase.dims)

			data := []elefas.DataFrame[T]{elefas.MakeDataFrame[T]([]int{3})}
			data[0].SetAt(T(testCase.MaxValue), 0)
			data[0].SetAt(T(testCase.NegativeSlope), 1)
			data[0].SetAt(T(testCase.Threshold), 2)

			testLayerByPython[T](t, pythonLayerData[T]{
				name:    "relu",
				weights: data,
			}, &elefas.ReLUActivation[T]{
				MaxValue:      T(testCase.MaxValue),
				NegativeSlope: T(testCase.NegativeSlope),
				Threshold:     T(testCase.Threshold),
			}, input, epsilon)
		})
	}
}

func TestReLU(t *testing.T) {
	t.Parallel()
	r := rand.New(rand.NewSource(0))
	testcases := []reluTestCase{
		{[]int{5}, math.Inf(1), 0, 0},
		{[]int{2, 3}, math.Inf(1), 2, 10},
		{[]int{10, 2, 5}, math.Inf(1), 1, 2},
	}
	t.Run("float32", func(t *testing.T) {
		reluTestFunc[float32](t, r, testcases, 1e-5)
	})
	t.Run("float64", func(t *testing.T) {
		reluTestFunc(t, r, testcases, 1e-5)
	})
}

func reluBenchFunc[T elefas.SizedNumber](b *testing.B, r *rand.Rand, testCases []reluTestCase) {
	for _, testCase := range testCases {
		b.Run(dimString(testCase.dims), func(b *testing.B) {
			input := randomDataFrame[T](r, testCase.dims)
			activation := &elefas.ReLUActivation[T]{
				MaxValue:      T(testCase.MaxValue),
				NegativeSlope: T(testCase.NegativeSlope),
				Threshold:     T(testCase.Threshold),
			}
			for i := 0; i < b.N; i++ {
				_ = activation.Apply(input)
			}
		})
	}
}

func BenchmarkReLU(b *testing.B) {
	r := rand.New(rand.NewSource(0))
	testcases := []reluTestCase{
		{[]int{5}, math.Inf(1), 0, 0},
		{[]int{2, 3}, math.Inf(1), 2, 10},
		{[]int{10, 2, 5}, math.Inf(1), 1, 2},
	}
	b.Run("float32", func(b *testing.B) {
		reluBenchFunc[float32](b, r, testcases)
	})
	b.Run("float64", func(b *testing.B) {
		reluBenchFunc[float64](b, r, testcases)
	})
}

func simpleActivationTestFunc[T elefas.SizedNumber](t *testing.T, r *rand.Rand, name string, layer elefas.Layer[T],
	testCases [][]int, epsilon T) {

	for _, testCase := range testCases {
		t.Run(dimString(testCase), func(t *testing.T) {
			input := randomDataFrame[T](r, testCase)

			testLayerByPython(t, pythonLayerData[T]{
				name: name,
			}, layer, input, epsilon)
		})
	}
}

func simpleActivationBenchFunc[T elefas.SizedNumber](b *testing.B, r *rand.Rand, layer elefas.Layer[T], testCases [][]int) {
	for _, testCase := range testCases {
		b.Run(dimString(testCase), func(b *testing.B) {
			input := randomDataFrame[T](r, testCase)

			for i := 0; i < b.N; i++ {
				_ = layer.Apply(input)
			}
		})
	}
}

func TestSigmoid(t *testing.T) {
	t.Parallel()
	r := rand.New(rand.NewSource(0))
	testcases := [][]int{
		{5},
		{2, 3},
		{10, 2, 5},
	}
	t.Run("float32", func(t *testing.T) {
		simpleActivationTestFunc[float32](t, r, "sigmoid", &elefas.SigmoidActivation[float32]{}, testcases, 1e-5)
	})
	t.Run("float64", func(t *testing.T) {
		simpleActivationTestFunc[float64](t, r, "sigmoid", &elefas.SigmoidActivation[float64]{}, testcases, 1e-5)
	})
}

func BenchmarkSigmoid(b *testing.B) {
	r := rand.New(rand.NewSource(0))
	testcases := [][]int{
		{5},
		{2, 3},
		{10, 2, 5},
	}
	b.Run("float32", func(b *testing.B) {
		simpleActivationBenchFunc[float32](b, r, &elefas.SigmoidActivation[float32]{}, testcases)
	})
	b.Run("float64", func(b *testing.B) {
		simpleActivationBenchFunc[float64](b, r, &elefas.SigmoidActivation[float64]{}, testcases)
	})
}

type softmaxTestCase struct {
	dims []int
	axis int
}

func softmaxTestFunc[T elefas.SizedNumber](t *testing.T, r *rand.Rand, testCases []softmaxTestCase, epsilon T) {
	for _, testCase := range testCases {
		t.Run(dimString(testCase.dims)+"_"+strconv.Itoa(testCase.axis), func(t *testing.T) {
			input := randomDataFrame[T](r, testCase.dims)

			data := []elefas.DataFrame[T]{elefas.MakeDataFrame[T]([]int{1})}
			data[0].SetAt(T(testCase.axis), 0)

			testLayerByPython[T](t, pythonLayerData[T]{
				name:    "softmax",
				weights: data,
			}, &elefas.SoftmaxActivation[T]{Axis: testCase.axis}, input, epsilon)
		})
	}
}

func TestSoftmax(t *testing.T) {
	t.Parallel()
	r := rand.New(rand.NewSource(0))
	testcases := []softmaxTestCase{
		{[]int{5}, -1},
		{[]int{5}, 0},
		{[]int{2, 3}, -1},
		{[]int{2, 3}, 0},
		{[]int{2, 3}, 1},
		{[]int{10, 2, 5}, -1},
		{[]int{10, 2, 5}, 0},
		{[]int{10, 2, 5}, 1},
		{[]int{10, 2, 5}, 2},
	}
	t.Run("float32", func(t *testing.T) {
		softmaxTestFunc[float32](t, r, testcases, 1e-5)
	})
	t.Run("float64", func(t *testing.T) {
		softmaxTestFunc(t, r, testcases, 1e-5)
	})
}

func softmaxBenchFunc[T elefas.SizedNumber](b *testing.B, r *rand.Rand, testCases []softmaxTestCase) {
	for _, testCase := range testCases {
		b.Run(dimString(testCase.dims)+"_"+strconv.Itoa(testCase.axis), func(b *testing.B) {
			input := randomDataFrame[T](r, testCase.dims)
			activation := &elefas.SoftmaxActivation[T]{Axis: testCase.axis}
			for i := 0; i < b.N; i++ {
				_ = activation.Apply(input)
			}
		})
	}
}

func BenchmarkSoftmax(b *testing.B) {
	r := rand.New(rand.NewSource(0))
	testcases := []softmaxTestCase{
		{[]int{5}, -1},
		{[]int{5}, 0},
		{[]int{2, 3}, -1},
		{[]int{2, 3}, 0},
		{[]int{2, 3}, 1},
		{[]int{10, 2, 5}, -1},
		{[]int{10, 2, 5}, 0},
		{[]int{10, 2, 5}, 1},
		{[]int{10, 2, 5}, 2},
		{[]int{4, 4, 4, 4, 4}, -1},
		{[]int{4, 4, 4, 4, 4}, 2},
		{[]int{4, 4, 4, 4, 4}, 0},
	}
	b.Run("float32", func(b *testing.B) {
		softmaxBenchFunc[float32](b, r, testcases)
	})
	b.Run("float64", func(b *testing.B) {
		softmaxBenchFunc[float64](b, r, testcases)
	})
}

func TestSoftplus(t *testing.T) {
	t.Parallel()
	r := rand.New(rand.NewSource(0))
	testcases := [][]int{
		{5},
		{2, 3},
		{10, 2, 5},
	}
	t.Run("float32", func(t *testing.T) {
		simpleActivationTestFunc[float32](t, r, "softplus", &elefas.SoftplusActivation[float32]{}, testcases, 1e-5)
	})
	t.Run("float64", func(t *testing.T) {
		simpleActivationTestFunc[float64](t, r, "softplus", &elefas.SoftplusActivation[float64]{}, testcases, 1e-5)
	})
}

func BenchmarkSoftplus(b *testing.B) {
	r := rand.New(rand.NewSource(0))
	testcases := [][]int{
		{5},
		{2, 3},
		{10, 2, 5},
	}
	b.Run("float32", func(b *testing.B) {
		simpleActivationBenchFunc[float32](b, r, &elefas.SoftplusActivation[float32]{}, testcases)
	})
	b.Run("float64", func(b *testing.B) {
		simpleActivationBenchFunc[float64](b, r, &elefas.SoftplusActivation[float64]{}, testcases)
	})
}

func TestSoftsign(t *testing.T) {
	t.Parallel()
	r := rand.New(rand.NewSource(0))
	testcases := [][]int{
		{5},
		{2, 3},
		{10, 2, 5},
	}
	t.Run("float32", func(t *testing.T) {
		simpleActivationTestFunc[float32](t, r, "softsign", &elefas.SoftsignActivation[float32]{}, testcases, 1e-5)
	})
	t.Run("float64", func(t *testing.T) {
		simpleActivationTestFunc[float64](t, r, "softsign", &elefas.SoftsignActivation[float64]{}, testcases, 1e-5)
	})
}

func BenchmarkSoftsign(b *testing.B) {
	r := rand.New(rand.NewSource(0))
	testcases := [][]int{
		{5},
		{2, 3},
		{10, 2, 5},
	}
	b.Run("float32", func(b *testing.B) {
		simpleActivationBenchFunc[float32](b, r, &elefas.SoftsignActivation[float32]{}, testcases)
	})
	b.Run("float64", func(b *testing.B) {
		simpleActivationBenchFunc[float64](b, r, &elefas.SoftsignActivation[float64]{}, testcases)
	})
}

func TestTanh(t *testing.T) {
	t.Parallel()
	r := rand.New(rand.NewSource(0))
	testcases := [][]int{
		{5},
		{2, 3},
		{10, 2, 5},
	}
	t.Run("float32", func(t *testing.T) {
		simpleActivationTestFunc[float32](t, r, "tanh", &elefas.TanhActivation[float32]{}, testcases, 1e-5)
	})
	t.Run("float64", func(t *testing.T) {
		simpleActivationTestFunc[float64](t, r, "tanh", &elefas.TanhActivation[float64]{}, testcases, 1e-5)
	})
}

func BenchmarkTanh(b *testing.B) {
	r := rand.New(rand.NewSource(0))
	testcases := [][]int{
		{5},
		{2, 3},
		{10, 2, 5},
	}
	b.Run("float32", func(b *testing.B) {
		simpleActivationBenchFunc[float32](b, r, &elefas.TanhActivation[float32]{}, testcases)
	})
	b.Run("float64", func(b *testing.B) {
		simpleActivationBenchFunc[float64](b, r, &elefas.TanhActivation[float64]{}, testcases)
	})
}

func TestSelu(t *testing.T) {
	t.Parallel()
	r := rand.New(rand.NewSource(0))
	testcases := [][]int{
		{5},
		{2, 3},
		{10, 2, 5},
	}
	t.Run("float32", func(t *testing.T) {
		simpleActivationTestFunc[float32](t, r, "selu", &elefas.SeluActivation[float32]{}, testcases, 1e-5)
	})
	t.Run("float64", func(t *testing.T) {
		simpleActivationTestFunc[float64](t, r, "selu", &elefas.SeluActivation[float64]{}, testcases, 1e-5)
	})
}

func BenchmarkSelu(b *testing.B) {
	r := rand.New(rand.NewSource(0))
	testcases := [][]int{
		{5},
		{2, 3},
		{10, 2, 5},
	}
	b.Run("float32", func(b *testing.B) {
		simpleActivationBenchFunc[float32](b, r, &elefas.SeluActivation[float32]{}, testcases)
	})
	b.Run("float64", func(b *testing.B) {
		simpleActivationBenchFunc[float64](b, r, &elefas.SeluActivation[float64]{}, testcases)
	})
}

type eluTestCase struct {
	dims  []int
	alpha float64
}

func eluTestFunc[T elefas.SizedNumber](t *testing.T, r *rand.Rand, testCases []eluTestCase, epsilon T) {
	for _, testCase := range testCases {
		t.Run(dimString(testCase.dims)+"_"+strconv.FormatFloat(float64(testCase.alpha), 'f', 2, 32), func(t *testing.T) {
			input := randomDataFrame[T](r, testCase.dims)

			data := []elefas.DataFrame[T]{elefas.MakeDataFrame[T]([]int{1})}
			data[0].SetAt(T(testCase.alpha), 0)

			testLayerByPython[T](t, pythonLayerData[T]{
				name:    "elu",
				weights: data,
			}, &elefas.EluActivation[T]{Alpha: T(testCase.alpha)}, input, epsilon)
		})
	}
}

func TestElu(t *testing.T) {
	t.Parallel()
	r := rand.New(rand.NewSource(0))
	testcases := []eluTestCase{
		{[]int{5}, 1},
		{[]int{2, 3}, 0.5},
		{[]int{10, 2, 5}, 1.5},
	}
	t.Run("float32", func(t *testing.T) {
		eluTestFunc[float32](t, r, testcases, 1e-5)
	})
	t.Run("float64", func(t *testing.T) {
		eluTestFunc(t, r, testcases, 1e-5)
	})
}

func eluBenchFunc[T elefas.SizedNumber](b *testing.B, r *rand.Rand, testCases []eluTestCase) {
	for _, testCase := range testCases {
		b.Run(dimString(testCase.dims)+"_"+strconv.FormatFloat(float64(testCase.alpha), 'f', 2, 32), func(b *testing.B) {
			input := randomDataFrame[T](r, testCase.dims)
			activation := &elefas.EluActivation[T]{Alpha: T(testCase.alpha)}
			for i := 0; i < b.N; i++ {
				_ = activation.Apply(input)
			}
		})
	}
}

func BenchmarkElu(b *testing.B) {
	r := rand.New(rand.NewSource(0))
	testcases := []eluTestCase{
		{[]int{5}, 1},
		{[]int{2, 3}, 0.5},
		{[]int{10, 2, 5}, 1.5},
	}
	b.Run("float32", func(b *testing.B) {
		eluBenchFunc[float32](b, r, testcases)
	})
	b.Run("float64", func(b *testing.B) {
		eluBenchFunc[float64](b, r, testcases)
	})
}

func TestExponential(t *testing.T) {
	t.Parallel()
	r := rand.New(rand.NewSource(0))
	testcases := [][]int{
		{5},
		{2, 3},
		{10, 2, 5},
	}
	t.Run("float32", func(t *testing.T) {
		simpleActivationTestFunc[float32](t, r, "exponential", &elefas.ExponentialActivation[float32]{}, testcases, 1e-5)
	})
	t.Run("float64", func(t *testing.T) {
		simpleActivationTestFunc[float64](t, r, "exponential", &elefas.ExponentialActivation[float64]{}, testcases, 1e-5)
	})
}

func BenchmarkExponential(b *testing.B) {
	r := rand.New(rand.NewSource(0))
	testcases := [][]int{
		{5},
		{2, 3},
		{10, 2, 5},
	}
	b.Run("float32", func(b *testing.B) {
		simpleActivationBenchFunc[float32](b, r, &elefas.ExponentialActivation[float32]{}, testcases)
	})
	b.Run("float64", func(b *testing.B) {
		simpleActivationBenchFunc[float64](b, r, &elefas.ExponentialActivation[float64]{}, testcases)
	})
}
