package layer

import (
	"math"

	"github.com/YohayAiTe/elefas"
)

type ReLUActivation[T elefas.SizedNumber] struct {
	MaxValue, NegativeSlope, Threshold T
}

func NewReLUActivation[T elefas.SizedNumber]() *ReLUActivation[T] {
	return &ReLUActivation[T]{
		MaxValue:      T(math.Inf(1)),
		NegativeSlope: 0,
		Threshold:     0,
	}
}

func (ra *ReLUActivation[T]) Apply(input elefas.DataFrame[T]) elefas.DataFrame[T] {
	output := elefas.MakeDataFrame[T](input.Dims)
	for i := 0; i < output.TotalSize(); i++ {
		if ra.MaxValue <= input.Data[i] {
			output.Data[i] = ra.MaxValue
		} else if ra.Threshold <= input.Data[i] {
			output.Data[i] = input.Data[i]
		} else {
			output.Data[i] = (input.Data[i] - ra.Threshold) * ra.NegativeSlope
		}
	}

	return output
}

type SigmoidActivation[T elefas.SizedNumber] struct{}

func (sa *SigmoidActivation[T]) Apply(input elefas.DataFrame[T]) elefas.DataFrame[T] {
	output := elefas.MakeDataFrame[T](input.Dims)
	for i := 0; i < output.TotalSize(); i++ {
		output.Data[i] = T(1 / (1 + math.Exp(-float64(input.Data[i]))))
	}
	return output
}

type SoftmaxActivation[T elefas.SizedNumber] struct {
	Axis int
}

func (sa *SoftmaxActivation[T]) Apply(input elefas.DataFrame[T]) elefas.DataFrame[T] {
	axis := sa.Axis
	if axis >= len(input.Dims) {
		panic("axis is greater than the number of dimensions")
	}
	if axis == -1 {
		axis = len(input.Dims) - 1
	}
	output := elefas.MakeDataFrame[T](input.Dims)

	postIdxMax := 1
	for i := len(input.Dims) - 1; i > axis; i-- {
		postIdxMax *= input.Dims[i]
	}
	preIdxDiff := postIdxMax * input.Dims[axis]

	for preIdx := 0; preIdx < input.TotalSize(); preIdx += preIdxDiff {
		for postIdx := 0; postIdx < postIdxMax; postIdx++ {
			var sum T
			for axisIdx := 0; axisIdx < input.Dims[axis]; axisIdx++ {
				idx := preIdx + postIdx + axisIdx*postIdxMax
				value := T(math.Exp(float64(input.Data[idx])))
				output.Data[idx] = value
				sum += value
			}
			for axisIdx := 0; axisIdx < input.Dims[axis]; axisIdx++ {
				idx := preIdx + postIdx + axisIdx*postIdxMax
				output.Data[idx] /= sum
			}
		}
	}
	return output
}

type SoftplusActivation[T elefas.SizedNumber] struct{}

func (sa *SoftplusActivation[T]) Apply(input elefas.DataFrame[T]) elefas.DataFrame[T] {
	output := elefas.MakeDataFrame[T](input.Dims)
	for i := 0; i < output.TotalSize(); i++ {
		output.Data[i] = T(math.Log(math.Exp(float64(input.Data[i])) + 1))
	}
	return output
}

type SoftsignActivation[T elefas.SizedNumber] struct{}

func (sa *SoftsignActivation[T]) Apply(input elefas.DataFrame[T]) elefas.DataFrame[T] {
	output := elefas.MakeDataFrame[T](input.Dims)
	for i := 0; i < output.TotalSize(); i++ {
		a := input.Data[i]
		if a < 0 {
			a = -a
		}
		output.Data[i] = a / (a + 1)
	}
	return output
}

type TanhActivation[T elefas.SizedNumber] struct{}

func (ta *TanhActivation[T]) Apply(input elefas.DataFrame[T]) elefas.DataFrame[T] {
	output := elefas.MakeDataFrame[T](input.Dims)
	for i := 0; i < output.TotalSize(); i++ {
		output.Data[i] = T(math.Tanh(float64(input.Data[i])))
	}
	return output
}

type SeluActivation[T elefas.SizedNumber] struct{}

func (sa *SeluActivation[T]) Apply(input elefas.DataFrame[T]) elefas.DataFrame[T] {
	const scale, alpha = 1.05070098, 1.67326324
	output := elefas.MakeDataFrame[T](input.Dims)
	for i := 0; i < output.TotalSize(); i++ {
		if input.Data[i] >= 0 {
			output.Data[i] = T(scale * float64(input.Data[i]))
		} else {
			output.Data[i] = T(scale * alpha * (math.Exp(float64(input.Data[i]) - 1)))
		}
	}
	return output
}

type EluActivation[T elefas.SizedNumber] struct {
	Alpha T
}

func (ea *EluActivation[T]) Apply(input elefas.DataFrame[T]) elefas.DataFrame[T] {
	output := elefas.MakeDataFrame[T](input.Dims)
	for i := 0; i < output.TotalSize(); i++ {
		if input.Data[i] >= 0 {
			output.Data[i] = input.Data[i]
		} else {
			output.Data[i] = ea.Alpha * T((math.Exp(float64(input.Data[i]) - 1)))
		}
	}
	return output
}

type ExponentialActivation[T elefas.SizedNumber] struct{}

func (ea *ExponentialActivation[T]) Apply(input elefas.DataFrame[T]) elefas.DataFrame[T] {
	output := elefas.MakeDataFrame[T](input.Dims)
	for i := 0; i < output.TotalSize(); i++ {
		output.Data[i] = T(math.Exp(float64(input.Data[i])))
	}
	return output
}

//TODO: add 'activation layers' from Keras: LeakyReLU, PReLU, and ThresholdedReLU.
