package elefas

import "math"

type ReLUActivation[T SizedNumber] struct {
	MaxValue, NegativeSlope, Threshold T
}

func NewReLUActivation[T SizedNumber]() *ReLUActivation[T] {
	return &ReLUActivation[T]{
		MaxValue:      T(math.Inf(1)),
		NegativeSlope: 0,
		Threshold:     0,
	}
}

func (ra *ReLUActivation[T]) Apply(input DataFrame[T]) DataFrame[T] {
	output := MakeDataFrame[T](input.dims)
	for i := 0; i < output.TotalSize(); i++ {
		if ra.MaxValue <= input.data[i] {
			output.data[i] = ra.MaxValue
		} else if ra.Threshold <= input.data[i] {
			output.data[i] = input.data[i]
		} else {
			output.data[i] = (input.data[i] - ra.Threshold) * ra.NegativeSlope
		}
	}

	return output
}

type SigmoidActivation[T SizedNumber] struct{}

func (sa *SigmoidActivation[T]) Apply(input DataFrame[T]) DataFrame[T] {
	output := MakeDataFrame[T](input.dims)
	for i := 0; i < output.TotalSize(); i++ {
		output.data[i] = T(1 / (1 + math.Exp(-float64(input.data[i]))))
	}
	return output
}

type SoftmaxActivation[T SizedNumber] struct {
	Axis int
}

func (sa *SoftmaxActivation[T]) Apply(input DataFrame[T]) DataFrame[T] {
	axis := sa.Axis
	if axis >= len(input.dims) {
		panic("axis is greater than the number of dimensions")
	}
	if axis == -1 {
		axis = len(input.dims) - 1
	}
	output := MakeDataFrame[T](input.dims)

	postIdxMax := 1
	for i := len(input.dims) - 1; i > axis; i-- {
		postIdxMax *= input.dims[i]
	}
	preIdxDiff := postIdxMax * input.dims[axis]

	for preIdx := 0; preIdx < input.TotalSize(); preIdx += preIdxDiff {
		for postIdx := 0; postIdx < postIdxMax; postIdx++ {
			var sum T
			for axisIdx := 0; axisIdx < input.dims[axis]; axisIdx++ {
				idx := preIdx + postIdx + axisIdx*postIdxMax
				value := T(math.Exp(float64(input.data[idx])))
				output.data[idx] = value
				sum += value
			}
			for axisIdx := 0; axisIdx < input.dims[axis]; axisIdx++ {
				idx := preIdx + postIdx + axisIdx*postIdxMax
				output.data[idx] /= sum
			}
		}
	}
	return output
}

type SoftplusActivation[T SizedNumber] struct{}

func (sa *SoftplusActivation[T]) Apply(input DataFrame[T]) DataFrame[T] {
	output := MakeDataFrame[T](input.dims)
	for i := 0; i < output.TotalSize(); i++ {
		output.data[i] = T(math.Log(math.Exp(float64(input.data[i])) + 1))
	}
	return output
}

type SoftsignActivation[T SizedNumber] struct{}

func (sa *SoftsignActivation[T]) Apply(input DataFrame[T]) DataFrame[T] {
	output := MakeDataFrame[T](input.dims)
	for i := 0; i < output.TotalSize(); i++ {
		a := input.data[i]
		if a < 0 {
			a = -a
		}
		output.data[i] = a / (a + 1)
	}
	return output
}

type TanhActivation[T SizedNumber] struct{}

func (ta *TanhActivation[T]) Apply(input DataFrame[T]) DataFrame[T] {
	output := MakeDataFrame[T](input.dims)
	for i := 0; i < output.TotalSize(); i++ {
		output.data[i] = T(math.Tanh(float64(input.data[i])))
	}
	return output
}

type SeluActivation[T SizedNumber] struct{}

func (sa *SeluActivation[T]) Apply(input DataFrame[T]) DataFrame[T] {
	const scale, alpha = 1.05070098, 1.67326324
	output := MakeDataFrame[T](input.dims)
	for i := 0; i < output.TotalSize(); i++ {
		if input.data[i] >= 0 {
			output.data[i] = T(scale * float64(input.data[i]))
		} else {
			output.data[i] = T(scale * alpha * (math.Exp(float64(input.data[i]) - 1)))
		}
	}
	return output
}

type EluActivation[T SizedNumber] struct {
	Alpha T
}

func (ea *EluActivation[T]) Apply(input DataFrame[T]) DataFrame[T] {
	output := MakeDataFrame[T](input.dims)
	for i := 0; i < output.TotalSize(); i++ {
		if input.data[i] >= 0 {
			output.data[i] = input.data[i]
		} else {
			output.data[i] = ea.Alpha * T((math.Exp(float64(input.data[i]) - 1)))
		}
	}
	return output
}

type ExponentialActivation[T SizedNumber] struct{}

func (ea *ExponentialActivation[T]) Apply(input DataFrame[T]) DataFrame[T] {
	output := MakeDataFrame[T](input.dims)
	for i := 0; i < output.TotalSize(); i++ {
		output.data[i] = T(math.Exp(float64(input.data[i])))
	}
	return output
}

//TODO: add 'activation layers' from Keras: LeakyReLU, PReLU, and ThresholdedReLU.
