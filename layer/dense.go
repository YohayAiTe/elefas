package layer

import (
	"unsafe"

	"github.com/YohayAiTe/elefas"
	"github.com/YohayAiTe/elefas/layer/internal/optimization"
)

type Dense[T elefas.SizedNumber] struct {
	inputUnits, outputUnits int

	kernel elefas.DataFrame[T]
	bias   elefas.DataFrame[T]
}

func NewDense[T elefas.SizedNumber](kernel, bias elefas.DataFrame[T]) Dense[T] {
	if len(kernel.Dims) != 2 {
		panic("kernel must have 2 dimensions")
	}
	if len(bias.Dims) != 1 {
		panic("kernel must have 1 dimension")
	}
	if kernel.Dims[1] != bias.Dims[0] {
		panic("the dimensions of kernel and bias do not match")
	}

	// transpose kernel
	transposedKernel := elefas.MakeDataFrame[T]([]int{kernel.Dims[1], kernel.Dims[0]})
	for i := 0; i < kernel.Dims[0]; i++ {
		for j := 0; j < kernel.Dims[1]; j++ {
			transposedKernel.SetAt(kernel.At(i, j), j, i)
		}
	}

	return Dense[T]{
		inputUnits:  kernel.Dims[0],
		outputUnits: kernel.Dims[1],
		kernel:      transposedKernel, bias: bias,
	}
}

func (d Dense[T]) Apply(input elefas.DataFrame[T]) elefas.DataFrame[T] {
	if len(input.Dims) < 1 {
		panic("dense layer's input must have at least one dimension")
	}
	if d.inputUnits != input.Dims[len(input.Dims)-1] {
		panic("dense layer's input does not match the specified input units")
	}
	outputDims := make([]int, len(input.Dims))
	copy(outputDims, input.Dims)
	outputDims[len(outputDims)-1] = d.outputUnits
	output := elefas.MakeDataFrame[T](outputDims)

	batchCount := output.TotalSize() / d.outputUnits
	if _, isF32 := any(T(0)).(float32); isF32 && optimization.HasDenseApplyF32 {
		optimization.DenseApplyF32(
			(*float32)(unsafe.Pointer(unsafe.SliceData(input.Data))),
			(*float32)(unsafe.Pointer(unsafe.SliceData(d.kernel.Data))),
			(*float32)(unsafe.Pointer(unsafe.SliceData(d.bias.Data))),
			(*float32)(unsafe.Pointer(unsafe.SliceData(output.Data))),
			int64(batchCount), int64(d.inputUnits), int64(d.outputUnits))
		return output
	}

	var acc T
	for batch := 0; batch < batchCount; batch++ {
		kernelIndex := 0
		for j := 0; j < d.outputUnits; j++ {
			outputIndex := batch*d.outputUnits + j
			acc = d.bias.Data[j]

			for i := 0; i < d.inputUnits; i++ {
				acc += input.Data[batch*d.inputUnits+i] * d.kernel.Data[kernelIndex] // kernelIndex = i*dl.outputUnits+j
				kernelIndex++
			}
			output.Data[outputIndex] = acc
		}
	}

	return output
}
