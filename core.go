package elefas

import "unsafe"

type DenseLayer[T SizedNumber] struct {
	inputUnits, outputUnits int

	kernel DataFrame[T]
	bias   DataFrame[T]
}

func NewDenseLayer[T SizedNumber](kernel, bias DataFrame[T]) DenseLayer[T] {
	if len(kernel.dims) != 2 {
		panic("kernel must have 2 dimensions")
	}
	if len(bias.dims) != 1 {
		panic("kernel must have 1 dimension")
	}
	if kernel.dims[1] != bias.dims[0] {
		panic("the dimensions of kernel and bias do not match")
	}

	// transpose kernel
	transposedKernel := MakeDataFrame[T]([]int{kernel.dims[1], kernel.dims[0]})
	for i := 0; i < kernel.dims[0]; i++ {
		for j := 0; j < kernel.dims[1]; j++ {
			transposedKernel.SetAt(kernel.At(i, j), j, i)
		}
	}

	return DenseLayer[T]{
		inputUnits:  kernel.dims[0],
		outputUnits: kernel.dims[1],
		kernel:      transposedKernel, bias: bias,
	}
}

func (dl DenseLayer[T]) Apply(input DataFrame[T]) DataFrame[T] {
	if len(input.dims) < 1 {
		panic("dense layer's input must have at least one dimension")
	}
	if dl.inputUnits != input.dims[len(input.dims)-1] {
		panic("dense layer's input does not match the specified input units")
	}
	outputDims := make([]int, len(input.dims))
	copy(outputDims, input.dims)
	outputDims[len(outputDims)-1] = dl.outputUnits
	output := MakeDataFrame[T](outputDims)

	batchCount := output.TotalSize() / dl.outputUnits
	if sizedNumberToDtype[T]() == "f4" && hasDenseApplyOptimizedF32 {
		denseApplyOptimizedF32(
			(*float32)(unsafe.Pointer(unsafe.SliceData(input.data))),
			(*float32)(unsafe.Pointer(unsafe.SliceData(dl.kernel.data))),
			(*float32)(unsafe.Pointer(unsafe.SliceData(dl.bias.data))),
			(*float32)(unsafe.Pointer(unsafe.SliceData(output.data))),
			int64(batchCount), int64(dl.inputUnits), int64(dl.outputUnits))
		return output
	}

	var acc T
	for batch := 0; batch < batchCount; batch++ {
		kernelIndex := 0
		for j := 0; j < dl.outputUnits; j++ {
			outputIndex := batch*dl.outputUnits + j
			acc = dl.bias.data[j]

			for i := 0; i < dl.inputUnits; i++ {
				acc += input.data[batch*dl.inputUnits+i] * dl.kernel.data[kernelIndex] // kernelIndex = i*dl.outputUnits+j
				kernelIndex++
			}
			output.data[outputIndex] = acc
		}
	}

	return output
}

type Layer[T SizedNumber] interface {
	Apply(input DataFrame[T]) DataFrame[T]
}

type (
	LayerData[T SizedNumber] struct {
		model *Model[T]

		layer   Layer[T]
		input   *LayerData[T]
		outputs []*LayerData[T]
	}

	outputLayer[T SizedNumber] struct{ df DataFrame[T] }

	Model[T SizedNumber] struct {
		input      *LayerData[T]
		layersData []*LayerData[T]
		outputs    []*LayerData[T]
	}
)

func (ol *outputLayer[T]) Apply(input DataFrame[T]) DataFrame[T] {
	ol.df = input
	return DataFrame[T]{}
}

func NewModel[T SizedNumber](outputs int) *Model[T] {
	return &Model[T]{
		input:   &LayerData[T]{layer: nil, input: nil, outputs: nil},
		outputs: make([]*LayerData[T], outputs),
	}
}

func (m *Model[T]) AddLayer(layer Layer[T], input *LayerData[T]) *LayerData[T] {
	if input == nil {
		input = m.input
	}
	current := &LayerData[T]{
		model: m,
		layer: layer,
		input: input,
	}
	m.layersData = append(m.layersData, current)
	input.outputs = append(input.outputs, current)
	return current
}

func (d *LayerData[T]) AddLayer(layer Layer[T]) *LayerData[T] {
	return d.model.AddLayer(layer, d)
}

func (m *Model[T]) SetOutput(input *LayerData[T], index int) {
	m.outputs[index] = &LayerData[T]{
		layer: &outputLayer[T]{},
		input: input,
	}
	input.outputs = append(input.outputs, m.outputs[index])
}

func (ld *LayerData[T]) runLayer(input DataFrame[T]) {
	output := ld.layer.Apply(input)
	for _, l := range ld.outputs {
		l.runLayer(output)
	}
}

func (m *Model[T]) Predict(input DataFrame[T]) []DataFrame[T] {
	for _, l := range m.input.outputs {
		l.runLayer(input)
	}

	outputs := make([]DataFrame[T], len(m.outputs))
	for i := 0; i < len(outputs); i++ {
		outputs[i] = (m.outputs[i].layer).(*outputLayer[T]).df
	}
	return outputs
}
