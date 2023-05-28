package elefas

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
