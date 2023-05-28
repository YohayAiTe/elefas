package layer

import "github.com/YohayAiTe/elefas"

type Flatten[T elefas.SizedNumber] struct{}

func (f Flatten[T]) Apply(input elefas.DataFrame[T]) elefas.DataFrame[T] {
	if input.DimCount() == 0 {
		panic("cannot flatten dataframe with no dimensions")
	} else if input.DimCount() == 1 {
		output := elefas.MakeDataFrame[T]([]int{input.Dim(0), 1})
		copy(output.Data, input.Data)
		return output
	}
	output := elefas.MakeDataFrame[T]([]int{input.Dim(0), input.TotalSize() / input.Dim(0)})
	copy(output.Data, input.Data)
	return output
}
