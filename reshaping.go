package elefas

type FlattenLayer[T SizedNumber] struct{}

func (fl FlattenLayer[T]) Apply(input DataFrame[T]) DataFrame[T] {
	if input.DimCount() == 0 {
		panic("cannot flatten dataframe with no dimensions")
	} else if input.DimCount() == 1 {
		output := MakeDataFrame[T]([]int{input.Dim(0), 1})
		copy(output.Data, input.Data)
		return output
	}
	output := MakeDataFrame[T]([]int{input.Dim(0), input.TotalSize() / input.Dim(0)})
	copy(output.Data, input.Data)
	return output
}
