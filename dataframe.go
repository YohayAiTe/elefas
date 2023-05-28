package elefas

import (
	"fmt"
)

type SizedNumber interface {
	int8 | int16 | int32 | int64 |
		uint8 | uint16 | uint32 | uint64 |
		float32 | float64
}

type DataFrame[T SizedNumber] struct {
	Dims []int
	Data []T
}

func MakeDataFrame[T SizedNumber](dims []int) DataFrame[T] {
	if len(dims) == 0 {
		return DataFrame[T]{}
	}
	totalSize := 1
	for i := 0; i < len(dims); i++ {
		totalSize *= dims[i]
	}
	return DataFrame[T]{
		Dims: dims,
		Data: make([]T, totalSize),
	}
}

func (df DataFrame[T]) DimCount() int        { return len(df.Dims) }
func (df DataFrame[T]) Dim(i int) int        { return df.Dims[i] }
func (df DataFrame[T]) TotalSize() int       { return len(df.Data) }
func (df DataFrame[T]) FlatAt(i int) T       { return df.Data[i] }
func (df DataFrame[T]) SetFlatAt(v T, i int) { df.Data[i] = v }

func (df DataFrame[T]) Index(indices ...int) int {
	if len(indices) != len(df.Dims) {
		panic("number of indices does not match number of dims")
	}
	for i := 0; i < len(df.Dims); i++ {
		if indices[i] >= df.Dims[i] {
			panic(fmt.Sprintf("index %d is too big for dimension %d which is %d", indices[i], i, df.Dims[i]))
		}
	}

	idx := indices[0]
	for i := 1; i < len(df.Dims); i++ {
		idx = df.Dims[i]*idx + indices[i]
	}
	return idx
}

func (df DataFrame[T]) At(indices ...int) T       { return df.Data[df.Index(indices...)] }
func (df DataFrame[T]) SetAt(v T, indices ...int) { df.Data[df.Index(indices...)] = v }

func (df DataFrame[T]) Sub(index int) DataFrame[T] {
	size := df.TotalSize() / df.Dims[0]
	return DataFrame[T]{
		Dims: df.Dims[1:],
		Data: df.Data[size*index : size*(index+1)],
	}
}
func (df DataFrame[T]) Slice(start, end int) DataFrame[T] {
	if start > end {
		panic("start must be less than end")
	}
	ndims := make([]int, len(df.Dims))
	copy(ndims, df.Dims)
	ndims[0] = end - start
	subSize := df.TotalSize() / df.Dims[0]
	return DataFrame[T]{
		Dims: ndims,
		Data: df.Data[subSize*start : subSize*end],
	}
}

func CastDf[T, U SizedNumber](df DataFrame[T]) DataFrame[U] {
	res := DataFrame[U]{
		Dims: df.Dims,
		Data: make([]U, len(df.Data)),
	}
	for i := 0; i < len(df.Data); i++ {
		res.Data[i] = U(df.Data[i])
	}
	return res
}
