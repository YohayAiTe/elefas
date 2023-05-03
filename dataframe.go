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
	dims []int
	data []T
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
		dims: dims,
		data: make([]T, totalSize),
	}
}

func (df DataFrame[T]) DimCount() int        { return len(df.dims) }
func (df DataFrame[T]) Dim(i int) int        { return df.dims[i] }
func (df DataFrame[T]) TotalSize() int       { return len(df.data) }
func (df DataFrame[T]) FlatAt(i int) T       { return df.data[i] }
func (df DataFrame[T]) SetFlatAt(v T, i int) { df.data[i] = v }

func (df DataFrame[T]) Index(indices ...int) int {
	if len(indices) != len(df.dims) {
		panic("number of indices does not match number of dims")
	}
	for i := 0; i < len(df.dims); i++ {
		if indices[i] >= df.dims[i] {
			panic(fmt.Sprintf("index %d is too big for dimension %d which is %d", indices[i], i, df.dims[i]))
		}
	}

	idx := indices[0]
	for i := 1; i < len(df.dims); i++ {
		idx = df.dims[i]*idx + indices[i]
	}
	return idx
}

func (df DataFrame[T]) At(indices ...int) T       { return df.data[df.Index(indices...)] }
func (df DataFrame[T]) SetAt(v T, indices ...int) { df.data[df.Index(indices...)] = v }

func (df DataFrame[T]) Sub(index int) DataFrame[T] {
	size := df.TotalSize() / df.dims[0]
	return DataFrame[T]{
		dims: df.dims[1:],
		data: df.data[size*index : size*(index+1)],
	}
}
func (df DataFrame[T]) Slice(start, end int) DataFrame[T] {
	if start > end {
		panic("start must be less than end")
	}
	ndims := make([]int, len(df.dims))
	copy(ndims, df.dims)
	ndims[0] = end - start
	subSize := df.TotalSize() / df.dims[0]
	return DataFrame[T]{
		dims: ndims,
		data: df.data[subSize*start : subSize*end],
	}
}

func CastDf[T, U SizedNumber](df DataFrame[T]) DataFrame[U] {
	res := DataFrame[U]{
		dims: df.dims,
		data: make([]U, len(df.data)),
	}
	for i := 0; i < len(df.data); i++ {
		res.data[i] = U(df.data[i])
	}
	return res
}
