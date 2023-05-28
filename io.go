package elefas

import (
	"archive/zip"
	"bytes"
	"io"
	"strconv"
	"unsafe"

	"github.com/kshedden/gonpy"
)

func castAndCopy[T, U SizedNumber](dst []T, src []U) {
	for i := 0; i < len(dst); i++ {
		dst[i] = T(src[i])
	}
}

func sizedNumberToDtype[T SizedNumber]() string {
	var zero T
	var t any = zero
	switch t.(type) {
	case int8:
		return "i1"
	case int16:
		return "i2"
	case int32:
		return "i4"
	case int64:
		return "i8"
	case uint8:
		return "u1"
	case uint16:
		return "u2"
	case uint32:
		return "u4"
	case uint64:
		return "u8"
	case float32:
		return "f4"
	case float64:
		return "f8"
	default:
		panic("unknown number type")
	}
}

func LoadNumpyDataFrame[T SizedNumber](r io.Reader) (DataFrame[T], error) {
	npyr, err := gonpy.NewReader(r)
	if err != nil {
		return DataFrame[T]{}, err
	}
	if npyr.Dtype != sizedNumberToDtype[T]() {
		return DataFrame[T]{}, ErrDifferentDataType
	}

	shape := npyr.Shape
	if len(shape) == 0 {
		shape = []int{1}
	}
	df := MakeDataFrame[T](shape)

	switch npyr.Dtype {
	case "c16", "c8":
		return df, ErrUnsupportedType
	case "f8":
		data, err := npyr.GetFloat64()
		if err != nil {
			return df, err
		}
		castAndCopy(df.Data, data)
	case "f4":
		data, err := npyr.GetFloat32()
		if err != nil {
			return df, err
		}
		castAndCopy(df.Data, data)
	case "u8":
		data, err := npyr.GetUint64()
		if err != nil {
			return df, err
		}
		castAndCopy(df.Data, data)
	case "u4":
		data, err := npyr.GetUint32()
		if err != nil {
			return df, err
		}
		castAndCopy(df.Data, data)
	case "u2":
		data, err := npyr.GetUint16()
		if err != nil {
			return df, err
		}
		castAndCopy(df.Data, data)
	case "u1":
		data, err := npyr.GetUint8()
		if err != nil {
			return df, err
		}
		castAndCopy(df.Data, data)
	case "i8":
		data, err := npyr.GetInt64()
		if err != nil {
			return df, err
		}
		castAndCopy(df.Data, data)
	case "i4":
		data, err := npyr.GetInt32()
		if err != nil {
			return df, err
		}
		castAndCopy(df.Data, data)
	case "i2":
		data, err := npyr.GetInt16()
		if err != nil {
			return df, err
		}
		castAndCopy(df.Data, data)
	case "i1":
		data, err := npyr.GetInt8()
		if err != nil {
			return df, err
		}
		castAndCopy(df.Data, data)
	}
	return df, nil
}

func LoadNumpyDataFrames[T SizedNumber](r io.Reader) ([]DataFrame[T], error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}
	zr, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return nil, err
	}
	dfs := make([]DataFrame[T], len(zr.File))
	for i, fileinfo := range zr.File {
		f, err := zr.Open(fileinfo.Name)
		if err != nil {
			return dfs, err
		}
		dfs[i], err = LoadNumpyDataFrame[T](f)
		if err != nil {
			return dfs, err
		}
	}
	return dfs, nil
}

type writerCloserExtender struct {
	io.Writer
}

func (e writerCloserExtender) Close() error {
	return nil
}

func SaveNumpyDataFrame[T SizedNumber](df DataFrame[T], w io.Writer) error {
	npyw, err := gonpy.NewWriter(writerCloserExtender{w}) // no nead to actualy close
	if err != nil {
		return err
	}
	npyw.Shape = df.Dims

	// dataHeader := *(*reflect.SliceHeader)(unsafe.Pointer(&df.data))
	dataPointer := unsafe.SliceData(df.Data)
	switch sizedNumberToDtype[T]() {
	case "i1":
		// return npyw.WriteInt8(*(*[]int8)(unsafe.Pointer(&dataHeader)))
		return npyw.WriteInt8(unsafe.Slice((*int8)(unsafe.Pointer(dataPointer)), len(df.Data)))
	case "i2":
		// return npyw.WriteInt16(*(*[]int16)(unsafe.Pointer(&dataHeader)))
		return npyw.WriteInt16(unsafe.Slice((*int16)(unsafe.Pointer(dataPointer)), len(df.Data)))
	case "i4":
		// return npyw.WriteInt32(*(*[]int32)(unsafe.Pointer(&dataHeader)))
		return npyw.WriteInt32(unsafe.Slice((*int32)(unsafe.Pointer(dataPointer)), len(df.Data)))
	case "i8":
		// return npyw.WriteInt64(*(*[]int64)(unsafe.Pointer(&dataHeader)))
		return npyw.WriteInt64(unsafe.Slice((*int64)(unsafe.Pointer(dataPointer)), len(df.Data)))
	case "u1":
		// return npyw.WriteUint8(*(*[]uint8)(unsafe.Pointer(&dataHeader)))
		return npyw.WriteUint8(unsafe.Slice((*uint8)(unsafe.Pointer(dataPointer)), len(df.Data)))
	case "u2":
		// return npyw.WriteUint16(*(*[]uint16)(unsafe.Pointer(&dataHeader)))
		return npyw.WriteUint16(unsafe.Slice((*uint16)(unsafe.Pointer(dataPointer)), len(df.Data)))
	case "u4":
		// return npyw.WriteUint32(*(*[]uint32)(unsafe.Pointer(&dataHeader)))
		return npyw.WriteUint32(unsafe.Slice((*uint32)(unsafe.Pointer(dataPointer)), len(df.Data)))
	case "u8":
		// return npyw.WriteUint64(*(*[]uint64)(unsafe.Pointer(&dataHeader)))
		return npyw.WriteUint64(unsafe.Slice((*uint64)(unsafe.Pointer(dataPointer)), len(df.Data)))
	case "f4":
		// return npyw.WriteFloat32(*(*[]float32)(unsafe.Pointer(&dataHeader)))
		return npyw.WriteFloat32(unsafe.Slice((*float32)(unsafe.Pointer(dataPointer)), len(df.Data)))
	case "f8":
		// return npyw.WriteFloat64(*(*[]float64)(unsafe.Pointer(&dataHeader)))
		return npyw.WriteFloat64(unsafe.Slice((*float64)(unsafe.Pointer(dataPointer)), len(df.Data)))
	default:
		panic("unknown number type")
	}
}

func SaveNumpyDataFrames[T SizedNumber](dfs []DataFrame[T], w io.Writer) error {
	zw := zip.NewWriter(w)
	for i, df := range dfs {
		f, err := zw.Create("arr_" + strconv.Itoa(i) + ".npy")
		if err != nil {
			return err
		}
		err = SaveNumpyDataFrame(df, f)
		if err != nil {
			return err
		}
	}
	zw.Close()
	return nil
}
