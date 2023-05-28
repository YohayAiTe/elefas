//go:build !amd64

package optimization

const (
	HasDenseApplyF32 = false
)

func DenseApplyF32(input_ptr *float32, kernel_ptr *float32, bias_ptr *float32, output_ptr *float32,
	batch_count int64, input_units int64, output_units int64) {
	panic("does not have optimized dense apply for float32")
}
