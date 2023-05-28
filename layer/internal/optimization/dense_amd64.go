//go:build amd64

package optimization

const (
	HasDenseApplyF32 = true
)

//go:generate python3 -m peachpy.x86_64 -mabi=goasm -S -o dense_amd64.s dense_amd64.py
func DenseApplyF32(input_ptr *float32, kernel_ptr *float32, bias_ptr *float32, output_ptr *float32,
	batch_count int64, input_units int64, output_units int64)
