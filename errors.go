package elefas

import (
	"errors"
)

var (
	ErrUnsupportedType   = errors.New("unsupported type")
	ErrDifferentDataType = errors.New("different data type")
)
