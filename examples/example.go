package main

import (
	"fmt"
	"log"
	"os"

	"github.com/YohayAiTe/elefas"
)

func main() {
	f, err := os.Open("test.npy")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	df, err := elefas.LoadNumpyDataFrame[float32](f)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%+v\n", df)

	f, err = os.Create("test_.npy")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	err = elefas.SaveNumpyDataFrame(elefas.CastDf[float32, int64](df), f)
	if err != nil {
		log.Fatal(err)
	}
}
