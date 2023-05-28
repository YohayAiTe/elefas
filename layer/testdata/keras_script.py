from tensorflow.keras.layers import Input, Dense, ReLU, Activation, Softmax, ELU
from tensorflow.keras.models import Sequential
import numpy as np
import sys

if __name__ == "__main__":
    layer_name = sys.argv[1]
    layer_weights_file = sys.argv[2]
    input_file = sys.argv[3]
    output_file = sys.argv[4]
    print(layer_name, layer_weights_file, input_file, output_file)

    weights = np.load(layer_weights_file)
    input = np.load(input_file)

    model = Sequential()
    model.add(Input(shape=input.shape, dtype=input.dtype))

    if layer_name == "dense":
        model.add(Dense(weights['arr_0'].shape[1], dtype=input.dtype))
        model.set_weights([weights['arr_0'], weights['arr_1']])
    elif layer_name == "relu":
        model.add(ReLU(weights['arr_0'][0], weights['arr_0'][1], weights['arr_0'][2], dtype=input.dtype))
    elif layer_name == "sigmoid":
        model.add(Activation("sigmoid", dtype=input.dtype))
    elif layer_name == "softmax":
        axis = int(weights['arr_0'][0])
        model.add(Softmax(-1 if axis == -1 else axis+1, dtype=input.dtype))  # increase axis to counter the batch
    elif layer_name == "softplus":
        model.add(Activation("softplus", dtype=input.dtype))
    elif layer_name == "softsign":
        model.add(Activation("softsign", dtype=input.dtype))
    elif layer_name == "tanh":
        model.add(Activation("tanh", dtype=input.dtype))
    elif layer_name == "selu":
        model.add(Activation("selu", dtype=input.dtype))
    elif layer_name == "elu":
        model.add(ELU(weights['arr_0'][0], dtype=input.dtype))
    elif layer_name == "exponential":
        model.add(Activation("exponential", dtype=input.dtype))
    else:
        print("unknown layer name:", layer_name, file=sys.stderr)
        exit(-1)
    
    output = model.predict(input.reshape((1,) + input.shape))
    np.save(output_file, output[0])
