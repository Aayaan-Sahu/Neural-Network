# main.py

import nnfs
from nnfs.datasets import spiral_data


from NeuralNetwork.DenseLayer import DenseLayer
from NeuralNetwork.ActivationReLU import ActivationReLU
from NeuralNetwork.ActivationSoftmax import ActivationSoftmax
from NeuralNetwork.CategoricalCrossEntropy import CategoricalCrossEntropy

nnfs.init()


def create_one_hots(y, classes):
    one_hots = []
    for i in range(len(y)):
        temp_list = []
        for j in range(classes):
            if y[i] == j:
                temp_list.append(1)
            else:
                temp_list.append(0)
        one_hots.append(temp_list)
    return one_hots


X, y = spiral_data(samples=100, classes=3)

# Layers
dense1 = DenseLayer(number_of_inputs=2, number_of_neurons=3)
activation1 = ActivationReLU()
dense2 = DenseLayer(number_of_inputs=3, number_of_neurons=3)
activation2 = ActivationSoftmax()
c = CategoricalCrossEntropy()

# Passing Through
dense1.read_weights(file_name="weights/weights1.in")
dense1.forward(inputs=X, transpose=True)
activation1.forward(dense1.outputs)
dense2.read_weights("weights/weights2.in")
dense2.forward(inputs=activation1.output, transpose=True)
activation2.forward(dense2.outputs)
batch_loss = c.calculate(inputs=activation2.outputs, y=y)

print(batch_loss)
