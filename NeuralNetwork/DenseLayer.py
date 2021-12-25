# DenseLayer.py

import numpy as np

from NeuralNetwork.Vector import Vector
from NeuralNetwork.Matrix import Matrix


class DenseLayer:
    """
    @brief: a dense layer of a certain number of nodes that can be used in a model
    @params: number_of_inputs -> the number of inputs coming into the dense layer
    @params: number_of_neurons -> the number of neurons within the dense layer

    ex: DenseLayer(number_of_inputs=4, number_of_neurons=3)
        results in
            |
            |                  | - the dense layer
            |                  v
            |
            |               o
            |               o  o
            |->             o  o
                            o  o
    """

    def __init__(self, number_of_inputs: int, number_of_neurons: int):
        """
        @brief: initializes random matrix with size (n_neurons x n_inputs)
        @brief: initializes random horizontal vector with size (n_neurons)
        @params: number_of_inputs -> the number of inputs coming into the dense layer
        @params: number_of_neurons -> the number of neurons within the dense layer
        """
        self.weights: list[list[float]] = Matrix.scalar_matrix_multiplication(
            0.01, np.random.randn(number_of_neurons, number_of_inputs)
        )

        temp_biases = np.zeros((1, number_of_neurons))
        self.biases: list[float] = []
        for i in range(number_of_neurons):
            self.biases.append(temp_biases[0][i])
        self.outputs: list[list[float]] = []

    def read_weights(self, file_name: str) -> None:
        """
        @brief: Overrides self.weights to the weights from a file
        @params: file_name -> the file to read the weights from
        @ret: None
        """
        with open(file_name, "r") as f:
            self.weights = []
            rows = int(f.readline())
            cols = int(f.readline())
            for i in range(rows):
                self.weights.append([])
                for j in range(cols):
                    self.weights[i].append(float(f.readline()))
            self.weights = Matrix.transpose(self.weights)

    def forward(self, inputs: list[list[float]], transpose: bool = False) -> None:
        """
        @brief: Perform the calculations required for the dense layer
        @params: inputs -> A matrix of inputs that are cross multiplied with the weights
        @params: transpose (optional, defaults to `False`) -> Transposes weights if needed
        @ret: none
        """

        if transpose:
            self.outputs = Matrix.matrix_multiplication(
                inputs, Matrix.transpose(self.weights)
            )
        else:
            self.outputs = Matrix.matrix_multiplication(inputs, self.weights)
        self.outputs = Matrix.add_vector(self.outputs, self.biases)
