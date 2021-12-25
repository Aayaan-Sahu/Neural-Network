# ActivationSoftmax.py

import math
import numpy as np
from NeuralNetwork.Vector import Vector


class ActivationSoftmax:
    """
    @brief: an implementation of the Softmax activation function
    @params: inputs -> the inputs to the activation function
    @ret: None
    """

    def forward(self, inputs: list[list[float]]) -> None:
        """
        @brief: implements the Softmax activation function for a matrix
        @params: inputs -> the inputs in the forms of a matrix
        @ret: None
        """
        # Make the inputs `safe` for the softmax function
        safe_inputs: list[list[float]] = []
        for i in range(len(inputs)):
            safe_inputs.append([])
            max_value = np.max(inputs[i])
            for j in range(len(inputs[i])):
                safe_inputs[i].append(inputs[i][j] - max_value)

        # Change all the elements to e^(element)
        exp_values: list[list[float]] = []
        for i in range(len(safe_inputs)):
            exp_values.append([])
            for j in range(len(inputs[i])):
                exp_values[i].append(math.e ** safe_inputs[i][j])

        # For each row in the matrix, the normalized base is the sum of that row
        normalized_bases: list[float] = []
        for i in range(len(exp_values)):
            normalized_bases.append(Vector.vector_sum(exp_values[i]))

        # Normalize all the values by dividing by the corresponding normalized base
        normalized_values: list[list[float]] = []
        for i in range(len(exp_values)):
            normalized_values.append([])
            for j in range(len(exp_values[i])):
                normalized_values[i].append(exp_values[i][j] / normalized_bases[i])

        self.outputs = normalized_values
