# Loss.py

from NeuralNetwork.Vector import Vector


class Loss:
    """
    @brief: a base loss class for calculating the loss of a batch of data
    """
    def calculate(self, inputs: list[list[float]], y) -> float:
        """
        @brief: a base loss class for calculating the loss of a batch of data
        @params: inputs -> a matrix holding probability distributions
        @params: y -> in the form of sparse or one-hots containing the correct class
        @ret: data_loss -> the loss for a batch of data
        """
        sample_losses = self.forward(inputs, y)  # Difference for every loss function
        data_loss = Vector.mean(sample_losses)
        return data_loss
