# ActivationReLU.py


class ActivationReLU:
    """
    @brief: an implementation of the ReLU activation function
    @params: inputs -> the inputs to the activation function
    @ret: None
    """

    def forward(self, inputs: list[list[float]]) -> None:
        """
        @brief: implements the ReLU activation function for a matrix
        @params: inputs -> the inputs in the forms of a matrix
        @ret: None
        """
        self.output: list[list[float]] = inputs  # Copy the inputs to outputs
        for i in range(len(inputs)):
            for j in range(len(inputs[i])):
                # Keep the element the same if it is greater than 0
                if self.output[i][j] > 0:
                    continue
                # If not, then the element is 0
                else:
                    self.output[i][j] = 0
