# CategoricalCrossEntropy.py

import math
from NeuralNetwork.Loss import Loss
from NeuralNetwork.Matrix import Matrix


class CategoricalCrossEntropy(Loss):
    """
    @brief: a derivative of the Loss base class that calculates categorical cross entropy loss
    """
    def forward(self, y_predictions: list[list[float]], y_true) -> list[float]:
        """
        @brief: calculates the losses for each probability distribution
        @params: y_predictions -> a matrix that's a list of probability distributions
        @params: y_true -> the correct classes (in the form of sparse or one-hot) that
                           is needed to get the losses
        @ret: self.losses -> the losses for each probability distribution
        """
        # For determining how the correct confidences will be acquired
        y_true_are_one_hots: bool = None

        # Check whether y_true is sparse of a list of one-hots
        try:
            Matrix.shape(y_true)
            y_true_are_one_hots = True
        except Exception:
            y_true_are_one_hots = False

        # Clip the inputs to stop unwanted calculations
        y_predictions_clipped: list[list[float]] = Matrix.clip(
            y_predictions,
            1e-7,
            1-1e-7
        )

        # Quick function for getting the correct confidences when y_true are one-hots
        def find_the_index_of_the_one(y_true, index):
            for i in range(len(y_true[index])):
                if (y_true[index][i] == 1):
                    return i

        confidences: list[float] = []

        # If the inputs are sparse
        if (y_true_are_one_hots is False):
            for i in range(len(y_true)):
                confidences.append(y_predictions_clipped[i][y_true[i]])
        # If the inputs are one_hots
        elif (y_true_are_one_hots):
            for i in range(len(y_true)):
                index = find_the_index_of_the_one(y_true, i)
                confidences.append(y_predictions_clipped[i][index])

        self.losses: list[float] = []
        # Get the actual loss by taking the -log of the correct confidence
        for confidence in confidences:
            self.losses.append(-1 * math.log(confidence))
        return self.losses
