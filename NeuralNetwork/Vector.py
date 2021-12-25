# Vector.py

from colored import fg, bg, attr


class Vector:
    """
    A namespace for utilities that will be performed in relation to vectors
    within the neural network
    """

    @staticmethod
    def display(vector: list[float]) -> None:
        """
        @brief: pretty prints a given vector
        @params: vector -> the vector to print
        @ret: None
        """
        print("[  ", end="")
        for i in range(len(vector)):
            print(f"{vector[i]}  ", end="")
        print("]")

    @staticmethod
    def shape(vector: list[float]) -> list[float]:
        """
        @brief: returns the 2D shape of a vector
        @params: vector -> the vector to get the shape
        @ret: [ 1, len(vector) ] -> the 2D shape of the vector
        """
        return [1, len(vector)]

    @staticmethod
    def dot(v1: list[float], v2: list[float]) -> float:
        """
        @brief: calculates the dot product between two equal-length vectors
        @params: v1 -> the first vector
        @params: v2 -> the second vector
        @ret: sum -> the dot product of v1 and v2 ( v1 • v2 )
        """
        sum: float = 0
        if len(v1) == len(v2):
            for i in range(len(v1)):
                sum += v1[i] * v2[i]
        else:
            print(f"{fg('red')}{bg('white')}INVALID DOT PRODUCT{attr(0)}")
            return
        return sum

    @staticmethod
    def add(v1: list[float], v2: list[float]) -> list[float]:
        """
        @brief: adds two equal length vectors together
        @params: v1 -> the first vector
        @params: v2 -> the second vector
        @ret: sum -> the sum of v1 and v2 ( v1 + v2 )
        """
        v: list[float] = []
        if len(v1) == len(v2):
            for i in range(len(v1)):
                v.append(v1[i] + v2[i])
        else:
            print(f"{fg('red')}{bg('white')}INVALID ADDITION{attr(0)}")
            return
        return v

    @staticmethod
    def vector_sum(vector: list[float]) -> float:
        """
        @brief: returns the sum of all the elements within a vector
        @params: vector -> the vector to take the sum of
        @ret: sum -> the total sum of all the elements within a vector
        """
        sum: float = 0
        for i in vector:
            sum += i
        return sum

    @staticmethod
    def mean(vector: list[float]) -> float:
        """
        @brief: returns the mean of all the elements of a vector
        @params: vector -> the vector to get the mean from
        @ret: (Vector.vector_sum(vector) / len(vector)) -> the average
        """
        return (Vector.vector_sum(vector) / len(vector))

    @staticmethod
    def clip(vector: list[float], left: float, right: float) -> list[float]:
        """
        @brief: clips all the values of a vector between two extremes
        @params: vector -> the vector to get the elements from
        @params: left -> the lower bound
        @params: right -> the upper bound
        @ret: v -> the vector with the values clipped
        """
        v: list[float] = []
        for i in vector:
            if (i > right):
                v.append(right)
            elif (i < left):
                v.append(left)
            else:
                v.append(i)
        return v
