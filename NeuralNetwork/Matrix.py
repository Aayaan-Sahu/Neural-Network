# Matrix.py

from NeuralNetwork.Vector import Vector


class Matrix:
    """
    A namespace for utilities that will be performed in relation to matrices
    within the neural network
    """

    @staticmethod
    def display(matrix: list[list[float]]) -> None:
        """
        @brief: pretty prints a given matrix
        @params: matrix -> the matrix to print
        @ret: None
        """
        for i in range(len(matrix)):
            Vector.display(matrix[i])

    @staticmethod
    def shape(matrix: list[list[float]]) -> list[int]:
        """
        @brief: returns the 2D shape of a matrix
        @params: matrix -> the matrix to get the shape of
        @ret: [ len(matrix), len(matrix[0]) ] -> the rows and columns of the matrix
        """
        return [len(matrix), len(matrix[0])]

    @staticmethod
    def return_row(index: int, matrix: list[list[float]]) -> list[float]:
        """
        @brief: return a row at an index of a matrix
        @params: index -> the index of the row of the matrix to return
        @params: matrix -> the matrix supplied
        @ret: matrix[index] -> the vector at the index
        """
        return matrix[index]

    @staticmethod
    def return_column(index: int, matrix: list[list[float]]) -> list[float]:
        """
        @brief: return a column at an index of a matrix
        @params: index -> the index of the column of the matrix to return
        @params: matrix -> the matrix supplied
        @ret: v -> the vector at the index
        """
        v: list[float] = []
        for i in range(len(matrix)):
            v.append(matrix[i][index])
        return v

    @staticmethod
    def scalar_matrix_multiplication(
        scalar: float, matrix: list[list[float]]
    ) -> list[list[float]]:
        """
        @brief: multiplies a scalar with an entire matrix
        @params: scalar -> the scalar to multiply by
        @params: matrix -> the matrix to multiply by
        @ret: m -> the matrix multiplied by the scalar
        """
        m = matrix
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                m[i][j] = scalar * m[i][j]
        return m

    @staticmethod
    def transpose(matrix: list[list[float]]) -> list[list[float]]:
        """
        @brief: transposes a matrix

            ex: 1 2 3   ->  1 4
                4 5 6       2 5
                            3 6

        @params: matrix -> the matrix to transpose
        @ret: m -> the transposed matrix
        """
        m: list[list[float]] = []

        rows: int = len(matrix)
        columns: int = len(matrix[0])

        for i in range(columns):
            m.append([])
            for j in range(rows):
                m[i].append(matrix[j][i])

        return m

    @staticmethod
    def add_vector(matrix: list[list[float]], vector: list[float]) -> list[list[float]]:
        """
        @brief: adds a vector to all the rows within a matrix
        @params: matrix -> the matrix that is an addend
        @params: vector -> the vector that is an addend
        @ret: m -> a matrix with all the rows added by the vector
        """
        m: list[list[float]] = []
        for i in range(len(matrix)):
            m.append(Vector.add(matrix[i], vector))
        return m

    @staticmethod
    def is_valid_matrix_multiplication(
        m1: list[list[float]], m2: list[list[float]]
    ) -> bool:
        """
        @brief: checks if the multiplication of two matrices is valid
        @params: m1 -> the first matrix
        @params: m2 -> the second matrix
        @ret: True/False -> whether the multiplication is valid or not
        """
        if len(m1[0]) == len(m2):
            return True
        return False

    @staticmethod
    def matrix_multiplication(
        m1: list[list[float]], m2: list[list[float]]
    ) -> list[list[float]]:
        """
        @brief: multiplies two matrices (first's no. columns == second's no. rows)
        @params: m1 -> the first matrix
        @params: m2 -> the second matrix
        @ret: m -> the product of the two matrices
        """
        if len(m1[0]) != len(m2):
            print(
                "Invalid matrix multiplication -> Number of columns in m1 should equal number of rows in m2"
            )
            return
        else:
            m: list[list[float]] = []  # The return variable

            # Getting the dimensions of the result matrix
            rows_in_result: int = len(m1)
            columns_in_result: int = len(m2[0])

            for i in range(rows_in_result):
                m.append([])
                row: list[float] = Matrix.return_row(i, m1)
                for j in range(columns_in_result):
                    col: list[float] = Matrix.return_column(j, m2)

                    dot_product: float = Vector.dot(
                        row, col
                    )  # Every new value is the dot product of its row and column
                    m[i].append(dot_product)

            return m

    @staticmethod
    def clip(matrix: list[list[float]], left: float, right: float) -> list[list[float]]:
        """
        @brief: clips all the values of a matrix between two extremes
        @params: matrix -> the matrix to get the elements from
        @params: left -> the lower bound
        @params: right -> the upper bound
        @ret: v -> the matrix with the values clipped
        """
        m: list[list[float]] = []
        for i in range(len(matrix)):
            m.append([])
            for j in range(len(matrix[i])):
                if (matrix[i][j] > right):
                    m[i].append(right)
                elif (matrix[i][j] < left):
                    m[i].append(left)
                else:
                    m[i].append(matrix[i][j])
        return m
