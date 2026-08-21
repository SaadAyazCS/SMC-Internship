class Matrix:
    def __init__(self, data):
        self.data = data

    def display(self):
        for row in self.data:
            print(row)

    def add(self, other):
        result = []

        for i in range(len(self.data)):
            row = []
            for j in range(len(self.data[0])):
                row.append(self.data[i][j] + other.data[i][j])
            result.append(row)

        return Matrix(result)

    def subtract(self, other):
        result = []

        for i in range(len(self.data)):
            row = []
            for j in range(len(self.data[0])):
                row.append(self.data[i][j] - other.data[i][j])
            result.append(row)

        return Matrix(result)

    def scalar_multiply(self, scalar):
        result = []

        for row in self.data:
            new_row = []
            for value in row:
                new_row.append(value * scalar)
            result.append(new_row)

        return Matrix(result)

    def transpose(self):
        result = []

        for j in range(len(self.data[0])):
            row = []
            for i in range(len(self.data)):
                row.append(self.data[i][j])
            result.append(row)

        return Matrix(result)

    def multiply(self, other):
        rows = len(self.data)
        cols = len(other.data[0])
        common = len(other.data)

        result = []

        for i in range(rows):
            row = []

            for j in range(cols):

                total = 0

                for k in range(common):
                    total += self.data[i][k] * other.data[k][j]

                row.append(total)

            result.append(row)

        return Matrix(result)

    @staticmethod
    def identity(size):

        matrix = []

        for i in range(size):

            row = []

            for j in range(size):

                if i == j:
                    row.append(1)
                else:
                    row.append(0)

            matrix.append(row)

        return Matrix(matrix)

    def inverse_2x2(self):

        if len(self.data) != 2 or len(self.data[0]) != 2:
            print("Inverse only implemented for 2x2 matrices.")
            return None

        a = self.data[0][0]
        b = self.data[0][1]
        c = self.data[1][0]
        d = self.data[1][1]

        determinant = a * d - b * c

        if determinant == 0:
            print("Matrix has no inverse.")
            return None

        result = [
            [d / determinant, -b / determinant],
            [-c / determinant, a / determinant]
        ]

        return Matrix(result)