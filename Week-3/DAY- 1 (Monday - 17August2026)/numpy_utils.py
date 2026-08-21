import numpy as np


class MatrixOperations:

    def __init__(self):
        pass

    def addition(self, a, b):
        return np.add(a, b)

    def subtraction(self, a, b):
        return np.subtract(a, b)

    def multiplication(self, a, b):
        return np.matmul(a, b)

    def transpose(self, a):
        return np.transpose(a)

    def inverse(self, a):
        return np.linalg.inv(a)

    def determinant(self, a):
        return np.linalg.det(a)

    def dot_product(self, a, b):
        return np.dot(a, b)

    def reshape(self, a, shape):
        return np.reshape(a, shape)

    def flatten(self, a):
        return a.flatten()

    def mean(self, a):
        return np.mean(a)

    def maximum(self, a):
        return np.max(a)

    def minimum(self, a):
        return np.min(a)

    def random_matrix(self, rows, cols):
        return np.random.randint(0, 10, (rows, cols))