import numpy as np

from numpy_utils import MatrixOperations
from performance_test import python_loop, numpy_loop
from image_processing import ImageProcessor


matrix = MatrixOperations()

A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

print("========== MATRIX OPERATIONS ==========\n")

print("Matrix A")
print(A)

print("\nMatrix B")
print(B)

print("\nAddition")
print(matrix.addition(A, B))

print("\nSubtraction")
print(matrix.subtraction(A, B))

print("\nMultiplication")
print(matrix.multiplication(A, B))

print("\nTranspose")
print(matrix.transpose(A))

print("\nDeterminant")
print(matrix.determinant(A))

print("\nInverse")
print(matrix.inverse(A))

print("\nDot Product")
print(matrix.dot_product(A, B))

print("\nFlatten")
print(matrix.flatten(A))

print("\nMean")
print(matrix.mean(A))

print("\nRandom Matrix")
print(matrix.random_matrix(3, 3))

print("\n========== PERFORMANCE ==========\n")

python_time = python_loop()
numpy_time = numpy_loop()

print(f"Python Loop : {python_time:.6f} seconds")
print(f"NumPy       : {numpy_time:.6f} seconds")

print("\n========== IMAGE PROCESSING ==========\n")

processor = ImageProcessor("images/sample.jpg")

processor.grayscale().save("images/grayscale.jpg")
processor.negative().save("images/negative.jpg")
processor.flip_horizontal().save("images/horizontal.jpg")
processor.flip_vertical().save("images/vertical.jpg")
processor.rotate90().save("images/rotated.jpg")

print("Processed images saved successfully.")