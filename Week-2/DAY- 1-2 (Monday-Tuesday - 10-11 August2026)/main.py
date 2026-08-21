from matrix import Matrix

matrix1 = Matrix([
    [1, 2],
    [3, 4]
])

matrix2 = Matrix([
    [5, 6],
    [7, 8]
])

print("Matrix 1")
matrix1.display()

print("\nMatrix 2")
matrix2.display()

print("\nAddition")
matrix1.add(matrix2).display()

print("\nSubtraction")
matrix1.subtract(matrix2).display()

print("\nScalar Multiplication")
matrix1.scalar_multiply(2).display()

print("\nTranspose")
matrix1.transpose().display()

print("\nMatrix Multiplication")
matrix1.multiply(matrix2).display()

print("\nIdentity Matrix")
Matrix.identity(3).display()

print("\nInverse of Matrix 1")
inverse = matrix1.inverse_2x2()

if inverse:
    inverse.display()