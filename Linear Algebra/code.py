# Linear Algebra Problems

# Basic Matrix Operations
from matplotlib.pylab import beta
import numpy as np
import numpy.linalg as la

# Define two matrices
A = np.array([[2, -1], [0, 3]])
B = np.array([[1, 2], [3, 4]])
print("Matrix A:\n", A)
print("\n Matrix B:\n", B)

# Matrix Addition
C = A + B
print("\n Matrix Addition:\n A + B =", C)

# Matrix Mulitplication
D = np.dot(A, B)
# We can also use D=A@B and A.dot(B)
print("\n Matrix Multiplication:\n A * B =", D)

# Matrix Subtraction
E = 3 * A - 2 * B
print("\n Matrix Subtraction:\n 3A - 2B =", E)

# Determinant and Inverse of a Matrix
# Determinant of matrix A
det_A = la.det(A)
print("\n Determinant of A:", det_A)


# Inverse of matrix A
if det_A != 0:
    inv_A = la.inv(A)
    print("\n Inverse of A:\n", inv_A)


# Eigenvalues and Eigenvectors
# Matrix for eigenvalue problem
C = np.array([[4, 1], [-2, 1]])
eigenvalues, eigenvectors = la.eig(C)

# Print the results
print("\n Matrix C:")
print(C)
print("\n Eigenvalue (first): ")
print(eigenvalues[0])
print("\n Corresponding Eigenvector:")
print(eigenvectors[:, 0])


# You are given a dataset of cars with the following features:Number of cylinders (x1​)Engine size in liters (x2​)Horsepower (x3)
# car 1:
# x1: 4
# x2:  1.5
# x3: 100
# and price: 20

# car 2:
# x1: 6
# x2:  2.0
# x3: 150
# and price: 30

# Then, find the price of a car 3 with features
# x1: 8
# x2:  3.0
# x3: 200
# price:?

# Define the feature matrix X and price vector y
X = np.array([[4, 1.5, 100], [6, 2.0, 150]])
y = np.array([20, 30])

# β calculation using the normal equation
beta = la.inv(X.T @ X) @ X.T @ y

# # Predict the price of car 3
X_car3 = np.array([[8, 3.0, 200]])
price_car3 = X_car3 @ beta

print("Coefficients (β):", beta)
print("Predicted price of car 3:", price_car3[0])
