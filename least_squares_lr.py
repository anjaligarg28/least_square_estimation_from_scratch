"""
linear regression using least square method:
finding y = b0 + b1.x + e
such that SSE (sum of squared residuals) is minimised:
SSE = (e1)^2 + (e2)^2 + ..... (en)^2

equations:
y1 = b0 + b1.x1 + e1
y2 = b0 + b1.x2 + e2
.
.
.
yn = b0 + b1.xn + en

matrix form of equations:
[Y](n x 1) = [X](n x p).[B](p x 1) + [E](n x 1)

n: number of data points
p: number of independent variables + 1 for b0

Now the algorithm's goal is to minimise SSE i.e., minimise the vector Et(transpose).E (sum of E^2)

SSE = (Y - XB)t.(Y - XB)
SSE = Yt.Y - 2.Yt.X.B + Bt.Xt.X.B
derivative of SSE wrt B = -2.Xt.Y + 2.Xt.X.B - (1)
putting (1) to 0
B = (Xt.X)inv.(Xt.Y)

Our goal is to find this B
"""
# import numpy as np
# import pandas as pd
from linear_algebra import LinearAlgebra

def least_squares_fit(X, y):
    # X = X.to_numpy()
    # y = y.to_numpy()
    # beta = np.linalg.inv(X.T @ X) @ X.T @ y

    X = X.values.tolist()
    y = y.values.tolist()

    linearAlgebra = LinearAlgebra()

    # Add a column of ones to X for the intercept in the least squares method
    ones_column = [1]*len(X)
    X = linearAlgebra.add_column(X, ones_column, 0)
    xt = linearAlgebra.transpose(X)
    xt_x = linearAlgebra.multiplication(xt, X)
    # xt_x_inv = linearAlgebra.inverse_using_cofactors(xt_x) # not suitable for larger matrices
    xt_x_inv = linearAlgebra.inverse_matrix_gaussian_elimination(xt_x) #this method uses gaussian elimination to find the inverse
    xt_y = linearAlgebra.multiplication(xt, y)
    beta = linearAlgebra.multiplication(xt_x_inv, xt_y)
    beta = linearAlgebra.flatten(beta)
    return beta
