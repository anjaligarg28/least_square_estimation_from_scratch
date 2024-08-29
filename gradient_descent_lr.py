from linear_algebra import LinearAlgebra
import numpy as np

def gradient_descent_multiple(X, y, learning_rate=0.01, epochs=1000):
    X = X.values.tolist()
    y = y.values.tolist()

    linearAlgebra = LinearAlgebra()

    # Add a column of ones to X for the intercept in the least squares method
    ones_column = [1]*len(X)
    X = linearAlgebra.add_column(X, ones_column, 0)

    n, p = linearAlgebra.get_row_col(X)
    beta = np.array([[0.0]]*p)

    for _ in range(epochs):
        y_pred = linearAlgebra.multiplication(X, beta.tolist())
        gradients = (-2/n) * np.array(linearAlgebra.multiplication(linearAlgebra.transpose(X), linearAlgebra.matrix_subtraction(y, y_pred)))
        beta -= learning_rate * gradients
        # print(beta)

    return beta
