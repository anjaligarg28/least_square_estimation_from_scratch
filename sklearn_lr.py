from sklearn.linear_model import LinearRegression

def sklearn_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model.intercept_, model.coef_