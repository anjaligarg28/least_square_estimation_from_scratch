import pandas as pd
from least_squares_lr import least_squares_fit
from sklearn_lr import sklearn_linear_regression
from gradient_descent_lr import gradient_descent_multiple

def rename_columns(data):
    # Rename columns to snake case
    data.columns = data.columns.str.strip().str.replace(r"\s+", " ", regex=True).str.lower().str.replace(" ", "_").str.replace("-", "_").str.replace("/", "_")
    return data

def main():
    # import data
    life_expectancy_data = pd.read_csv(r"life_expectancy_data.csv")
    # print(life_expectancy_data)
    life_expectancy_data = rename_columns(life_expectancy_data)
    life_expectancy_data["year"] = life_expectancy_data["year"].astype("str")
    life_expectancy_data.dropna(inplace=True)

    print(life_expectancy_data)

    target_variable = {"life_expectancy"}
    numerical_variables = set(life_expectancy_data.select_dtypes(["int", "float"]).columns).difference(target_variable)
    categorical_variables = set(life_expectancy_data.columns).difference(set(life_expectancy_data.select_dtypes(["int", "float"]).columns))

    print("Target Variable: ", target_variable)
    print("Numerical Variables: ", numerical_variables)
    print("Categorical Variables: ",categorical_variables)

    X = life_expectancy_data[list(numerical_variables)]
    y = life_expectancy_data[list(target_variable)]

    X_normalized = (X - X.mean()) / X.std()

    # Fit the model using scikit-learn
    intercept, coefficients = sklearn_linear_regression(X_normalized, y)
    print(f"Intercept (sklearn): {intercept}")
    print(f"Coefficients (sklearn): {coefficients}")
    # Compute the coefficients using the least squares method
    beta_ls = least_squares_fit(X_normalized, y)
    print(f"Coefficients (Least Squares): {beta_ls}")
    # Compute the coefficients using gradient descent
    beta_gd = gradient_descent_multiple(X_normalized, y, learning_rate=0.02, epochs=5000)
    print(f"Coefficients (Gradient Descent): {beta_gd}")

if __name__ == "__main__":
    main()
