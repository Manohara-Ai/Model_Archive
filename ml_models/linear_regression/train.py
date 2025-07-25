import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from linear_regression_gradient_descent import LinearRegression as LinearRegressionGD
from linear_regression_normal_equation import LinearRegression as LinearRegressionNE

def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=25)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36)

plt.figure(figsize=(10, 5))
plt.scatter(X_train[:, 0], y_train, color='crimson', label='Training Data')
plt.scatter(X_test[:, 0], y_test, color='blue', label='Test Data')
plt.legend()
plt.title("Train vs Test Data")
plt.show()

model_gd = LinearRegressionGD()
model_gd.fit(X_train, y_train)
y_pred_gd = model_gd.predict(X_test)
mse_gd = MSE(y_test, y_pred_gd)
rmse_gd = np.sqrt(mse_gd)

model_ne = LinearRegressionNE()
model_ne.fit(X_train, y_train)
y_pred_ne = model_ne.predict(X_test)
mse_ne = MSE(y_test, y_pred_ne)
rmse_ne = np.sqrt(mse_ne)

print("Gradient Descent Linear Regression:")
print(f"  MSE:  {mse_gd:.4f}")
print(f"  RMSE: {rmse_gd:.4f}\n")

print("Normal Equation Linear Regression:")
print(f"  MSE:  {mse_ne:.4f}")
print(f"  RMSE: {rmse_ne:.4f}")

X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line_gd = model_gd.predict(X_line)
y_line_ne = model_ne.predict(X_line)

plt.figure(figsize=(10, 5))
plt.scatter(X, y, color='gray', label='Data Points', alpha=0.6)
plt.plot(X_line, y_line_gd, color='green', label='Gradient Descent Prediction')
plt.plot(X_line, y_line_ne, color='blue', linestyle='--', label='Normal Equation Prediction')
plt.legend()
plt.title("Comparison of Linear Regression Methods")
plt.xlabel("X")
plt.ylabel("y")
plt.show()
