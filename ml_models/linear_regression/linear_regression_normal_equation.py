import numpy as np

class LinearRegression:
    def __init__(self, w_lambda=0.0):
        self.weights = None
        self.w_lambda = w_lambda

    def fit(self, X, y):
        n_samples, n_features = X.shape
        ones = np.ones((n_samples, 1))
        X_b = np.hstack((ones, X))

        I = np.eye(X_b.shape[1])
        I[0, 0] = 0 
        self.weights = np.dot(np.linalg.inv(np.dot(X_b.T, X_b) + self.w_lambda * I), np.dot(X_b.T, y))

    def predict(self, X):
        n_samples, n_features = X.shape
        ones = np.ones((n_samples, 1))
        X_b = np.hstack((ones, X))
        return np.dot(X_b, self.weights)