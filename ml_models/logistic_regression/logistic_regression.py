import numpy as np 

class LogisticRegression:
    def __init__(self, lr=0.01, num_iters=1000, w_lambda=0.0):
        self.lr = lr
        self.num_iters = num_iters
        self.weights = None
        self.bias = None
        self.w_lambda = w_lambda
    
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        
        for _ in range(self.num_iters):
            linear_predictions = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_predictions)

            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y)) + self.w_lambda * self.weights
            db = (1 / num_samples) * np.sum(y_predicted - y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return (1 / (1 + np.exp(-z)))
    
    def predict(self, X):
        linear_predictions = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_predictions)
        predicted_class = [0 if i < 0.5 else 1 for i in y_predicted]
        return np.array(predicted_class)