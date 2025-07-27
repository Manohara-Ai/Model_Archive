import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        cov = np.cov(X.T)

        eigen_vectors, eigen_values = np.linalg.eig(cov)
        eigen_vectors = eigen_vectors.T

        idxs = np.argsort(eigen_values)[::-1]
        eigen_values = eigen_values[idxs]
        eigen_vectors = eigen_vectors[idxs]

        self.components = eigen_vectors[:self.n_components]
 
    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components.T)