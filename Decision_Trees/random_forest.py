import numpy as np
from decision_tree import DecisionTree
from collections import Counter

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        # Initialize the Random Forest with parameters for tree creation.
        self.n_trees = n_trees                     # Number of decision trees in the forest.
        self.max_depth = max_depth                  # Maximum depth of each decision tree.
        self.min_samples_split = min_samples_split  # Minimum number of samples required to split a node.
        self.n_features = n_features                # Number of features to consider when looking for the best split.
        self.trees = []                             # List to hold the individual decision trees.

    def fit(self, X, y):
        # Fit the random forest model to the training data X and labels y.
        self.trees = []  # Reset the list of trees for a new fit.
        for _ in range(self.n_trees):
            # Create and fit a new decision tree for each iteration.
            tree = DecisionTree(max_depth=self.max_depth, 
                                min_samples_split=self.min_samples_split,
                                n_features=self.n_features)
            X_sample, y_sample = self._bootstrap_samples(X, y)  # Generate bootstrap samples.
            tree.fit(X_sample, y_sample)  # Fit the decision tree on the sampled data.
            self.trees.append(tree)  # Add the fitted tree to the forest.

    def _bootstrap_samples(self, X, y):
        # Create bootstrap samples from the dataset X and labels y.
        n_samples = X.shape[0]  # Get the number of samples in the dataset.
        idxs = np.random.choice(n_samples, n_samples, replace=True)  # Randomly select indices for sampling with replacement.
        return X[idxs], y[idxs]  # Return the sampled features and labels.
    
    def _most_common_label(self, y):
        # Return the most common label in the array y.
        counter = Counter(y)  # Count occurrences of each label.
        most_common = counter.most_common(1)[0][0]  # Get the most common label.
        return most_common
    
    def predict(self, X):
        # Make predictions for the input data X using the ensemble of trees.
        predictions = np.array([tree.predict(X) for tree in self.trees])  # Get predictions from each tree.
        tree_preds = np.swapaxes(predictions, 0, 1)  # Transpose predictions to organize by sample.
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])  # Aggregate predictions by majority vote.
        return predictions  # Return the final predictions for the input data.
