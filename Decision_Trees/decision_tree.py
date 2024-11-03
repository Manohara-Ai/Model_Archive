import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        # Initialize a node in the decision tree.
        self.feature = feature         # The feature index that the node splits on.
        self.threshold = threshold     # The threshold value to split on.
        self.left = left               # Pointer to the left child node.
        self.right = right             # Pointer to the right child node.
        self.value = value             # The class label if the node is a leaf node.
    
    def is_leaf_node(self):
        # Check if the node is a leaf node (i.e., it has a class label).
        return self.value is not None
    
class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=1000, n_features=None):
        # Initialize the decision tree with parameters for splitting and depth.
        self.min_samples_split = min_samples_split  # Minimum number of samples required to split a node.
        self.max_depth = max_depth                  # Maximum depth of the tree.
        self.n_features = n_features                # Number of features to consider when looking for the best split.
        self.root = None                            # The root of the decision tree.

    def fit(self, X, y):
        # Fit the model to the training data X and labels y.
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)  # Grow the tree starting from the root.

    def _grow_tree(self, X, y, depth=0):
        # Recursively grow the decision tree.
        n_samples, n_feats = X.shape               # Number of samples and features.
        n_labels = np.unique(y)                    # Unique class labels in the target variable.

        # Base cases for stopping the recursion:
        # 1. Maximum depth reached.
        # 2. Only one class label present.
        # 3. Not enough samples to split further.
        if (depth >= self.max_depth or len(n_labels) == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)  # Assign the most common label as the leaf value.
            return Node(value=leaf_value)
        
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)  # Randomly select feature indices.
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)  # Find the best feature and threshold to split on.

        # Check if a valid split was found
        if best_feature is None or best_thresh is None:
            leaf_value = self._most_common_label(y)  # Assign the most common label if no split is possible.
            return Node(value=leaf_value)
    
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)  # Split the dataset into left and right subsets.

        # Safeguard against empty splits
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            leaf_value = self._most_common_label(y)  # Assign the most common label if splits are empty.
            return Node(value=leaf_value)

        # Recursively grow the left and right child nodes.
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature, best_thresh, left, right)  # Return a new node with the best split.

    def _most_common_label(self, y):
        # Return the most common label in the array y.
        counter = Counter(y)  # Count occurrences of each label.
        value = counter.most_common(1)[0][0]  # Get the most common label.
        return value
    
    def _best_split(self, X, y, feat_idxs):
        # Find the best feature and threshold for splitting the dataset.
        best_gain = -1  # Initialize the best gain as negative.
        split_idx, split_threshold = None, None  # Initialize best feature and threshold.

        for feat_idx in feat_idxs:
            X_Column = X[:, feat_idx]  # Extract the feature column.
            thresholds = np.unique(X_Column)  # Find unique values to use as thresholds.

            for thr in thresholds:
                gain = self._information_gain(y, X_Column, thr)  # Calculate information gain for the split.

                if gain > best_gain:
                    best_gain = gain  # Update best gain if a better one is found.
                    split_idx = feat_idx  # Update the best feature index.
                    split_threshold = thr  # Update the best threshold.
        
        return split_idx, split_threshold  # Return the best feature and threshold.
    
    def _information_gain(self, y, X_Column, threshold):
        # Calculate the information gain from a split at the given threshold.
        parent_entropy = self._entropy(y)  # Entropy of the parent node.
        left_idxs, right_idxs = self._split(X_Column, threshold)  # Split the data into left and right based on the threshold.

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0  # Return 0 gain if one side is empty.
        
        n = len(y)  # Total number of samples.
        n_left, n_right = len(left_idxs), len(right_idxs)  # Number of samples in left and right splits.
        entropy_left, entropy_right = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])  # Entropy of the left and right splits.

        # Calculate the weighted average of child entropies.
        child_entropy = (n_left / n) * entropy_left + (n_right / n) * entropy_right

        # Information gain is the difference between parent and child entropy.
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _entropy(self, y):
        # Calculate the entropy of the labels y.
        hist = np.bincount(y)  # Count occurrences of each class.
        ps = hist / len(y)  # Calculate probabilities.
        return -np.sum([p * np.log(p) for p in ps if p > 0])  # Calculate entropy using the probabilities.
    
    def _split(self, X_Column, split_thresh):
        # Split the data based on the threshold.
        left_idx = np.argwhere(X_Column <= split_thresh).flatten()  # Indices where the feature value is less than or equal to the threshold.
        right_idx = np.argwhere(X_Column > split_thresh).flatten()  # Indices where the feature value is greater than the threshold.
        return left_idx, right_idx  # Return left and right indices.
    
    def predict(self, X):
        # Make predictions for the input data X.
        return np.array([self._traverse_tree(x, self.root) for x in X])  # Traverse the tree for each sample.
    
    def _traverse_tree(self, x, node):
        # Traverse the decision tree to make a prediction for a sample x.
        if node.is_leaf_node():
            return node.value  # Return the leaf value if it's a leaf node.
        
        # Navigate to the left or right child based on the feature value.
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)  # Traverse left if the feature value is less than or equal to the threshold.
        return self._traverse_tree(x, node.right)  # Traverse right otherwise.
