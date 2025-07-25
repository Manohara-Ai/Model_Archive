import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from knn import KNN

iris = datasets.load_iris()
X, y = iris.data,iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

plt.figure(figsize=(10, 5))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', label='Train Data', alpha=0.6)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='x', label='Test Data', alpha=0.6)
plt.title('Train and Test Data Visualization')
plt.legend()
plt.show()

clf = KNN(k=5)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print("Predictions:", predictions)

accuracy = np.mean(predictions == y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')