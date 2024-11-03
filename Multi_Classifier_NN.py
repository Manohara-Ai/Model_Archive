# Import necessary libraries
import torch
from torch import nn
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from helper_functions import plot_decision_boundary

# Generate synthetic data with 4 centers (classes) and 2 features
X, y = make_blobs(n_samples=1000,
                  n_features=2,
                  centers=4,
                  cluster_std=1.5,
                  random_state=45)

# Convert data to PyTorch tensors (X as float, y as LongTensor for classification)
X, y = torch.from_numpy(X).type(torch.float), torch.from_numpy(y).type(torch.LongTensor)

# Split data into training and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

# Set device to GPU if available, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define a multi-layer classifier model
class MultiClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Layer 1: Input layer (2 features) to 10 units
        self.Layer_1 = nn.Linear(in_features=2, out_features=10)
        # Layer 2: Hidden layer (10 units) to 5 units
        self.Layer_2 = nn.Linear(in_features=10, out_features=5)
        # Layer 3: Output layer (5 units) to 4 units (one for each class)
        self.Layer_3 = nn.Linear(in_features=5, out_features=4)

    # Define the forward pass
    def forward(self, x):
        # Pass input through layers sequentially
        return self.Layer_3(self.Layer_2(self.Layer_1(x)))

# Define a function to calculate accuracy
def accuracy_fn(y_true, y_pred):
    # Count where predicted labels match true labels
    correct = torch.eq(y_true, y_pred).sum().item()
    # Calculate accuracy as a percentage
    acc = (correct / len(y_pred)) * 100 
    return acc

# Instantiate the model and move it to the specified device
model_0 = MultiClassifier().to(device)

# Define loss function for multi-class classification
loss_fn = nn.CrossEntropyLoss()

# Define the Adam optimizer with a learning rate of 0.1
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.1)

# Set random seed for reproducibility
torch.manual_seed(42)

# Set number of epochs for training
epochs = 100

# Move training and test data to the specified device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Training loop
for epoch in range(epochs):
    model_0.train()  # Set model to training mode

    # Forward pass: Compute logits for training data
    y_logits = model_0(X_train)
    # Get predicted class labels by applying softmax and taking the argmax
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

    # Calculate the training loss between logits and true labels
    loss = loss_fn(y_logits, y_train)
    # Calculate training accuracy
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

    # Zero gradients from the previous step to prevent accumulation
    optimizer.zero_grad()
    # Backward pass: Compute gradients
    loss.backward()
    # Update model parameters based on gradients
    optimizer.step()

    # Evaluation on the test set
    model_0.eval()  # Set model to evaluation mode
    with torch.inference_mode():  # Disable gradient calculation for efficiency
        # Forward pass: Compute logits for test data
        test_logits = model_0(X_test)
        # Get predicted class labels for test data
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)

        # Calculate test loss and accuracy
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

    # Print progress every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")

# Final evaluation on the test set
model_0.eval()
with torch.inference_mode():
    # Compute logits for the test data
    y_logits = model_0(X_test)

# Compute predicted probabilities and labels for plotting
y_pred_probs = torch.softmax(y_logits, dim=1)
y_preds = y_pred_probs.argmax(dim=1)

# Plot decision boundaries for train and test data
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_0, X_train, y_train)  # Plot train data with decision boundary
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_0, X_test, y_test)  # Plot test data with decision boundary
plt.show()

# Print final test accuracy
print(f"Test accuracy: {accuracy_fn(y_true=y_test, y_pred=y_preds)}%")
