from sklearn.datasets import make_circles
import torch
from sklearn.model_selection import train_test_split
from torch import nn
import matplotlib.pyplot as plt
from helper_functions import plot_decision_boundary

# Generate synthetic circular data with some noise
X, y = make_circles(1000, noise=0.05, random_state=42)

# Convert data to PyTorch tensors
X = torch.from_numpy(X).type(torch.float) 
y = torch.from_numpy(y).type(torch.float)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set device to GPU if available, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define a binary classifier for circular data
class CircleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Input layer to hidden layer (2 input features, 10 units in hidden layer)
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        # First hidden layer to second hidden layer (10 units to 5 units)
        self.layer_2 = nn.Linear(in_features=10, out_features=5)
        # Second hidden layer to output layer (5 units to 1 output for binary classification)
        self.layer_3 = nn.Linear(in_features=5, out_features=1)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation function to introduce non-linearity

    # Define forward pass
    def forward(self, x):
        # Apply layers sequentially with activation
        return self.layer_3(self.layer_2(self.sigmoid(self.layer_1(x))))

# Function to calculate accuracy of predictions
def accuracy_fn(y_true, y_pred):
    # Count number of correct predictions
    correct = torch.eq(y_true, y_pred).sum().item()
    # Calculate accuracy as percentage
    acc = (correct / len(y_pred)) * 100 
    return acc

# Initialize the model and move it to the device
model_0 = CircleClassifier().to(device)

# Test initial untrained predictions
untrained_preds = model_0(X_test.to(device))  # Predictions with untrained model (for reference)

# Define the binary cross-entropy loss with logits
loss_fn = nn.BCEWithLogitsLoss()

# Define the Adam optimizer with learning rate of 0.1
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.1)

# Check initial logits, probabilities, and rounded predictions for a sample batch
y_logits = model_0(X_test.to(device))[:5]
y_preds_probs = torch.sigmoid(y_logits)  # Apply sigmoid to logits to get probabilities
y_preds = torch.round(y_preds_probs)  # Round probabilities to get binary predictions
y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))

# Print to confirm that rounding probabilities gives expected labels
print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))

# Set random seed for reproducibility
torch.manual_seed(42)

# Set number of epochs
epochs = 100

# Move training and test data to the device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Training loop
for epoch in range(epochs):
    model_0.train()  # Set model to training mode

    # Forward pass: calculate logits and apply sigmoid to get predictions
    y_logits = model_0(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))  # Convert logits to probabilities, then round for predictions
  
    # Calculate the training loss
    loss = loss_fn(y_logits, y_train)
    # Calculate training accuracy
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

    # Zero gradients to prevent accumulation from previous steps
    optimizer.zero_grad()

    # Backward pass: compute gradients
    loss.backward()

    # Update model parameters based on gradients
    optimizer.step()

    # Evaluation mode for test set
    model_0.eval()
    with torch.inference_mode():
        # Compute logits, predictions, loss, and accuracy for the test set
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))  # Get rounded predictions
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

    # Print progress every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

# Plot decision boundaries for train and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_0, X_train, y_train)  # Custom function to plot decision boundary for training set
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_0, X_test, y_test)  # Custom function to plot decision boundary for test set
plt.show()
