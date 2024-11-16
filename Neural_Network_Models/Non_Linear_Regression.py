import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
import torch
from torch import nn
import matplotlib.pyplot as plt

# Define the regression model using a neural network
class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        
        # Define the layers of the neural network
        self.Layer_1 = nn.Linear(in_features=8, out_features=128)
        self.Layer_2 = nn.Linear(in_features=128, out_features=256)  
        self.Layer_3 = nn.Linear(in_features=256, out_features=512)
        self.Layer_4 = nn.Linear(in_features=512, out_features=256)
        self.Layer_5 = nn.Linear(in_features=256, out_features=128)
        self.Layer_6 = nn.Linear(in_features=128, out_features=1)

        # Define activation function and regularization techniques
        self.relu = nn.ReLU()  # ReLU activation function
        self.dropout = nn.Dropout(p=0.3)  # Dropout layer with 30% dropout probability
        self.batch_norm_1 = nn.BatchNorm1d(128)  
        self.batch_norm_2 = nn.BatchNorm1d(256)
        self.batch_norm_3 = nn.BatchNorm1d(512)

    # Define the forward pass through the network
    def forward(self, x):
        x = self.relu(self.batch_norm_1(self.Layer_1(x)))
        x = self.dropout(x)
        
        x = self.relu(self.batch_norm_2(self.Layer_2(x)))
        x = self.dropout(x) 

        x = self.relu(self.batch_norm_3(self.Layer_3(x)))
        x = self.dropout(x)
        
        x = self.relu(self.Layer_4(x))
        x = self.dropout(x)
        
        x = self.relu(self.Layer_5(x))
        x = self.dropout(x)
        
        return self.Layer_6(x)

# Function to calculate accuracy as Mean Absolute Error (MAE)
def accuracy_fn(y_true, y_pred):
    mae = torch.mean(torch.abs(y_true - y_pred))  
    return mae.item()

# Function to visualize actual vs predicted values
def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_true)), y_true.cpu(), color='blue', label='Actual Values', alpha=0.6)  # Plot actual values in blue
    plt.scatter(range(len(y_pred)), y_pred.cpu(), color='red', label='Predicted Values', alpha=0.6)  # Plot predicted values in red
    plt.xlabel('Sample Index') 
    plt.ylabel('Salary in USD')
    plt.title('Actual vs Predicted Salaries')
    plt.legend()  
    plt.show()

# Main code execution
if __name__ == '__main__':
    # Selecting the features (X) and the target variable (y)
    X, y = make_regression(n_samples=200, n_features=8, noise=10.0, random_state=42)
    y = np.exp(0.1 * X[:, 0]) + 5 * np.sin(2 * X[:, 0]) - 3 * np.log(np.abs(X[:, 0] + 1)) + y

    # Convert X and y to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # Reshape y to match the output shape of the model

    # Split the dataset into training and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Check if GPU is available, otherwise use CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Instantiate the model and move it to the device (GPU or CPU)
    model_0 = Regressor().to(device)

    # Define loss function (Mean Squared Error for regression)
    loss_fn = nn.MSELoss()
    # Optimizer (Adam optimizer with a learning rate of 0.001)
    optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)

    # Move the training and test data to the same device as the model
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    epochs = 1000  # Number of training epochs

    # Training loop
    for epoch in range(epochs):
        model_0.train()  # Set the model to training mode

        y_pred = model_0(X_train)  # Get predictions from the model
        loss = loss_fn(y_pred, y_train)  # Calculate the loss (MSE loss)

        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()  # Backpropagation to compute gradients
        optimizer.step()  # Update model parameters

        # Evaluation loop (compute validation/test loss)
        model_0.eval()  # Set the model to evaluation mode
        with torch.inference_mode():  # Inference mode to disable gradient calculation
            test_pred = model_0(X_test)  # Get predictions for the test set
            test_loss = loss_fn(test_pred, y_test)  # Calculate the test loss
            test_mae = accuracy_fn(y_true=y_test, y_pred=test_pred)  # Calculate Mean Absolute Error (MAE)

        # Print the progress every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}, Test MAE: {test_mae:.4f}")

    # Final evaluation
    model_0.eval()
    with torch.inference_mode():
        y_pred = model_0(X_test)  # Get predictions for the test set

    # Print predicted salaries
    print("Predicted salaries:")
    for i in range(len(X_test)):
        print(X_test[i], y_pred[i])

    # Plot the actual vs predicted salaries
    plot_predictions(y_test, y_pred)
