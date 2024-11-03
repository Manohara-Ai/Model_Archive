import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define the true parameters for the synthetic data
weight = 0.7
bias = 0.3

# Create feature data X (inputs) as a tensor from 0 to 24 with step size of 1
X = torch.arange(start=0, end=25, step=1)
# Create target data y (labels) based on a simple linear relationship with some weight and bias
y = weight * X + bias

# Split data into training and test sets using sklearn's train_test_split function
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define a function to plot training and test data along with model predictions
def plot_predictions(train_data=X_train,
                     train_label=y_train,
                     test_data=X_test,
                     test_label=y_test,
                     predictions=None):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_label, c='b', s=4, label='Train Data')
    plt.scatter(test_data, test_label, c='g', s=4, label='Test Data')

    if predictions is not None:
        plt.scatter(test_data, predictions, c='r', s=4, label='Predictions')

    plt.legend(prop={'size': 14})
    plt.show()

# Define a simple Linear Regression model by subclassing nn.Module
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        # Define a trainable parameter for weight initialized randomly; this will be optimized during training
        self.weight = nn.Parameter(torch.randn(1, dtype=torch.float),
                                    requires_grad=True)
        # Define a trainable parameter for bias, also initialized randomly
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float),
                                requires_grad=True)

    # Define the forward pass, which applies the linear transformation to the input x
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias  # y = weight * x + bias

# Instantiate the model
model_0 = LinearRegression()

# Make predictions on test data using the initial random parameters (before training)
# torch.inference_mode() turns off gradient tracking, improving performance and memory usage during inference
with torch.inference_mode():
    y_preds = model_0(X_test)

# Define the loss function as Mean Squared Error (MSE), common for regression tasks
loss_fn = nn.MSELoss()

# Define the optimizer (Adam optimizer in this case) which will update model parameters to minimize the loss
# model_0.parameters() tells the optimizer which parameters to optimize
optimizer = torch.optim.Adam(params=model_0.parameters(),
                             lr=0.1)  # learning rate of 0.1

# Training loop - number of epochs (iterations over the full dataset)
epochs = 100
for epoch in range(epochs):
    # Set the model to training mode to enable gradient calculation
    model_0.train()

    # Perform the forward pass on the training data to get predictions
    y_pred = model_0(X_train)
    
    # Calculate the training loss (MSE between predictions and actual targets)
    loss = loss_fn(y_pred, y_train)

    # Zero out the gradients of model parameters to avoid accumulating gradients from previous steps
    optimizer.zero_grad()

    # Perform backpropagation to compute gradients of the loss w.r.t model parameters
    loss.backward()

    # Step the optimizer to update the model parameters based on computed gradients
    optimizer.step()

    # Set the model to evaluation mode for testing (no gradient calculation)
    model_0.eval()

    # Make predictions on the test data
    with torch.inference_mode():
        test_pred = model_0(X_test)
        # Calculate the loss on test data to track model performance
        test_loss = loss_fn(test_pred, y_test.type(torch.float))

# After training, inspect the final values of the model's parameters (weight and bias)
print(model_0.state_dict())

# Make final predictions with the trained model on the test data
model_0.eval()
with torch.inference_mode():
    y_preds = model_0(X_test)

# Print actual test values and the predicted values for comparison
print('Test values:', y_test)
print('Predicted values:', y_preds)

# Plot the training and test data along with the model's predictions
plot_predictions(predictions=y_preds)
