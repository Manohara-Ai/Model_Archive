import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from helper_functions import accuracy_fn
from tqdm.auto import tqdm  # For progress tracking
from timeit import default_timer as timer  # To measure training time
import random  # For random sampling

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)

# Set device to GPU if available, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Define the CNN model architecture
class CNN_Model(nn.Module):
    def __init__(self, input_shape:int, hidden_units:int, output_shape:int):
        super().__init__()
        # Define the first block of convolutional layers
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3,  # 3x3 convolutional filter
                      stride=1, 
                      padding=1), 
            nn.ReLU(),  # ReLU activation
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Downsampling with max pooling, Output Size = ((Input Size - Kernel Size + 2 * Padding) / Stride) + 1
        )
        # Define the second block of convolutional layers
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Downsampling with max pooling, Output Size = ((Input Size - Kernel Size + 2 * Padding) / Stride) + 1
        )
        # Define the classifier that maps features to output classes
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Flatten the tensor
            nn.Linear(in_features=hidden_units*7*7,  # Adjust input size based on previous layer output
                      out_features=output_shape)  # Output layer for class predictions
        )

    # Define the forward pass through the network
    def forward(self, x:torch.Tensor):
        x = self.block_1(x)  # Pass input through block 1
        x = self.block_2(x)  # Pass input through block 2
        x = self.classifier(x)  # Pass input through classifier
        return x


# Function to print training time
def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start  # Calculate total training time
    print(f"Train time on {device}: {total_time:.3f} seconds")  # Print training time
    return total_time


# Function to perform a single training step
def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    train_loss, train_acc = 0, 0  # Initialize loss and accuracy
    model.to(device)  # Move model to device
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)  # Move batch data to device

        y_pred = model(X)  # Forward pass: compute predictions

        loss = loss_fn(y_pred, y)  # Compute loss
        train_loss += loss  # Accumulate training loss
        train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))  # Compute accuracy

        optimizer.zero_grad()  # Zero gradients from previous step
        loss.backward()  # Backward pass: compute gradients
        optimizer.step()  # Update model parameters

    # Average the training loss and accuracy over the number of batches
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")  # Print training metrics


# Function to perform a single testing step
def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    test_loss, test_acc = 0, 0  # Initialize test loss and accuracy
    model.to(device)  # Move model to device
    model.eval()  # Set model to evaluation mode
    with torch.inference_mode():  # Disable gradient tracking
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)  # Move batch data to device
            
            test_pred = model(X)  # Forward pass: compute predictions
            
            test_loss += loss_fn(test_pred, y)  # Compute test loss
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))  # Compute accuracy
        
        # Average the test loss and accuracy over the number of batches
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")  # Print testing metrics


# Function to evaluate the model on a dataset
def eval_model(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               accuracy_fn,
               device: torch.device = device):

    loss, acc = 0, 0  # Initialize total loss and accuracy
    model.to(device)  # Move model to device
    model.eval()  # Set model to evaluation mode
    with torch.inference_mode():  # Disable gradient tracking
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)  # Move batch data to device
            
            y_pred = model(X)  # Forward pass: compute predictions
            
            loss += loss_fn(y_pred, y).item()  # Accumulate loss
            acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))  # Accumulate accuracy
        
        # Average the loss and accuracy over the number of batches
        loss /= len(data_loader)
        acc /= len(data_loader)
    
    return {"model_name": model.__class__.__name__,
            "model_loss": loss,
            "model_acc": acc}  # Return evaluation metrics


# Function to make predictions on a list of samples
def make_predictions(model: torch.nn.Module, data: list, device: torch.device = device):
    pred_probs = []  # List to hold prediction probabilities
    model.eval()  # Set model to evaluation mode
    with torch.inference_mode():  # Disable gradient tracking
        for sample in data:
            sample = torch.unsqueeze(sample, dim=0).to(device)  # Add batch dimension and move to device

            pred_logit = model(sample)  # Forward pass: compute logits

            # Apply softmax to get probabilities
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)  # Perform softmax on the logits dimension

            pred_probs.append(pred_prob.cpu())  # Append predictions to the list
            
    return torch.stack(pred_probs)  # Return stacked probabilities


# Load the FashionMNIST dataset
train_data = datasets.FashionMNIST(root='data', train=True, download=True,
                                   transform=ToTensor(), target_transform=None)

test_data = datasets.FashionMNIST(root='data', train=False, download=True,
                                   transform=ToTensor())

# Set batch size for DataLoader
BATCH_SIZE = 32
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)  # Shuffle training data
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)  # No shuffle for test data
class_names = train_data.classes  # Get class names for labeling predictions

# Initialize the CNN model
model_2 = CNN_Model(input_shape=1, hidden_units=10, output_shape=len(class_names)).to(device)

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()  # For multi-class classification
optimizer = torch.optim.SGD(params=model_2.parameters(), lr=0.1)  # Stochastic Gradient Descent

# Start timer for training duration
train_time_start_model_2 = timer()

# Set number of epochs
epochs = 3
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")  # Print current epoch
    # Train the model for one epoch
    train_step(data_loader=train_dataloader, 
        model=model_2, 
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn,
        device=device
    )
    # Test the model after training
    test_step(data_loader=test_dataloader,
        model=model_2,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        device=device
    )

# End timer for training duration
train_time_end_model_2 = timer()
# Print total training time
total_train_time_model_2 = print_train_time(start=train_time_start_model_2,
                                           end=train_time_end_model_2,
                                           device=device)

# Evaluate the model on the test dataset
model_2_results = eval_model(
    model=model_2,
    data_loader=test_dataloader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn
)
print(model_2_results)  # Print evaluation results

# Randomly sample 9 test images for prediction visualization
test_samples = []
test_labels = []
for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)

# Make predictions for the sampled images
pred_probs= make_predictions(model=model_2, data=test_samples)
pred_classes = pred_probs.argmax(dim=1)  # Get predicted class indices

# Set up a plot for visualizing predictions
plt.figure(figsize=(9, 9))
nrows = 3
ncols = 3
for i, sample in enumerate(test_samples):
  plt.subplot(nrows, ncols, i+1)

  plt.imshow(sample.squeeze(), cmap="gray")  # Display image

  pred_label = class_names[pred_classes[i]]  # Get predicted label

  truth_label = class_names[test_labels[i]]  # Get true label 

  title_text = f"Pred: {pred_label} | Truth: {truth_label}"  # Prepare title text
  
  # Color the title based on prediction accuracy
  if pred_label == truth_label:
      plt.title(title_text, fontsize=10, c="g")  # green text if correct
  else:
      plt.title(title_text, fontsize=10, c="r")  # red text if wrong
  plt.axis(False)  # Hide axis ticks
plt.show()  # Show the plot
