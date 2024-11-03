import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torchvision
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

import os
import zipfile

from pathlib import Path

import requests

from typing import List

def set_seeds(random_seed: int=42, torch_seed: int=42):
    """Sets random seeds for various operations to ensure reproducibility.

    Args:
        random_seed (int, optional): Random seed to set for random operations. Defaults to 42.
        torch_seed (int,  optional): Random seed to set for random torch operations. Defaults to 42.
    """
    # Set the seed for Python's built-in random module
    random.seed(random_seed)
    # Set the seed for torch operations on the CPU
    torch.manual_seed(torch_seed)
    # Set the seed for torch operations on the GPU (CUDA)
    torch.cuda.manual_seed(torch_seed)

def download_data(source: str, 
                  destination: str,
                  remove_source: bool = True) -> Path:
    """Downloads a zipped dataset from source and unzips to destination.

    Args:
        source (str): A link to a zipped file containing data.
        destination (str): A target directory to unzip data to.
        remove_source (bool): Whether to remove the source after downloading and extracting.
    
    Returns:
        pathlib.Path to downloaded data.
    
    Example usage:
        download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                      destination="pizza_steak_sushi")
    """
    # Setup path to data folder
    data_path = Path("data/")
    image_path = data_path / destination

    # If the image folder doesn't exist, download it and prepare it... 
    if image_path.is_dir():
        print(f"[INFO] {image_path} directory exists, skipping download.")
    else:
        print(f"[INFO] Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)
        
        # Download the files
        target_file = Path(source).name
        with open(data_path / target_file, "wb") as f:
            request = requests.get(source)
            print(f"[INFO] Downloading {target_file} from {source}...")
            f.write(request.content)

        # Unzip the files
        with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
            print(f"[INFO] Unzipping {target_file} data...") 
            zip_ref.extractall(image_path)

        # Remove .zip file
        if remove_source:
            os.remove(data_path / target_file)
    
    return image_path

def print_train_time(start, end, device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to cuda if available

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time

def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between true labels and predictions.

    Args:
        y_true (torch.Tensor): Ground truth labels.
        y_pred (torch.Tensor): Predicted labels.

    Returns:
        float: Accuracy percentage
    """
    correct = torch.eq(y_true, y_pred).sum().item()  # Count correct predictions
    acc = (correct / len(y_pred)) * 100  # Calculate accuracy percentage
    return acc

def plot_loss_curves(results):
    """Plots training and validation loss and accuracy curves.

    Args:
        results (dict): Dictionary containing lists of values
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))  # Number of epochs

    plt.figure(figsize=(10, 5))

    # Plot loss curves
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Plots a confusion matrix using Seaborn heatmap.

    Args:
        y_true (torch.Tensor or list): Ground truth labels.
        y_pred (torch.Tensor or list): Predicted labels.
        class_names (list, optional): List of class names to display on the axes.

    Returns:
        plt.figure: A confusion matrix figure.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot the confusion matrix as a heatmap
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=class_names, yticklabels=class_names)

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.xticks(rotation=45, fontsize=14)
    plt.yticks(rotation=0, fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_decision_boundary(model, X, y, class_names, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Plots the decision boundary of a model across multiple features using PCA for dimensionality reduction.

    Args:
        model (torch.nn.Module): The trained model.
        X (torch.Tensor): The input features.
        y (torch.Tensor): The true labels.
        class_names (list): List of class names.
        device (str, optional): The device to use for computation. Defaults to 'cuda' if available.
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Convert the data to the appropriate device
    X, y = X.to(device), y.to(device)

    # Apply PCA to reduce the dimensionality to 2D for visualization
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X.cpu().numpy())
    
    # Define the grid for plotting
    x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
    y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    
    # Prepare the grid data and project it to the original space using inverse PCA transform
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_original = pca.inverse_transform(grid)
    grid_tensor = torch.from_numpy(grid_original).float().to(device)
    
    # Get model predictions for each point in the grid
    with torch.inference_mode():
        model_preds = model(grid_tensor)
        zz = model_preds.argmax(dim=1).cpu().numpy().reshape(xx.shape)
    
    plt.figure(figsize=(7, 5))
    # Plot the decision boundary
    plt.contourf(xx, yy, zz, cmap=plt.cm.Spectral, alpha=0.8)
    
    # Scatter plot the original data points with colors according to their true labels
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y.cpu(), cmap=plt.cm.Spectral, edgecolors='k')
    
    # Add a legend with class names
    plt.legend(handles=scatter.legend_elements()[0], labels=class_names)
    
    plt.title("Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.show()

def plot_pred_image(
    model: torch.nn.Module,
    image_path: str,
    class_names: List[str] = None,
    transform=None,
    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"
):
    """Makes a prediction on a target image with a trained model and plots the image.

    Args:
        model (torch.nn.Module): trained PyTorch image classification model.
        image_path (str): filepath to target image.
        class_names (List[str], optional): different class names for target image. Defaults to None.
        transform (_type_, optional): transform of target image. Defaults to None.
        device (torch.device, optional): target device to compute on. Defaults to "cuda" if torch.cuda.is_available() else "cpu".
    
    Returns:
        Matplotlib plot of target image and model prediction as title.

    Example usage:
        pred_and_plot_image(model=model,
                            image="some_image.jpeg",
                            class_names=["class_1", "class_2", "class_3"],
                            transform=torchvision.transforms.ToTensor(),
                            device=device)
    """

    # 1. Load in image and convert the tensor values to float32
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    # 2. Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255.0

    # 3. Transform if necessary
    if transform:
        target_image = transform(target_image)

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image.to(device))

    # 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 7. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 8. Plot the image alongside the prediction and prediction probability
    plt.figure(figsize=(7, 5))
    plt.imshow(
        target_image.squeeze().permute(1, 2, 0)
    )  # make sure it's the right size for matplotlib
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else:
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)
    plt.show()

def train_and_eval(model: torch.nn.Module, 
                   train_loader: torch.utils.data.DataLoader, 
                   test_loader: torch.utils.data.DataLoader, 
                   loss_fn: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer, 
                   accuracy_fn, 
                   epochs: int,
                   device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Trains and evaluates the model for a specified number of epochs, returning performance metrics.

    Args:
        model (torch.nn.Module): The model to train and evaluate.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        test_loader (torch.utils.data.DataLoader): DataLoader for testing data.
        loss_fn (torch.nn.Module): Loss function to calculate loss.
        optimizer (torch.optim.Optimizer): Optimizer to update model weights.
        accuracy_fn (function): Function to calculate accuracy.
        epochs (int): Number of epochs to train for.
        device (torch.device): Device to perform computations on (CPU or GPU).
    
    Returns:
        dict: Dictionary containing lists of train and test losses, accuracies, and predictions for each epoch.
    """
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "test_predictions": [],
        "test_labels": []
    }

    model.to(device)  # Move the model to the target device

    for epoch in range(epochs):
        # Training step
        model.train()  # Set model to training mode
        train_loss, train_acc = 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)  # Move data to the target device

            # Forward pass: compute predictions
            y_pred = model(X)

            # Calculate the loss
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()  # Accumulate the loss
            train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))  # Accumulate accuracy

            optimizer.zero_grad()  # Zero the gradients before backward pass

            loss.backward()  # Backward pass: compute gradients

            optimizer.step()  # Update model parameters

        # Compute average training loss and accuracy over the entire dataset
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        # Testing step
        model.eval()  # Set model to evaluation mode
        test_loss, test_acc = 0, 0
        test_preds = []
        test_labels = []
        
        with torch.inference_mode():  # Disable gradient computation for efficiency
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)  # Move data to the target device

                # Forward pass: compute predictions
                test_pred = model(X)

                # Calculate and accumulate the loss and accuracy
                test_loss += loss_fn(test_pred, y).item()
                test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
                
                # Store predictions and true labels
                test_preds.append(test_pred.argmax(dim=1).cpu())
                test_labels.append(y.cpu())

            # Compute average testing loss and accuracy over the entire dataset
            test_loss /= len(test_loader)
            test_acc /= len(test_loader)

        # Concatenate all predictions and labels into single tensors
        results["test_predictions"] = torch.cat(test_preds)
        results["test_labels"] = torch.cat(test_labels)

        # Store the results for this epoch
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        # Print the results for this epoch
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

    return results


def test_model(model: torch.nn.Module, 
               X: torch.Tensor, 
               device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"):
    """Evaluates the model and returns the predictions using the given input data X.

    Args:
        model (torch.nn.Module): The model to evaluate.
        X (torch.Tensor): Input data to evaluate the model on.
        device (torch.device): Device to perform the computations on (CPU or GPU).

    Returns:
        dict: Dictionary containing model name and predictions.
    """
    model.to(device)  # Move the model to the target device
    model.eval()  # Set model to evaluation mode

    with torch.inference_mode():  # Disable gradient computation for efficiency
        X = X.to(device)  # Move data to the target device
        
        # Forward pass: compute predictions
        y_pred = model(X)
        
        # Store predictions
        all_preds = y_pred.argmax(dim=1).cpu()

    # Return results as a dictionary
    return {
        "model_name": model.__class__.__name__,
        "predictions": all_preds
    }
