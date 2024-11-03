import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Define the first linear layer with input size and hidden size
        self.linear1 = nn.Linear(input_size, hidden_size)
        # Define the second linear layer with hidden size and output size
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Apply the first linear transformation followed by ReLU activation
        x = F.relu(self.linear1(x))
        # Apply the second linear transformation to produce the output
        x = self.linear2(x)     
        return x

    def save(self, file_name='model.pth'):
        # Create a directory to save the model if it doesn't exist
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        # Save the model state dictionary to the specified file
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        # Learning rate for the optimizer
        self.lr = lr
        # Discount factor for future rewards
        self.gamma = gamma
        # Initialize the model
        self.model = model
        # Use Adam optimizer for training
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # Mean Squared Error loss for training
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # Convert numpy arrays to PyTorch tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # Check if the state has only one dimension (single instance)
        if len(state.shape) == 1:
            # Unsqueeze to add batch dimension
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )  # Convert to a tuple for consistency

        # 1: Get the predicted Q values for the current state
        pred = self.model(state)

        # Clone the predicted values to create the target Q values
        target = pred.clone()
        for idx in range(len(done)):
            # Calculate the new Q value from the reward
            Q_new = reward[idx]
            if not done[idx]:
                # If the state is not terminal, add discounted future reward
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            # Update the target for the action taken
            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Perform a gradient descent step to minimize the loss
        self.optimizer.zero_grad()  # Clear previous gradients
        # Calculate the loss between target and predicted Q values
        loss = self.criterion(target, pred)
        loss.backward()  # Backpropagation to calculate gradients

        self.optimizer.step()  # Update the model parameters
