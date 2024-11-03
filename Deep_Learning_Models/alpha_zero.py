import numpy as np
import math
import random

from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):
    def __init__(self, game, num_resBlocks, num_hidden, device):
        super().__init__()

        # Store the device (CPU/GPU) for model operations
        self.device = device
        
        # Initial convolutional block that processes the input image
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),  # Convolutional layer
            nn.BatchNorm2d(num_hidden),                          # Batch normalization for stable training
            nn.ReLU()                                            # Activation function
        )
        
        # List of residual blocks forming the backbone of the network
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]  # Create specified number of ResBlocks
        )
        
        # Policy head for predicting actions
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),  # Convolutional layer
            nn.BatchNorm2d(32),                                    # Batch normalization
            nn.ReLU(),                                            # Activation function
            nn.Flatten(),                                         # Flatten the output for linear layer
            nn.Linear(32 * game.row_count * game.column_count, game.action_size)  # Fully connected layer
        )
        
        # Value head for predicting the value of the state
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),   # Convolutional layer
            nn.BatchNorm2d(3),                                     # Batch normalization
            nn.ReLU(),                                            # Activation function
            nn.Flatten(),                                         # Flatten the output for linear layer
            nn.Linear(3 * game.row_count * game.column_count, 1),  # Fully connected layer
            nn.Tanh()                                            # Output activation for value head
        )
        
        # Move the model to the specified device
        self.to(device)

    def forward(self, x):
        # Forward pass through the initial block
        x = self.startBlock(x)
        
        # Forward pass through all residual blocks
        for resBlock in self.backBone:
            x = resBlock(x)
        
        # Generate policy and value predictions
        policy = self.policyHead(x)
        value = self.valueHead(x)
        
        return policy, value  # Return both policy and value outputs
        
        
class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        
        # Two convolutional layers with batch normalization for residual learning
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        
    def forward(self, x):
        residual = x  # Store the input for the skip connection
        
        # Forward pass through the first convolutional layer with ReLU activation
        x = F.relu(self.bn1(self.conv1(x)))
        # Forward pass through the second convolutional layer
        x = self.bn2(self.conv2(x))
        
        # Add the residual connection
        x += residual
        # Apply ReLU activation after adding the residual
        x = F.relu(x)
        
        return x  # Return the output of the residual block
 
class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0):
        # Initialize the node with game state, arguments, parent node, and other attributes
        self.game = game  # The game environment
        self.args = args  # Configuration or hyperparameters for the node
        self.state = state  # Current game state represented by the node
        self.parent = parent  # Parent node in the search tree
        self.action_taken = action_taken  # Action taken to reach this node
        self.prior = prior  # Prior probability of reaching this node
        
        self.children = []  # List to store child nodes
        
        self.visit_count = visit_count  # Number of times this node has been visited
        self.value_sum = 0  # Sum of values from backpropagation
        
    def is_fully_expanded(self):
        # Check if the node has any children (i.e., if it has been fully expanded)
        return len(self.children) > 0
    
    def select(self):
        # Select the child node with the highest Upper Confidence Bound (UCB) value
        best_child = None  # Initialize the best child
        best_ucb = -np.inf  # Initialize the best UCB value to negative infinity
        
        for child in self.children:
            ucb = self.get_ucb(child)  # Calculate the UCB for the child
            if ucb > best_ucb:  # Update best child if current UCB is greater
                best_child = child
                best_ucb = ucb
                
        return best_child  # Return the child with the highest UCB
    
    def get_ucb(self, child):
        # Calculate the Upper Confidence Bound (UCB) for a given child node
        if child.visit_count == 0:
            q_value = 0  # If the child has not been visited, set Q-value to 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2  # Compute the average value
        
        # UCB formula: Q-value + exploration term
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior
    
    def expand(self, policy):
        # Expand the node by adding child nodes based on the given policy
        for action, prob in enumerate(policy):
            if prob > 0:  # Only consider actions with a non-zero probability
                child_state = self.state.copy()  # Create a copy of the current state
                child_state = self.game.get_next_state(child_state, action, 1)  # Get the next state after the action
                child_state = self.game.change_perspective(child_state, player=-1)  # Change perspective for the opponent

                # Create a new child node and append it to the children list
                child = Node(self.game, self.args, child_state, self, action, prob)
                self.children.append(child)
                
        return child  # Return the last added child node
            
    def backpropagate(self, value):
        # Backpropagate the value from this node up to its parent nodes
        self.value_sum += value  # Update the sum of values for this node
        self.visit_count += 1  # Increment the visit count
        
        # Get the opponent's value and backpropagate if a parent exists
        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)  # Recursively backpropagate to the parent

class MCTS:
    def __init__(self, game, args, model):
        # Initialize MCTS with the game environment, arguments, and the model
        self.game = game  # The game environment
        self.args = args  # Configuration or hyperparameters for MCTS
        self.model = model  # The neural network model used for predictions
        
    @torch.no_grad()  # Disable gradient calculations for efficiency during inference
    def search(self, state):
        # Perform the MCTS search algorithm for a given game state
        root = Node(self.game, self.args, state, visit_count=1)  # Create the root node
        
        # Get the initial policy from the model
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(state), device=self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()  # Apply softmax to get probabilities
        
        # Apply Dirichlet noise for exploration
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)
        
        valid_moves = self.game.get_valid_moves(state)  # Get valid moves for the current state
        policy *= valid_moves  # Mask the policy with valid moves
        policy /= np.sum(policy)  # Normalize the policy
        root.expand(policy)  # Expand the root node with the computed policy
        
        # Perform the specified number of searches
        for search in range(self.args['num_searches']):
            node = root  # Start from the root node
            
            # Traverse down the tree until a node that is not fully expanded is found
            while node.is_fully_expanded():
                node = node.select()
                
            # Get the value and check if the state is terminal
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)  # Get opponent's value
            
            if not is_terminal:  # If the state is not terminal, expand further
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state), device=self.model.device).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()  # Apply softmax
                valid_moves = self.game.get_valid_moves(node.state)  # Get valid moves for the node
                policy *= valid_moves  # Mask the policy with valid moves
                policy /= np.sum(policy)  # Normalize the policy
                
                value = value.item()  # Convert value to a scalar
                
                node.expand(policy)  # Expand the node with the computed policy
                
            node.backpropagate(value)  # Backpropagate the value up the tree
            
        # Prepare action probabilities based on visit counts of child nodes
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)  # Normalize the action probabilities
        return action_probs  # Return the action probabilities


class MCTSParallel:
    def __init__(self, game, args, model):
        # Initialize MCTSParallel with the game environment, arguments, and the model
        self.game = game  # The game environment
        self.args = args  # Configuration or hyperparameters for MCTS
        self.model = model  # The neural network model used for predictions
        
    @torch.no_grad()  # Disable gradient calculations for efficiency during inference
    def search(self, states, spGames):
        # Perform parallel MCTS search for multiple game states
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
        )
        policy = torch.softmax(policy, axis=1).cpu().numpy()  # Apply softmax to get probabilities
        
        # Apply Dirichlet noise for exploration for each state
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size, size=policy.shape[0])
        
        # Initialize root nodes for each self-play game
        for i, spg in enumerate(spGames):
            spg_policy = policy[i]  # Get the policy for the current self-play game
            valid_moves = self.game.get_valid_moves(states[i])  # Get valid moves for the current state
            spg_policy *= valid_moves  # Mask the policy with valid moves
            spg_policy /= np.sum(spg_policy)  # Normalize the policy

            # Create the root node for the self-play game
            spg.root = Node(self.game, self.args, states[i], visit_count=1)
            spg.root.expand(spg_policy)  # Expand the root node with the computed policy
        
        # Perform the specified number of searches
        for search in range(self.args['num_searches']):
            for spg in spGames:
                spg.node = None  # Reset the node for the self-play game
                node = spg.root  # Start from the root node

                # Traverse down the tree until a node that is not fully expanded is found
                while node.is_fully_expanded():
                    node = node.select()

                # Get the value and check if the state is terminal
                value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
                value = self.game.get_opponent_value(value)  # Get opponent's value
                
                if is_terminal:  # If the state is terminal, backpropagate the value
                    node.backpropagate(value)
                else:
                    spg.node = node  # Store the current node for future processing
                    
            # Get the indices of self-play games that can be expanded
            expandable_spGames = [mappingIdx for mappingIdx in range(len(spGames)) if spGames[mappingIdx].node is not None]
                    
            if len(expandable_spGames) > 0:  # If there are expandable games
                states = np.stack([spGames[mappingIdx].node.state for mappingIdx in expandable_spGames])  # Get states for expansion
                
                # Get policies and values for the expandable states from the model
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
                )
                policy = torch.softmax(policy, axis=1).cpu().numpy()  # Apply softmax
                value = value.cpu().numpy()  # Move value to CPU for processing
                
            # Expand the nodes for the expandable self-play games
            for i, mappingIdx in enumerate(expandable_spGames):
                node = spGames[mappingIdx].node  # Get the current node
                spg_policy, spg_value = policy[i], value[i]  # Get the corresponding policy and value
                
                valid_moves = self.game.get_valid_moves(node.state)  # Get valid moves for the node
                spg_policy *= valid_moves  # Mask the policy with valid moves
                spg_policy /= np.sum(spg_policy)  # Normalize the policy

                node.expand(spg_policy)  # Expand the node with the computed policy
                node.backpropagate(spg_value)  # Backpropagate the value

class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        # Initialize the AlphaZero algorithm with the model, optimizer, game environment, and parameters
        self.model = model  # The neural network model used for predictions
        self.optimizer = optimizer  # The optimizer for training the model
        self.game = game  # The game environment
        self.args = args  # Configuration or hyperparameters for AlphaZero
        self.mcts = MCTSParallel(game, args, model)  # Initialize the parallel MCTS instance
        
    def selfPlay(self):
        # Perform self-play to generate training data
        return_memory = []  # List to store the memory of self-play games
        player = 1  # Start with player 1
        spGames = [SPG(self.game) for _ in range(self.args['num_parallel_games'])]  # Create self-play games
        
        while len(spGames) > 0:  # While there are ongoing self-play games
            states = np.stack([spg.state for spg in spGames])  # Get the current states of all self-play games
            neutral_states = self.game.change_perspective(states, player)  # Change perspective for the current player
            
            self.mcts.search(neutral_states, spGames)  # Search for optimal actions using MCTS
            
            for i in range(len(spGames))[::-1]:  # Iterate backwards to safely remove finished games
                spg = spGames[i]
                
                action_probs = np.zeros(self.game.action_size)  # Initialize action probabilities
                for child in spg.root.children:  # Aggregate visit counts from child nodes
                    action_probs[child.action_taken] = child.visit_count
                
                total_visit_count = np.sum(action_probs)  # Total visits to children
                if total_visit_count > 0:
                    action_probs /= total_visit_count  # Normalize action probabilities
                
                spg.memory.append((spg.root.state, action_probs, player))  # Store state, action probabilities, and player

                # Apply temperature for exploration
                temperature_action_probs = action_probs ** (1 / self.args['temperature'])  # Apply temperature scaling
                total_temp_prob = np.sum(temperature_action_probs)
                if total_temp_prob > 0:
                    temperature_action_probs /= total_temp_prob  # Normalize again
                
                # Sample an action based on the temperature-scaled probabilities
                action = np.random.choice(self.game.action_size, p=temperature_action_probs)

                # Get the next state based on the action taken
                spg.state = self.game.get_next_state(spg.state, action, player)

                # Get the value and check if the game state is terminal
                value, is_terminal = self.game.get_value_and_terminated(spg.state, action)

                if is_terminal:  # If the game is finished
                    # Store the outcome for all states in memory
                    for hist_neutral_state, hist_action_probs, hist_player in spg.memory:
                        hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                        return_memory.append((  # Add to return memory
                            self.game.get_encoded_state(hist_neutral_state),
                            hist_action_probs,
                            hist_outcome
                        ))
                    del spGames[i]  # Remove finished game from the list
                    
            player = self.game.get_opponent(player)  # Switch to the opponent player
            
        return return_memory  # Return the generated memory from self-play

    def train(self, memory):
        # Train the model using the collected memory from self-play
        random.shuffle(memory)  # Shuffle memory for stochastic training
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])]  # Sample a batch
            state, policy_targets, value_targets = zip(*sample)  # Unzip the sample into states, policies, and values
            
            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)
            
            # Convert arrays to tensors for training
            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)
            
            # Forward pass through the model
            out_policy, out_value = self.model(state)
            
            # Calculate loss for policy and value predictions
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss  # Total loss
            
            self.optimizer.zero_grad()  # Zero the gradients
            loss.backward()  # Backpropagate the loss
            self.optimizer.step()  # Update the model parameters
    
    def learn(self):
        # Main learning loop for AlphaZero
        for iteration in range(self.args['num_iterations']):
            memory = []  # Initialize memory for this iteration
            
            self.model.eval()  # Set model to evaluation mode
            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations'] // self.args['num_parallel_games']):
                memory += self.selfPlay()  # Perform self-play and add to memory
                
            self.model.train()  # Set model to training mode
            for epoch in trange(self.args['num_epochs']):
                self.train(memory)  # Train the model with the collected memory
            
        # Save the trained model parameters to a file
        torch.save(self.model.state_dict(), f"C:/Users/Manohara/Projects/Mahoraga/models/model_{self.game}.pth")

class SPG:
    def __init__(self, game):
        # Initialize a self-play game instance
        self.state = game.get_initial_state()  # Get the initial state of the game
        self.memory = []  # Memory to store states, action probabilities, and outcomes
        self.root = None  # Root node for MCTS
        self.node = None  # Current node during MCTS
