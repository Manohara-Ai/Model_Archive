import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Data Generation
N = 100  # Number of sine waves
L = 1000  # Length of each wave
T = 20  # Period of sine wave

# Generate sine wave data with random phase shifts
x = np.empty((N, L), np.float32)
x[:] = np.array(range(L)) + np.random.randint(-4 + T, 4 + T, N).reshape(N, 1)
y = np.sin(x / 1.0 / T).astype(np.float32)

# LSTM Model Definition
class LSTMPredictor(nn.Module):
    def __init__(self, n_hidden=51):
        super().__init__()
        self.n_hidden = n_hidden
        self.lstm1 = nn.LSTMCell(1, self.n_hidden)
        self.lstm2 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, 1)

    def forward(self, x, future=0):
        outputs = []
        n_samples = x.size(0)

        # Initialize hidden states
        h_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32, device=x.device)
        c_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32, device=x.device)
        h_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32, device=x.device)
        c_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32, device=x.device)

        # Predict for input sequence
        for input_t in x.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)

        # Predict future sequence
        for _ in range(future):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)

        outputs = torch.cat(outputs, dim=1)
        return outputs

# Main Execution
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare data for training and testing
    train_input = torch.from_numpy(y[3:, :-1]).to(device)
    train_target = torch.from_numpy(y[3:, 1:]).to(device)
    test_input = torch.from_numpy(y[:3, :-1]).to(device)
    test_target = torch.from_numpy(y[:3, 1:]).to(device)

    # Initialize model, loss function, and optimizer
    model = LSTMPredictor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.LBFGS(model.parameters(), lr=0.8)

    # Lists to store losses
    train_loss = []
    test_loss = []

    print("Training in progress...")
    n_steps = 10

    # Training Loop
    for step in range(n_steps):
        def closure():
            optimizer.zero_grad()
            out = model(train_input)
            loss = criterion(out, train_target)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        train_loss.append(loss.item())

        # Evaluate on test set
        with torch.no_grad():
            future = 1000
            pred = model(test_input, future=future)
            loss = criterion(pred[:, :-future], test_target)
            test_loss.append(loss.item())

            print(f"Step {step + 1}/{n_steps}, Train Loss: {train_loss[step]:.6f}, Test Loss: {test_loss[step]:.6f}")

            def plot_series(y_pred, color):
                n = train_input.shape[1]
                plt.plot(np.arange(n), y_pred[:n], color, linewidth=2)
                plt.plot(np.arange(n, n + future), y_pred[n:], color + ":", linewidth=2)

    print("Training complete.")

    # Plot Loss Curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(train_loss, color='blue', linewidth=2)
    axes[0].set_title('Train Loss')
    axes[0].set_xlabel('Training Steps')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)

    axes[1].plot(test_loss, color='yellow', linewidth=2)
    axes[1].set_title('Test Loss')
    axes[1].set_xlabel('Training Steps')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    # Plot Predictions
    plt.figure(figsize=(10, 6))
    plt.title("Sine Wave Prediction")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    plot_series(pred[0].cpu().numpy(), 'r')  # First prediction
    plot_series(pred[1].cpu().numpy(), 'b')  # Second prediction
    plot_series(pred[2].cpu().numpy(), 'g')  # Third prediction

    plt.show()
