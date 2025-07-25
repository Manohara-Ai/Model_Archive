import torch 
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from linear_regression import LinearRegression

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=25)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = torch.tensor(X, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36)

plt.figure(figsize=(10, 5))
plt.scatter(X_train[:, 0].cpu(), y_train.cpu(), color='crimson', label='Training Data')
plt.scatter(X_test[:, 0].cpu(), y_test.cpu(), color='blue', label='Test Data')
plt.legend()
plt.title("Train vs Test Data")
plt.show()

model = LinearRegression().to(device)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
w_lambda = 0.0

epochs = 500
for epoch in range(epochs):
    model.train()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train) + w_lambda * torch.sum(model.linear.weight** 2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Train Loss = {loss.item():.4f}")

model.eval()
with torch.inference_mode():
    test_pred = model(X_test)
    test_loss = loss_fn(test_pred, y_test)
    print(f"\nFinal Test Loss: {test_loss.item():.4f}")

pred_line = model(X)
plt.figure(figsize=(10, 5))
plt.scatter(X.cpu(), y.cpu(), color='gray', label='Data Points', alpha=0.6)
plt.plot(X.cpu().detach().numpy(), pred_line.cpu().detach().numpy(), color='green', label='ANN Prediction')
plt.legend()
plt.title("Linear Regression Prediction")
plt.show()