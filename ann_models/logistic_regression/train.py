import torch
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from logistic_regression import LogisticRegression

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = torch.tensor(X, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36)

plt.figure(figsize=(10, 6))
plt.scatter(X_train[:, 0].cpu(), X_train[:, 1].cpu(), c=y_train.cpu(), cmap='coolwarm', edgecolors='k', s=50, label='Train')
plt.scatter(X_test[:, 0].cpu(), X_test[:, 1].cpu(), c=y_test.cpu(), cmap='coolwarm', s=50, marker='x', label='Test')
plt.title('Training Data Visualization')
plt.legend()
plt.show()

model = LogisticRegression(in_features=X.shape[1]).to(device)
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
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
    print(f"\nTest Loss: {test_loss.item():.4f}")

    predicted_classes = (test_pred >= 0.5).float()
    accuracy = (predicted_classes == y_test).float().mean()
    print(f'Accuracy: {accuracy * 100:.2f}%')
