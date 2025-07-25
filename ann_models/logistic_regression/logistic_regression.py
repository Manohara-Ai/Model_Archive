import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, in_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, X):
        return self.linear(X)