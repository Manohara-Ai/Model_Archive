import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self, in_features=1):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(in_features, 1) 

    def forward(self, X):
        return self.linear(X)
