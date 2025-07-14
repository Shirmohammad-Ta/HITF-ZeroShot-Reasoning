import torch
import torch.nn as nn
import torch.nn.functional as F

class Selector(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256):
        super(Selector, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        lambda_weight = torch.sigmoid(self.fc2(h))
        return lambda_weight.squeeze()
