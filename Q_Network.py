import torch
from torch import nn
from torch.utils.data import DataLoader


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        x = self.flatten(x)
        output = self.linear_relu_stack(x)
        return output
