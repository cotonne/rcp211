import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Critic(nn.Module):
    def __init__(self):
        """Initialize."""
        super(Critic, self).__init__()
        self.hidden1 = nn.Sequential(
               nn.Linear(in_features=128, out_features=64),
               nn.ReLU())
        self.hidden2 = nn.Sequential(
               nn.Linear(in_features=64, out_features=32),
               nn.ReLU())
        self.out = nn.Linear(32, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = state
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)

        return x
