import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Critic(nn.Module):
    def __init__(self):
        """Initialize."""
        super(Critic, self).__init__()
        
        self.econv1 = nn.Sequential(
               nn.Conv2d(kernel_size=4, in_channels=1, out_channels=32, stride=2, padding=1),
               nn.ReLU())
        self.econv2 = nn.Sequential(
               nn.Conv2d(kernel_size=4, in_channels=32, out_channels=64, stride=2, padding=1),
               nn.ReLU())
        self.out = nn.Sequential(
               nn.Flatten(),
               nn.Linear(in_features=64 * 23 * 21, out_features=1))
        # initialize_uniformly(self.out)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = state
        x = self.econv1(x)
        x = self.econv2(x)
        x = self.out(x)
        x = F.relu(x)
        
        return x
