import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Critic(nn.Module):
    def __init__(self):
        """Initialize."""
        super(Critic, self).__init__()
        
        # initialize_uniformly(self.out)
        self.econv1 = nn.Sequential(
               nn.Conv2d(kernel_size=8, in_channels=1, out_channels=32, stride=4, padding=1),
               nn.ReLU())
        self.econv2 = nn.Sequential(
               nn.Conv2d(kernel_size=4, in_channels=32, out_channels=64, stride=2, padding=1),
               nn.ReLU())
        self.econv3 = nn.Sequential(
               nn.Conv2d(kernel_size=3, in_channels=64, out_channels=64, stride=1, padding=1),
               nn.ReLU())
        self.eline1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64 * 11 * 10, out_features=512),
               nn.ReLU())
        self.eline2 = nn.Sequential(
               nn.Flatten(),
               nn.Linear(in_features=512, out_features=128),
               nn.ReLU())
        self.eline3 = nn.Sequential(
               nn.Flatten(),
               nn.Linear(in_features=128, out_features=32),
               nn.ReLU())

        self.out = nn.Linear(32, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = state
        x = self.econv1(x)
        x = self.econv2(x)
        x = self.econv3(x)
        x = self.eline1(x)
        x = self.eline2(x)
        x = self.eline3(x)
        x = self.out(x)

        return x
