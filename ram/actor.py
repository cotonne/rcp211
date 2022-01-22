
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

def initialize_uniformly(layer: nn.Linear, init_w: float = 3e-3):
    """Initialize the weights and bias in [-init_w, init_w]."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

class Actor(nn.Module):
    def __init__(self, action_size: int):
        """Initialize."""
        super(Actor, self).__init__()
        self.hidden1 = nn.Sequential(
               nn.Linear(in_features=128, out_features=64),
               nn.ReLU())
        self.hidden2 = nn.Sequential(
               nn.Linear(in_features=64, out_features=32),
               nn.ReLU())
        self.mu_layer = nn.Linear(32, action_size)
        # self.log_std_layer = nn.Linear(512, action_size)
        initialize_uniformly(self.mu_layer)
        # initialize_uniformly(self.log_std_layer)

    def forward(self, state: torch.Tensor):
        """Forward method implementation."""
        x = state
        x = self.hidden1(x)
        x = self.hidden2(x)
        mu = F.softmax(self.mu_layer(x), dim=-1).squeeze(0)# * 2
        action = mu.multinomial(num_samples=1).detach()
        log_prob = F.log_softmax(mu, dim=-1).sum(dim=-1)
        return action, log_prob
