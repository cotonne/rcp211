
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
        # TODO: USe LeakyReLU?
        # cf. https://github.com/deepmind/dqn/blob/master/dqn/convnet_atari3.lua
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
               nn.Linear(in_features=64 * 11 * 10, out_features=512))
        
        self.mu_layer = nn.Linear(512, action_size)     
        # self.log_std_layer = nn.Linear(512, action_size)
        initialize_uniformly(self.mu_layer)
        # initialize_uniformly(self.log_std_layer)

    def forward(self, state: torch.Tensor):
        """Forward method implementation."""
        x = state
        x = self.econv1(x)
        x = self.econv2(x)
        x = self.econv3(x)
        x = self.eline1(x)
        x = F.relu(x)
        # mu = (F.softmax(self.mu_layer(x)).squeeze(0))
        # action = torch.argmax(mu)
        # return action, torch.log(mu).sum()
        mu = F.softmax(self.mu_layer(x), dim=-1).squeeze(0)# * 2
        # To avoid value explosion with ReLU/SoftPlus, use Tanh to limit the value
        # log_std = F.softmax(self.log_std_layer(x))
        # log_std = F.softplus(self.log_std_layer(x), threshold=2)
        # std = torch.exp(log_std)
        # dist = Normal(mu, std)
        # action = dist.sample()
        # log_prob = dist.log_prob(action).sum(dim=-1)
        # prob = F.softmax(mu, dim=-1)
        action = mu.multinomial(num_samples=1).detach()
        # print(mu)
        log_prob = F.log_softmax(mu, dim=-1).sum(dim=-1)
        return action, log_prob
