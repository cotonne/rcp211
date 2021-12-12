#!/usr/bin/env python
import numpy as np
from ale_py import ALEInterface, SDL_SUPPORT
from ale_py.roms import Pacman

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torchvision import transforms



torch.autograd.set_detect_anomaly(True)

def initialize_uniformly(layer: nn.Linear, init_w: float = 3e-3):
    """Initialize the weights and bias in [-init_w, init_w]."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

weights_luminance = torch.tensor([0.2126, 0.7152, 0.0722])
p = transforms.Resize(size=84)
trans = transforms.ToPILImage()
trans1 = transforms.ToTensor()

def preprocessing(previous_state, state):
    # Remove flickering by taking the maximum value of each pixel colour
    state = torch.max(torch.stack([previous_state, state]), dim=0).values
    # crop the image, we don't need the header and the score bar
    state = state[20:-50,:,:]
    # Resize to 84x84
    state = trans1(p(trans(state.permute(2, 0, 1)))).permute(1, 2, 0)
    # Extract luminance (https://en.wikipedia.org/wiki/Relative_luminance)
    state = (state * weights_luminance).sum(-1)
    return state

class Actor(nn.Module):
    def __init__(self, action_size: int):
        """Initialize."""
        super(Actor, self).__init__()
        self.econv1 = nn.Sequential(
               nn.Conv2d(kernel_size=4, in_channels=1, out_channels=32, stride=4, padding=1),
               nn.ReLU())
        self.econv2 = nn.Sequential(
               nn.Conv2d(kernel_size=4, in_channels=32, out_channels=64, stride=2, padding=1),
               nn.ReLU())
        self.econv3 = nn.Sequential(
               nn.Conv2d(kernel_size=4, in_channels=64, out_channels=64, stride=1, padding=1),
               nn.ReLU())
        self.eline1 = nn.Sequential(
               nn.Flatten(),
               nn.Linear(in_features=64 * 11 * 9, out_features=512))
        
        self.mu_layer = nn.Linear(512, action_size)     
        self.log_std_layer = nn.Linear(512, action_size) 

        # initialize_uniformly(self.mu_layer)
        # initialize_uniformly(self.log_std_layer)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = state
        # torch.Size([1, 1, 250, 160])
        x = self.econv1(x)
        # print(x.min(), x.max())
        # print(x)
        # torch.Size([1, 32, 125, 80])
        x = self.econv2(x)
        x = self.econv3(x)
        # print(x.shape)
        # torch.Size([1, 64, 62, 40])
        x = self.eline1(x)
        x = F.relu(x)
        mu = torch.tanh(self.mu_layer(x)) * 2
        # To avoid value explosion with ReLU, use Sigmoid to limit the value
        log_std = torch.sigmoid(self.log_std_layer(x))
        std = torch.exp(log_std)
        
        dist = Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return np.argmax(action[3]).item(), log_prob

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


system = ALEInterface()
γ = 0.9



# # Get & Set the desired settings
# system.setInt("random_seed", 123)

# # Check if we can display the screen
if SDL_SUPPORT:
    system.setBool("sound", True)
    system.setBool("display_screen", True)

# Load the ROM file
system.loadROM(Pacman)

# Get the list of legal actions
legal_actions = system.getLegalActionSet()

# To define
V_critic = Critic()
V_critic.train()
critic_optimizer = optim.Adam(V_critic.parameters(), lr=1e-3)
actor = Actor(action_size=len(legal_actions))
actor.train()
actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)

previous_state = torch.zeros(250, 160, 3)

batch = torch.zeros(4, 1, 94, 84)

# Play 10 episodes
for episode in range(10):
    total_reward = 0
    state = torch.from_numpy(system.getScreenRGB())

    # Shift images
    batch = torch.roll(batch, 3, 0) 
    batch[3] = preprocessing(previous_state, state)
    previous_state = state

    while not system.game_over():
        action, log_prob = actor(batch)
        a_t = legal_actions[action]
        print("Action = ")
        print(a_t)
        # Apply an action and get the resulting reward
        # State?
        # Perform action a_t accoring to policy π(a_t / s_t; θ)
        # Receive reward r_t 
        r_t = system.act(a_t)
        state_t_plus_1 = torch.from_numpy(system.getScreenRGB())
        batch = torch.roll(batch, 3, 0) 
        batch[3] = preprocessing(previous_state, state_t_plus_1)
        V = V_critic(batch)
        Vp = V_critic(batch)
        
        # update value
        value_loss = F.smooth_l1_loss(V, Vp.detach())
        critic_optimizer.zero_grad()
        value_loss.backward()
        critic_optimizer.step()

        V_ = V.detach()
        Vp_ = Vp.detach()
        # 𝐴(𝑠,𝑎)=𝑟+𝛾𝑉(𝑠′)−𝑉(𝑠)
        # L’erreur TD est un estimateur de l’avantage
        # δπθ = r + γVπθ(s′) −Vπθ(s)
        A = (r_t + γ * Vp_ - V_)

        # Le gradient est donné par :
        # ∇θJ(θ) = Eπθ[∇θ log πθ(a|s)δπθ(s,a)]
        policy_loss = -A * log_prob
        
        actor_optimizer.zero_grad()
        policy_loss.backward()
        actor_optimizer.step()
        total_reward += r_t

        # state = state_t_plus_1
    print("Episode %d ended with score: %d" % (episode, total_reward))
    system.reset_game()

