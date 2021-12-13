#!/usr/bin/env python
from queue import SimpleQueue
import numpy as np
from ale_py import ALEInterface, SDL_SUPPORT
from ale_py.roms import Pacman

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from actor import Actor
from critic import Critic
from runner import Runner
from experience_replay import ReplayMemory 

torch.autograd.set_detect_anomaly(True)

def initialize_uniformly(layer: nn.Linear, init_w: float = 3e-3):
    """Initialize the weights and bias in [-init_w, init_w]."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)


class Central:
    def __init__(self) -> None:
        self.weight_memory = {
            "actor": actor.state_dict(),
            "critic": V_critic.state_dict(),
        }
        self.queue = SimpleQueue()
        self.memory = ReplayMemory(10000)
        self.BATCH_SIZE = 128

def start_runner(episode: int, central: Central, rom):
    runner = Runner(episode, central, rom)
    return runner.run()


# Load the ROM file
system = ALEInterface()
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

central = Central()

def worker():
    print('Booting worker...')
    while True:
        item = central.queue.get(block=True)
        print(f'Item received')
        central.weight_memory = {
            "actor": item["actor"],
            "critic": item["critic"],
        }

import threading
threading.Thread(target=worker, daemon=True).start()

# Play 10 episodes
NUMBER_OF_EPISODES = 10
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=2) as executor:
    futures = []
    for episode in range(NUMBER_OF_EPISODES):
        futures.append(executor.submit(start_runner, episode, central, Pacman))
    
    for episode in range(NUMBER_OF_EPISODES):
        print(futures[episode].result())

