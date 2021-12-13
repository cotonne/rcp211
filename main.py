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
from runner import Runner, γ 
from experience_replay import ReplayMemory, Transition

torch.autograd.set_detect_anomaly(True)

def initialize_uniformly(layer: nn.Linear, init_w: float = 3e-3):
    """Initialize the weights and bias in [-init_w, init_w]."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)


class Central:
    def __init__(self, actor, critic) -> None:
        self.weight_memory = {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
        }
        self.queue = SimpleQueue()
        self.memory = ReplayMemory(10000)
        

def start_runner(episode: int, central: Central, rom):
    runner = Runner(episode, central, rom)
    return runner.run()


# Load the ROM file
system = ALEInterface()
system.loadROM(Pacman)

# Get the list of legal actions
legal_actions = system.getLegalActionSet()

V_critic_target = Critic()
critic_optimizer = optim.Adam(V_critic_target.parameters(), lr=1e-3)
actor_target = Actor(action_size=len(legal_actions))
actor_optimizer = optim.Adam(actor_target.parameters(), lr=1e-4)

central = Central(actor=actor_target, critic=V_critic_target)
BATCH_SIZE = 128
def worker():
    print('Booting worker...')
    while True:
        item = central.queue.get(block=True)
        print(f'Item received')
        V_critic = Critic()
        V_critic.load_state_dict(item["critic"])
        print(f'Memory site = {len(central.memory)}')
        if len(central.memory) < BATCH_SIZE:
            continue
        print(f'Experience replay...')
        transitions = central.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        # action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = V_critic(state_batch) # .gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE)
        next_state_values[non_final_mask] = V_critic_target(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * γ) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        critic_optimizer.zero_grad()
        loss.backward()
        for param in V_critic.parameters():
            param.grad.data.clamp_(-1, 1)
        critic_optimizer.step() 
        central.weight_memory = {
            "actor": item["actor"],
            "critic": V_critic.state_dict(),
        }
        print(f'Experience replay done')

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

