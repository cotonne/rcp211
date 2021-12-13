# From https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# We’ll be using experience replay memory for training our DQN.
#  It stores the transitions that the agent observes,
# allowing us to reuse this data later.
# By sampling from it randomly, the transitions that build up a batch are decorrelated.
# It has been shown that this greatly stabilizes and improves the DQN training procedure.
from collections import namedtuple, deque
import random

import torch

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
