#!/usr/bin/env python
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import SimpleQueue, Empty

import torch
import torch.nn as nn
import torch.optim as optim
from ale_py import ALEInterface
from ale_py.roms import Pacman
from simple_chalk import chalk

from actor import Actor
from critic import Critic
from experience_replay import ReplayMemory, Transition
from runner import Runner, γ

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(f"Device = {device}")

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
    runner = Runner(device, episode, central, rom)
    return runner.run()


# Load the ROM file
system = ALEInterface()
system.loadROM(Pacman)

# Get the list of legal actions
legal_actions = system.getLegalActionSet()

V_critic_target = Critic().to(device)
critic_optimizer = optim.Adam(V_critic_target.parameters(), lr=1e-3)
actor_target = Actor(action_size=len(legal_actions)).to(device)
actor_optimizer = optim.Adam(actor_target.parameters(), lr=1e-4)

central = Central(actor=actor_target, critic=V_critic_target)
BATCH_SIZE = 128


def worker():
    alpha = 1e-4
    print('Booting worker...')
    while True:
        gradients = central.queue.get(block=True)
        try:
            while True:
                print(chalk.green('Item received'))
                # Asynchronous Gradient Descent
                dθv = gradients["critic"]
                d_critic = Critic().to(device)
                d_critic.load_state_dict(central.weight_memory["critic"])
                with torch.no_grad():
                    for name, param in d_critic.named_parameters():
                        param -= alpha * dθv[name]

                dθ = gradients["actor"]
                d_actor = Actor(action_size=len(legal_actions)).to(device)
                d_actor.load_state_dict(central.weight_memory["actor"])
                with torch.no_grad():
                    for name, param in d_actor.named_parameters():
                        param -= alpha * dθ[name]

                central.weight_memory = {
                    "actor": d_actor.state_dict(),
                    "critic": d_critic.state_dict(),
                }
                gradients = central.queue.get(block=False)
        except Empty:
            print(chalk.green("Empty queue"))
            pass

        # Experience Replay
        V_critic = Critic().to(device)
        V_critic.load_state_dict(central.weight_memory["critic"])
        print(chalk.green(f'Memory site = {len(central.memory)}'))
        if len(central.memory) < BATCH_SIZE:
            continue
        print(chalk.green('Experience replay...'))
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
        state_batch = torch.cat(batch.state).to(device)
        # action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward).to(device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = V_critic(state_batch)  # .gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE).to(device)
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
            "actor": central.weight_memory["actor"],
            "critic": V_critic.state_dict(),
        }
        print(chalk.green('Experience replay done'))


threading.Thread(target=worker, daemon=True).start()

NUMBER_OF_EPISODES = 200

with open('execution.log', 'w') as f:
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for episode in range(NUMBER_OF_EPISODES):
            futures.append(executor.submit(start_runner, episode, central, Pacman))

        for episode in range(NUMBER_OF_EPISODES):
            result = futures[episode].result()
            print(chalk.green(result))
            f.write(result + "\n")
            f.flush()
