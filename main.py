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

from ram.actor import Actor
from ram.critic import Critic
from ram.runner import Runner, γ

from experience_replay import ReplayMemory, Transition

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


def start_runner(episode: int, central: Central, rom, display_screen):
    runner = Runner(device, episode, central, rom, display_screen=display_screen)
    return runner.run()


# Load the ROM file
system = ALEInterface()
system.loadROM(Pacman)

# Get the list of legal actions
legal_actions = system.getLegalActionSet()
print(legal_actions)

V_critic_target = Critic().to(device)
critic_optimizer = optim.Adam(V_critic_target.parameters(), lr=1e-3)
actor_target = Actor(action_size=len(legal_actions)).to(device)
actor_optimizer = optim.Adam(actor_target.parameters(), lr=1e-4)

central = Central(actor=actor_target, critic=V_critic_target)
BATCH_SIZE = 128


def worker():
    alpha = 1e-3
    # alpha = 1
    lr = 1e-2
    lr_decay = 1e-2 # learning rate decay
    state_sum_critic = 0
    state_sum_actor = 0
    weight_decay = 1e-2
    eps = 1e-10
    step = 0

    state_critic = {}
    d_critic = Critic().to(device)
    for name, param in d_critic.named_parameters():
        state_critic[name] = torch.full_like(param, 0, memory_format=torch.preserve_format)


    state_actor = {}
    d_actor = Actor(action_size=len(legal_actions)).to(device)
    for name, param in d_actor.named_parameters():
        state_actor[name] = torch.full_like(param, 0, memory_format=torch.preserve_format)

    print('Booting worker...')
    while True:
        gradients = central.queue.get(block=True)
        try:
            while True:
                print(chalk.green('Item received'))
                clr = lr / (1 + (step - 1) * lr_decay)
                step = step + 1
                # Asynchronous Gradient Descent
                dθv = gradients["critic"]
                d_critic = Critic().to(device)
                d_critic.load_state_dict(central.weight_memory["critic"])
                with torch.no_grad():
                    for name, param in d_critic.named_parameters():
                        # ADAGRAD

                        dθv_param = dθv[name].add(param, alpha = weight_decay)
                        state_critic[name].addcmul_(dθv_param, dθv_param, value=1)
                        std = state_critic[name].sqrt().add_(eps)
                        param.addcdiv_(dθv_param, std, value=-clr)

                dθ = gradients["actor"]
                d_actor = Actor(action_size=len(legal_actions)).to(device)
                d_actor.load_state_dict(central.weight_memory["actor"])
                # print("BEFORE")
                # print(d_actor.state_dict()["eline3.1.weight"])
                with torch.no_grad():
                    for name, param in d_actor.named_parameters():
                        dθ_param = dθ[name].add(param, alpha = weight_decay)
                        state_actor[name].addcmul_(dθ_param, dθ_param, value=1)
                        std = state_actor[name].sqrt().add_(eps)
                        param.addcdiv_(dθ_param, std, value=-clr)
                # print("AFTER")
                # print(d_actor.state_dict()["eline3.1.weight"])

                central.weight_memory = {
                    "actor": d_actor.state_dict(),
                    "critic": d_critic.state_dict(),
                }
                gradients = central.queue.get(block=False)
        except Empty:
            print(chalk.green("Empty queue"))
            pass

        if step % 10 == 0:
            # # Experience Replay
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

            state_batch = torch.cat(batch.state).to(device)
            reward_batch = torch.cat(batch.reward).to(device)

            # Compute V(s_t)
            V_s_t = V_critic(state_batch)

            # Compute V(s_{t+1})
            V_s_t_plus_1 = (V_s_t.detach() * γ) + reward_batch.unsqueeze(1).unsqueeze(1)

            # Compute Huber loss
            criterion = nn.SmoothL1Loss()
            loss = criterion(V_s_t, V_s_t_plus_1)

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

NUMBER_OF_EPISODES = 10000
MAX_WORKERS = 4

with open('execution.log', 'w') as f:
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for episode in range(NUMBER_OF_EPISODES):
            futures.append(executor.submit(start_runner, episode, central, Pacman, MAX_WORKERS == 1))

        for episode in range(NUMBER_OF_EPISODES):
            result = futures[episode].result()
            print(chalk.green(result))
            f.write(result + "\n")
            f.flush()
