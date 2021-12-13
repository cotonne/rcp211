from ale_py import ALEInterface

import torch
import torch.nn.functional as F
import torch.optim as optim

from critic import Critic
from actor import Actor
from preprocessing import preprocessing
from recorder import Recorder

γ = 0.9

class Runner():
    def __init__(self, episode, central, rom) -> None:
        self.episode = episode
        self.system = ALEInterface()
        self.system.loadROM(rom)
        self.legal_actions = self.system.getLegalActionSet()
        self.V_critic = Critic()
        self.actor = Actor(action_size=len(self.legal_actions))
        self.central = central
        self.recorder = Recorder(episode)


    def run(self):
        self.V_critic.train()
        self.V_critic.load_state_dict(self.central.weight_memory["critic"])
        critic_optimizer = optim.Adam(self.V_critic.parameters(), lr=1e-3)
        self.actor.train()
        self.actor.load_state_dict(self.central.weight_memory["actor"])
        actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        previous_state = torch.zeros(250, 160, 3)
        batch = torch.zeros(4, 1, 94, 84)
        total_reward = 0
        state = torch.from_numpy(self.system.getScreenRGB())

        # Shift images
        batch = torch.roll(batch, 3, 0) 
        batch[3] = preprocessing(previous_state, state)
        previous_state = state

        self.recorder.start()
        while not self.system.game_over():
            action, log_policy = self.actor(batch[3].unsqueeze(0))
            a_t = self.legal_actions[action]
            
            # Apply an action and get the resulting reward
            # State?
            # Perform action a_t accoring to policy π(a_t / s_t; θ)
            # Receive reward r_t 
            # Skip k frames. No big difference between two close frames
            r_t = 0
            for _ in range(4):
                r_t += self.system.act(a_t)

            state_t_plus_1 = torch.from_numpy(self.system.getScreenRGB())
            self.recorder.save(self.system)
            batch = torch.roll(batch, 3, 0) 
            batch[3] = preprocessing(previous_state, state_t_plus_1)
            previous_state = state_t_plus_1

            V = self.V_critic(batch[2].unsqueeze(0))
            Vp = self.V_critic(batch[3].unsqueeze(0))
            
            # update value
            value_loss = F.smooth_l1_loss(V, Vp.detach())
            critic_optimizer.zero_grad()
            value_loss.backward()
            critic_optimizer.step()

            V_ = V.detach().squeeze().squeeze()
            Vp_ = Vp.detach().squeeze().squeeze()
            # 𝐴(𝑠,𝑎)=𝑟+𝛾𝑉(𝑠′)−𝑉(𝑠)
            # L’erreur TD est un estimateur de l’avantage
            # δπθ = r + γVπθ(s′) −Vπθ(s)
            A = (r_t + γ * Vp_ - V_)

            # Le gradient est donné par :
            # ∇θJ(θ) = Eπθ[∇θ log πθ(a|s)δπθ(s,a)]
            policy_loss = -A * log_policy

            actor_optimizer.zero_grad()
            policy_loss.backward()
            actor_optimizer.step()
            total_reward += r_t

            self.central.memory.push(batch[2].unsqueeze(0), torch.tensor([action]), batch[3].unsqueeze(0), torch.tensor([r_t]))

            # state = state_t_plus_1
        self.central.queue.put({
            "actor": self.actor.state_dict(),
            "critic": self.V_critic.state_dict()
        })
        self.system.reset_game()
        self.recorder.stop()
        return "Episode %d ended with score: %d" % (self.episode, total_reward)

