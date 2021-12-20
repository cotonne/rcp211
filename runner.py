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
    def __init__(self, device, episode, central, rom) -> None:
        self.device = device
        self.episode = episode
        self.system = ALEInterface()
        self.system.loadROM(rom)
        self.legal_actions = self.system.getLegalActionSet()
        self.V_critic = Critic().to(device)
        self.actor = Actor(action_size=len(self.legal_actions)).to(device)
        self.central = central
        self.recorder = Recorder(episode)
        self.entropy_weight = 1e-2


    def run(self):
        self.V_critic.train()
        self.V_critic.load_state_dict(self.central.weight_memory["critic"])
        critic_optimizer = optim.Adam(self.V_critic.parameters(), lr=1e-3)
        self.actor.train()
        self.actor.load_state_dict(self.central.weight_memory["actor"])
        actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        previous_state = torch.zeros(250, 160, 3).to(self.device)
        batch = torch.zeros(4, 1, 94, 84).to(self.device)
        total_reward = 0
        state = torch.from_numpy(self.system.getScreenRGB()).to(self.device)

        # Shift images
        batch = torch.roll(batch, 3, 0) 
        batch[3] = preprocessing(previous_state, state)
        previous_state = state

        dθ = {name: torch.zeros(param.shape).to(self.device) for name, param in self.actor.named_parameters()}
        dθv = {name: torch.zeros(param.shape).to(self.device) for name, param in self.V_critic.named_parameters()}

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

            state_t_plus_1 = torch.from_numpy(self.system.getScreenRGB()).to(self.device)
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
            gradients_v = {name: param.grad.detach() for name, param in self.V_critic.named_parameters()}
            dθv = {key: (value + gradients_v[key]) for key, value in dθv.items()}

            V_ = V.detach().squeeze().squeeze()
            Vp_ = Vp.detach().squeeze().squeeze()
            # 𝐴(𝑠,𝑎)=𝑟+𝛾𝑉(𝑠′)−𝑉(𝑠)
            # L’erreur TD est un estimateur de l’avantage
            # δπθ = r + γVπθ(s′) −Vπθ(s)
            print(A)
            A = (r_t + γ * Vp_ - V_)

            # Le gradient est donné par :
            # ∇θJ(θ) = Eπθ[∇θ log πθ(a|s)δπθ(s,a)]
            # et on fait une montée de gradient
            # Equivalent à calculer -∇θJ(θ) et faire une descente de gradient
            policy_loss = -A * log_policy
            # Regularization: Entropy = - 	Σ p(a) log p(a)
            policy_loss += self.entropy_weight * -log_policy

            actor_optimizer.zero_grad()
            policy_loss.backward()
            actor_optimizer.step()
            gradients = {name: param.grad.detach() for name, param in self.actor.named_parameters()}
            dθ = {key: (value + gradients[key]) for (key, value) in dθ.items()}
            total_reward += r_t

            self.central.memory.push(batch[2].unsqueeze(0), torch.tensor([action]), batch[3].unsqueeze(0), torch.tensor([r_t]))

            # state = state_t_plus_1
        self.central.queue.put({
            "actor": dθ,
            "critic": dθv
        })
        self.system.reset_game()
        self.recorder.stop()
        return "Episode %d ended with score: %d" % (self.episode, total_reward)

