from ale_py import ALEInterface, LoggerMode

import torch
import torch.nn.functional as F
import torch.optim as optim

from ram.critic import Critic
from ram.actor import Actor

γ = 0.9

class Runner():
    def __init__(self, device, episode, central, rom, display_screen = False) -> None:
        self.device = device
        self.episode = episode
        self.system = ALEInterface()
        self.system.setLoggerMode(LoggerMode.Error)
        self.system.setBool("display_screen", display_screen)
        self.system.loadROM(rom)
        self.legal_actions = self.system.getLegalActionSet()
        self.V_critic = Critic().to(device)
        self.actor = Actor(action_size=len(self.legal_actions)).to(device)
        self.central = central
        self.entropy_weight = 1e-2


    def run(self):
        print(f"Starting runner {self.episode}")
        self.V_critic.train()
        self.V_critic.load_state_dict(self.central.weight_memory["critic"])
        critic_optimizer = optim.Adam(self.V_critic.parameters(), lr=1e-3)
        self.actor.train()
        self.actor.load_state_dict(self.central.weight_memory["actor"])
        actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        batch = torch.zeros(4, 1, 128).to(self.device)
        total_reward = 0
        state = torch.from_numpy(self.system.getRAM()).to(self.device)

        # Shift images
        batch = torch.roll(batch, 3, 0)
        batch[3] = state

        dθ = {name: torch.zeros(param.shape).to(self.device) for name, param in self.actor.named_parameters()}
        dθv = {name: torch.zeros(param.shape).to(self.device) for name, param in self.V_critic.named_parameters()}
        min_policy_loss, min_value_loss = float("inf"), float("inf")
        max_policy_loss, max_value_loss = -float("inf"), -float("inf")
        while not self.system.game_over():
            action, log_policy = self.actor(batch[3].unsqueeze(0))
            a_t = self.legal_actions[action]

            # Apply an action and get the resulting reward
            # State?
            # Perform action a_t according to policy π(a_t / s_t; θ)
            # Receive reward r_t 
            # Skip k frames. No big difference between two close frames
            r_t = 0
            frames_skipped = 4
            for i in range(frames_skipped):
                r_t += self.system.act(a_t)

            state_t_plus_1 = torch.from_numpy(self.system.getRAM()).to(self.device)
            # The function w from algorithm 1 described below applies this preprocess-
            # ing to the m most recent frames and stacks them to produce the input to the
            # Q-function, in which m=4, although the algorithm is robust to different values of
            # m (for example, 3 or 5).
            V = self.V_critic(batch[3].unsqueeze(0))
            
            batch = torch.roll(batch, 3, 0) 
            batch[3] = state_t_plus_1

            # v(s_t) = R_t+1 + γ v(s_t+1)
            target_V = r_t + γ * self.V_critic(batch[3].unsqueeze(0)).detach()
            
            # update value
            value_loss = F.smooth_l1_loss(V, target_V)
            critic_optimizer.zero_grad()
            value_loss.backward()
            critic_optimizer.step()
            gradients_v = {name: param.grad.detach() for name, param in self.V_critic.named_parameters()}
            dθv = {key: (value + gradients_v[key]) for key, value in dθv.items()}

            V_ = V.detach().squeeze().squeeze()
            Vp_ = target_V.detach().squeeze().squeeze()
            # 𝐴(𝑠,𝑎)=𝑟+𝛾𝑉(𝑠′)−𝑉(𝑠)
            # L’erreur TD est un estimateur non-biaisé de l’avantage
            # δπθ = r + γVπθ(s′) −Vπθ(s)
            A = (r_t + γ * Vp_ - V_)

            # Le gradient est donné par :
            # ∇θJ(θ) = Eπθ[∇θ log πθ(a|s)A(s,a)]
            # et on fait une montée de gradient
            # Equivalent à calculer -∇θJ(θ) et faire une descente de gradient
            policy_loss = -A * log_policy
            # Regularization: Entropy = - Σ p(a) log p(a)
            policy_loss += self.entropy_weight * -log_policy

            min_policy_loss = min(min_policy_loss, policy_loss)
            max_policy_loss = max(max_policy_loss, policy_loss)
            min_value_loss = min(min_value_loss, value_loss)
            max_value_loss = max(max_value_loss, value_loss)

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
        return "Episode %d ended with score: %d, policy_loss: %f - %f, value_loss: %f - %s" % (
            self.episode, total_reward, min_policy_loss, max_policy_loss, min_value_loss, max_value_loss)

