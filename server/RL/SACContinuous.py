import os
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(self, N, hidden_dim, action_dim=2):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(3 * N + 2, hidden_dim)
        self.l_mean = nn.Linear(hidden_dim, action_dim)
        self.l_std = nn.Linear(hidden_dim, action_dim)
        self.N = N

        nn.init.orthogonal_(self.l1.weight, gain=np.sqrt(2))
        nn.init.constant_(self.l1.bias, 0.0)
        nn.init.uniform_(self.l_mean.weight, -1e-3, 1e-3)
        nn.init.constant_(self.l_mean.bias, 0.0)
        nn.init.constant_(self.l_std.weight, 0.0)
        nn.init.constant_(self.l_std.bias, -1.0)

    def forward(self, x):
        x = F.relu(self.l1(x))
        means = self.l_mean(x)
        stds = F.softplus(self.l_std(x))
        stds = torch.clamp(stds, min=1e-6)
        return means, stds


class QValueNet(nn.Module):
    def __init__(self, N, action_dim, hidden_dim):
        super(QValueNet, self).__init__()
        input_dim = 3 * N + 2 + action_dim
        self.l1 = nn.Linear(input_dim, hidden_dim * 2)
        self.l2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        for layer in [self.l1, self.l2]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)

        nn.init.uniform_(self.l3.weight, -1e-3, 1e-3)
        nn.init.constant_(self.l3.bias, 0.0)

    def forward(self, state, action):
        state = state.float()
        action = action.float()
        if state.dim() == 3:
            B, C, S = state.shape
            state = state.view(B * C, S)
        if action.dim() == 3:
            B, C, A = action.shape
            action = action.view(B * C, A)

        if action.dim() == 1:
            action = action.unsqueeze(-1)

        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)


class SACContinuous:
    def __init__(
        self,
        N,
        hidden_dim,
        action_dim,
        actor_lr,
        critic_lr,
        alpha_lr,
        target_entropy,
        tau,
        gamma,
        device,
        model_path=None,
    ):
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "RLModel",
                "SACContinuous.pth",
            )

        self.N = N
        self.device = device
        self.actor = Actor(N, hidden_dim, action_dim).to(device)
        self.critic_1 = QValueNet(N, action_dim, hidden_dim).to(device)
        self.critic_2 = QValueNet(N, action_dim, hidden_dim).to(device)
        self.target_critic_1 = QValueNet(N, action_dim, hidden_dim).to(device)
        self.target_critic_2 = QValueNet(N, action_dim, hidden_dim).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(
            self.critic_1.parameters(), lr=critic_lr
        )
        self.critic_2_optimizer = torch.optim.Adam(
            self.critic_2.parameters(), lr=critic_lr
        )

        self.log_alpha = torch.tensor(np.log(0.001), dtype=torch.float32, device=device)
        self.log_alpha.requires_grad_(True)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau
        self.model_path = model_path
        self.epochs = 0

        if os.path.exists(model_path):
            print(f"Loading model from {self.model_path}...")
            self.load_model()
        else:
            self.target_critic_1.load_state_dict(self.critic_1.state_dict())
            self.target_critic_2.load_state_dict(self.critic_2.state_dict())
    @staticmethod
    def _to_2d(x):
        if x.dim() == 3:
            B, C, F = x.shape
            return x.view(B * C, F)
        elif x.dim() == 2:
            return x
        elif x.dim() == 1:
            return x.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported tensor dim: {x.dim()} for shape {x.shape}")

    @staticmethod
    def _to_col(x):
        if x.dim() == 3:
            B, C, _ = x.shape
            return x.view(B * C, 1)
        elif x.dim() == 2:
            if x.shape[1] == 1:
                return x
            else:
                return x
        elif x.dim() == 1:
            return x.unsqueeze(1)
        else:
            raise ValueError(f"Unsupported tensor dim in _to_col: {x.dim()}")

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        if state.dim() == 2:
            C, S = state.shape
            state_in = state.view(C, S)
        elif state.dim() == 3:
            B, C, S = state.shape
            state_in = state.view(B * C, S)
        else:
            raise ValueError(f"Unsupported state shape in take_action: {state.shape}")

        self.epochs += 1
        means, stds = self.actor(state_in)

        dist = Normal(means, stds)
        sampled = dist.rsample()
        sampled = torch.clamp(sampled, 0.0, 1.0)
        o_raw = sampled[:, 0]
        xi = torch.clamp(sampled[:, 1], 0.001, 0.999)
        o_int = torch.floor(o_raw * (self.N + 1)).long()
        o_int = torch.clamp(o_int, max=self.N)

        return o_int.detach().cpu().numpy(), xi.detach().cpu().numpy()

    def calc_target(self, rewards, next_states, dones):
        rewards = rewards.to(self.device).float()
        dones = dones.to(self.device).float()
        next_states = next_states.to(self.device).float()

        means, stds = self.actor(next_states)
        dist = Normal(means, stds)

        actions_raw = dist.rsample()
        actions_raw = torch.clamp(actions_raw, 0.0, 1.0)

        o = actions_raw[:, 0:1]
        xi = torch.clamp(actions_raw[:, 1:2], 0.001, 0.999)
        next_actions = torch.cat([o, xi], dim=1)

        log_probs = dist.log_prob(actions_raw).sum(dim=1, keepdim=True)
        entropy = -log_probs

        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        min_q = torch.minimum(q1_value, q2_value)                 

        next_value = min_q + self.log_alpha.exp() * entropy       

        td_target = rewards + self.gamma * next_value * (1.0 - dones)
        return td_target.detach()

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(
                param_target.data * (1.0 - self.tau) + param.data * self.tau
            )

    def update(self, transition_dict):
        states = transition_dict["states"].to(self.device).float()
        actions = transition_dict["actions"].to(self.device).float()
        rewards = transition_dict["rewards"].to(self.device).float()
        next_states = transition_dict["next_states"].to(self.device).float()
        dones = transition_dict["dones"].to(self.device).float()

        B, C, S = states.shape

        states_flat = states.view(B * C, S)
        actions_flat = actions.view(B * C, -1)
        next_states_flat = next_states.view(B * C, S)

        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)
        rewards_bc = rewards.repeat(1, C).view(B * C, 1)

        if dones.dim() == 1:
            dones = dones.unsqueeze(1)
        dones_bc = dones.repeat(1, C).view(B * C, 1)    

        td_target = self.calc_target(rewards_bc, next_states_flat, dones_bc)

        q1 = self.critic_1(states_flat, actions_flat)
        q2 = self.critic_2(states_flat, actions_flat)

        critic_1_loss = F.mse_loss(q1, td_target)
        critic_2_loss = F.mse_loss(q2, td_target)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        means, stds = self.actor(states_flat)
        dist = Normal(means, stds)
        actions_raw = dist.rsample()
        actions_raw = torch.clamp(actions_raw, 0.0, 1.0)

        o = actions_raw[:, 0:1]                     
        xi = torch.clamp(actions_raw[:, 1:2], 0.001, 0.999)
        new_actions = torch.cat([o, xi], dim=1)

        log_probs = dist.log_prob(actions_raw).sum(dim=1, keepdim=True)
        entropy = -log_probs

        q1_new = self.critic_1(states_flat, new_actions)
        q2_new = self.critic_2(states_flat, new_actions)
        min_q_new = torch.minimum(q1_new, q2_new)

        actor_loss = (self.log_alpha.exp() * log_probs - min_q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (entropy + self.target_entropy).detach()).mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

    def save_model(self):
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_1_state_dict": self.critic_1.state_dict(),
                "critic_2_state_dict": self.critic_2.state_dict(),
                "target_critic_1_state_dict": self.target_critic_1.state_dict(),
                "target_critic_2_state_dict": self.target_critic_2.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_1_optimizer_state_dict": self.critic_1_optimizer.state_dict(),
                "critic_2_optimizer_state_dict": self.critic_2_optimizer.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "log_alpha_optimizer_state_dict": self.log_alpha_optimizer.state_dict(),
                "epochs": self.epochs,
            },
            self.model_path,
        )
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic_1.load_state_dict(checkpoint["critic_1_state_dict"])
        self.critic_2.load_state_dict(checkpoint["critic_2_state_dict"])
        self.target_critic_1.load_state_dict(checkpoint["target_critic_1_state_dict"])
        self.target_critic_2.load_state_dict(checkpoint["target_critic_2_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_1_optimizer.load_state_dict(
            checkpoint["critic_1_optimizer_state_dict"]
        )
        self.critic_2_optimizer.load_state_dict(
            checkpoint["critic_2_optimizer_state_dict"]
        )
        self.log_alpha = checkpoint["log_alpha"].to(self.device)
        self.log_alpha.requires_grad_(True)
        self.log_alpha_optimizer.load_state_dict(
            checkpoint["log_alpha_optimizer_state_dict"]
        )
        self.epochs = checkpoint["epochs"]
        print(f"Model loaded from {self.model_path}")
