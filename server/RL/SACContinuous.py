import os
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.distributions import Normal, Dirichlet

# class AttentionLayer(nn.Module):
#     def __init__(self, state_dim, hidden_dim):
#         super(AttentionLayer, self).__init__()
#         self.query = nn.Linear(state_dim, hidden_dim)
#         self.key = nn.Linear(state_dim, hidden_dim)
#         self.value = nn.Linear(state_dim, hidden_dim)

#     def forward(self, ti, other_states):
#         Q = self.query(ti.unsqueeze(-1))  # (bsz, 1, hidden_dim)
#         K = self.key(other_states.unsqueeze(-1))  # (bsz, k' - 1, hidden_dim)
#         V = self.value(other_states.unsqueeze(-1)) # (bsz, k' - 1, hidden_dim)
#         attention_scores = torch.matmul(Q, K.transpose(-2, -1)) \
#             / (K.size(-1) ** 0.5) # (bsz, 1, k' - 1)
#         attention_weights = nn.Softmax(attention_scores, dim = -1)  # (bsz, 1, k' - 1)
#         context = torch.matmul(attention_weights, V)  # (bsz, 1, hidden_dim)
        
#         return context.squeeze(dim = 1), \
#             attention_weights.squeeze(dim = 1) # (bsz, hidden_dim), (bsz, k' - 1)

class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l_out = nn.Linear(hidden_dim, 1)

        for layer in [self.l1, self.l2]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)

        nn.init.uniform_(self.l_out.weight, -1e-3, 1e-3)
        nn.init.constant_(self.l_out.bias, 0.5)

    def forward(self, states):
        B, N, _ = states.shape
        x = states.view(B * N, -1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        psi = F.softplus(self.l_out(x)) + 1e-6
        psi = psi.view(B, N)
        return psi

class QValueNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim * 2)
        self.l2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        for layer in [self.l1, self.l2]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)

        nn.init.uniform_(self.l3.weight, -1e-3, 1e-3)
        nn.init.constant_(self.l3.bias, 0.0)

    def forward(self, state_flat, action):
        x = torch.cat([state_flat, action], dim=-1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)

class SACContinuous:
    def __init__(
        self,
        N,
        hidden_dim,
        actor_lr,
        critic_lr,
        alpha_lr,
        target_entropy,
        tau,
        gamma,
        device,
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RLModel', 'SACContinuous.pth')
    ):
        self.N = N
        self.state_dim = 3 * N
        self.action_dim = N
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.model_path = model_path

        self.actor = Actor(3, hidden_dim).to(device)
        self.critic_1 = QValueNet(self.state_dim, self.action_dim, hidden_dim).to(device)
        self.critic_2 = QValueNet(self.state_dim, self.action_dim, hidden_dim).to(device)
        self.target_critic_1 = QValueNet(self.state_dim, self.action_dim, hidden_dim).to(device)
        self.target_critic_2 = QValueNet(self.state_dim, self.action_dim, hidden_dim).to(device)

        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = target_entropy

    def sample_action(self, states):
        psi = self.actor(states)
        dist = torch.distributions.Dirichlet(psi)
        action = dist.rsample()                # (B, N)
        log_prob = dist.log_prob(action)       # (B,)
        entropy = dist.entropy()               # (B,)
        return action, log_prob, entropy, psi

    def soft_update(self, net, target_net):
        for p, tp in zip(net.parameters(), target_net.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def update(self, transition_dict):
        states = torch.as_tensor(transition_dict['states'], device=self.device)         # (B,N,3)
        actions = torch.as_tensor(transition_dict['actions'], device=self.device)       # (B,N)
        rewards = torch.as_tensor(transition_dict['rewards'], device=self.device)       # (B,1)
        next_states = torch.as_tensor(transition_dict['next_states'], device=self.device)
        dones = torch.as_tensor(transition_dict['dones'], device=self.device)

        B = states.size(0)
        state_flat = states.view(B, -1)
        next_state_flat = next_states.view(B, -1)

        with torch.no_grad():
            next_action, next_logp, _, _ = self.sample_action(next_states)
            q1_next = self.target_critic_1(next_state_flat, next_action)
            q2_next = self.target_critic_2(next_state_flat, next_action)
            q_next = torch.min(q1_next, q2_next)
            target_q = rewards + self.gamma * (q_next - self.log_alpha.exp() * next_logp.unsqueeze(-1)) * (1 - dones)

        q1 = self.critic_1(state_flat, actions)
        q2 = self.critic_2(state_flat, actions)
        critic_1_loss = F.mse_loss(q1, target_q)
        critic_2_loss = F.mse_loss(q2, target_q)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        new_action, logp, entropy, _ = self.sample_action(states)
        q1_pi = self.critic_1(state_flat, new_action)
        q2_pi = self.critic_2(state_flat, new_action)
        q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = torch.mean(self.log_alpha.exp() * logp.unsqueeze(-1) - q_pi)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = torch.mean(
            self.log_alpha.exp() * (-entropy.detach() - self.target_entropy)
        )
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)
        
    def save_model(self):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_1_state_dict': self.critic_1.state_dict(),
            'critic_2_state_dict': self.critic_2.state_dict(),
            'target_critic_1_state_dict': self.target_critic_1.state_dict(),
            'target_critic_2_state_dict': self.target_critic_2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_1_optimizer_state_dict': self.critic_1_optimizer.state_dict(),
            'critic_2_optimizer_state_dict': self.critic_2_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'log_alpha_optimizer_state_dict': self.log_alpha_optimizer.state_dict()
        }, self.model_path)
        print(f"Model saved to {self.model_path}")
        
    def load_model(self):
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
        self.critic_2.load_state_dict(checkpoint['critic_2_state_dict'])
        self.target_critic_1.load_state_dict(checkpoint['target_critic_1_state_dict'])
        self.target_critic_2.load_state_dict(checkpoint['target_critic_2_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_1_optimizer.load_state_dict(
            checkpoint['critic_1_optimizer_state_dict']
        )
        self.critic_2_optimizer.load_state_dict(
            checkpoint['critic_2_optimizer_state_dict']
        )
        self.log_alpha = checkpoint['log_alpha']
        self.log_alpha_optimizer.load_state_dict(
            checkpoint['log_alpha_optimizer_state_dict']
        )
        print(f"Model loaded from {self.model_path}")