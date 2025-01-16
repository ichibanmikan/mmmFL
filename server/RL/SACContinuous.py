import os
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.distributions import Normal
from RL.utils import ReplayBuffer

class AttentionLayer(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.query = nn.Linear(state_dim, hidden_dim)
        self.key = nn.Linear(state_dim, hidden_dim)
        self.value = nn.Linear(state_dim, hidden_dim)

    def forward(self, ti, other_states):
        Q = self.query(ti.unsqueeze(0))  # (bsz, 1, hidden_dim)
        K = self.key(other_states)         # (bsz, n - 1, hidden_dim)
        V = self.value(other_states)       # (bsz, n - 1, hidden_dim)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) \
            / (K.size(-1) ** 0.5) # (bsz, 1, n - 1)
        attention_weights = nn.Softmax(attention_scores, dim = -1)  # (bsz, 1, n - 1)
        context = torch.matmul(attention_weights, V)  # (bsz, 1, hidden_dim)
        
        return context, attention_weights # (bsz, 1, hidden_dim), (bsz, 1, n - 1)

class Actor(nn.Module):
    def __init__(self, N, hidden_dim):
        super(Actor, self).__init__()
        self.N = N
        self.attention = AttentionLayer(1, hidden_dim)
        self.l1 = nn.Linear(hidden_dim, hidden_dim)
        self.l_mean = nn.Linear(hidden_dim, 1)
        self.l_std = nn.Linear(hidden_dim, 1) 

    def forward(self, x):
        # input x: (bsz, N)
        x, _ = self.attention(x[:, 0], x[:, 1:])
        x = F.relu(self.l1(x))
        
        x_mean = self.l_mean(x) 
        x_std = F.softplus(self.l_std(x))
        x_std = torch.clamp(x_std, min=1e-6)        
        dist = Normal(x_mean, x_std)
        normal_sample = dist.rsample()
        log_prob = dist.log_prob(normal_sample)
        action = (torch.tanh(normal_sample) + 1) / 2
        log_prob -= torch.log(1 - action.pow(2) + 1e-7)
        
        return action, log_prob

class QValueNet(nn.Module):
    def __init__(self, N, hidden_dim):
        super(QValueNet, self).__init__()
        self.l1 = nn.Linear(4 * N + 1 + 1, hidden_dim * 2) 
        # state: [4N+1]; action: [1]. (b, 4 * N + 2) -> (b, h_d * 2)
        self.l2 = nn.Linear(hidden_dim * 2, hidden_dim) 
        # (b, h_d * 2) -> (b, h_d)
        self.l3 = nn.Linear(hidden_dim, 1) 
        #(b, h_d) -> (b, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim = -1)  
        # ((batch_size, 4N + 1), (batch_size, 1)) -> 
            # (batch_size, 4N + 2)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)
 
class SAC:
    def __init__(
        self, N, hidden_dim, actor_lr, critic_lr, alpha_lr,\
            target_entropy, tau, gamma, device,\
                model_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), 'RLModel', 'SAC.pth'
        )):
        self.actor = Actor(N, hidden_dim).to(device)
        self.critic_1 = QValueNet(N, hidden_dim).to(device)
        self.critic_2 = QValueNet(N, hidden_dim).to(device)
        self.target_critic_1 = QValueNet(N, hidden_dim).to(device)
        self.target_critic_2 = QValueNet(N, hidden_dim).to(device)
        
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr
        )
        self.critic_1_optimizer = torch.optim.Adam(
            self.critic_1.parameters(), lr=critic_lr
        )
        self.critic_2_optimizer = torch.optim.Adam(
            self.critic_2.parameters(), lr=critic_lr
        )

        self.log_alpha = torch.tensor(np.log(0.0001), dtype=torch.float)
        self.log_alpha.requires_grad = True 
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr
        )
        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.model_path = model_path
        if os.path.exists(model_path):
            print(f"Loading model from {self.model_path}...")
            self.load_model()
        else:
            self.target_critic_1.load_state_dict(self.critic_1.state_dict())
            self.target_critic_2.load_state_dict(self.critic_2.state_dict())
            
    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device) # (N)
        action = self.actor(state) # (1)
        return action.cpu().detach().numpy()

    def calc_target(self, rewards, next_states, dones): 
        next_actions, log_prob = self.actor(next_states)
        entropy = -log_prob
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        
        next_value = torch.minimum(q1_value, q2_value) \
            + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target.float()

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data \
                * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = transition_dict['states']  # (b, 1, 4 * N + 1)
        actions = transition_dict['actions']  # (b, 1, 4)
        rewards = transition_dict['rewards'].unsqueeze(-1)  # (b, 1, 1)
        next_states = transition_dict['next_states']  # (b, 1, 4 * N + 1)
        dones = transition_dict['dones'].unsqueeze(-1)  # (b, 1, 1)

        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_loss = torch.mean(
            F.mse_loss(self.critic_1(states, actions), td_target.detach()))
        critic_2_loss = torch.mean(
            F.mse_loss(self.critic_2(states, actions), td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        new_actions, log_prob = self.actor(states)
        log_prob = log_prob.sum(dim = 1, keepdim = True)
        entropy = -log_prob
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
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
            'log_alpha_optimizer_state_dict': self.log_alpha_optimizer.state_dict(),
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