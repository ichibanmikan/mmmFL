import os
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.distributions import Normal

class Actor(nn.Module):
    def __init__(self, N, hidden_dim, action_dim = 2):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(4 * N + 1, hidden_dim)
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
        x_means = self.l_mean(x) 
        x_stds = F.softplus(self.l_std(x))
        x_stds = torch.clamp(x_stds, min=1e-6)
          
        o_dist = Normal(x_means[:, 0], x_stds[:, 0])
        o_sampled = o_dist.rsample()
        clipped = torch.clamp(o_sampled, 0.0, 1.0)
        o = torch.floor(clipped * (self.N+1)).long()
        o = torch.clamp(o, max=self.N)
        
        xi_dist = Normal(x_means[:, 1], x_stds[:, 1])
        xi_sampled = xi_dist.rsample()
        xi = torch.clamp(xi_sampled, 0.0, 1.0)
        xi = torch.clamp(xi, min=0.001, max=0.999)              
        return o.cpu().detach().numpy(), xi.cpu().detach().numpy()

class QValueNet(nn.Module):
    def __init__(self, N, action_dim, hidden_dim):
        super(QValueNet, self).__init__()
        self.l1 = nn.Linear(4 * N + 1 + action_dim, (hidden_dim) * 2) 
        self.l2 = nn.Linear((hidden_dim) * 2, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)
        for layer in [self.l1, self.l2]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)

        nn.init.uniform_(self.l3.weight, -1e-3, 1e-3)
        nn.init.constant_(self.l3.bias, 0.0)   
     
    def forward(self, state, action):
        if action.dim() == 1:
            action = action.unsqueeze(-1)
        x = torch.cat([state, action], dim = -1) # (bsz, h_d + 2)
        x = F.relu(self.l1(x)) # (bsz, (hidden_dim + 2) * 2)
        x = F.relu(self.l2(x)) # (bsz, (hidden_dim + 2))
        return self.l3(x) # (bsz, 1)
 
class SACContinuous:
    def __init__(
        self, N, hidden_dim, action_dim, actor_lr, critic_lr, alpha_lr,\
            target_entropy, tau, gamma, device,\
                model_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), 'RLModel', 'SACContinuous.pth'
        )):
        self.actor = Actor(N, hidden_dim, action_dim).to(device)
        self.critic_1 = QValueNet(N, action_dim, hidden_dim).to(device)
        self.critic_2 = QValueNet(N, action_dim, hidden_dim).to(device)
        self.target_critic_1 = QValueNet(N, action_dim, hidden_dim).to(device)
        self.target_critic_2 = QValueNet(N, action_dim, hidden_dim).to(device)
        
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr
        )
        self.critic_1_optimizer = torch.optim.Adam(
            self.critic_1.parameters(), lr=critic_lr
        )
        self.critic_2_optimizer = torch.optim.Adam(
            self.critic_2.parameters(), lr=critic_lr
        )

        self.log_alpha = torch.tensor(np.log(0.001), dtype=torch.float)
        self.log_alpha.requires_grad = True 
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr
        )
        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.model_path = model_path
        self.epochs = 0
        if os.path.exists(model_path):
            print(f"Loading model from {self.model_path}...")
            self.load_model()
        else:
            self.target_critic_1.load_state_dict(self.critic_1.state_dict())
            self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.N = N
    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        self.epochs += 1
        o, xi = self.actor(state)
        return o, xi

    def calc_target(self, rewards, next_states = None, dones = 1): 
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)
        if dones.dim() == 1:
            dones = dones.unsqueeze(-1)
            
        means, stds = self.actor(next_states) 
        o_dist = Normal(means[:, 0], stds[:, 0])
        xi_dist = Normal(means[:, 1], stds[:, 1])
        entropy = o_dist.entropy() + xi_dist.entropy()
        q1_value = self.target_critic_1(next_states, torch.cat([means, stds], dim = 1)) # (bsz, 1)
        q2_value = self.target_critic_2(next_states, torch.cat([means, stds], dim = 1)) # (bsz, 1)
        
        next_value = torch.minimum(q1_value, q2_value) \
            + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target.float()

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data \
                * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = transition_dict['states']         # (b, 3)
        actions = transition_dict['actions']         # (b, 1)
        rewards = transition_dict['rewards']  # (b, 1)
        next_states = transition_dict['next_states'] # (b, 3)
        dones = transition_dict['dones']      # (b, 1)
        
        filtered_states = []
        filtered_actions = []
        filtered_rewards = []
        filtered_dones = []
        filtered_next_states = []

        if not filtered_actions:
            return
        
        states_tensor = torch.stack(filtered_states).float()
        actions_tensor = torch.stack(filtered_actions).float()
        rewards_tensor = torch.stack(filtered_rewards).float()
        dones_tensor = torch.stack(filtered_dones).float()
        next_states_tensor = torch.stack(filtered_next_states).float()

        states = states_tensor.to(self.device)
        actions = actions_tensor.to(self.device)
        rewards = rewards_tensor.to(self.device)
        dones = dones_tensor.to(self.device)
        next_states = next_states_tensor.to(self.device)
        
        # print("low transition_dict state shape is: ", states.shape)
        
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

        new_actions, log_probs = self.actor(states) # (bsz, 1)
        entropy = -log_probs # (bsz, 1)
        q1_value = self.critic_1(states, new_actions) # (bsz, 1)
        q2_value = self.critic_2(states, new_actions) # (bsz, 1)
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
            'epochs': self.epochs
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
        self.epochs = checkpoint['epochs']
        print(f"Model loaded from {self.model_path}")