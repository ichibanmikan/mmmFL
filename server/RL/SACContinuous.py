import os
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.distributions import Normal

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
    def __init__(self, hidden_dim):
        super(Actor, self).__init__()
        # self.attention = AttentionLayer(1, hidden_dim)
        # self.l1 = nn.Linear(hidden_dim + 2, hidden_dim + 2)
        self.l1 = nn.Linear(3, hidden_dim)
        self.l_mean = nn.Linear(hidden_dim, 1)
        self.l_std = nn.Linear(hidden_dim, 1) 

    def forward(self, x):
        # Input x: (bsz, 3), 
        # Ιnclude ti, model_size, T_remain.
        # s_r = x[:, -2:] # (bsz, 2)
        # x, _ = self.attention(x[:, 0].unsqueeze(-1), x[:, 1:-2]) # (bsz, h_d)
        # x = torch.cat([x, s_r]) # (bsz, h_d + 2)
        x = F.relu(self.l1(x))
        
        x_mean = self.l_mean(x) 
        x_std = F.softplus(self.l_std(x))
        x_std = torch.clamp(x_std, min=1e-6)        
        dist = Normal(x_mean, x_std)
        normal_sample = dist.rsample()
        log_prob = dist.log_prob(normal_sample)
        action = (torch.tanh(normal_sample) + 1) / 2
        action = action.clamp(min=1e-4, max=1.0 - 1e-4)
        log_prob -= torch.log(1 - action.pow(2) + 1e-7)
        
        return action, log_prob

class QValueNet(nn.Module):
    def __init__(self, hidden_dim):
        super(QValueNet, self).__init__()
        # self.attention = AttentionLayer(1, hidden_dim)
        # self.l1 = nn.Linear(hidden_dim + 2, (hidden_dim + 2) * 2) 
        # self.l2 = nn.Linear((hidden_dim + 2) * 2, hidden_dim + 2)
        # self.l3 = nn.Linear(hidden_dim + 2, 1)
        self.l1 = nn.Linear(3 + 1, (hidden_dim) * 2) 
        self.l2 = nn.Linear((hidden_dim) * 2, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        # action.unsqueeze(-1)
        # print("1111111111111")
        # print(state.shape)
        # print(action.shape)
        if action.dim() == 1:
            action = action.unsqueeze(-1)
        x = torch.cat([state, action], dim = -1) # (bsz, h_d + 2)
        x = F.relu(self.l1(x)) # (bsz, (hidden_dim + 2) * 2)
        x = F.relu(self.l2(x)) # (bsz, (hidden_dim + 2))
        return self.l3(x) # (bsz, 1)
 
class SACContinuous:
    def __init__(
        self, hidden_dim, actor_lr, critic_lr, alpha_lr,\
            target_entropy, tau, gamma, device,\
                model_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), 'RLModel', 'SACContinuous.pth'
        )):
        self.actor = Actor(hidden_dim).to(device)
        self.critic_1 = QValueNet(hidden_dim).to(device)
        self.critic_2 = QValueNet(hidden_dim).to(device)
        self.target_critic_1 = QValueNet(hidden_dim).to(device)
        self.target_critic_2 = QValueNet(hidden_dim).to(device)
        
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
        self.epochs = 0
        if os.path.exists(model_path):
            print(f"Loading model from {self.model_path}...")
            self.load_model()
        else:
            self.target_critic_1.load_state_dict(self.critic_1.state_dict())
            self.target_critic_2.load_state_dict(self.critic_2.state_dict())
            
    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device) # (N)
        state = state.squeeze(0) # batch_size: 1
        self.epochs += 1
        if self.epochs <= 50:
            b = np.random.rand()
            if b < 1e-4:
                b = 1e-4
            elif b >= 1 - 1e-4:
                b = 1 - 1e-4
            return b
        action = self.actor(state)[0] # (1)
        return action.cpu().detach().item()

    def calc_target(self, rewards, next_states = None, dones = 1): 
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)
        if dones.dim() == 1:
            dones = dones.unsqueeze(-1)
        next_actions, log_probs = self.actor(next_states) # (bsz, 1), (bsz, 1)
        entropy = -log_probs # (bsz, 1)
        q1_value = self.target_critic_1(next_states, next_actions) # (bsz, 1)
        q2_value = self.target_critic_2(next_states, next_actions) # (bsz, 1)
        
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
        
        for idx, s in enumerate(states):
            if s[0] == -1.0 and s[1] == -1.0 and s[2] == -1.0:
                continue
            else:
                filtered_states.append(s)
                filtered_actions.append(actions[idx])
                filtered_rewards.append(rewards[idx])
                filtered_dones.append(dones[idx])
                ns = next_states[idx]
                if ns[0] == -1.0 and ns[1] == -1.0 and ns[2] == -1.0:
                    filtered_next_states.append(torch.full_like(ns, 0))
                else:
                    filtered_next_states.append(ns)

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