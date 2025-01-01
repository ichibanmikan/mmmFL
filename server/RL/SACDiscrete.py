import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, action_dim)
        
    def forward(self, x):
        x = F.relu(self.l1(x))
        return F.softmax(self.l2(x), dim = 1)
 
class QValueNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width):
        super(QValueNet, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, action_dim)

    def forward(self, x):  
        x = F.relu(self.l1(x))  
        return self.l2(x)
 
class SAC:
    def __init__(self, state_dim, action_dim, hidden_width,
                 actor_lr, critic_lr, alpha_lr,
                 target_entropy, tau, gamma, device):

        self.actor = Actor(state_dim, action_dim, hidden_width).to(device)
        
        self.critic_1 = QValueNet(state_dim, action_dim, hidden_width).to(device)
        self.critic_2 = QValueNet(state_dim, action_dim, hidden_width).to(device)

        self.target_critic_1 = QValueNet(state_dim, action_dim, hidden_width).to(device)
        self.target_critic_2 = QValueNet(state_dim, action_dim, hidden_width).to(device)
 
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
 
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True
        
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau
        self.device = device

    def take_action(self, state):
        # numpy[n_states]-->tensor[1,n_states]
        state = torch.tensor(state[np.newaxis,:], dtype=torch.float).to(self.device)
        # [1,n_actions]
        probs = self.actor(state)
        
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample().item()
        
        return action
    
    # now state_value
    def calc_target(self, rewards, next_states, dones):
        # [b,n_states]-->[b,n_actions]
        next_probs = self.actor(next_states)
        # [b,n_actions]
        next_log_probs = torch.log(next_probs + 1e-8)
        # [b,1]
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdims=True)
        # [b,n_actions]
        q1_value = self.target_critic_1(next_states)
        q2_value = self.target_critic_2(next_states)
        # [b, 1]
        min_qvalue = torch.sum(next_probs * torch.min(q1_value,q2_value), dim=1, keepdims=True)
        # [b, 1]
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        # [b, n_actions]
        td_target = rewards + self.gamma * next_value * (1-dones)
        return td_target.float()
    
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data*(1-self.tau) + param.data*self.tau)
 
    def update(self, transition_dict):
        states = transition_dict['states']  # [b,n_states]
        actions = transition_dict['actions'].view(-1,1)  # [b,1]
        rewards = rewards = transition_dict['rewards'].view(-1, 1)  # [b,1]
        next_states = transition_dict['next_states']  # [b,n_states]
        dones = transition_dict['dones'].view(-1,1)  # [b,1]

        td_target = self.calc_target(rewards, next_states, dones)
        
        critic_1_qvalues = self.critic_1(states).gather(1, actions)
        critic_1_loss = torch.mean(F.mse_loss(critic_1_qvalues, td_target.detach()))
        
        critic_2_qvalues = self.critic_2(states).gather(1, actions)
        critic_2_loss = torch.mean(F.mse_loss(critic_2_qvalues, td_target.detach()))
        
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()

        critic_1_loss.backward()
        critic_2_loss.backward()

        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()
 
        probs = self.actor(states)  # [b,n_actions]
        log_probs = torch.log(probs + 1e-8)  # [b,n_actions]
        # [b,1]
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)

        q1_value = self.critic_1(states)  # [b,n_actions]
        q2_value = self.critic_2(states)
        # [b,1]
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)
 
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
 
        alpha_loss = torch.mean((entropy-self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
 
        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)
