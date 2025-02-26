import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import os
import random

class Actor(nn.Module):
    def __init__(self, N, hidden_width = 128, action_width = 4):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(3 * N + 1, hidden_width) 
        # (bsz, 3N + 1) @ (3N + 1, hidden_width)
        self.l2 = nn.Linear(hidden_width, action_width) 
        # (bsz, hidden_width) @ (hidden_width, 4)
        
        nn.init.kaiming_normal_(self.l1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.l1.bias, 0.0)
        nn.init.uniform_(self.l2.weight, -1e-3, 1e-3)
        nn.init.constant_(self.l2.bias, 0.0)
        
    def forward(self, x):
        x = F.relu(self.l1(x))
        return F.softmax(self.l2(x), dim = -1)
        # (batch_size, 4), means \pai(.|x)
    #Action a means select job (a - 1) unless a == 0

class QValueNet(nn.Module):
    def __init__(self, N, hidden_width, action_width = 4):
        super(QValueNet, self).__init__()
        self.l1 = nn.Linear(3 * N + 1, hidden_width)
        # (bsz, 3 * N + 1) @ (3 * N + 1, h_d)
        self.l2 = nn.Linear(hidden_width, action_width)
        # (bsz, h_d) @ (h_d, 4)
        
        nn.init.kaiming_normal_(self.l1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.l1.bias, 0.0)
        nn.init.uniform_(self.l2.weight, -1e-3, 1e-3)
        nn.init.constant_(self.l2.bias, 0.0)

    def forward(self, state):  
        # state: (bsz, 3 * N + 1) 
        state = F.relu(self.l1(state))
        return self.l2(state)
        # (bsz, 4)
 
class SACDiscrete:
    def __init__(self, N, hidden_dim,
                 actor_lr, critic_lr, alpha_lr,
                 target_entropy, tau, gamma, device, 
                 model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RLModel', 'SACDiscrete.pth')
                 ):

        self.actor = Actor(N = N, hidden_width = hidden_dim).to(device)
        
        self.critic_1 = QValueNet(N = N, hidden_width = hidden_dim).to(device)
        self.critic_2 = QValueNet(N = N, hidden_width = hidden_dim).to(device)

        self.target_critic_1 = QValueNet(N = N, hidden_width = hidden_dim).to(device)
        self.target_critic_2 = QValueNet(N = N, hidden_width = hidden_dim).to(device)
        
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
        self.model_path = model_path
        if os.path.exists(model_path):
            print(f"Loading model from {self.model_path}...")
            self.load_model()
        else:
            self.target_critic_1.load_state_dict(self.critic_1.state_dict())
            self.target_critic_2.load_state_dict(self.critic_2.state_dict())
            
    def take_action(self, state, take_next = False):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        probs = self.actor(state)
        probs = torch.clamp(probs, min=1e-8)
        action_dist = torch.distributions.Categorical(probs)
        
        action = action_dist.sample().item()
        return action
    
    # now state_value
    def calc_target(self, rewards, next_states, dones):
        next_prob = self.actor(next_states) # (batch_size, 4)
        next_prob = torch.clamp(next_prob, min=1e-8) # (batch_size, 4)
        
        next_log_probs = torch.log(next_prob) # (batch_size, 4)
        
        entropy = -torch.sum(next_prob * next_log_probs, dim = -1, keepdims=True) 
        # (batch_size, 1)

        q1 = self.target_critic_1(next_states) 
        # (batch_size, 4)
        q2= self.target_critic_2(next_states)
        # (batch_size, 4)
        
        min_qvalue = torch.sum(next_prob * torch.min(q1, q2), dim = -1, keepdims=True)
        # (batch_size, 1)

        next_value = min_qvalue + self.log_alpha.exp() * entropy
        # (batch_size, 1)
        
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

        # print("high transition_dict state shape is: ", transition_dict['states'].shape)

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
 
        probs = self.actor(states)
        action_dist = torch.distributions.Categorical(probs)
        log_probs = action_dist.log_prob(actions.squeeze())
        entropy = -torch.mean(log_probs)

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