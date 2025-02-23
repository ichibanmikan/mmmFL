import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class reward_decoder_model(nn.Module):
    def __init__(
        self, 
        input_dim, 
        output_dim = 2,
        device = 'cuda',
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RLModel', 'RewardDecoder.pth')
        ):
        super().__init__()
        self.path = path
        if os.path.exists(path):
            self.load_model(path)
        else:
            self.model = nn.Linear(input_dim, output_dim)
        self.device = device
        self.to(device)
        nn.init.xavier_normal_(self.model.weight)
        nn.init.zeros_(self.model.bias)
        
    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        return torch.round(self.model(x))
    
    def save_model(self):
        torch.save(self.state_dict(), self.path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))

class RewardDecoder:
    def __init__(self, input_dim, output_dim = 1):
        self.model = reward_decoder_model(input_dim, output_dim)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = 0.001)
        self.loss = nn.MSELoss()
    
    def episode_reward(self, p, m, t):
        return -math.log((p - t) / (m - t))
    
    def train(self, practice_length, max_length, min_length, latent_rewards, episode_length):
        start_pos = 0
        for i in range(len(episode_length)):
            l = latent_rewards[start_pos : start_pos + episode_length[i]]
            e_r = self.episode_reward(practice_length, max_length, min_length)
            self.optimizer.zero_grad()
            dense_rewards = self.model(l)
            d_r = torch.sum(dense_rewards, dim=0)
            e_r = torch.tensor(e_r, dtype=torch.float32, device=self.model.device)
            print(e_r.shape)
            print(d_r.shape)
            loss = self.loss(e_r, d_r)
            loss.backward()
            self.optimizer.step()
            start_pos = start_pos + episode_length[i]
    
    def get_dense_rewards(self, x):
        return self.model(x)
    
    def save_model(self):
        self.model.save_model()
    
    def load_model(self):
        self.model.load_model()
