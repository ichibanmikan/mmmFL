import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class reward_decoder_model(nn.Module):
    def __init__(
        self, 
        input_dim, 
        output_dim = 1,
        device = 'cuda'
        ):
        super().__init__()
        self.model = nn.Linear(input_dim, output_dim)
        self.device = device
        self.to(device)
        nn.init.constant_(self.model.weight, 0.125)
        nn.init.constant_(self.model.bias, 1e-4)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(self.device)
        return self.model(x)
    
    # def save_model(self):
    #     torch.save({'RewardDecoder': self.state_dict()}, self.path,)
    
    # def load_model(self, path):
    #     checkpoint = torch.load(path, weights_only=True)
    #     new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    #     self.load_state_dict(new_state_dict)

class RewardDecoder:
    def __init__(self, 
                 input_dim, 
                 output_dim = 1,
                 path = os.path.join(
                     os.path.dirname(
                         os.path.dirname(os.path.abspath(__file__))), 'RLModel', 'RewardDecoder.pth')):
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.model = reward_decoder_model(input_dim, output_dim, self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = 0.001)
        self.loss = nn.MSELoss()
        self.path = path
        if os.path.exists(self.path):
            self.load_model()
    
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
            e_r = e_r.expand_as(d_r)
            loss = self.loss(e_r, d_r)
            loss.backward()
            self.optimizer.step()
            start_pos = start_pos + episode_length[i]
    
    def get_dense_rewards(self, x):
        return self.model(x)
    
    def save_model(self):
        torch.save({'RewardDecoder': self.model.state_dict(),
                    'RewardDecoderOptimizer': self.optimizer.state_dict()}, 
                   self.path)
    
    def load_model(self):
        checkpoint = torch.load(self.path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['RewardDecoder'])
        self.optimizer.load_state_dict(checkpoint['RewardDecoderOptimizer'])
