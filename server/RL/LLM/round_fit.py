import torch
from torch import nn

class round_fit_model(nn.Module):
    def __init__(self, input_dim, output_dim = 1, device = 'cuda'):
        super().__init__()
        self.model = nn.Linear(input_dim, output_dim)
        self.device = device
        self.to(device)

    def forward(self, x):
        x = torch.tensor(x).float().to(self.device)
        return self.model(x)

class round_fit:
    def __init__(self, input_dim, output_dim = 1, device = 'cuda'):
        self.model = round_fit_model(input_dim, output_dim, device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.001)
        self.loss = nn.MSELoss()
        
    def train(self, accs):
        self.optimizer.zero_grad()
        pred = self.model(accs).squeeze(-1)
        labels = torch.arange(1, len(accs) + 1).float().to(self.model.device)
        loss = self.loss(pred, labels)
        loss.backward()
        self.optimizer.step()
    
    def get_prob_length(self, x):
        return torch.round(self.model(x)).detach().cpu().numpy()