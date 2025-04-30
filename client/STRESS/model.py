import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class STRESSClassifier(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=3):
        super(STRESSClassifier, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(6, 32),  # Input layer: 6 -> 256
            nn.Linear(32, 64),  # Hidden layer: 256 -> 128
            nn.Linear(64, 128),   # Output layer: 128 -> num_classes
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.sequential(x)
        return x