import pdb
import torch
import numpy as np
import torch.nn as nn

from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence

# typing import
from typing import Dict, Iterable, Optional

class FuseBaseSelfAttention(nn.Module):
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8421023
    def __init__(
        self, 
        d_hid:  int=64,
        d_head: int=4,
        dropout: float=0.5
    ):
        super().__init__()
        self.att_fc1 = nn.Linear(d_hid, 1024)
        self.att_pool = nn.Tanh()
        self.att_fc2 = nn.Linear(1024, d_head)

        self.d_hid = d_hid
        self.d_head = d_head
        
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        val_a=None,
        val_b=None,
        a_len=None
    ):
        att = self.att_pool(self.att_fc1(x))
        # att = self.att_fc2(att).squeeze(-1)
        # att = self.dropout(att)
        att = self.att_fc2(att)
        att = att.transpose(1, 2)
        if val_a is not None:
            for idx in range(len(val_a)):
                att[idx, :, val_a[idx]:a_len] = -1e5
                att[idx, :, a_len+val_b[idx]:] = -1e5
        att = torch.softmax(att, dim=2)
        # x = torch.matmul(att, x).mean(axis=1)
        x = torch.matmul(att, x)
        # x = self.dropout(x)
        x = x.reshape(x.shape[0], self.d_head*self.d_hid)
        return x

class ImageTextClassifier(nn.Module):
    def __init__(
        self, 
        num_classes: int,       # Number of classes 
        img_input_dim: int,     # Image data input dim
        text_input_dim: int,    # Text data input dim
        d_hid: int=64,          # Hidden Layer size
        d_head: int=6           # Head dim
    ):
        super(ImageTextClassifier, self).__init__()
        self.dropout_p = 0.8
        
        # Projection head
        self.img_proj = nn.Sequential(
            nn.Linear(img_input_dim, d_hid),
            # nn.BatchNorm1d(d_hid*2),
            # nn.ReLU(),
            # nn.Dropout(self.dropout_p),
            # nn.Linear(d_hid*2, d_hid),
            nn.BatchNorm1d(d_hid),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(d_hid, d_hid),
            # nn.Dropout(self.dropout_p),
        )
            
        # RNN module
        self.text_rnn = nn.GRU(
            input_size=text_input_dim, 
            hidden_size=d_hid, 
            num_layers=1, 
            batch_first=True, 
            # dropout=self.dropout_p, 
            bidirectional=False
        )
        
        self.dropout = nn.Dropout(self.dropout_p)

        self.fuse_att = FuseBaseSelfAttention(
            d_hid=d_hid,
            d_head=d_head
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_hid*d_head, 64),
            # nn.BatchNorm1d(d_hid),
            # nn.ReLU(),
            # nn.Dropout(self.dropout_p),
            # nn.Linear(d_hid, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
            
        self.init_weight()
        
    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv1d:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x_img, x_text, len_i, len_t):
        # 1. img proj
        x_img = self.img_proj(x_img[:, 0, :])
        
        x_text = pack_padded_sequence(
            x_text, 
            len_t.cpu().numpy(), 
            batch_first=True, 
            enforce_sorted=False
        )
        x_text, _ = self.text_rnn(x_text)
        x_text, _ = pad_packed_sequence(x_text, batch_first=True)

        x_mm = torch.cat((x_img.unsqueeze(dim=1), x_text), dim=1)
        x_mm = self.fuse_att(x_mm, len_i, len_t, 1)
        # 4. MM embedding and predict
        preds = self.classifier(x_mm)
        return preds.squeeze(1), x_mm