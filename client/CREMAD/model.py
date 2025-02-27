import torch
import torch.nn as nn 
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence

class Conv1dEncoder(nn.Module):
    def __init__(self, input_dim: int, n_filters: int, dropout: float=0.1):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(input_dim, n_filters, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            nn.Conv1d(n_filters, n_filters*2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            nn.Conv1d(n_filters*2, n_filters*4, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout)
        )
        self.residual = nn.Sequential(
            nn.Conv1d(input_dim, n_filters*4, kernel_size=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        # self.pooling = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.float().permute(0, 2, 1)
        residual = self.residual(x)
        x = self.convs(x) + residual
        # x = self.pooling(x)
        # x = self.dropout(x)
        x = x.permute(0, 2, 1)
        return x

class FuseBaseSelfAttention(nn.Module):
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8421023
    def __init__(
        self, 
        d_hid:  int=64,
        d_head: int=4
    ):
        super().__init__()
        self.att_fc1 = nn.Linear(d_hid*2, 512)
        self.att_pool = nn.Tanh()
        self.att_fc2 = nn.Linear(512, d_head)

        self.d_hid = d_hid
        self.d_head = d_head

    def forward(
        self,
        x: Tensor,
        val_a=None, # 600
        val_b=None, # 6
        a_len=None # 300
    ):
        att = self.att_pool(self.att_fc1(x))
        # att = self.att_fc2(att).squeeze(-1)
        att = self.att_fc2(att) # (bsz, 306, d_head)
        att = att.transpose(1, 2) # (bsz, d_head, 306)
        if val_a is not None:
            for idx in range(len(val_a)):
                att[idx, :, val_a[idx]:a_len] = -1e5
                att[idx, :, a_len+val_b[idx]:] = -1e5
        att = torch.softmax(att, dim=2)
        # x = torch.matmul(att, x).mean(axis=1)
        x = torch.matmul(att, x) 
        # # (bsz, d_head, 306) @ (bsz, 306, d_head) -> (bsz, d_head, 2*d_hid)
        x = x.reshape(x.shape[0], self.d_head*self.d_hid*2) #(bsz, d_head*2*d_hid)
        return x

class MMActionClassifier(nn.Module):
    def __init__(
        self, 
        num_classes: int=4,       # Number of classes 
        audio_input_dim: int=80,   # Audio feature input dim
        video_input_dim: int=1280,   # Frame-wise video feature input dim
        d_hid: int=128,         # Hidden Layer size
        n_filters: int=32,      # number of filters
        d_head: int=6           # Head dim
    ):
        super(MMActionClassifier, self).__init__()
        self.dropout_p = 0
        
        # Conv Encoder module
        self.audio_conv = Conv1dEncoder(
            input_dim=audio_input_dim, 
            n_filters=n_filters, 
            dropout=self.dropout_p, 
        )
        
        # RNN module
        self.audio_rnn = nn.GRU(n_filters*4, d_hid, num_layers=2, \
            batch_first=True, bidirectional=True)
        self.video_rnn = nn.GRU(video_input_dim, d_hid, num_layers=2, \
            batch_first=True, bidirectional=True)

        self.fuse_att = FuseBaseSelfAttention(
            d_hid=d_hid,
            d_head=d_head
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_hid*d_head*2, d_hid),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(d_hid, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(64, num_classes)
        )
            
         # Projection head
        self.init_weight()

    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv1d:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x_audio, x_video, len_a, len_v):
        # 1. Conv forward
        x_audio = self.audio_conv(x_audio) # (bsz, 600, 80) -> (bsz, 300, n_filters*4)
        
        # 2. Rnn forward
        # max pooling, time dim reduce by 8 times
        len_a = len_a//8
        x_audio = pack_padded_sequence(
            x_audio, 
            len_a, 
            batch_first=True, 
            enforce_sorted=False
        )
        x_video = pack_padded_sequence(
            x_video, 
            len_v, 
            batch_first=True, 
            enforce_sorted=False
        )

        x_audio, _ = self.audio_rnn(x_audio) # (batch_size, 300, 2*d_hid)
        x_video, _ = self.video_rnn(x_video) # (batch_size, 6, 2*d_hid)

        x_audio, _ = pad_packed_sequence(   
            x_audio, 
            batch_first=True
        )
        
        x_video, _ = pad_packed_sequence(
            x_video, 
            batch_first=True
        )
        a_max_len = x_audio.shape[1] # 300
        x_mm = torch.cat((x_audio, x_video), dim=1)
        # (batch_size, 300, 2*d_hid) + (batch_size, 6, 2*d_hid) -> (batch_size, 306, 2*d_hid)
        x_mm = self.fuse_att(x_mm, len_a, len_v, a_max_len) #(bsz, d_head*2*d_hid)

        # 6. MM embedding and predict
        preds = self.classifier(x_mm)
        return preds

# class Conv1dEncoder(nn.Module):
#     def __init__(
#         self,
#         input_dim: int, 
#         n_filters: int,
#         dropout: float=0.1
#     ):
#         super().__init__()
#         # conv module
#         self.conv1 = nn.Conv1d(input_dim, n_filters, kernel_size=5, padding=2)
#         self.conv2 = nn.Conv1d(n_filters, n_filters*2, kernel_size=5, padding=2)
#         self.conv3 = nn.Conv1d(n_filters*2, n_filters*4, kernel_size=5, padding=2)
#         self.relu = nn.ReLU()
#         self.pooling = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(
#             self,
#             x: Tensor   # shape => [batch_size (B), num_data (T), feature_dim (D)]
#         ):
#         x = x.float()
#         x = x.permute(0, 2, 1)
#         # conv1
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.pooling(x)
#         x = self.dropout(x)
#         # conv2
#         x = self.conv2(x)
#         x = self.relu(x)
#         x = self.pooling(x)
#         x = self.dropout(x)
#         # conv3
#         x = self.conv3(x)
#         x = self.relu(x)
#         x = self.pooling(x)
#         x = self.dropout(x)
#         x = x.permute(0, 2, 1)
#         return x

# class FuseBaseSelfAttention(nn.Module):
#     # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8421023
#     def __init__(
#         self, 
#         d_hid:  int=64,
#         d_head: int=4
#     ):
#         super().__init__()
#         self.att_fc1 = nn.Linear(d_hid, 512)
#         self.att_pool = nn.Tanh()
#         self.att_fc2 = nn.Linear(512, d_head)

#         self.d_hid = d_hid
#         self.d_head = d_head

#     def forward(
#         self,
#         x: Tensor,
#         val_a=None,
#         val_b=None,
#         a_len=None
#     ):
#         att = self.att_pool(self.att_fc1(x))
#         # att = self.att_fc2(att).squeeze(-1)
#         att = self.att_fc2(att)
#         att = att.transpose(1, 2)
#         if val_a is not None:
#             for idx in range(len(val_a)):
#                 att[idx, :, val_a[idx]:a_len] = -1e5
#                 att[idx, :, a_len+val_b[idx]:] = -1e5
#         att = torch.softmax(att, dim=2)
#         # x = torch.matmul(att, x).mean(axis=1)
#         x = torch.matmul(att, x)
#         x = x.reshape(x.shape[0], self.d_head*self.d_hid)
#         return x

# class MMActionClassifier(nn.Module):
#     def __init__(
#         self, 
#         num_classes: int=4,       # Number of classes 
#         audio_input_dim: int=80,   # Audio feature input dim
#         video_input_dim: int=1280,   # Frame-wise video feature input dim
#         d_hid: int=128,         # Hidden Layer size
#         n_filters: int=32,      # number of filters
#         en_att: bool=True,     # Enable self attention or not
#         att_name: str='fuse_base',       # Attention Name
#         d_head: int=6           # Head dim
#     ):
#         super(MMActionClassifier, self).__init__()
#         self.dropout_p = 0
#         self.en_att = en_att
#         self.att_name = att_name
        
#         # Conv Encoder module
#         self.audio_conv = Conv1dEncoder(
#             input_dim=audio_input_dim, 
#             n_filters=n_filters, 
#             dropout=self.dropout_p, 
#         )
        
#         # RNN module
#         self.audio_rnn = nn.GRU(
#             input_size=n_filters*4, 
#             hidden_size=d_hid,
#             num_layers=1, 
#             batch_first=True, 
#             dropout=self.dropout_p, 
#             bidirectional=False
#         )

#         self.video_rnn = nn.GRU(
#             input_size=video_input_dim, 
#             hidden_size=d_hid, 
#             num_layers=1, 
#             batch_first=True, 
#             dropout=self.dropout_p, 
#             bidirectional=False
#         )
        

#         self.fuse_att = FuseBaseSelfAttention(
#             d_hid=d_hid,
#             d_head=d_head
#         )

#         self.classifier = nn.Sequential(
#             nn.Linear(d_hid*d_head, 64),
#             nn.ReLU(),
#             nn.Dropout(self.dropout_p),
#             nn.Linear(64, num_classes)
#         )
            
#          # Projection head
#         self.init_weight()

#     def init_weight(self):
#         for m in self._modules:
#             if type(m) == nn.Linear:
#                 torch.nn.init.xavier_uniform(m.weight)
#                 m.bias.data.fill_(0.01)
#             if type(m) == nn.Conv1d:
#                 torch.nn.init.xavier_uniform(m.weight)
#                 m.bias.data.fill_(0.01)

#     def forward(self, x_audio, x_video, len_a, len_v):
#         # 1. Conv forward
#         x_audio = self.audio_conv(x_audio)
        
#         # 2. Rnn forward
#         # max pooling, time dim reduce by 8 times
#         len_a = len_a//8
#         x_audio = pack_padded_sequence(
#             x_audio, 
#             len_a, 
#             batch_first=True, 
#             enforce_sorted=False
#         )
#         x_video = pack_padded_sequence(
#             x_video, 
#             len_v, 
#             batch_first=True, 
#             enforce_sorted=False
#         )

#         x_audio, _ = self.audio_rnn(x_audio) 
#         x_video, _ = self.video_rnn(x_video) 

#         x_audio, _ = pad_packed_sequence(   
#             x_audio, 
#             batch_first=True
#         )
        
#         x_video, _ = pad_packed_sequence(
#             x_video, 
#             batch_first=True
#         )
#         a_max_len = x_audio.shape[1]
#         x_mm = torch.cat((x_audio, x_video), dim=1)
#         x_mm = self.fuse_att(x_mm, len_a, len_v, a_max_len)

#         # 6. MM embedding and predict
#         preds = self.classifier(x_mm)
#         return preds