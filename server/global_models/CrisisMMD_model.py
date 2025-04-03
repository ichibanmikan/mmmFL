import pdb
import pickle
import torch
import numpy as np
import torch.nn as nn
import os
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence

from torch.utils.data import DataLoader, Dataset

class FuseBaseSelfAttention(nn.Module):
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8421023
    def __init__(
        self, 
        d_hid:  int=64,
        d_head: int=4,
        dropout: float=0.5
    ):
        super().__init__()
        self.att_fc1 = nn.Linear(d_hid, 512)
        self.att_pool = nn.Tanh()
        self.att_fc2 = nn.Linear(512, d_head)

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
        att = self.dropout(att)
        att = self.att_fc2(att)
        att = att.transpose(1, 2)
        if val_a is not None:
            for idx in range(len(val_a)):
                att[idx, :, val_a[idx]:a_len] = -1e5
                att[idx, :, a_len+val_b[idx]:] = -1e5
        att = torch.softmax(att, dim=2)
        # x = torch.matmul(att, x).mean(axis=1)
        x = torch.matmul(att, x)
        x = self.dropout(x)
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
        self.dropout_p = 0.5
        
        # Projection head
        self.img_proj = nn.Sequential(
            nn.Linear(img_input_dim, d_hid),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            # nn.Linear(d_hid*2, d_hid),
            # nn.ReLU(),
            # nn.Dropout(self.dropout_p),
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
        
        # self.dropout = nn.Dropout(self.dropout_p)

        self.fuse_att = FuseBaseSelfAttention(
            d_hid=d_hid,
            d_head=d_head,
            dropout=self.dropout_p
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_hid*d_head, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            # nn.Linear(d_hid*2, d_hid),
            # nn.ReLU(),
            # nn.Dropout(self.dropout_p),
            # nn.Linear(d_hid, 64),
            # nn.ReLU(),
            # nn.Dropout(self.dropout_p),
            nn.Linear(64, num_classes)
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
        
        # x_text = self.dropout(x_text)

        # get attention output
        x_mm = torch.cat((x_img.unsqueeze(dim=1), x_text), dim=1)
        x_mm = self.fuse_att(x_mm, len_i, len_t, 1)
        # 4. MM embedding and predict
        preds = self.classifier(x_mm)
        # if(torch.isnan(preds).any()):
        #     print("preds is nan")
        #     print(x_mm)
        preds = torch.log_softmax(preds, dim=1)
        return preds, x_mm

# class FuseBaseSelfAttention(nn.Module):
#     # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8421023
#     def __init__(
#         self, 
#         d_hid:  int=64,
#         d_head: int=4,
#         dropout: float=0.5
#     ):
#         super().__init__()
#         self.att_fc1 = nn.Linear(d_hid, 512)
#         self.att_pool = nn.Tanh()
#         self.att_fc2 = nn.Linear(512, d_head)

#         self.d_hid = d_hid
#         self.d_head = d_head
        
#         self.dropout = nn.Dropout(dropout)

#     def forward(
#         self,
#         x: Tensor,
#         val_a=None,
#         val_b=None,
#         a_len=None
#     ):
#         att = self.att_pool(self.att_fc1(x))
#         # att = self.att_fc2(att).squeeze(-1)
#         att = self.dropout(att)
#         att = self.att_fc2(att)
#         att = att.transpose(1, 2)
#         if val_a is not None:
#             for idx in range(len(val_a)):
#                 att[idx, :, val_a[idx]:a_len] = -1e5
#                 att[idx, :, a_len+val_b[idx]:] = -1e5
#         att = torch.softmax(att, dim=2)
#         # x = torch.matmul(att, x).mean(axis=1)
#         x = torch.matmul(att, x)
#         x = self.dropout(x)
#         x = x.reshape(x.shape[0], self.d_head*self.d_hid)
#         return x

# class ImageTextClassifier(nn.Module):
#     def __init__(
#         self, 
#         num_classes: int,       # Number of classes 
#         img_input_dim: int,     # Image data input dim
#         text_input_dim: int,    # Text data input dim
#         d_hid: int=64,          # Hidden Layer size
#         d_head: int=6           # Head dim
#     ):
#         super(ImageTextClassifier, self).__init__()
#         self.dropout_p = 0.5
        
#         # Projection head
#         self.img_proj = nn.Sequential(
#             nn.Linear(img_input_dim, d_hid*2),
#             nn.BatchNorm1d(d_hid*2),
#             nn.Dropout(self.dropout_p),
#             nn.ReLU(),
#             nn.Linear(d_hid*2, d_hid),
#             nn.BatchNorm1d(d_hid),
#             nn.Dropout(self.dropout_p),
#             nn.ReLU(),
#             nn.Linear(d_hid, d_hid),
#         )
            
#         # RNN module
#         self.text_rnn = nn.GRU(
#             input_size=text_input_dim, 
#             hidden_size=d_hid, 
#             num_layers=2, 
#             batch_first=True, 
#             dropout=self.dropout_p, 
#             bidirectional=False
#         )
        
#         # self.dropout = nn.Dropout(self.dropout_p)

#         self.fuse_att = FuseBaseSelfAttention(
#             d_hid=d_hid,
#             d_head=d_head,
#             dropout=self.dropout_p
#         )

#         self.classifier = nn.Sequential(
#             nn.Linear(d_hid*d_head, d_hid),
#             nn.BatchNorm1d(d_hid),
#             nn.Dropout(self.dropout_p),
#             nn.ReLU(),
#             nn.Linear(d_hid, 64),
#             nn.BatchNorm1d(64),
#             nn.Dropout(self.dropout_p),
#             nn.ReLU(),
#             nn.Linear(64, num_classes)
#         )
            
#         self.init_weight()
        
#     def init_weight(self):
#         for m in self._modules:
#             if type(m) == nn.Linear:
#                 torch.nn.init.xavier_uniform(m.weight)
#                 m.bias.data.fill_(0.01)
#             if type(m) == nn.Conv1d:
#                 torch.nn.init.xavier_uniform(m.weight)
#                 m.bias.data.fill_(0.01)

#     def forward(self, x_img, x_text, len_i, len_t):
#         # 1. img proj
#         x_img = self.img_proj(x_img[:, 0, :])
        
#         x_text = pack_padded_sequence(
#             x_text, 
#             len_t.cpu().numpy(), 
#             batch_first=True, 
#             enforce_sorted=False
#         )
#         x_text, _ = self.text_rnn(x_text)
#         x_text, _ = pad_packed_sequence(x_text, batch_first=True)
        
#         # x_text = self.dropout(x_text)

#         # get attention output
#         x_mm = torch.cat((x_img.unsqueeze(dim=1), x_text), dim=1)
#         x_mm = self.fuse_att(x_mm, len_i, len_t, 1)
#         # 4. MM embedding and predict
#         preds = self.classifier(x_mm)
#         # if(torch.isnan(preds).any()):
#         #     print("preds is nan")
#         #     print(x_mm)
#         return preds, x_mm

def pad_tensor(vec, pad, dim=0):
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size, dtype=vec.dtype, device=vec.device)], dim=dim)

def collate_fn_padd(batch):
    texts, imgs, text_len, img_len, labels = zip(*batch)

    max_text_len = max(t.shape[0] for t in texts)
    max_img_len = max(i.shape[0] for i in imgs) if imgs[0].ndim > 0 else 0

    padded_texts = []
    for text in texts:
        if text.ndim == 1:
            padded_text = pad_tensor(text, max_text_len, 0)
        else:
            padded_text = pad_tensor(text, max_text_len, 0)
        padded_texts.append(padded_text)

    padded_imgs = []
    for img in imgs:
        if img.ndim == 1:
            padded_img = pad_tensor(img, max_img_len, 0) if max_img_len > 0 else img
        else:
            padded_img = pad_tensor(img, max_img_len, 0) if max_img_len > 0 else img
        padded_imgs.append(padded_img)

    padded_texts = torch.stack(padded_texts, dim=0)
    padded_imgs = torch.stack(padded_imgs, dim=0) if max_img_len > 0 else torch.stack(imgs, dim=0)
    labels = torch.stack(labels, dim=0)

    text_len = torch.stack(text_len, dim=0)
    img_len = torch.stack(img_len, dim=0)
    
    return padded_texts, padded_imgs, text_len, img_len, labels

class CrisisMMDSet(Dataset):
    def __init__(self, data_dir, device="cuda"):
        self.data = []
        pkl_path = os.path.join(data_dir, 'test.pkl')
        if os.path.exists(pkl_path):
            self.data.extend(pickle.load(open(pkl_path, "rb")))
        self.device = device
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text_feature = self.data[index]["text"]
        img_feature = self.data[index]["img"]
        label = self.data[index]["label"]
        return (torch.tensor(text_feature, dtype=torch.float32),
                torch.tensor(img_feature, dtype=torch.float32),
                torch.tensor(text_feature.shape[0], dtype=torch.long),
                torch.tensor(img_feature.shape[0], dtype=torch.long),
                torch.tensor(label, dtype=torch.long))


class CrisisMMD:
    def __init__(self, device):
        self.model = ImageTextClassifier(
            num_classes=8,
            img_input_dim=1280,
            text_input_dim=512,
            d_hid=256,
            d_head=8
        ) 
        self.device = device
        self.load_model(
            os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)), f'models/{self.get_model_name()}.pth'))
        self.model = self.model.to(device)
        self.test_loader = \
            DataLoader(CrisisMMDSet('/home/chenxu/codes/ichibanFATE/server/test_datasets/CrisisMMD'), \
                batch_size=16, num_workers=16, collate_fn=collate_fn_padd)
        self.Tester = Tester(self.model, test_loader=self.test_loader, device=device)
    def get_model_params(self):
            
        params = []
        for param in self.model.parameters():
            if torch.cuda.is_available() or torch.backends.mps.is_available():
                params.extend(param.view(-1).cpu().detach().numpy())
            else :
                params.extend(param.view(-1).detach().numpy())
            # print(param)

        # model_params = params.cpu().numpy()
        model_params = np.array(params)
        # print("Shape of model weight: ", model_params.shape) # 823468

        return model_params

    def reset_model_parameter(self, new_params):
        temp_index = 0
        with torch.no_grad():
            for param in self.model.parameters():
                if len(param.shape) == 2:

                    para_len = int(param.shape[0] * param.shape[1])
                    temp_weight = new_params[temp_index : temp_index + para_len].astype(float)
                    param.copy_(torch.Tensor(temp_weight.reshape(param.shape[0], param.shape[1])).to(param.device))
                    temp_index += para_len

                elif len(param.shape) == 3:

                    para_len = int(param.shape[0] * param.shape[1] * param.shape[2])
                    temp_weight = new_params[temp_index : temp_index + para_len].astype(float)
                    param.copy_(torch.Tensor(temp_weight.reshape(param.shape[0], param.shape[1], param.shape[2])).to(param.device))
                    temp_index += para_len 

                elif len(param.shape) == 4:

                    para_len = int(param.shape[0] * param.shape[1] * param.shape[2] * param.shape[3])
                    temp_weight = new_params[temp_index : temp_index + para_len].astype(float)
                    param.copy_(torch.Tensor(temp_weight.reshape(param.shape[0], param.shape[1], param.shape[2], param.shape[3])).to(param.device))
                    temp_index += para_len  

                elif len(param.shape) == 5:

                    para_len = int(param.shape[0] * param.shape[1] * param.shape[2] * param.shape[3] * param.shape[4])
                    temp_weight = new_params[temp_index : temp_index + para_len].astype(float)
                    param.copy_(torch.Tensor(temp_weight.reshape(param.shape[0], param.shape[1], param.shape[2], param.shape[3], param.shape[4])).to(param.device))
                    temp_index += para_len  

                else:

                    para_len = param.shape[0]
                    temp_weight = new_params[temp_index : temp_index + para_len].astype(float)
                    param.copy_(torch.Tensor(temp_weight).to(param.device))
                    temp_index += para_len    
                    
    def save_model(self, save_file):
        print('==> Saving...')

        torch.save(self.model.cpu().state_dict(), save_file)

    def load_model(self, load_file):
        if os.path.exists(load_file):
            print(f'==> Loading model from {load_file}...')
            self.model.load_state_dict(torch.load(load_file, map_location=self.device, weights_only=True))
            self.model.to(self.device)
        else:
            print(f'==> Model file {load_file} not found. Using initialized model.')


    def get_model_name(self):
        return "CrisisMMD"
        
    
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        # print(correct)

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Tester:
    def __init__(self, model, test_loader, device):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        
    def test(self):
        if next(self.model.parameters()).device != self.device:
            self.model = self.model.to(self.device)
        self.model.eval()
        accs = AverageMeter()

        with torch.no_grad():
            for text, img, text_len, img_len, label in self.test_loader:
                output = None
                
                bsz = text.shape[0]
                text, img = text.to(self.device), img.to(self.device)
                text_len, img_len = text_len.to(self.device), img_len.to(self.device)
                label = label.to(self.device)
                output, _ = self.model(img, text, img_len, text_len)
                acc, _ = accuracy(output, label, topk=(1, 5))
                accs.update(acc, bsz)
        # print(accs.avg)
        return accs.avg.cpu().item()