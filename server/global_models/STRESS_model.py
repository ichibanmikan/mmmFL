import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd

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

class STRESS_set(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_csv(
            os.path.join(os.path.dirname(__file__), '../test_datasets/STRESS/test.csv')).to_numpy()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return torch.tensor(self.data[index][0:6], dtype=torch.float32), \
            torch.tensor(self.data[index][6], dtype=torch.long)

class STRESS:
    def __init__(self, device):
        self.now_loss = 999
        self.model = STRESSClassifier()
        self.device = device
        self.load_model(
            os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)), f'models/{self.get_model_name()}.pth'))
        self.model = self.model.to(device)
        
        dl = DataLoader(STRESS_set(), batch_size=1024, num_workers = 4)
        self.Tester = Tester(self.model, dl, device)
    
    def get_model_name(self):
        return "STRESS"
    
    def get_model_params(self):
    
        params = []
        for param in self.model.parameters():
            if torch.cuda.is_available() or torch.backends.mps.is_available():
                params.extend(param.view(-1).cpu().detach().numpy())
            else:
                params.extend(param.view(-1).detach().numpy())
            # print(param)

        # model_params = params.cpu().numpy()
        model_params = np.array(params)
        # print("Shape of model weight: ", model_params.shape)#39456

        return model_params

    def reset_model_parameter(self, new_params):
        
        temp_index = 0

        with torch.no_grad():
            for param in self.model.parameters():

                # print(param.shape)

                if len(param.shape) == 2:

                    para_len = int(param.shape[0] * param.shape[1])
                    # print(para_len)
                    # print(temp_index)
                    # print(len(new_params))
                    temp_weight = new_params[temp_index : temp_index + para_len].astype(float)
                    # print(len(temp_weight))
                    param.copy_(torch.Tensor(temp_weight.reshape(param.shape[0], param.shape[1])))
                    temp_index += para_len

                elif len(param.shape) == 4:

                    para_len = int(param.shape[0] * param.shape[1] * param.shape[2] * param.shape[3])
                    temp_weight = new_params[temp_index : temp_index + para_len].astype(float)
                    param.copy_(torch.Tensor(temp_weight.reshape(param.shape[0], param.shape[1], param.shape[2], param.shape[3])))
                    temp_index += para_len  

                elif len(param.shape) == 5:

                    para_len = int(param.shape[0] * param.shape[1] * param.shape[2] * param.shape[3] * param.shape[4])
                    temp_weight = new_params[temp_index : temp_index + para_len].astype(float)
                    param.copy_(torch.Tensor(temp_weight.reshape(param.shape[0], param.shape[1], param.shape[2], param.shape[3], param.shape[4])))
                    temp_index += para_len  

                else:

                    para_len = param.shape[0]
                    temp_weight = new_params[temp_index : temp_index + para_len].astype(float)
                    param.copy_(torch.Tensor(temp_weight))
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
            for dt, labels in self.test_loader:
                dt = dt.to(self.device)
                labels = labels.to(self.device)
                bsz = len(dt)
                output = self.model(dt)
                acc = accuracy(output, labels)
                accs.update(acc[0], bsz)

        return accs.avg.cpu().item()