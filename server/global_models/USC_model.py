import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
# import numpy as np

class encoder_acc(nn.Module):
    """
    CNN layers applied on acc sensor data to generate pre-softmax
    ---
    params for __init__():
        input_size: e.g. 1
        num_classes: e.g. 6
    forward():
        Input: data
        Output: pre-softmax
    """
    def __init__(self, input_size):
        super().__init__()

        # Extract features, 2D conv layers
        self.features = nn.Sequential(
            nn.Conv2d(input_size, 32, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 16, -1)#[bsz, 16, 1, 198]

        return x


class encoder_gyr(nn.Module):
    """
    CNN layers applied on acc sensor data to generate pre-softmax
    ---
    params for __init__():
        input_size: e.g. 1
        num_classes: e.g. 6
    forward():
        Input: data
        Output: pre-softmax
    """
    def __init__(self, input_size):
        super().__init__()

        # Extract features, 2D conv layers
        self.features = nn.Sequential(
            nn.Conv2d(input_size, 32, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 16, -1)

        return x


class Encoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.encoder_acc = encoder_acc(input_size)
        self.encoder_gyr = encoder_gyr(input_size)

    def forward(self, x1, x2):
        
        acc_output = self.encoder_acc(x1)
        gyro_output = self.encoder_gyr(x2)

        return acc_output, gyro_output



class MMModel(nn.Module):
    """Model for human-activity-recognition."""
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.encoder = Encoder(input_size)

        self.gru = nn.GRU(198, 60, 1, batch_first=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(960, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, data_1, data_2):

        acc_output, gyro_output = self.encoder(data_1, data_2)

        fused_feature = (acc_output + gyro_output) / 2 # weighted sum

        fused_feature, _ = self.gru(fused_feature)
        fused_feature = fused_feature.contiguous().view(fused_feature.size(0), -1)

        output = self.classifier(fused_feature)

        return output
            

class USC_set(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.acc_data = torch.tensor(np.load('/home/chenxu/codes/ichibanFATE/server/test_datasets/USC/acc.npy'), dtype=torch.float32)
        self.gyr = torch.tensor(np.load('/home/chenxu/codes/ichibanFATE/server/test_datasets/USC/gyr.npy'), dtype=torch.float32)
        self.action = torch.tensor(np.load('/home/chenxu/codes/ichibanFATE/server/test_datasets/USC/action.npy'), dtype=torch.long)

    def __len__(self):
        return len(self.acc_data)
    
    def __getitem__(self, index):
        return self.acc_data[index].unsqueeze(0), \
            self.gyr[index].unsqueeze(0), \
                self.action[index]

class USC:
    def __init__(self, device):
        self.now_loss = 999
        self.model = MMModel(1, 12)      

        self.device = device
        self.model = self.model.to(device)
        
        dl = DataLoader(USC_set(), batch_size=16, num_workers = 4)
        self.Tester = Tester(self.model, dl, device)
    
    def get_model_name(self):
        return "USC"
    
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
            for acc_data, gyr, labels in self.test_loader:
                acc_data = acc_data.to(self.device)
                gyr = gyr.to(self.device)
                labels = labels.to(self.device)
                bsz = len(acc_data)
                output = self.model(acc_data, gyr)
                acc, _ = accuracy(output, labels, topk=(1, 5))
                accs.update(acc, bsz)

        return accs.avg.cpu().item()