# Copyright 2024 ichibanmikan
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn 
import numpy as np
from torch.utils.data import Dataset, DataLoader

class gps_encoder(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
        nn.Conv1d(2, 20, 3, padding = 1),
        nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
        nn.Conv1d(20, 40, 3, padding = 1),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(2,padding = 1)
        )

        self.layer3 = nn.Sequential(
        nn.Conv1d(40, 80, 3, padding = 1),
        nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
        nn.Conv1d(80, 40, 3, padding = 1),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(2,padding = 1),
        nn.Flatten()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

## lidar input: [bsz, 20, 20, 20]
class lidar_encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.channel = 32
        self.dropProb = 0.3


        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=32, kernel_size = (3,3), padding = (1,1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = (3,3), padding = (1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
        
            nn.Conv2d(64, 128, kernel_size = (3,3), padding = (1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),

            nn.Conv2d(128, 32, kernel_size = (3,3), padding = (1,1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True)

        )

        self.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 3)),
            nn.Dropout(p = self.dropProb)
        )

        self.maxpool_ = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p = self.dropProb)
        )

        self.flatten_layer = nn.Sequential(
            nn.Flatten()
        )



    def forward(self, x):
        a = self.layer1(x)
        x = a + self.layer2(a)
        x = self.maxpool(x) # b

        b = x
        x = self.layer2(x) + b
        x = self.maxpool(x) #c

        c = x 
        x = self.layer2(x) + c 
        x = self.maxpool_(x) #d

        d = x 
        x = self.layer2(x) + d 

        x = self.flatten_layer(x)

        return x


## image input: [bsz, 3, 112, 112]
class image_encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.channel = 32
        self.dropProb = 0.25


        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.channel, kernel_size = (7,7), padding = (1,1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True)
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(self.channel, 32, kernel_size = (3,3), padding = (1,1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = (3,3), padding = (1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
        
            nn.Conv2d(64, 128, kernel_size = (3,3), padding = (1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),

            nn.Conv2d(128, 64, kernel_size = (3,3), padding = (1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),

            nn.Conv2d(64, 32, kernel_size = (3,3), padding = (1,1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),

        )

        self.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=(6, 6)),
            nn.Dropout(p = self.dropProb)
        )

        self.maxpool_ = nn.Sequential(
            nn.MaxPool2d(kernel_size=(6, 6)),
            nn.Dropout(p = self.dropProb),
            nn.Flatten()
        )


    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        b = x 
        x = self.layer2(x) + b
        x = self.maxpool(x)
        c = x 
        x = self.layer2(x) + c 
        x = self.maxpool_(x)


        return x

class Encoder3(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_1 = gps_encoder()
        self.encoder_2 = lidar_encoder()
        self.encoder_3 = image_encoder()

    def forward(self, x1, x2, x3):  
        feature_1 = self.encoder_1(x1)
        feature_2 = self.encoder_2(x2)
        feature_3 = self.encoder_3(x3)

        return feature_1, feature_2, feature_3


class My3Model(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.encoder = Encoder3()

        self.classifier = nn.Sequential(
            nn.Linear(488, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )
     
    def forward(self, x1, x2, x3):
        feature_1, feature_2, feature_3 = self.encoder(x1, x2, x3)

        feature = torch.cat((feature_1, feature_2, feature_3), dim=1)
        output = self.classifier(feature).float()
        return output

class FLASH_set(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.gps_data = np.load('/home/chenxu/codes/ichibanFATE/server/test_datasets/FLASH/gps.npz')['gps']
        self.lidar_data = np.load('/home/chenxu/codes/ichibanFATE/server/test_datasets/FLASH/lidar.npz')['lidar']
        self.image_data = np.load('/home/chenxu/codes/ichibanFATE/server/test_datasets/FLASH/image.npz')['image']
        self.label_data = np.load('/home/chenxu/codes/ichibanFATE/server/test_datasets/FLASH/rf.npz')['rf']

    def __len__(self):
        return len(self.label_data)
    
    def __getitem__(self, index):
        return torch.tensor(self.gps_data[index], dtype=torch.float32), torch.tensor(self.lidar_data[index], dtype=torch.float32), torch.tensor(self.image_data[index], dtype=torch.float32), torch.tensor(self.label_data[index], dtype=torch.long)

class FLASH:
    def __init__(self, device):
        self.model = My3Model(64)
        self.model = self.model.to(device)
        self.test_loader = DataLoader(FLASH_set(), batch_size=16, num_workers=16)
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
        # print("Shape of model weight: ", model_params.shape)#39456

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

    def get_model_name(self):
        return "FLASH"

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
        self.model.eval()
        accs = AverageMeter()

        with torch.no_grad():
            for gps_data, lidar_data, image_data, labels in self.test_loader:
                gps_data = gps_data.to(self.device)
                lidar_data = lidar_data.to(self.device)
                image_data = image_data.to(self.device)
                labels = labels.to(self.device)
                output = self.model(gps_data, lidar_data, image_data)
                acc, _ = accuracy(output, labels, topk=(1, 5))

                # calculate and store confusion matrix
                accs.update(acc, gps_data.size(0))

        return accs.avg.cpu().item()