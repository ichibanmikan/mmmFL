import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
# import numpy as np

class acc_encoder(nn.Module):
    """
    CNN layers applied on acc sensor data to generate pre-softmax
    ---
    params for __init__():

    forward():
        Input: data [bsz, 1, 60, 9]
        Output: feature [bsz, 128]
    """
    def __init__(self):
        super().__init__()

        # Extract features, 2D conv layers
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = (2,2), padding=(1, 1)),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(32, 64, kernel_size = (2,2), padding=(1, 1)),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(64, 64, kernel_size = (2,2), padding=(1, 1)),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            )

        self.gru = nn.GRU(64, 16, 2, batch_first=True, dropout=0.3)

    def forward(self, x):

        # self.gru.flatten_parameters()

        x = self.features(x)# [bsz, 128, 8, 2]
        # print("original acc feature:", x.shape)

        x = x.view(x.size(0), 16, -1)#[bsz, 16, 64]
        # print("acc feature:", x.shape)

        x, _ = self.gru(x)#.reshape(x.size(0), -1)


        feature = x.reshape(x.size(0), -1)
        # print("acc gru feature:", feature.shape)#[bsz, 256]

        return feature


class skeleton_encoder(nn.Module):
    """
    CNN layers applied on acc sensor data to generate pre-softmax
    ---
    params for __init__():

    forward():
        Input: [bsz, 1, 60, 3, 35]
        Output: pre-softmax
    """
    def __init__(self):
        super().__init__()

        # Extract features, 2D conv layers
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 2, 3), padding=(1, 1, 1)),
            nn.Dropout3d(0.2),
            nn.BatchNorm3d(32),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=(2, 2, 2), padding=(1, 1, 1)),
            nn.Dropout3d(0.2),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

            nn.Conv3d(64, 128, kernel_size=(2, 2, 2), padding=(1, 1, 1)),
            nn.Dropout3d(0.2),
            nn.BatchNorm3d(128),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

            nn.Conv3d(128, 256, kernel_size=(2, 2, 2), padding=(1, 1, 1)),
            nn.Dropout3d(0.2),
            nn.BatchNorm3d(256),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

            # nn.Conv3d(256, 512, kernel_size=(2, 2, 2), padding=(1, 1, 1)),
            # nn.BatchNorm3d(512),
            # nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

            )

        self.gru = nn.GRU(192, 16, 2, batch_first=True, dropout=0.3)

    def forward(self, x):

        x = self.features(x)#[16, 256, 4, 1, 3]
        # print("original skeleton feature:", x.shape)

        x = x.view(x.size(0), 16, -1)#[bsz, 16, 192]
        # print("skeleton feature:", x.shape)

        x, _ = self.gru(x)
        # print("skeleton gru feature:", x.shape)#[bsz, 256]

        feature = x.reshape(x.size(0), -1)
        # print("skeleton gru feature:", feature.shape)#[bsz, 256]

        return feature


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_1 = acc_encoder()
        self.encoder_2 = skeleton_encoder()

    def forward(self, x1, x2):

        feature_1 = self.encoder_1(x1)
        feature_2 = self.encoder_2(x2)

        return feature_1, feature_2



class MyMMModel(nn.Module):
    """Model for human-activity-recognition."""
    def __init__(self, num_classes):
        super().__init__()

        self.encoder = Encoder()

        # Classify output, fully connected layers
        # self.classifier = nn.Linear(1920, num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x1, x2):

        feature_1, feature_2 = self.encoder(x1, x2)

        fused_feature = (feature_1 + feature_2) / 2 # weighted sum
        # fused_feature = torch.cat((acc_output,gyro_output), dim=1) #concate
        # print(fused_feature.shape)

        output = self.classifier(fused_feature).float()

        return output

# class acc_encoder(nn.Module):
#     """
#     CNN layers applied on acc sensor data to generate pre-softmax
#     ---
#     params for __init__():

#     forward():
#         Input: data [bsz, 1, 60, 9]
#         Output: feature [bsz, 128]
#     """
#     def __init__(self):
#         super().__init__()

#         # Extract features, 2D conv layers
#         self.features = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size = (2,2), padding=(1, 1)),
#             nn.BatchNorm2d(32),
#             nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

#             nn.Conv2d(32, 64, kernel_size = (2,2), padding=(1, 1)),
#             nn.BatchNorm2d(64),
#             nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

#             nn.Conv2d(64, 64, kernel_size = (2,2), padding=(1, 1)),
#             nn.BatchNorm2d(64),
#             nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

#             )

#         self.gru = nn.GRU(64, 16, 2, batch_first=True)

#     def forward(self, x):

#         # self.gru.flatten_parameters()

#         x = self.features(x)#bsz, 128, 8, 2]
#         # print("original acc feature:", x.shape)

#         x = x.view(x.size(0), 16, -1)#[bsz, 16, 64]
#         # print("acc feature:", x.shape)

#         x, _ = self.gru(x)#.reshape(x.size(0), -1)


#         feature = x.reshape(x.size(0), -1)
#         # print("acc gru feature:", feature.shape)#[bsz, 256]

#         return feature


# class skeleton_encoder(nn.Module):
#     """
#     CNN layers applied on acc sensor data to generate pre-softmax
#     ---
#     params for __init__():

#     forward():
#         Input: [bsz, 1, 60, 3, 35]
#         Output: pre-softmax
#     """
#     def __init__(self):
#         super().__init__()

#         # Extract features, 2D conv layers
#         self.features = nn.Sequential(
#             nn.Conv3d(1, 32, kernel_size=(3, 2, 3), padding=(1, 1, 1)),
#             nn.BatchNorm3d(32),
#             nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

#             nn.Conv3d(32, 64, kernel_size=(2, 2, 2), padding=(1, 1, 1)),
#             nn.BatchNorm3d(64),
#             nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

#             nn.Conv3d(64, 128, kernel_size=(2, 2, 2), padding=(1, 1, 1)),
#             nn.BatchNorm3d(128),
#             nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

#             nn.Conv3d(128, 256, kernel_size=(2, 2, 2), padding=(1, 1, 1)),
#             nn.BatchNorm3d(256),
#             nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

#             # nn.Conv3d(256, 512, kernel_size=(2, 2, 2), padding=(1, 1, 1)),
#             # nn.BatchNorm3d(512),
#             # nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

#             )

#         self.gru = nn.GRU(192, 16, 2, batch_first=True)

#     def forward(self, x):

#         x = self.features(x)#[16, 256, 4, 1, 3]
#         # print("original skeleton feature:", x.shape)

#         x = x.view(x.size(0), 16, -1)#[bsz, 16, 192]
#         # print("skeleton feature:", x.shape)

#         x, _ = self.gru(x)
#         # print("skeleton gru feature:", x.shape)#[bsz, 256]

#         feature = x.reshape(x.size(0), -1)
#         # print("skeleton gru feature:", feature.shape)#[bsz, 256]

#         return feature


# class Encoder(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.encoder_1 = acc_encoder()
#         self.encoder_2 = skeleton_encoder()

#     def forward(self, x1, x2):

#         feature_1 = self.encoder_1(x1)
#         feature_2 = self.encoder_2(x2)

#         return feature_1, feature_2



# class MyMMModel(nn.Module):
#     """Model for human-activity-recognition."""
#     def __init__(self, num_classes):
#         super().__init__()

#         self.encoder = Encoder()

#         # Classify output, fully connected layers
#         # self.classifier = nn.Linear(1920, num_classes)
#         self.classifier = nn.Sequential(

#             nn.Linear(256, 64),
#             nn.BatchNorm1d(64),

#             nn.Linear(64, num_classes),
#             )

#     def forward(self, x1, x2):

#         feature_1, feature_2 = self.encoder(x1, x2)

#         fused_feature = (feature_1 + feature_2) / 2 # weighted sum
#         # fused_feature = torch.cat((acc_output,gyro_output), dim=1) #concate
#         # print(fused_feature.shape)

#         output = self.classifier(fused_feature)

#         return output

class Mhad_set(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.x1_data = torch.tensor(np.load('/home/chenxu/codes/ichibanFATE/server/test_datasets/MHAD/x1.npy'), dtype=torch.float32)
        self.x2_data = torch.tensor(np.load('/home/chenxu/codes/ichibanFATE/server/test_datasets/MHAD/x2.npy'), dtype=torch.float32)
        self.label_data = torch.tensor(np.load('/home/chenxu/codes/ichibanFATE/server/test_datasets/MHAD/y.npy'), dtype=torch.long)

    def __len__(self):
        return len(self.label_data)
    
    def __getitem__(self, index):
        return torch.unsqueeze(self.x1_data[index], dim=0), torch.unsqueeze(self.x2_data[index], dim=0), self.label_data[index]

class MHAD:
    def __init__(self, device):
        self.model = MyMMModel(11)
        self.device = device
        self.load_model(
            os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)), f'models/{self.get_model_name()}.pth'))
        self.model = self.model.to(device)
        self.test_loader = DataLoader(Mhad_set(), batch_size=16, num_workers=16)
        self.Tester = Tester(self.model, self.test_loader, device)
    def get_model_params(self):
    
        params = []
        for param in self.model.parameters():
            if torch.cuda.is_available():
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
                    temp_weight = new_params[temp_index : temp_index + para_len].astype(float)
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
            self.model.load_state_dict(torch.load(load_file, map_location=self.device), weights_only=True)
            self.model.to(self.device)
        else:
            print(f'==> Model file {load_file} not found. Using initialized model.')        
    def get_model_name(self):
        return "MHAD"
    
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
            for x1_data, x2_data, labels in self.test_loader:
                x1_data = x1_data.to(self.device)
                x2_data = x2_data.to(self.device)
                labels = labels.to(self.device)  
                output = self.model(x1_data, x2_data)
                acc, _ = accuracy(output, labels, topk=(1, 5))

                # calculate and store confusion matrix
                accs.update(acc, x1_data.size(0))

        return accs.avg.cpu().item()