import torch
import torch.nn as nn 
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset
import librosa
import torchvision.models as models
from torchvision import transforms
from moviepy import VideoFileClip
import numpy as np
import os
    
class Conv1dEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int, 
        n_filters: int,
        dropout: float=0.1
    ):
        super().__init__()
        # conv module
        self.conv1 = nn.Conv1d(input_dim, n_filters, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(n_filters, n_filters*2, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(n_filters*2, n_filters*4, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
            self,
            x: Tensor   # shape => [batch_size (B), num_data (T), feature_dim (D)]
        ):
        x = x.float()
        x = x.permute(0, 2, 1)
        # conv1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.dropout(x)
        # conv2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.dropout(x)
        # conv3
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.dropout(x)
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
        self.att_fc1 = nn.Linear(d_hid, 512)
        self.att_pool = nn.Tanh()
        self.att_fc2 = nn.Linear(512, d_head)

        self.d_hid = d_hid
        self.d_head = d_head

    def forward(
        self,
        x: Tensor,
        val_a=None,
        val_b=None,
        a_len=None
    ):
        att = self.att_pool(self.att_fc1(x))
        # att = self.att_fc2(att).squeeze(-1)
        att = self.att_fc2(att)
        att = att.transpose(1, 2)
        if val_a is not None:
            for idx in range(len(val_a)):
                att[idx, :, val_a[idx]:a_len] = -1e5
                att[idx, :, a_len+val_b[idx]:] = -1e5
        att = torch.softmax(att, dim=2)
        # x = torch.matmul(att, x).mean(axis=1)
        x = torch.matmul(att, x)
        x = x.reshape(x.shape[0], self.d_head*self.d_hid)
        return x

class MMActionClassifier(nn.Module):
    def __init__(
        self, 
        num_classes: int=4,       # Number of classes 
        audio_input_dim: int=80,   # Audio feature input dim
        video_input_dim: int=1280,   # Frame-wise video feature input dim
        d_hid: int=128,         # Hidden Layer size
        n_filters: int=32,      # number of filters
        en_att: bool=True,     # Enable self attention or not
        att_name: str='fuse_base',       # Attention Name
        d_head: int=6           # Head dim
    ):
        super(MMActionClassifier, self).__init__()
        self.dropout_p = 0
        self.en_att = en_att
        self.att_name = att_name
        
        # Conv Encoder module
        self.audio_conv = Conv1dEncoder(
            input_dim=audio_input_dim, 
            n_filters=n_filters, 
            dropout=self.dropout_p, 
        )
        
        # RNN module
        self.audio_rnn = nn.GRU(
            input_size=n_filters*4, 
            hidden_size=d_hid,
            num_layers=1, 
            batch_first=True, 
            dropout=self.dropout_p, 
            bidirectional=False
        )

        self.video_rnn = nn.GRU(
            input_size=video_input_dim, 
            hidden_size=d_hid, 
            num_layers=1, 
            batch_first=True, 
            dropout=self.dropout_p, 
            bidirectional=False
        )
        

        self.fuse_att = FuseBaseSelfAttention(
            d_hid=d_hid,
            d_head=d_head
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_hid*d_head, 64),
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
        x_audio = self.audio_conv(x_audio)
        
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

        x_audio, _ = self.audio_rnn(x_audio) 
        x_video, _ = self.video_rnn(x_video) 

        x_audio, _ = pad_packed_sequence(   
            x_audio, 
            batch_first=True
        )

        x_video, _ = pad_packed_sequence(
            x_video, 
            batch_first=True
        )
        a_max_len = x_audio.shape[1]
        x_mm = torch.cat((x_audio, x_video), dim=1)
        x_mm = self.fuse_att(x_mm, len_a, len_v, a_max_len)

        # 6. MM embedding and predict
        preds = self.classifier(x_mm)
        return preds

class CREMADSet(Dataset):
    def __init__(self, data_dir, device = "cuda"):
        super().__init__()
        self.data_dir = data_dir
        self.wav_dir = os.path.join(data_dir, "AudioWAV")
        self.flv_dir = os.path.join(data_dir, "VideoFlash")

        self.wav_files = sorted(
            [f for f in os.listdir(self.wav_dir) if f.endswith(".wav")]
        )
        self.flv_files = sorted(
            [f for f in os.listdir(self.flv_dir) if f.endswith(".flv")]
        )

        # self.mobilenet = models.mobilenet_v2(
        #     weights=models.MobileNet_V2_Weights.DEFAULT
        # ).to(device)
        # self.mobilenet.classifier = torch.nn.Sequential(
        #     *list(self.mobilenet.classifier.children())[:-1]
        # )
        # self.mobilenet.eval()

        # self.preprocess = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], \
        #         std=[0.229, 0.224, 0.225]),
        # ])

        for wav_file, flv_file in zip(self.wav_files, self.flv_files):
            if os.path.splitext(wav_file)[0] != \
                    os.path.splitext(flv_file)[0]:
                raise ValueError(f"Not match: {wav_file} and {flv_file}")

        self.num_samples = len(self.wav_files)

        self.emotion_map = {
            "HAP": 0,  # Happy
            "SAD": 1,  # Sad
            "ANG": 2,  # Anger
            "FEA": 3,  # Fear
            "DIS": 4,  # Disgust
            "NEU": 5,  # Neutral
        }

        self.data = []
        self.device = device
        for i in range(self.num_samples):
            file_name = os.path.splitext(self.wav_files[i])[0]
            wav_file = os.path.join(self.wav_dir, f"{file_name}.wav")
            flv_file = os.path.join(self.flv_dir, f"{file_name}.npy")

            wav = self.extract_audio_features(wav_file).astype(np.float32)
            # flv = self.extract_video_features(flv_file)
            flv = np.load(flv_file).astype(np.float32)

            emotion_code = file_name.split("_")[2]
            label = self.emotion_map.get(emotion_code, -1)

            self.data.append((wav, flv, len(wav), len(flv), label))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.data[index]

    def extract_audio_features(self, wav_path, n_mfcc=80, max_frames=600):
        audio_data, sample_rate = librosa.load(wav_path, sr=None)
        mfcc_features = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc)
        mfcc_features = mfcc_features.T  # [n_frames, n_mfcc]

        if mfcc_features.shape[0] < max_frames:
            padding = np.zeros((max_frames - mfcc_features.shape[0], n_mfcc))
            mfcc_features = np.vstack((mfcc_features, padding))
        else:
            mfcc_features = mfcc_features[:max_frames, :]
            
        feature_mean = np.mean(mfcc_features, axis=0)
        feature_std = np.std(mfcc_features, axis=0)
        mfcc_features = (mfcc_features - feature_mean) / (feature_std + 1e-5)
        
        return mfcc_features  # [600, 80]

    # def extract_video_features(self, flv_path, max_frames=6):
    #     video = VideoFileClip(flv_path)
    #     features = []

    #     for frame in video.iter_frames():
    #         frame_tensor = self.preprocess(frame).unsqueeze(0).to(self.device)  # [1, 3, 224, 224]
    #         with torch.no_grad():
    #             feature = self.mobilenet(frame_tensor)  # [1, 1280]
    #         features.append(feature.squeeze(0).cpu().numpy())

    #     features = np.array(features)  # [n_frames, 1280]

    #     if features.shape[0] < max_frames:
    #         padding = np.zeros((max_frames - features.shape[0], 1280))
    #         features = np.vstack((features, padding))
    #     else:
    #         features = features[:max_frames, :]

    #     return features  # [6, 1280]

class CREMAD:
    def __init__(self, device):
        self.model = MMActionClassifier(num_classes=6)
        self.model = self.model.to(device)
        self.test_loader = \
            DataLoader(CREMADSet('/home/chenxu/codes/ichibanFATE/server/test_datasets/CREMAD'), \
                batch_size=16, num_workers=16)
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

    def get_model_name(self):
        return "CREMAD"
        
    
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
        self.model.to(self.device)
        self.model.eval()
        accs = AverageMeter()

        with torch.no_grad():
            for wav, flv, len_wav, len_flv, label in self.test_loader:
                output = None

                wav = wav.to(self.device)
                flv = flv.to(self.device)
                labels = label.to(self.device)
                
                bsz = wav.shape[0]
                
                output = self.model(wav, flv, len_wav, len_flv)
                acc, _ = accuracy(output, labels, topk=(1, 5))

                # calculate and store confusion matrix
                accs.update(acc, bsz)

        return accs.avg.cpu().item()