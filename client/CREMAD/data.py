import os
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import torch
import librosa
import torchvision.models as models
from torchvision import transforms
from moviepy import VideoFileClip


class DataSet(Dataset):
    def __init__(self, data_dir, device = "cuda"):
        super().__init__()
        self.data_dir = data_dir
        self.wav_dir = os.path.join(data_dir, "AudioWAV")
        self.flv_dir = os.path.join(data_dir, "VideoFlash")

        self.wav_files = sorted(
            [f for f in os.listdir(self.wav_dir) if f.endswith(".wav")]
        )
        self.flv_files = sorted(
            [f for f in os.listdir(self.flv_dir) if f.endswith(".npy")]
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
    
class DataFactory:
    def __init__(self, data_dir, config):
        self.dataset = DataSet(data_dir)
        self.config = config
    def get_dataloader(self):
        train_size = int(0.8 * len(self.dataset))  # 80% 
        val_size = len(self.dataset) - train_size  # 20% 
        train_dataset, val_dataset = \
            random_split(self.dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False
        )
        
        return train_loader, val_loader