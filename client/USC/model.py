import torch
import torch.nn as nn
import torch.nn.functional as F
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
            nn.Conv2d(input_size, 64, 2),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv2d(64, 64, 2),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv2d(64, 32, 1),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv2d(32, 16, 1),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

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
            nn.Conv2d(input_size, 64, 2),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv2d(64, 64, 2),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv2d(64, 32, 1),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv2d(32, 16, 1),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

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

        self.gru = nn.GRU(198, 120, 2, batch_first=True, dropout=0.3)

        # Classify output, fully connected layers
        self.classifier = nn.Sequential(

            nn.Linear(1920, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1280, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes),
            )

    def forward(self, data_1, data_2):

        acc_output, gyro_output = self.encoder(data_1, data_2)

        fused_feature = (acc_output + gyro_output) / 2 # weighted sum

        fused_feature, _ = self.gru(fused_feature)
        fused_feature = fused_feature.contiguous().view(fused_feature.size(0), 1920)

        output = self.classifier(fused_feature)

        return output
            