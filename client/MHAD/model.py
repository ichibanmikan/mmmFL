import torch
import torch.nn as nn
import torch.nn.functional as F
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

        x = self.features(x)#bsz, 128, 8, 2]
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