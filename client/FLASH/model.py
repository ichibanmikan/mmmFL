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