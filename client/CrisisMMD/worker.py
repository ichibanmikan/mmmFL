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

import time
from CrisisMMD.model import *
from CrisisMMD.train_tools import *
from CrisisMMD.data import *

class Trainer:
    def __init__(self, config, model, train_loader, device):
        self.config = config
        self.device = device
        self.model = model
        self.criterion = nn.NLLLoss().to(device)
        # self.criterion = nn.CrossEntropyLoss().to(device)
        self.train_tools = train_tools(self.model, config)
        self.train_loader = train_loader
        self.best_acc = -100
        self.now_epoch = 0
        
    def every_epoch_train(self):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        
        end = time.time()
        for text, img, text_len, img_len, label in self.train_loader:
            data_time.update(time.time() - end)
            output = None
            
            bsz = text.shape[0]
            text, img = text.to(self.device), img.to(self.device)
            text_len, img_len = text_len.to(self.device), img_len.to(self.device)
            label = label.to(self.device)
            output, _ = self.model(img, text, img_len, text_len)
            
            loss = self.criterion(output, label)

            acc, _ = accuracy(output, label, topk=(1, 5))
            # update metric
            losses.update(loss.item(), bsz)
            top1.update(acc[0], bsz)
            # print(loss.item())
            # SGD
            self.train_tools.optimizer.zero_grad()
            loss.backward()
            self.train_tools.optimizer.step()
            # self.scheduler.step()
            batch_time.update(time.time() - end)
            end = time.time()
        print("loss: %f", loss.item())
        return losses.avg, acc[0]
    
    def train(self):
        record_loss = np.zeros(self.config.epochs)
        record_acc = np.zeros(self.config.epochs)
        self.model.to(self.device)
        self.model.train()
        for epoch in range(0, self.config.epochs):
            self.now_epoch += 1
            self.train_tools.adjust_learning_rate(self.now_epoch)
            time1 = time.time()
            loss, acc = self.every_epoch_train()
            time2 = time.time()
            print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
            record_loss[epoch] = loss
            # evaluation
            record_acc[epoch] = acc
            if acc > self.best_acc:
                self.best_acc = acc
            self.model.train()
        # print(record_acc)
        return record_loss[self.config.epochs - 1], record_acc[self.config.epochs - 1]
    
    def sample_one_epoch(self):
        self.model.to(self.device)
        self.model.train()
        time1 = time.time()
        loss, _ = self.every_epoch_train()
        time2 = time.time()
        return time2 - time1, loss