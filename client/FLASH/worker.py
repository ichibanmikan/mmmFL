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

import os
import time
from FLASH.model import *
from FLASH.train_tools import *
from FLASH.data import *

class Trainer:
    def __init__(self, config, model, train_loader, valid_loader, device, state):
        self.config = config
        self.device = device
        self.model = model
        self.model.train()
        self.criterion = torch.nn.CrossEntropyLoss().to(device)
        self.train_tools = train_tools(self.model, config)
        self.train_loader = train_loader
        self.validater = Validater(self.model, valid_loader, config, self.criterion, device)
        self.best_acc = -100
    def every_epoch_train(self):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        
        end = time.time()
        for data_list, labels in self.train_loader:
            data_time.update(time.time() - end)
            output = None
            data_1 = data_list[0]
            data_1 = data_1.to(self.device)
            labels = labels.to(self.device)
            bsz = data_1.shape[0]
            
            if len(data_list) == 1:
                output = self.model(data_1)  
            elif len(data_list) == 2:
                data_2 = data_list[1]
                data_2 = data_2.to(self.device)
                
                output = self.model(data_1, data_2)
            else:
                data_2 = data_list[1]
                data_3 = data_list[2]
                data_2 = data_2.to(self.device)
                data_3 = data_3.to(self.device)
                
                output = self.model(data_1, data_2, data_3)
                
                
            
            loss = self.criterion(output, labels)

            acc, _ = accuracy(output, labels, topk=(1, 5))

            # update metric
            losses.update(loss.item(), bsz)
            top1.update(acc[0], bsz)

            # SGD
            self.train_tools.optimizer.zero_grad()
            loss.backward()
            self.train_tools.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        print("loss: %f", loss.item())
        return losses.avg
    
    def train(self):
        record_loss = np.zeros(self.config.epochs)
        record_acc = np.zeros(self.config.epochs)
        for epoch in range(0, self.config.epochs + 1):
            self.train_tools.adjust_learning_rate(epoch)
            time1 = time.time()
            loss = self.every_epoch_train()
            time2 = time.time()
            print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
            record_loss[epoch-1] = loss
            # evaluation
            loss, val_acc, _ = self.validater.validate()
            record_acc[epoch-1] = val_acc
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                # best_confusion = confusion
            # if self.best_acc > 65.01:
            #     self.train_tools.save_model(epoch, os.path.join(os.getcwd(), 'model/best.pth'))
            #     break;
        print(record_acc)

class Validater:
    def __init__(self, model, valid_loader, config, criterion, device):
        self.model = model
        self.config = config
        self.criterion = criterion
        self.valid_loader = valid_loader
        self.device = device
    def validate(self):
        self.model.eval()
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        confusion = np.zeros((self.config.num_classes, self.config.num_classes))

        with torch.no_grad():
            end = time.time()
            for data_list, labels in self.valid_loader:  
                output = None
                data_1 = data_list[0]
                data_1 = data_1.to(self.device)
                labels = labels.to(self.device)
                bsz = data_1.shape[0]
                
                if len(data_list) == 1:
                    output = self.model(data_1)  
                elif len(data_list) == 2:
                    data_2 = data_list[1]
                    data_2 = data_2.to(self.device)
                    
                    output = self.model(data_1, data_2)
                else:
                    data_2 = data_list[1]
                    data_3 = data_list[2]
                    data_2 = data_2.to(self.device)
                    data_3 = data_3.to(self.device)
                    
                    output = self.model(data_1, data_2, data_3)
                
                loss = self.criterion(output, labels)

                # update metric
                acc, _ = accuracy(output, labels, topk=(1, 5))
                losses.update(loss.item(), bsz)
                top1.update(acc[0], bsz)

                # calculate and store confusion matrix
                # rows = labels.cpu().numpy()
                # cols = output.max(1)[1].cpu().numpy()
                # for label_index in range(labels.shape[0][0]):
                #     confusion[rows[label_index], cols[label_index]] += 1

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

        return losses.avg, top1.avg, confusion

class Tester:
    def __init__(self, model, test_loader, device):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        
    def test(self):
        self.model.eval()
        accs = AverageMeter()

        with torch.no_grad():
            for data_list, labels in self.test_loader:
                output = None
                data_1 = data_list[0]
                data_1 = data_1.to(self.device)
                labels = labels.to(self.device)
                bsz = data_1.shape[0]
                
                if len(data_list) == 1:
                    output = self.model(data_1)  
                elif len(data_list) == 2:
                    data_2 = data_list[1]
                    data_2 = data_2.to(self.device)
                    
                    output = self.model(data_1, data_2)
                else:
                    data_2 = data_list[1]
                    data_3 = data_list[2]
                    data_2 = data_2.to(self.device)
                    data_3 = data_3.to(self.device)
                    
                    output = self.model(data_1, data_2, data_3)
                acc, _ = accuracy(output, labels, topk=(1, 5))

                # calculate and store confusion matrix
                accs.update(acc, data_1.size(0))

        return accs.avg