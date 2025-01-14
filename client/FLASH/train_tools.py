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

import math
import torch
import numpy as np
import torch.optim as optim

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

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
'''作用: 计算模型预测的准确率。支持计算多个 top-k 的准确率，比如 top-1 或 top-5。

output 是模型的输出，通常是未经处理的 logits。
target 是真实标签。
topk 指定需要计算的 k 个准确率，例如 topk=(1, 5) 会计算 top-1 和 top-5 准确率。
函数返回一个列表，包含每个 k 对应的准确率（以百分比表示）。'''


class train_tools:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.set_optimizer()
        
    def set_optimizer(self):
        self.optimizer = optim.Adam([ 
            {'params': self.model.encoder.parameters(), 'lr': 1e-4},
            {'params': self.model.classifier.parameters(), 'lr': self.config.learning_rate}],
            weight_decay=self.config.weight_decay)
    
    def adjust_learning_rate(self, epoch):
        if(self.config.total_epochs <= epoch):
            epoch = self.config.total_epochs
        lr = self.config.learning_rate
        eta_min = lr * (self.config.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / self.config.total_epochs)) / 2

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


    def warmup_learning_rate(self, epoch, batch_id, total_batches):
        if self.config.warm and epoch <= self.config.warm_epochs:
            p = (batch_id + (epoch - 1) * total_batches) / \
                (self.config.warm_epochs * total_batches)
            lr = self.config.warmup_from + p * (self.config.warmup_to - self.config.warmup_from)

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr


    def save_model(self, save_file):
        print('==> Saving...')

        torch.save(self.model.cpu().state_dict(), save_file)
        # state = {
        #     'opt': self.config,
        #     'model': self.model.state_dict(),
        #     'optimizer': self.optimizer.state_dict(),
        #     'epoch': epoch,
        # }
        # torch.save(state, save_file)
        # del state


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