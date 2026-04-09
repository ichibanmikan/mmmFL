import time

import torch
import numpy as np

try:
    from FMNIST.train_tools import AverageMeter, train_tools
except ModuleNotFoundError:
    from train_tools import AverageMeter, train_tools


class Trainer:
    def __init__(self, config, model, train_loader, node_id, device):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.node_id = node_id
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss().to(device)
        self.train_tools = train_tools(self.model, self.config)

    def every_epoch_train(self):
        losses = AverageMeter()
        correct = 0
        total = 0

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            self.train_tools.optimizer.zero_grad()
            loss.backward()
            self.train_tools.optimizer.step()

            batch_size = labels.size(0)
            losses.update(loss.item(), batch_size)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += batch_size

        acc = 100.0 * correct / total if total else 0.0
        print(f"Node {self.node_id} loss: {losses.avg:.6f}, acc: {acc:.2f}%")
        return losses.avg

    def train(self):
        record_loss = np.zeros(self.config.epochs, dtype=np.float32)
        for epoch in range(self.config.epochs):
            self.model.train()
            start_time = time.time()
            epoch_loss = self.every_epoch_train()
            used_time = time.time() - start_time
            record_loss[epoch] = epoch_loss
            print(f"Node {self.node_id} epoch {epoch}, total time {used_time:.2f}")
        return float(record_loss[-1])

    def sample_one_epoch(self):
        self.model.train()
        start_time = time.time()
        loss = self.every_epoch_train()
        used_time = time.time() - start_time
        return used_time, loss
