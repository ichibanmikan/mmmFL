import time

import torch
import numpy as np

try:
    from CIFAR.train_tools import AverageMeter, accuracy, train_tools
except ModuleNotFoundError:
    from train_tools import AverageMeter, accuracy, train_tools


class Trainer:
    def __init__(self, config, model, train_loader, validation_loader, test_loader, node_id, device):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.node_id = node_id
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss().to(device)
        self.train_tools = train_tools(self.model, config)
        self.now_epoch = 0
        self.last_metrics = {}

    def every_epoch_train(self):
        losses = AverageMeter()
        top1 = AverageMeter()
        end = time.time()

        for images, labels in self.train_loader:
            _ = time.time() - end
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            self.train_tools.optimizer.zero_grad()
            loss.backward()
            self.train_tools.optimizer.step()

            batch_size = labels.size(0)
            losses.update(loss.item(), batch_size)
            top1.update(accuracy(outputs, labels), batch_size)
            end = time.time()

        print(f"Node {self.node_id} train loss: {losses.avg:.6f}, acc: {top1.avg:.2f}%")
        return losses.avg, top1.avg

    def evaluate(self, loader, split):
        self.model.eval()
        losses = AverageMeter()
        top1 = AverageMeter()

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                batch_size = labels.size(0)
                losses.update(loss.item(), batch_size)
                top1.update(accuracy(outputs, labels), batch_size)

        print(f"Node {self.node_id} {split} loss: {losses.avg:.6f}, acc: {top1.avg:.2f}%")
        return losses.avg, top1.avg

    def train(self):
        record_loss = np.zeros(self.config.epochs, dtype=np.float32)
        final_metrics = {}

        for epoch in range(self.config.epochs):
            self.model.train()
            self.train_tools.adjust_learning_rate(self.now_epoch)
            self.now_epoch += 1

            time_start = time.time()
            train_loss, train_acc = self.every_epoch_train()
            validation_loss, validation_acc = self.evaluate(self.validation_loader, "validation")
            elapsed = time.time() - time_start

            record_loss[epoch] = train_loss
            final_metrics = {
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "validation_loss": float(validation_loss),
                "validation_acc": float(validation_acc),
                "epoch_time": float(elapsed),
            }
            # print(f"Node {self.node_id} epoch {epoch}, total time {elapsed:.2f}")

        test_loss, test_acc = self.evaluate(self.test_loader, "test")
        final_metrics["test_loss"] = float(test_loss)
        final_metrics["test_acc"] = float(test_acc)
        self.last_metrics = final_metrics
        return float(record_loss[-1]), final_metrics

    def sample_one_epoch(self):
        self.model.train()
        self.train_tools.adjust_learning_rate(self.now_epoch)
        self.now_epoch += 1
        time_start = time.time()
        train_loss, _ = self.every_epoch_train()
        elapsed = time.time() - time_start
        return elapsed, train_loss
