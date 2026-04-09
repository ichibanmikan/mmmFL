from pathlib import Path

import torch
import torch.optim as optim


def accuracy(output, target):
    with torch.no_grad():
        prediction = output.argmax(dim=1)
        correct = prediction.eq(target).sum().item()
        return 100.0 * correct / target.size(0) if target.size(0) else 0.0


class train_tools:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )

    def adjust_learning_rate(self, epoch):
        lr = self.config.learning_rate
        for milestone in self.config.lr_milestones:
            if epoch >= milestone:
                lr *= self.config.lr_gamma
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def save_model(self, save_file):
        save_path = Path(save_file)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        state_dict = {key: value.detach().cpu() for key, value in self.model.state_dict().items()}
        torch.save(state_dict, save_path)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0
