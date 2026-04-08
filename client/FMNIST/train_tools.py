from pathlib import Path

import torch
import torch.optim as optim


class train_tools:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def save_model(self, save_file):
        save_path = Path(save_file)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        state_dict = {name: value.detach().cpu() for name, value in self.model.state_dict().items()}
        torch.save(state_dict, save_path)


class AverageMeter:
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
