import json
from pathlib import Path

import torch
import numpy as np

try:
    from FMNIST.data import data_factory
    from FMNIST.model import FMNISTModel
    from FMNIST.worker import Trainer
except ModuleNotFoundError:
    from data import data_factory
    from model import FMNISTModel
    from worker import Trainer


class Config:
    def __init__(self, config_path):
        self.config_path = config_path
        self.load_config()

    def load_config(self):
        with open(self.config_path, "r") as f:
            config_data = json.load(f)
        self.batch_size = config_data.get("batch_size", 32)
        self.num_workers = config_data.get("num_workers", 0)
        self.epochs = config_data.get("epochs", 1)
        self.learning_rate = config_data.get("learning_rate", 0.001)
        self.weight_decay = config_data.get("weight_decay", 0.0)
        self.num_classes = config_data.get("num_classes", 10)
        self.MACs = config_data.get("MACs", 390410)


class FMNIST_main:
    def __init__(self, modality, node_id, model_size):
        self.modality = modality
        self.node_id = node_id
        self.model_size = model_size
        self.now_loss = 999

        self.base_dir = Path(__file__).parent
        self.config = Config(self.base_dir / "config.json")
        self.model = FMNISTModel(self.config.num_classes)

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)
        data_dir = self.base_dir / "datasets" / f"node_{node_id}"
        data_f = data_factory(data_dir, self.config)
        self.tr = Trainer(self.config, self.model, data_f.get_dataset(), node_id, self.device)
        self.MACs = data_f.sample_length * self.config.MACs

    def main(self):
        self.now_loss = self.tr.train()
        self.save_model(0)
        return self.get_model_param()

    def get_model_param(self):
        params = []
        for param in self.model.parameters():
            params.extend(param.view(-1).detach().cpu().numpy())
        return np.array(params)

    def reset_model_parameter(self, new_params):
        offset = 0
        with torch.no_grad():
            for param in self.model.parameters():
                numel = param.numel()
                values = new_params[offset:offset + numel].astype(float)
                values = torch.tensor(values, dtype=param.dtype, device=param.device).view(param.shape)
                param.copy_(values)
                offset += numel

    def sample_time(self):
        return self.tr.sample_one_epoch()

    def save_model(self, round):
        _ = round
        modality_name = "".join(self.modality) if self.modality else "img"
        save_path = self.base_dir / "models" / f"{modality_name}_node_{self.node_id}.pth"
        self.tr.train_tools.save_model(save_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train one FMNIST client")
    parser.add_argument("--client_id", type=int, required=True)
    args = parser.parse_args()

    trainer = FMNIST_main(["img"], args.client_id, 390410)
    trainer.main()
