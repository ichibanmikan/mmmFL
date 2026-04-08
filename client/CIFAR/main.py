import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
import numpy as np

try:
    from CIFAR.data import data_factory
    from CIFAR.model import ResNet
    from CIFAR.worker import Trainer
except ModuleNotFoundError:
    from data import data_factory
    from model import ResNet
    from worker import Trainer


class Config:
    def __init__(self, config_path):
        self.config_path = config_path
        self.load_config()

    def load_config(self):
        with open(self.config_path, "r") as f:
            config_data = json.load(f)
        self.batch_size = config_data.get("batch_size", 128)
        self.num_workers = config_data.get("num_workers", 4)
        self.epochs = config_data.get("epochs", 1)
        self.total_epochs = config_data.get("total_epochs", 180)
        self.learning_rate = config_data.get("learning_rate", 0.1)
        self.weight_decay = config_data.get("weight_decay", 1e-4)
        self.momentum = config_data.get("momentum", 0.9)
        self.num_classes = config_data.get("num_classes", 10)
        self.resnet_n = config_data.get("resnet_n", 9)
        self.lr_milestones = config_data.get("lr_milestones", [90, 135])
        self.lr_gamma = config_data.get("lr_gamma", 0.1)
        self.MACs = config_data.get("MACs", 125485568)


class CIFAR_main:
    def __init__(self, modality, node_id, model_size):
        self.modality = modality
        self.node_id = node_id
        self.now_loss = 999.0
        self.model_size = model_size
        self.metrics = {}

        base_dir = Path(__file__).parent
        self.config = Config(base_dir / "config.json")
        self.model = ResNet(num_classes=self.config.num_classes, n=self.config.resnet_n)

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)
        data_f = data_factory(base_dir / "datasets", node_id, self.config)
        train_loader = data_f.get_train_loader()
        validation_loader = data_f.get_validation_loader()
        test_loader = data_f.get_test_loader()
        self.tr = Trainer(
            self.config,
            self.model,
            train_loader,
            validation_loader,
            test_loader,
            node_id,
            self.device,
        )
        self.MACs = data_f.sample_length * self.config.MACs

    def main(self):
        self.now_loss, self.metrics = self.tr.train()
        self.save_model(0)
        self.save_metrics()
        return self.get_model_param()

    def get_model_param(self):
        params = []
        for param in self.model.parameters():
            params.extend(param.view(-1).detach().cpu().numpy())
        return np.array(params)

    def reset_model_parameter(self, new_params):
        temp_index = 0
        with torch.no_grad():
            for param in self.model.parameters():
                para_len = int(param.numel())
                temp_weight = new_params[temp_index:temp_index + para_len].astype(float)
                reshaped = torch.tensor(temp_weight, dtype=param.dtype).view(param.shape)
                param.copy_(reshaped.to(param.device))
                temp_index += para_len

    def sample_time(self):
        return self.tr.sample_one_epoch()

    def save_model(self, round):
        _ = round
        modalities = "".join(self.modality) if self.modality else "cifar"
        save_path = Path(__file__).parent / "models" / f"{modalities}_node_{self.node_id}.pth"
        self.tr.train_tools.save_model(save_path)

    def save_metrics(self):
        metrics_path = Path(__file__).parent / "results" / f"node_{self.node_id}_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset_name": "CIFAR",
            "node_id": self.node_id,
            "model_size": self.model_size,
            "metrics": self.metrics,
        }
        with metrics_path.open("w") as f:
            json.dump(payload, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train one CIFAR client")
    parser.add_argument("--client_id", type=int, required=True, help="CIFAR client id")
    args = parser.parse_args()

    trainer = CIFAR_main(["img"], args.client_id, 853018)
    trainer.main()
