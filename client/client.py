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

import argparse
import importlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from communication import ClientHandler


TRAINER_REGISTRY = {
    "AC": ("AC.main", "AC_main"),
    "CIFAR": ("CIFAR.main", "CIFAR_main"),
    "CREMAD": ("CREMAD.main", "CREMAD_main"),
    "CrisisMMD": ("CrisisMMD.main", "CrisisMMD_main"),
    "FLASH": ("FLASH.main", "FLASH_main"),
    "FMNIST": ("FMNIST.main", "FMNIST_main"),
    "HatefulMemes": ("HatefulMemes.main", "HatefulMemes_main"),
    "MHAD": ("MHAD.main", "MHAD_main"),
    "MNIST": ("MNIST.main", "MNIST_main"),
    "USC": ("USC.main", "USC_main"),
}


def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.set_default_dtype(torch.float32)
    generator = torch.Generator()
    generator.manual_seed(seed)
    torch.set_rng_state(generator.get_state())


def load_active_datasets(data):
    if "dataset_profiles" not in data:
        return data["datasets"]
    active_profile = data.get("active_profile")
    if active_profile not in data["dataset_profiles"]:
        raise KeyError(f"Unknown dataset profile: {active_profile}")
    return data["dataset_profiles"][active_profile]


def load_trainer_class(dataset_name):
    module_name, class_name = TRAINER_REGISTRY[dataset_name]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


class Config:
    def __init__(self):
        base_dir = Path(__file__).resolve().parent
        config_path = base_dir / "client.json"
        with config_path.open("r", encoding="utf-8") as js:
            data = json.load(js)

        parser = argparse.ArgumentParser(description="Process node ID.")
        parser.add_argument("--node_id", type=int, required=True, help="Node ID of the client")
        args = parser.parse_args()

        self.node_id = args.node_id
        self.server_address = data["Host"]["server_address"]
        self.port = data["Host"]["port"]
        self.datasets = load_active_datasets(data)
        self.active_profile = data.get("active_profile", "default")
        self.active_client_count = data.get("active_client_count", len(data["Ability"]["ability"]))
        self.random_seed = data["random_seed"]

        if self.node_id >= self.active_client_count:
            raise ValueError(
                f"node_id {self.node_id} is outside active_client_count {self.active_client_count}"
            )

        self.kappa = data["Ability"]["ability"][self.node_id]
        self.distance = data["Energy"]["distance"][self.node_id]
        self.tx_power_dbm = data["Energy"]["tx_power_dbm"][self.node_id]
        self.rho = data["Energy"]["rho"][self.node_id]
        self.noise_dbm = data["Energy"]["noise_dbm"]
        self.bandwidth_hz = data["Energy"]["bandwidth_hz"]
        self.total_energy = data["Energy"]["total_energy"][self.node_id]

    def modality(self, row):
        return self.datasets[row]["modalities_name"]


class Client:
    def __init__(self, config):
        self.config = config

    def start(self):
        set_all_seeds(self.config.random_seed)
        trainers = []

        for dataset in self.config.datasets:
            dataset_name = dataset["dataset_name"]
            if dataset_name not in TRAINER_REGISTRY:
                raise KeyError(f"Unsupported dataset: {dataset_name}")
            trainer_cls = load_trainer_class(dataset_name)
            trainer = trainer_cls(
                dataset["modalities_name"],
                self.config.node_id,
                dataset["model_size"],
            )
            trainers.append(trainer)

        handler = ClientHandler(self.config, trainers)
        handler.handle()
        print("all over")


if __name__ == "__main__":
    config = Config()
    client = Client(config)
    for _ in range(1):
        client.start()
        time.sleep(10)
