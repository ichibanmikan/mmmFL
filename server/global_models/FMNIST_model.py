from pathlib import Path

import torch
import torch.nn as nn

from global_models.image_classification_common import (
    ClassificationTester,
    IDXImageDataset,
    build_loader,
    flatten_model_params,
    load_state_if_exists,
    reset_model_from_vector,
    save_state,
)


class FMNISTModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.20),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.20),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.20),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.50),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class FMNIST:
    def __init__(self, device):
        self.device = device
        self.model = FMNISTModel().to(device)
        self.base_dir = Path(__file__).resolve().parent
        load_state_if_exists(self.model, self.base_dir / "models" / f"{self.get_model_name()}.pth", device)
        dataset = IDXImageDataset(
            self.base_dir / "../test_datasets/FMNIST/t10k-images-idx3-ubyte",
            self.base_dir / "../test_datasets/FMNIST/t10k-labels-idx1-ubyte",
        )
        self.Tester = ClassificationTester(self.model, build_loader(dataset, batch_size=256), device)

    def get_model_name(self):
        return "FMNIST"

    def get_model_params(self):
        return flatten_model_params(self.model)

    def reset_model_parameter(self, new_params):
        reset_model_from_vector(self.model, new_params)

    def save_model(self, save_file):
        save_state(self.model, Path(save_file))
