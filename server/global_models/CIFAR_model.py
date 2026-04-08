from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as torch_f

from global_models.image_classification_common import (
    CIFARPickleDataset,
    ClassificationTester,
    build_loader,
    flatten_model_params,
    load_state_if_exists,
    reset_model_from_vector,
    save_state,
)


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.stride != 1 or self.in_channel != self.out_channel:
            shortcut = x[:, :, ::self.stride, ::self.stride]
            shortcut = torch_f.pad(
                shortcut,
                (0, 0, 0, 0, 0, self.out_channel - self.in_channel),
                mode="constant",
                value=0,
            )
        else:
            shortcut = x
        return self.relu(out + shortcut)


class ResNet(nn.Module):
    def __init__(self, num_classes=10, n=9):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.stage1 = self._make_layers(16, 16, n, 1)
        self.stage2 = self._make_layers(16, 32, n, 2)
        self.stage3 = self._make_layers(32, 64, n, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layers(self, in_channel, out_channel, num_blocks, stride):
        layers = [BasicBlock(in_channel, out_channel, stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channel, out_channel, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn(self.conv1(x)))
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


class CIFAR:
    def __init__(self, device):
        self.device = device
        self.model = ResNet().to(device)
        self.base_dir = Path(__file__).resolve().parent
        load_state_if_exists(self.model, self.base_dir / "models" / f"{self.get_model_name()}.pth", device)
        dataset = CIFARPickleDataset(self.base_dir / "../test_datasets/CIFAR/test.pickle")
        self.Tester = ClassificationTester(self.model, build_loader(dataset, batch_size=256), device)

    def get_model_name(self):
        return "CIFAR"

    def get_model_params(self):
        return flatten_model_params(self.model)

    def reset_model_parameter(self, new_params):
        reset_model_from_vector(self.model, new_params)

    def save_model(self, save_file):
        save_state(self.model, Path(save_file))
