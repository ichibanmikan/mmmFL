from pathlib import Path
import struct

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from pickle_compat import load_pickle_file
except ModuleNotFoundError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from pickle_compat import load_pickle_file


def flatten_model_params(model):
    params = []
    for param in model.parameters():
        params.extend(param.view(-1).detach().cpu().numpy())
    return np.array(params)


def reset_model_from_vector(model, new_params):
    offset = 0
    with torch.no_grad():
        for param in model.parameters():
            numel = int(param.numel())
            values = new_params[offset:offset + numel].astype(float)
            reshaped = torch.tensor(values, dtype=param.dtype).view(param.shape)
            param.copy_(reshaped.to(param.device))
            offset += numel


def load_state_if_exists(model, load_path, device):
    if load_path.exists():
        model.load_state_dict(torch.load(load_path, map_location=device, weights_only=True))
        model.to(device)


def save_state(model, save_path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(state_dict, save_path)


class IDXImageDataset(Dataset):
    def __init__(self, image_path, label_path):
        raw_images = Path(image_path).read_bytes()
        magic, count, rows, cols = struct.unpack(">IIII", raw_images[:16])
        if magic != 2051:
            raise ValueError(f"unexpected image magic: {magic}")

        raw_labels = Path(label_path).read_bytes()
        label_magic, label_count = struct.unpack(">II", raw_labels[:8])
        if label_magic != 2049:
            raise ValueError(f"unexpected label magic: {label_magic}")
        if count != label_count:
            raise ValueError(f"image count {count} != label count {label_count}")

        self.images = torch.from_numpy(
            np.frombuffer(raw_images, dtype=np.uint8, offset=16).copy()
        ).to(torch.float32).view(count, 1, rows, cols) / 255.0
        self.labels = torch.from_numpy(
            np.frombuffer(raw_labels, dtype=np.uint8, offset=8).copy()
        ).to(torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


class CIFARPickleDataset(Dataset):
    def __init__(self, pickle_path):
        payload = load_pickle_file(pickle_path)
        self.images = torch.as_tensor(payload["x_test"], dtype=torch.float32)
        self.labels = torch.as_tensor(payload["y_test"], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


class ClassificationTester:
    def __init__(self, model, test_loader, device):
        self.model = model
        self.test_loader = test_loader
        self.device = device

    def test(self):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                predicted = outputs.argmax(dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        return 100.0 * correct / total if total else 0.0


def build_loader(dataset, batch_size=128, num_workers=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
