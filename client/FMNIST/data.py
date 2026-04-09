import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class FMNISTDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        data_path = Path(data_dir)
        self.images = self._read_images(data_path / "train-images-idx3-ubyte")
        self.labels = self._read_labels(data_path / "train-labels-idx1-ubyte")
        if len(self.images) != len(self.labels):
            raise ValueError("image count does not match label count")

    def _read_images(self, image_path):
        raw = image_path.read_bytes()
        if len(raw) < 16:
            raise ValueError(f"invalid image file: {image_path}")
        magic, count, rows, cols = struct.unpack(">IIII", raw[:16])
        if magic != 2051:
            raise ValueError(f"unexpected image magic: {magic}")
        expected_size = 16 + count * rows * cols
        if len(raw) != expected_size:
            raise ValueError(f"image file size mismatch: expected {expected_size}, got {len(raw)}")

        images = np.frombuffer(raw, dtype=np.uint8, offset=16).copy()
        images = torch.from_numpy(images).to(torch.float32).view(count, 1, rows, cols)
        return images / 255.0

    def _read_labels(self, label_path):
        raw = label_path.read_bytes()
        if len(raw) < 8:
            raise ValueError(f"invalid label file: {label_path}")
        magic, count = struct.unpack(">II", raw[:8])
        if magic != 2049:
            raise ValueError(f"unexpected label magic: {magic}")
        if len(raw) != 8 + count:
            raise ValueError(f"label file size mismatch: expected {8 + count}, got {len(raw)}")

        labels = np.frombuffer(raw, dtype=np.uint8, offset=8).copy()
        return torch.from_numpy(labels).to(torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


class data_factory:
    def __init__(self, data_dir, config):
        self.data_set = FMNISTDataset(data_dir)
        self.config = config
        self.sample_length = len(self.data_set)

    def get_dataset(self):
        return DataLoader(
            self.data_set,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )
